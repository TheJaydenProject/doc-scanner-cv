<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import type { ScanResult } from "../types";

const POLL_INTERVAL_MS = 1500;
// 120s budget (was 60s) — the VPS's CPU-only OCR routinely overruns the old
// budget on larger/handwritten scans, which killed otherwise-successful jobs.
const POLL_MAX_ATTEMPTS = 80;
// Matches app.py's MAX_CONTENT_LENGTH — checked client-side so an oversized
// file is rejected instantly instead of round-tripping to the server first.
const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;
// Matches the backend's resolution-gate message (api/documents.py) — this
// error is deterministic on retry, so the file selection gets cleared
// instead of just re-enabling the Scan button like other failures.
const RESOLUTION_ERROR_SIGNATURE = "too small to scan accurately";

// Used until /api/documents/metrics reports a real average (e.g. first scan
// after a fresh deploy, with no scan history yet).
const DEFAULT_ESTIMATE_MS = 8000;
const PROGRESS_TICK_MS = 150;
// We can't know up front whether the server will upscale this scan (decided
// after detection) or how long OCR will take, so we deliberately lean the
// estimate long: better to rise gradually and finish early (snap to 100) than
// to arrive early and stall. This is the main tuning knob — raise it if scans
// still routinely overrun.
const OVERESTIMATE_FACTOR = 2;
// The linear phase rises to this by the estimated finish time; only the poll
// result ever sets 100%, so the timer alone never claims completion.
const PROGRESS_SOFT_CAP = 90;
// If a scan still overruns the leaned estimate, the bar crawls from the soft
// cap toward this ceiling — it keeps inching up instead of freezing, but
// never reaches 100%.
const PROGRESS_CEILING = 99;
// Overrun crawl uses overrun/(overrun+OVERRUN_STEP_MS) rather than an
// exponential: a 1/x tail decays much slower than e^-x, so it keeps producing
// visible whole-percent ticks for minutes into an overrun instead of
// saturating near PROGRESS_CEILING within ~25s and then sitting frozen.
// Also sizes the ETA's renewal chunk (see startProgressTimer) — same "how
// big a step feels reasonable" question for both.
const OVERRUN_STEP_MS = 15000;

// The backend's average processing time is assumed to correspond to a scan of
// roughly this upload size; the estimate is scaled linearly by how the actual
// file compares. File bytes are only a rough proxy for OCR cost (which tracks
// image resolution and text density, not raw size), so the multiplier is
// clamped to keep an unusual file from producing a wild ETA.
const REFERENCE_FILE_SIZE_BYTES = 3 * 1024 * 1024;
const MIN_SIZE_MULTIPLIER = 0.5;
const MAX_SIZE_MULTIPLIER = 2;

const emit = defineEmits<{
  "scan-complete": [result: ScanResult, file: File];
}>();

const fileInput = ref<HTMLInputElement | null>(null);
const selectedFileName = ref("");
const loading = ref(false);
const error = ref("");
const scanDisabled = ref(true);
const progressPercent = ref(0);
const secondsRemaining = ref(0);

const progressLabel = computed(() => {
  // At completion the poll sets 100% / 0s; drop the "0s left" tail there.
  if (progressPercent.value >= 100) return "Processing… 100%";
  return `Processing… ${progressPercent.value}% · ≈${secondsRemaining.value}s left`;
});

// Average processing time from /metrics, before per-file size scaling.
let baseAverageMs = DEFAULT_ESTIMATE_MS;
// Size-scaled estimate for the current scan; set in startProgressTimer().
let estimatedDurationMs = DEFAULT_ESTIMATE_MS;
// Independent of the percent curve below: the wall-clock moment the ETA
// countdown is currently aimed at. Pushed back by OVERRUN_STEP_MS whenever
// elapsed time catches up to it but the job is still running, so the
// countdown always shows a real number ticking down instead of one that's
// forced toward zero just because percent looks close to the ceiling.
let etaTargetMs = DEFAULT_ESTIMATE_MS;
let progressTimer: ReturnType<typeof setInterval> | null = null;
let scanStartTime = 0;

onMounted(async () => {
  try {
    const res = await fetch("/api/documents/metrics");
    const data: { total_scans: number; avg_processing_time_ms: number } =
      await res.json();
    if (data.total_scans > 0 && data.avg_processing_time_ms > 0) {
      baseAverageMs = data.avg_processing_time_ms;
    }
  } catch {
    // Keep DEFAULT_ESTIMATE_MS — the progress bar still works, just less precisely.
  }
});

function startProgressTimer(fileSizeBytes: number) {
  const sizeMultiplier = Math.min(
    Math.max(fileSizeBytes / REFERENCE_FILE_SIZE_BYTES, MIN_SIZE_MULTIPLIER),
    MAX_SIZE_MULTIPLIER,
  );
  estimatedDurationMs = baseAverageMs * sizeMultiplier * OVERESTIMATE_FACTOR;
  etaTargetMs = estimatedDurationMs;

  scanStartTime = Date.now();
  progressPercent.value = 0;
  secondsRemaining.value = Math.ceil(estimatedDurationMs / 1000);

  progressTimer = setInterval(() => {
    const elapsed = Date.now() - scanStartTime;

    // Percent: whole numbers only, one curve, no phase-boundary jump.
    if (elapsed < estimatedDurationMs) {
      // Linear phase: rise from 0 to the soft cap over the estimated duration.
      const ratio = elapsed / estimatedDurationMs;
      progressPercent.value = Math.round(ratio * PROGRESS_SOFT_CAP);
    } else {
      // Overrun phase: estimate was too low. overrun/(overrun+step) starts at
      // 0 right where the linear phase left off (no jump) and keeps rising
      // toward 1 — slowly enough to stay incremental well past a minute of
      // overrun, but never actually reaching the ceiling.
      const overrun = elapsed - estimatedDurationMs;
      const headroom = PROGRESS_CEILING - PROGRESS_SOFT_CAP;
      progressPercent.value = Math.round(
        PROGRESS_SOFT_CAP + headroom * (overrun / (overrun + OVERRUN_STEP_MS)),
      );
    }

    // ETA: a separate, honest countdown against etaTargetMs. Deliberately not
    // derived from the percent curve above, so it never gets dragged toward
    // "1s left" just because percent is sitting near the ceiling — instead it
    // counts down normally and, if the job is still running when it would
    // hit zero, renews with another real step rather than freezing at zero.
    while (elapsed >= etaTargetMs) {
      etaTargetMs += OVERRUN_STEP_MS;
    }
    secondsRemaining.value = Math.ceil((etaTargetMs - elapsed) / 1000);
  }, PROGRESS_TICK_MS);
}

function stopProgressTimer() {
  if (progressTimer !== null) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function onFileChange(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];

  if (!file) {
    selectedFileName.value = "";
    scanDisabled.value = true;
    return;
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    error.value = `Selected file is ${sizeMb}MB — please choose an image under 20MB.`;
    selectedFileName.value = "";
    scanDisabled.value = true;
    target.value = "";
    return;
  }

  selectedFileName.value = file.name;
  scanDisabled.value = false;
}

async function onScan() {
  const file = fileInput.value?.files?.[0];
  if (!file) return;

  loading.value = true;
  error.value = "";
  scanDisabled.value = true;
  startProgressTimer(file.size);

  const formData = new FormData();
  formData.append("file", file);

  let jobId: string;

  try {
    const res = await fetch("/api/documents/scan", {
      method: "POST",
      body: formData,
    });

    let data: { error?: string; job_id?: string };
    try {
      data = await res.json();
    } catch {
      // Some failure responses (e.g. a 413 from Flask's MAX_CONTENT_LENGTH
      // guard) come back as an HTML error page, not JSON — res.json() throws.
      // That's a real server response, not a network failure, so it gets its
      // own message instead of falling into the generic catch below.
      showError(
        res.status === 413
          ? "File is too large. Please choose an image under 20MB."
          : `Upload failed (server returned HTTP ${res.status}).`,
      );
      return;
    }

    if (!res.ok) {
      showError(data.error ?? "Upload failed.");
      return;
    }
    jobId = data.job_id as string;
  } catch {
    showError("Network error. Could not reach the server.");
    return;
  }

  await pollJob(jobId, file);
}

function cancelJob(jobId: string) {
  // Fire-and-forget: lets the backend stop an abandoned job at its next
  // checkpoint instead of burning CPU/network to a result no one is waiting
  // for. Best-effort — a failed cancel just means the job runs to completion.
  fetch(`/api/documents/jobs/${jobId}`, { method: "DELETE" }).catch(() => {});
}

async function pollJob(jobId: string, file: File): Promise<void> {
  let attempts = 0;

  const interval = setInterval(async () => {
    attempts++;

    if (attempts > POLL_MAX_ATTEMPTS) {
      clearInterval(interval);
      cancelJob(jobId);
      showError("Processing timed out. Please try again.");
      return;
    }

    try {
      const res = await fetch(`/api/documents/jobs/${jobId}`);
      const job = await res.json();

      if (job.status === "complete") {
        clearInterval(interval);
        stopProgressTimer();
        progressPercent.value = 100;
        secondsRemaining.value = 0;
        setTimeout(() => {
          loading.value = false;
          scanDisabled.value = false;
          emit("scan-complete", job.result as ScanResult, file);
        }, 200);
      } else if (job.status === "failed") {
        clearInterval(interval);
        showError(job.error ?? "Processing failed.");
      }
    } catch {
      clearInterval(interval);
      showError("Network error while polling for result.");
    }
  }, POLL_INTERVAL_MS);
}

function showError(message: string) {
  stopProgressTimer();
  loading.value = false;
  error.value = message;

  if (message.includes(RESOLUTION_ERROR_SIGNATURE)) {
    selectedFileName.value = "";
    if (fileInput.value) fileInput.value.value = "";
    scanDisabled.value = true;
  } else {
    scanDisabled.value = false;
  }
}
</script>

<template>
  <section class="panel">
    <h2>Upload</h2>
    <label for="file-input">Choose image (JPEG or PNG, max 20MB)</label>

    <input
      ref="fileInput"
      type="file"
      id="file-input"
      class="sr-only"
      accept="image/jpeg,image/png"
      @change="onFileChange"
    />

    <div class="file-pick-group">
      <button type="button" class="file-pick-btn" @click="fileInput?.click()">
        {{ selectedFileName || "Choose File" }}
      </button>
      <p v-if="selectedFileName" class="file-hint">Click to choose a different file</p>
    </div>

    <button id="scan-btn" :disabled="scanDisabled" @click="onScan">Scan Document</button>
    <div v-if="loading" class="scan-progress" id="loading" role="status" aria-live="polite">
      <span>{{ progressLabel }}</span>
      <div class="progress-track">
        <div class="progress-bar" :style="{ width: progressPercent + '%' }"></div>
      </div>
    </div>
    <p v-if="error" class="error" id="error-msg" role="alert">
      <svg class="error-icon" viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <circle cx="10" cy="10" r="8.5" stroke="currentColor" stroke-width="1.3" />
        <line x1="10" y1="6" x2="10" y2="10.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" />
        <circle cx="10" cy="13.5" r="0.8" fill="currentColor" />
      </svg>
      <span>{{ error }}</span>
    </p>
  </section>
</template>
