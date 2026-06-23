<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from "vue";
import type { ScanResult } from "../types";

const POLL_INTERVAL_MS = 4000;
// 120s budget (was 60s) — the VPS's CPU-only OCR routinely overruns the old
// budget on larger/handwritten scans, which killed otherwise-successful jobs.
const POLL_MAX_ATTEMPTS = 30;
// Matches app.py's MAX_CONTENT_LENGTH — checked client-side so an oversized
// file is rejected instantly instead of round-tripping to the server first.
const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;
// Matches the backend's resolution-gate message (api/documents.py) — this
// error is deterministic on retry, so the file selection gets cleared
// instead of just re-enabling the Scan button like other failures.
const RESOLUTION_ERROR_SIGNATURE = "too small to scan accurately";
// A scan runs server-side and outlives a page refresh, but the frontend's
// in-memory state does not — so the running job's id is parked here and
// reclaimed on mount. Without this, a refresh orphans the job: the backend
// still has the IP marked busy, so the next scan is rejected with "a scan is
// already in progress" until that job finishes on its own.
const ACTIVE_SCAN_KEY = "docScannerActiveScan";

interface StoredScan {
  jobId: string;
  startTime: number;
  estimatedDurationMs: number;
}

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
const OVERRUN_STEP_MS = 15000;

// The backend's average processing time is assumed to correspond to a scan of
// roughly this upload size; the estimate is scaled linearly by how the actual
// file compares. File bytes are only a rough proxy for OCR cost (which tracks
// image resolution and text density, not raw size), so the multiplier is
// clamped to keep an unusual file from producing a wild ETA.
const REFERENCE_FILE_SIZE_BYTES = 3 * 1024 * 1024;
const MIN_SIZE_MULTIPLIER = 0.5;
const MAX_SIZE_MULTIPLIER = 2;

// The global average behind baseAverageMs is dominated by fast, non-upscaled
// scans, so it badly underestimates a scan that hits the upscale path (see
// upscale.md) — that path runs roughly this much longer. Applied once, the
// moment the backend reports job.stage === "upscaling", instead of baked into
// every estimate up front.
const UPSCALE_ESTIMATE_MULTIPLIER = 3;

const emit = defineEmits<{
  "scan-complete": [result: ScanResult, file: File | null];
}>();

const fileInput = ref<HTMLInputElement | null>(null);
const selectedFileName = ref("");
const previewUrl = ref<string | null>(null);
const loading = ref(false);
const error = ref("");
const scanDisabled = ref(true);
const progressPercent = ref(0);
// Phase 1 (within estimate): a countdown toward the estimated finish.
const secondsRemaining = ref(0);
// Phase 2 (past the estimate): a stopwatch of total time so far. Once the
// estimate is provably wrong we can't honestly predict what's left, so we
// stop counting down (a wrong countdown can only freeze or reset, both of
// which look broken) and switch to elapsed time, which only ever increases —
// never resets, never freezes, never lies about being almost done.
const secondsElapsed = ref(0);
const inOverrun = ref(false);

const progressLabel = computed(() => {
  // At completion the poll sets 100%; drop the time tail there.
  if (progressPercent.value >= 100) return "Processing… 100%";
  // Single labeled handoff: "≈Ns left" while the estimate holds, then
  // "Ns elapsed" once it doesn't. The label changing alongside the number is
  // what makes the one transition read as a different metric, not a reset.
  if (inOverrun.value) {
    return `Processing… ${progressPercent.value}% · ${secondsElapsed.value}s elapsed`;
  }
  return `Processing… ${progressPercent.value}% · ≈${secondsRemaining.value}s left`;
});

// The in-flight job's id, exposed so the Stop button knows what to cancel.
const activeJobId = ref<string | null>(null);

// Average processing time from /metrics, before per-file size scaling.
let baseAverageMs = DEFAULT_ESTIMATE_MS;
// Size-scaled estimate for the current scan; set in startProgressTimer().
let estimatedDurationMs = DEFAULT_ESTIMATE_MS;
let progressTimer: ReturnType<typeof setInterval> | null = null;
// Module-scoped (not local to pollJob) so Stop and resume can clear it too.
let pollInterval: ReturnType<typeof setInterval> | null = null;
let scanStartTime = 0;
// Guards rescaleEstimateForUpscale() to run at most once per scan.
let rescaledForUpscale = false;

function saveActiveScan(jobId: string) {
  try {
    const stored: StoredScan = { jobId, startTime: scanStartTime, estimatedDurationMs };
    localStorage.setItem(ACTIVE_SCAN_KEY, JSON.stringify(stored));
  } catch {
    // Private-mode / disabled storage: resume-on-refresh just won't work.
  }
}

function clearActiveScan() {
  try {
    localStorage.removeItem(ACTIVE_SCAN_KEY);
  } catch {
    // No-op — nothing to clear if storage is unavailable.
  }
}

function readActiveScan(): StoredScan | null {
  try {
    const raw = localStorage.getItem(ACTIVE_SCAN_KEY);
    return raw ? (JSON.parse(raw) as StoredScan) : null;
  } catch {
    return null;
  }
}

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

  // After metrics so a resumed scan that overruns can still lean on a real
  // average if it ever needs a fresh estimate.
  const stored = readActiveScan();
  if (stored) await resumeScan(stored);
});

onUnmounted(() => {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
  }
});

// Reconnect to a job left running by a previous page load.
async function resumeScan(stored: StoredScan): Promise<void> {
  try {
    const res = await fetch(`/api/documents/jobs/${stored.jobId}`);
    if (!res.ok) {
      // 404: job already evicted from the store. Nothing to resume.
      clearActiveScan();
      return;
    }
    const job = await res.json();

    if (job.status === "complete") {
      clearActiveScan();
      emit("scan-complete", job.result as ScanResult, null);
      return;
    }
    if (job.status === "processing") {
      loading.value = true;
      scanDisabled.value = true;
      activeJobId.value = stored.jobId;
      runProgressTimer(stored.startTime, stored.estimatedDurationMs);
      if (job.stage === "upscaling") rescaleEstimateForUpscale();
      pollJob(stored.jobId, null);
      return;
    }
    // failed / cancelled / anything else: nothing to show, just clear.
    clearActiveScan();
  } catch {
    // Network error reaching the server — don't strand the user in a
    // loading state; let them start fresh.
    clearActiveScan();
  }
}

function startProgressTimer(fileSizeBytes: number) {
  const sizeMultiplier = Math.min(
    Math.max(fileSizeBytes / REFERENCE_FILE_SIZE_BYTES, MIN_SIZE_MULTIPLIER),
    MAX_SIZE_MULTIPLIER,
  );
  const estimate = baseAverageMs * sizeMultiplier * OVERESTIMATE_FACTOR;
  runProgressTimer(Date.now(), estimate);
}

// Drives the progress bar from a start time + estimate. Split out from
// startProgressTimer so a resumed scan can pass the original (stored) start
// time, keeping the bar/elapsed continuous across a refresh.
function runProgressTimer(startTime: number, estimateMs: number) {
  scanStartTime = startTime;
  estimatedDurationMs = estimateMs;
  inOverrun.value = false;
  rescaledForUpscale = false;

  const tick = () => {
    const elapsed = Date.now() - scanStartTime;

    if (elapsed < estimatedDurationMs) {
      // Phase 1 — estimate still holds.
      // Percent: linear rise from 0 to the soft cap over the estimated duration.
      const ratio = elapsed / estimatedDurationMs;
      progressPercent.value = Math.round(ratio * PROGRESS_SOFT_CAP);
      // ETA: honest countdown toward the estimated finish.
      secondsRemaining.value = Math.ceil((estimatedDurationMs - elapsed) / 1000);
    } else {
      // Phase 2 — estimate was too low.
      inOverrun.value = true;
      // Percent: overrun/(overrun+step) starts at 0 right where the linear
      // phase left off (no jump) and keeps rising toward 1 — slowly enough to
      // stay incremental well past a minute of overrun, never reaching the
      // ceiling.
      const overrun = elapsed - estimatedDurationMs;
      const headroom = PROGRESS_CEILING - PROGRESS_SOFT_CAP;
      progressPercent.value = Math.round(
        PROGRESS_SOFT_CAP + headroom * (overrun / (overrun + OVERRUN_STEP_MS)),
      );
      // Time: count total elapsed up. Monotonic by construction, so it can
      // never reset, and it ticks every second, so it can never look frozen
      // even while percent is creeping slowly near the ceiling.
      secondsElapsed.value = Math.floor(elapsed / 1000);
    }
  };

  tick(); // Render immediately so a resume doesn't flash 0% for one tick.
  progressTimer = setInterval(tick, PROGRESS_TICK_MS);
}

// Widens the estimate once the backend reports it has entered the slow
// upscale path. Skipped once already past the original estimate — by then
// the overrun crawl already owns the percent, and widening it now would walk
// percent backwards (Phase 1's ratio = elapsed / newEstimate would compute
// lower than where the overrun crawl already is). In practice the gate fires
// within the first few seconds, long before that's a concern.
function rescaleEstimateForUpscale() {
  if (rescaledForUpscale || inOverrun.value) return;
  rescaledForUpscale = true;
  estimatedDurationMs *= UPSCALE_ESTIMATE_MULTIPLIER;
}

function stopProgressTimer() {
  if (progressTimer !== null) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function stopPolling() {
  if (pollInterval !== null) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}

function onFileChange(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];

  if (!file) {
    selectedFileName.value = "";
    scanDisabled.value = true;
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value);
      previewUrl.value = null;
    }
    return;
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    error.value = `Selected file is ${sizeMb}MB — please choose an image under 20MB.`;
    selectedFileName.value = "";
    scanDisabled.value = true;
    target.value = "";
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value);
      previewUrl.value = null;
    }
    return;
  }

  selectedFileName.value = file.name;
  scanDisabled.value = false;
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
  }
  previewUrl.value = URL.createObjectURL(file);
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

  activeJobId.value = jobId;
  // Park the job so a refresh can reconnect instead of orphaning it.
  saveActiveScan(jobId);
  pollJob(jobId, file);
}

function cancelJob(jobId: string) {
  // Fire-and-forget: lets the backend stop an abandoned job at its next
  // checkpoint instead of burning CPU/network to a result no one is waiting
  // for. Best-effort — a failed cancel just means the job runs to completion.
  fetch(`/api/documents/jobs/${jobId}`, { method: "DELETE" }).catch(() => {});
}

function pollJob(jobId: string, file: File | null): void {
  let attempts = 0;
  stopPolling(); // never run two poll loops at once (e.g. resume + a stray)

  pollInterval = setInterval(async () => {
    attempts++;

    if (attempts > POLL_MAX_ATTEMPTS) {
      stopPolling();
      cancelJob(jobId);
      clearActiveScan();
      activeJobId.value = null;
      showError("Processing timed out. Please try again.");
      return;
    }

    try {
      const res = await fetch(`/api/documents/jobs/${jobId}`);
      const job = await res.json();

      if (job.status === "processing") {
        if (job.stage === "upscaling") rescaleEstimateForUpscale();
      } else if (job.status === "complete") {
        stopPolling();
        stopProgressTimer();
        clearActiveScan();
        activeJobId.value = null;
        progressPercent.value = 100;
        secondsRemaining.value = 0;
        setTimeout(() => {
          loading.value = false;
          scanDisabled.value = false;
          emit("scan-complete", job.result as ScanResult, file);
        }, 200);
      } else if (job.status === "cancelled") {
        // Cancelled elsewhere (e.g. this job was Stopped in another tab).
        // It was intentional, so reset quietly without an error banner.
        resetToIdle();
      } else if (job.status === "failed") {
        stopPolling();
        clearActiveScan();
        activeJobId.value = null;
        showError(job.error ?? "Processing failed.");
      }
    } catch {
      stopPolling();
      clearActiveScan();
      activeJobId.value = null;
      showError("Network error while polling for result.");
    }
  }, POLL_INTERVAL_MS);
}

function onStop() {
  const jobId = activeJobId.value;
  if (jobId) cancelJob(jobId);
  resetToIdle();
}

// Tear down all scan state and return the panel to its pre-scan idle state.
function resetToIdle() {
  stopPolling();
  stopProgressTimer();
  clearActiveScan();
  activeJobId.value = null;
  loading.value = false;
  scanDisabled.value = false;
  error.value = "";
}

function showError(message: string) {
  stopProgressTimer();
  loading.value = false;
  error.value = message;

  if (message.includes(RESOLUTION_ERROR_SIGNATURE)) {
    selectedFileName.value = "";
    if (fileInput.value) fileInput.value.value = "";
    scanDisabled.value = true;
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value);
      previewUrl.value = null;
    }
  } else {
    scanDisabled.value = false;
  }
}
</script>

<template>
  <section class="panel">
    <div class="upload-layout">
      <div class="upload-controls">
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

        <button
          id="scan-btn"
          :class="{ 'is-cancel': loading }"
          :disabled="!loading && scanDisabled"
          @click="loading ? onStop() : onScan()"
        >
          {{ loading ? "Cancel" : "Scan Document" }}
        </button>
        <div v-if="loading" class="scan-progress" id="loading" role="status" aria-live="polite">
          <span>{{ progressLabel }}</span>
          <div class="progress-track">
            <div class="progress-bar" :style="{ width: progressPercent + '%' }"></div>
          </div>
        </div>
      </div>
      <div v-if="previewUrl" class="upload-preview">
        <img :src="previewUrl" alt="Selected image preview" />
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

<style scoped>
.upload-layout {
  display: flex;
  gap: var(--space-6);
  align-items: flex-start;
  max-width: 640px;
}

.upload-controls {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  flex: 1 1 auto;
  min-width: 0;
}

.upload-preview {
  flex: 0 0 auto;
  width: 180px;
  height: 180px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

@media (max-width: 768px) {
  .upload-layout {
    flex-direction: column;
  }
}
</style>
