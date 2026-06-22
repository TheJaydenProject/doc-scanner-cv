<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import type { ScanResult } from "../types";

const POLL_INTERVAL_MS = 1500;
const POLL_MAX_ATTEMPTS = 40;
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
// Never claim "done" from the local timer alone — only the poll result can do that.
const PROGRESS_CAP_PERCENT = 95;

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
  const base = `Processing… ${progressPercent.value}%`;
  return secondsRemaining.value > 0
    ? `${base} · ≈${secondsRemaining.value}s left`
    : base;
});

// Average processing time from /metrics, before per-file size scaling.
let baseAverageMs = DEFAULT_ESTIMATE_MS;
// Size-scaled estimate for the current scan; set in startProgressTimer().
let estimatedDurationMs = DEFAULT_ESTIMATE_MS;
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
  estimatedDurationMs = baseAverageMs * sizeMultiplier;

  scanStartTime = Date.now();
  progressPercent.value = 0;
  secondsRemaining.value = Math.ceil(estimatedDurationMs / 1000);

  progressTimer = setInterval(() => {
    const elapsed = Date.now() - scanStartTime;
    const ratio = Math.min(elapsed / estimatedDurationMs, 1);
    progressPercent.value = Math.min(
      Math.round(ratio * 100),
      PROGRESS_CAP_PERCENT,
    );
    secondsRemaining.value = Math.max(
      0,
      Math.ceil((estimatedDurationMs - elapsed) / 1000),
    );
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

async function pollJob(jobId: string, file: File): Promise<void> {
  let attempts = 0;

  const interval = setInterval(async () => {
    attempts++;

    if (attempts > POLL_MAX_ATTEMPTS) {
      clearInterval(interval);
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
