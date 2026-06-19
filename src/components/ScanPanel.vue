<script setup lang="ts">
import { ref } from "vue";
import type { ScanResult } from "../types";

const POLL_INTERVAL_MS = 1500;
const POLL_MAX_ATTEMPTS = 40;
// Matches the backend's resolution-gate message (api/documents.py) — this
// error is deterministic on retry, so the file selection gets cleared
// instead of just re-enabling the Scan button like other failures.
const RESOLUTION_ERROR_SIGNATURE = "too small to scan accurately";

const emit = defineEmits<{
  "scan-complete": [result: ScanResult, file: File];
}>();

const fileInput = ref<HTMLInputElement | null>(null);
const selectedFileName = ref("");
const loading = ref(false);
const error = ref("");
const scanDisabled = ref(true);

function onFileChange(event: Event) {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    selectedFileName.value = target.files[0].name;
    scanDisabled.value = false;
  } else {
    selectedFileName.value = "";
    scanDisabled.value = true;
  }
}

async function onScan() {
  const file = fileInput.value?.files?.[0];
  if (!file) return;

  loading.value = true;
  error.value = "";
  scanDisabled.value = true;

  const formData = new FormData();
  formData.append("file", file);

  let jobId: string;

  try {
    const res = await fetch("/api/documents/scan", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error ?? "Upload failed.");
      return;
    }
    jobId = data.job_id;
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
        loading.value = false;
        scanDisabled.value = false;
        emit("scan-complete", job.result as ScanResult, file);
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
    <label for="file-input">Choose image (JPEG or PNG, max 2MB)</label>

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
      <span>Processing…</span>
      <div class="sweep-track">
        <div class="sweep-bar"></div>
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
