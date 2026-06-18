// Chart.js is loaded via CDN script tag — declare the global so tsc can type-check usage.
declare class Chart {
  constructor(canvas: HTMLCanvasElement, config: object);
  destroy(): void;
}

interface ScanResult {
  text: string;
  char_count: number;
  word_count: number;
  processing_time_ms: number;
  image_b64: string;
  detection_count: number;
  doc_type: string;
  doc_type_confidence: number;
}

interface MetricsResponse {
  total_scans: number;
  avg_processing_time_ms: number;
  avg_char_count: number;
  recent: Array<{
    filename: string;
    char_count: number;
    processing_time_ms: number;
    created_at: string;
  }>;
}

const fileInput = document.getElementById("file-input") as HTMLInputElement;
const scanBtn = document.getElementById("scan-btn") as HTMLButtonElement;
const loadingEl = document.getElementById("loading") as HTMLParagraphElement;
const errorMsg = document.getElementById("error-msg") as HTMLParagraphElement;
const resultsPanel = document.getElementById("results-panel") as HTMLElement;
const originalImg = document.getElementById("original-img") as HTMLImageElement;
const scannedImg = document.getElementById("scanned-img") as HTMLImageElement;
const extractedText = document.getElementById("extracted-text") as HTMLTextAreaElement;
const copyBtn = document.getElementById("copy-btn") as HTMLButtonElement;
const totalScansEl = document.getElementById("total-scans") as HTMLSpanElement;
const avgTimeEl = document.getElementById("avg-time") as HTMLSpanElement;
const avgCharsEl = document.getElementById("avg-chars") as HTMLSpanElement;
const charCountCanvas = document.getElementById("char-count-chart") as HTMLCanvasElement;
const processingTimeCanvas = document.getElementById("processing-time-chart") as HTMLCanvasElement;
const detectionCountEl = document.getElementById("detection-count") as HTMLSpanElement;
const docTypeEl = document.getElementById("doc-type") as HTMLSpanElement;
const docTypeConfidenceEl = document.getElementById("doc-type-confidence") as HTMLSpanElement;

// Kept in module scope so they can be destroyed before re-render.
// Re-instantiating without destroying causes Chart.js to stack invisible instances.
let charCountChart: Chart | null = null;
let processingTimeChart: Chart | null = null;

const POLL_INTERVAL_MS = 1500;
const POLL_MAX_ATTEMPTS = 40; // 40 * 1.5s = 60s timeout

fileInput.addEventListener("change", () => {
  scanBtn.disabled = fileInput.files === null || fileInput.files.length === 0;
});

scanBtn.addEventListener("click", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  scanDocument(file);
});

copyBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(extractedText.value);
});

async function scanDocument(file: File): Promise<void> {
  loadingEl.hidden = false;
  resultsPanel.hidden = true;
  errorMsg.hidden = true;
  errorMsg.textContent = "";
  scanBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  let jobId: string;

  try {
    const submitRes = await fetch("/api/documents/scan", {
      method: "POST",
      body: formData,
    });

    const submitData = await submitRes.json();

    if (!submitRes.ok) {
      showError(submitData.error ?? "Upload failed.");
      return;
    }

    jobId = submitData.job_id;
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
        loadingEl.hidden = true;
        scanBtn.disabled = false;
        renderResults(job.result as ScanResult, file);
        loadDashboard();
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

function renderResults(result: ScanResult, file: File): void {
  originalImg.src = URL.createObjectURL(file);
  scannedImg.src = `data:image/png;base64,${result.image_b64}`;
  extractedText.value = result.text;
  detectionCountEl.textContent = String(result.detection_count);
  docTypeEl.textContent = result.doc_type;
  docTypeConfidenceEl.textContent = `${(result.doc_type_confidence * 100).toFixed(1)}%`;
  resultsPanel.hidden = false;
}

async function loadDashboard(): Promise<void> {
  try {
    const res = await fetch("/api/documents/metrics");
    const data: MetricsResponse = await res.json();

    totalScansEl.textContent = String(data.total_scans);
    avgTimeEl.textContent = `${data.avg_processing_time_ms}ms`;
    avgCharsEl.textContent = String(data.avg_char_count);

    const labels = data.recent.map((r) => r.filename);
    const charCounts = data.recent.map((r) => r.char_count);
    const processingTimes = data.recent.map((r) => r.processing_time_ms);

    charCountChart?.destroy();
    processingTimeChart?.destroy();

    charCountChart = new Chart(charCountCanvas, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Character Count",
            data: charCounts,
            backgroundColor: "#2563eb",
            borderRadius: 2,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#888888" }, grid: { color: "#2a2a2a" } },
          y: { ticks: { color: "#888888" }, grid: { color: "#2a2a2a" } },
        },
      },
    });

    processingTimeChart = new Chart(processingTimeCanvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Processing Time (ms)",
            data: processingTimes,
            borderColor: "#2563eb",
            backgroundColor: "transparent",
            tension: 0,
            pointRadius: 3,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#888888" }, grid: { color: "#2a2a2a" } },
          y: { ticks: { color: "#888888" }, grid: { color: "#2a2a2a" } },
        },
      },
    });
  } catch {
    // Dashboard failure is non-critical — main scan result is already shown.
    console.error("Failed to load dashboard metrics.");
  }
}

function showError(message: string): void {
  loadingEl.hidden = true;
  scanBtn.disabled = false;
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

loadDashboard();
