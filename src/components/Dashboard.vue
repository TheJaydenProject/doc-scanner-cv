<script setup lang="ts">
import { ref, onMounted } from "vue";
import Chart from "chart.js/auto";

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

const totalScans = ref("—");
const avgTime = ref("—");
const avgChars = ref("—");
const charCountCanvas = ref<HTMLCanvasElement | null>(null);
const processingTimeCanvas = ref<HTMLCanvasElement | null>(null);

let charCountChart: Chart | null = null;
let processingTimeChart: Chart | null = null;

async function refresh() {
  try {
    const res = await fetch("/api/documents/metrics");
    const data: MetricsResponse = await res.json();

    totalScans.value = String(data.total_scans);
    avgTime.value = `${data.avg_processing_time_ms}ms`;
    avgChars.value = String(data.avg_char_count);

    const fullLabels = data.recent.map((r) => r.filename);
    const labels = fullLabels.map((name) =>
      name.length > 10 ? name.slice(0, 10) + "..." : name,
    );
    const charCounts = data.recent.map((r) => r.char_count);
    const processingTimes = data.recent.map((r) => r.processing_time_ms);

    charCountChart?.destroy();
    processingTimeChart?.destroy();

    charCountChart = new Chart(charCountCanvas.value!, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Character Count",
            data: charCounts,
            backgroundColor: "#ffffff",
            borderRadius: 2,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: "Character Count — Recent Scans",
            color: "#ffffff",
            font: { size: 12, weight: 600 },
            padding: { bottom: 10 },
          },
          tooltip: {
            callbacks: {
              title: (items) => fullLabels[items[0].dataIndex],
            },
          },
        },
        scales: {
          x: { ticks: { color: "#666666" }, grid: { color: "#1f1f1f" } },
          y: { ticks: { color: "#666666" }, grid: { color: "#1f1f1f" } },
        },
      },
    });

    processingTimeChart = new Chart(processingTimeCanvas.value!, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Processing Time (ms)",
            data: processingTimes,
            borderColor: "#ffffff",
            backgroundColor: "transparent",
            tension: 0,
            pointRadius: 3,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: "Processing Time (ms) — Recent Scans",
            color: "#ffffff",
            font: { size: 12, weight: 600 },
            padding: { bottom: 10 },
          },
          tooltip: {
            callbacks: {
              title: (items) => fullLabels[items[0].dataIndex],
            },
          },
        },
        scales: {
          x: { ticks: { color: "#666666" }, grid: { color: "#1f1f1f" } },
          y: { ticks: { color: "#666666" }, grid: { color: "#1f1f1f" } },
        },
      },
    });
  } catch {
    // Dashboard failure is non-critical — main scan result is already shown.
    console.error("Failed to load dashboard metrics.");
  }
}

defineExpose({ refresh });

onMounted(refresh);
</script>

<template>
  <section class="panel" id="dashboard">
    <h2>Scan History</h2>
    <div id="stats-row">
      <div class="stat-card">
        <span class="stat-label">Total Scans</span>
        <span class="stat-value">{{ totalScans }}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Avg Processing Time</span>
        <span class="stat-value">{{ avgTime }}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Avg Characters</span>
        <span class="stat-value">{{ avgChars }}</span>
      </div>
    </div>
    <div id="charts-row">
      <div class="chart-box"><canvas ref="charCountCanvas"></canvas></div>
      <div class="chart-box"><canvas ref="processingTimeCanvas"></canvas></div>
    </div>
  </section>
</template>
