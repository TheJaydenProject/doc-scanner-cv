<script setup lang="ts">
import { ref } from "vue";
import ScanPanel from "./components/ScanPanel.vue";
import ResultPanel from "./components/ResultPanel.vue";
import Dashboard from "./components/Dashboard.vue";
import ExamplesPanel from "./components/ExamplesPanel.vue";
import type { ScanResult } from "./types";

const result = ref<ScanResult | null>(null);
const file = ref<File | null>(null);
const dashboardRef = ref<InstanceType<typeof Dashboard> | null>(null);
const rightView = ref<"examples" | "results">("examples");

function onScanComplete(scanResult: ScanResult, scanFile: File) {
  result.value = scanResult;
  file.value = scanFile;
  rightView.value = "results";
  dashboardRef.value?.refresh();
}
</script>

<template>
  <header class="topnav">
    <div class="nav-brand">
      <svg
        class="brand-mark"
        width="18"
        height="18"
        viewBox="0 0 20 20"
        fill="none"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linecap="round"
        aria-hidden="true"
      >
        <path d="M2 7V2h5" />
        <path d="M13 2h5v5" />
        <path d="M18 13v5h-5" />
        <path d="M7 18H2v-5" />
      </svg>
      <span>Doc Scanner CV</span>
    </div>
    <nav class="nav-tabs">
      <button
        type="button"
        class="nav-tab"
        :class="{ active: rightView === 'examples' }"
        @click="rightView = 'examples'"
      >
        Examples
      </button>
      <button
        type="button"
        class="nav-tab"
        :class="{ active: rightView === 'results' }"
        @click="rightView = 'results'"
      >
        Results
      </button>
      <a class="nav-tab" href="/docs">API Docs</a>
    </nav>
    <a
      class="nav-cta"
      href="https://github.com/TheJaydenProject/doc-scanner-cv"
      target="_blank"
      rel="noopener"
    >
      GitHub
    </a>
  </header>
  <main class="app-shell">
    <aside class="panel-left">
      <ScanPanel @scan-complete="onScanComplete" />
      <Dashboard ref="dashboardRef" />
    </aside>
    <section class="panel-right">
      <ExamplesPanel v-show="rightView === 'examples'" />
      <ResultPanel v-show="rightView === 'results'" :result="result" :file="file" />
    </section>
  </main>
</template>
