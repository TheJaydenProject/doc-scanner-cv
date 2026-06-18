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

function onScanComplete(scanResult: ScanResult, scanFile: File) {
  result.value = scanResult;
  file.value = scanFile;
  dashboardRef.value?.refresh();
}
</script>

<template>
  <header>
    <h1>Doc Scanner CV</h1>
    <nav>
      <a href="#examples">Examples</a>
      <a href="#upload-panel">Try It</a>
      <a href="/docs">API Docs</a>
    </nav>
  </header>
  <main>
    <ExamplesPanel />
    <ScanPanel @scan-complete="onScanComplete" />
    <ResultPanel :result="result" :file="file" />
    <Dashboard ref="dashboardRef" />
  </main>
</template>
