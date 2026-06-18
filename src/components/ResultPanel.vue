<script setup lang="ts">
import { ref, watch, computed } from "vue";
import type { ScanResult } from "../types";

const props = defineProps<{
  result: ScanResult | null;
  file: File | null;
}>();

const editableText = ref("");

watch(
  () => props.result,
  (newResult) => {
    editableText.value = newResult?.text ?? "";
  },
);

const originalSrc = computed(() =>
  props.file ? URL.createObjectURL(props.file) : "",
);

function copyText() {
  navigator.clipboard.writeText(editableText.value);
}
</script>

<template>
  <section v-if="result" class="panel" id="results-panel">
    <h2>Result</h2>
    <div id="image-comparison">
      <figure>
        <figcaption>Original</figcaption>
        <img :src="originalSrc" alt="Uploaded document" id="original-img" />
      </figure>
      <figure>
        <figcaption>Scanned (text regions highlighted)</figcaption>
        <img
          :src="`data:image/png;base64,${result.image_b64}`"
          alt="Processed document"
          id="scanned-img"
        />
      </figure>
    </div>
    <div id="scan-meta-row">
      <div class="stat-card">
        <span class="stat-label">Text Regions Detected</span>
        <span class="stat-value">{{ result.detection_count }}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Document Type</span>
        <span class="stat-value">{{ result.doc_type }}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Classifier Confidence</span>
        <span class="stat-value">{{ (result.doc_type_confidence * 100).toFixed(1) }}%</span>
      </div>
    </div>
    <label for="extracted-text">Extracted Text</label>
    <textarea id="extracted-text" rows="10" spellcheck="false" v-model="editableText"></textarea>
    <button id="copy-btn" @click="copyText">Copy Text</button>
  </section>
</template>
