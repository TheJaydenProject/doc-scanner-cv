<script setup lang="ts">
import { ref, watch, computed } from "vue";
import type { ScanResult } from "../types";
import Lightbox from "./Lightbox.vue";

type StageKey = "raw" | "warped";

const STAGES: { key: StageKey; label: string }[] = [
  { key: "raw", label: "Raw" },
  { key: "warped", label: "Warped" },
];

const props = defineProps<{
  result: ScanResult | null;
  file: File | null;
}>();

const editableText = ref("");
const lightboxSrc = ref("");
const lightboxVisible = ref(false);
const copied = ref(false);
const activeStage = ref<StageKey>("warped");

watch(
  () => props.result,
  (newResult) => {
    editableText.value = newResult?.text ?? "";
    activeStage.value = "warped";
  },
);

const originalSrc = computed(() =>
  props.file ? URL.createObjectURL(props.file) : "",
);

function stageSrc(key: StageKey): string {
  if (!props.result) return "";
  switch (key) {
    case "raw":
      return originalSrc.value;
    case "warped":
      return `data:image/png;base64,${props.result.warped_image_b64}`;
  }
}

function stageCaption(key: StageKey): string {
  if (!props.result) return "";
  switch (key) {
    case "raw":
      return "As uploaded, before any processing.";
    case "warped":
      return "Perspective-corrected and cropped to the document boundary.";
  }
}

function copyText() {
  navigator.clipboard.writeText(editableText.value).then(() => {
    copied.value = true;
    setTimeout(() => {
      copied.value = false;
    }, 2000);
  });
}

function openLightbox(src: string) {
  lightboxSrc.value = src;
  lightboxVisible.value = true;
}
</script>

<template>
  <section class="panel" id="results-panel">
    <h2>Result</h2>
    <template v-if="result">
      <div class="stage-tabs" role="tablist">
        <button
          v-for="(stage, i) in STAGES"
          :key="stage.key"
          type="button"
          class="stage-tab"
          :class="{ active: activeStage === stage.key }"
          role="tab"
          :aria-selected="activeStage === stage.key"
          @click="activeStage = stage.key"
        >
          <span class="stage-index">{{ i + 1 }}</span>
          {{ stage.label }}
        </button>
      </div>

      <figure id="stage-viewer">
        <div class="frame stage-frame">
          <img
            :src="stageSrc(activeStage)"
            :alt="`${activeStage} stage`"
            class="lightbox-trigger"
            @click="openLightbox(stageSrc(activeStage))"
          />
        </div>
        <figcaption>{{ stageCaption(activeStage) }}</figcaption>
      </figure>

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
          <span class="stat-value"
            >{{ (result.doc_type_confidence * 100).toFixed(1) }}%</span
          >
        </div>
      </div>
      <label for="extracted-text">Extracted Text</label>
      <textarea
        id="extracted-text"
        rows="10"
        spellcheck="false"
        v-model="editableText"
      ></textarea>
      <div class="copy-row">
        <button id="copy-btn" @click="copyText">Copy Text</button>
        <Transition name="copy-feedback">
          <span v-if="copied" class="copy-confirm">&#10003; Copied</span>
        </Transition>
      </div>
    </template>
    <p v-else class="empty-state">
      Upload a document on the left to see results here.
    </p>

    <Lightbox
      :src="lightboxSrc"
      :visible="lightboxVisible"
      @close="lightboxVisible = false"
    />
  </section>
</template>

<style scoped>
.copy-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 0 0 auto;
}

.copy-confirm {
  font-size: 12px;
  color: #4ade80;
  letter-spacing: 0.02em;
}

.copy-feedback-enter-active {
  transition:
    opacity 150ms ease,
    transform 150ms ease;
}

.copy-feedback-leave-active {
  transition:
    opacity 300ms ease,
    transform 300ms ease;
}

.copy-feedback-enter-from {
  opacity: 0;
  transform: translateX(-4px);
}

.copy-feedback-leave-to {
  opacity: 0;
  transform: translateX(4px);
}
</style>
