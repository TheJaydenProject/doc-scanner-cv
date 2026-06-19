<script setup lang="ts">
import { ref, watch, computed } from "vue";
import type { ScanResult } from "../types";
import Lightbox from "./Lightbox.vue";

type StageKey = "raw" | "warped" | "binarized" | "detected" | "compare";

const STAGES: { key: StageKey; label: string }[] = [
  { key: "raw", label: "Raw" },
  { key: "warped", label: "Warped" },
  { key: "binarized", label: "Binarized" },
  { key: "detected", label: "Detected" },
  { key: "compare", label: "Compare" },
];

const props = defineProps<{
  result: ScanResult | null;
  file: File | null;
}>();

const editableText = ref("");
const lightboxSrc = ref("");
const lightboxVisible = ref(false);
const copied = ref(false);
const activeStage = ref<StageKey>("detected");
const hoveredBox = ref<number | null>(null);
const naturalWidth = ref(0);
const naturalHeight = ref(0);
const compareNaturalWidth = ref(0);
const compareNaturalHeight = ref(0);

watch(
  () => props.result,
  (newResult) => {
    editableText.value = newResult?.text ?? "";
    activeStage.value = "detected";
    hoveredBox.value = null;
    naturalWidth.value = 0;
    naturalHeight.value = 0;
    compareNaturalWidth.value = 0;
    compareNaturalHeight.value = 0;
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
    case "binarized":
    case "detected":
      return `data:image/png;base64,${props.result.binarized_image_b64}`;
    case "compare":
      return "";
  }
}

function stageCaption(key: StageKey): string {
  if (!props.result) return "";
  switch (key) {
    case "raw":
      return "As uploaded, before any processing.";
    case "warped":
      return "Perspective-corrected and cropped to the document boundary.";
    case "binarized":
      return `Thresholded using the "${props.result.doc_type}" branch.`;
    case "detected": {
      const n = props.result.detection_count;
      return `${n} MSER text region${n === 1 ? "" : "s"} — hover a box to inspect it.`;
    }
    case "compare":
      return "";
  }
}

function onStageImageLoad(event: Event) {
  const img = event.target as HTMLImageElement;
  naturalWidth.value = img.naturalWidth;
  naturalHeight.value = img.naturalHeight;
}

function onCompareImageLoad(event: Event) {
  const img = event.target as HTMLImageElement;
  compareNaturalWidth.value = img.naturalWidth;
  compareNaturalHeight.value = img.naturalHeight;
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
          <span v-if="stage.key !== 'compare'" class="stage-index">{{
            i + 1
          }}</span>
          {{ stage.label }}
        </button>
      </div>

      <div v-if="activeStage === 'compare'" class="compare-view">
        <figure class="compare-half">
          <figcaption>Before</figcaption>
          <div class="frame compare-frame">
            <img
              :src="stageSrc('raw')"
              alt="Raw stage"
              class="lightbox-trigger"
              @click="openLightbox(stageSrc('raw'))"
            />
          </div>
        </figure>
        <figure class="compare-half">
          <figcaption>After</figcaption>
          <div class="frame compare-frame">
            <img
              :src="stageSrc('binarized')"
              alt="Detected stage"
              class="lightbox-trigger"
              @click="openLightbox(stageSrc('binarized'))"
              @load="onCompareImageLoad"
            />
            <svg
              v-if="compareNaturalWidth && compareNaturalHeight"
              class="detection-overlay"
              :viewBox="`0 0 ${compareNaturalWidth} ${compareNaturalHeight}`"
              preserveAspectRatio="none"
              style="pointer-events: none"
            >
              <rect
                v-for="(box, i) in result.detections"
                :key="i"
                :x="box[0]"
                :y="box[1]"
                :width="box[2]"
                :height="box[3]"
              />
            </svg>
          </div>
        </figure>
      </div>
      <figure v-else id="stage-viewer">
        <div class="frame stage-frame">
          <img
            :src="stageSrc(activeStage)"
            :alt="`${activeStage} stage`"
            class="lightbox-trigger"
            @click="openLightbox(stageSrc(activeStage))"
            @load="onStageImageLoad"
          />
          <svg
            v-if="activeStage === 'detected' && naturalWidth && naturalHeight"
            class="detection-overlay"
            :viewBox="`0 0 ${naturalWidth} ${naturalHeight}`"
            preserveAspectRatio="none"
          >
            <rect
              v-for="(box, i) in result.detections"
              :key="i"
              :x="box[0]"
              :y="box[1]"
              :width="box[2]"
              :height="box[3]"
              :class="{ hovered: hoveredBox === i }"
              @mouseenter="hoveredBox = i"
              @mouseleave="hoveredBox = null"
            />
          </svg>
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
