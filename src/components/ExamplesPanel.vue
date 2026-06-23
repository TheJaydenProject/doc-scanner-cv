<script setup lang="ts">
import { ref } from "vue";
import Lightbox from "./Lightbox.vue";

interface Example {
  id: string;
  label: string;
  originalExt: "jpg" | "png";
}

const examples: Example[] = [
  { id: "note2", label: "Handwritten Note", originalExt: "jpg" },
  { id: "note3", label: "Printed Document", originalExt: "png" },
];

const STAGES = ["raw", "warped", "binarized", "detected"] as const;

const lightboxSrc = ref("");
const lightboxVisible = ref(false);

function stageSrc(ex: Example, stage: (typeof STAGES)[number]): string {
  const ext = stage === "raw" ? ex.originalExt : "png";
  return `/examples/${ex.id}_${stage}.${ext}`;
}

function openLightbox(src: string) {
  lightboxSrc.value = src;
  lightboxVisible.value = true;
}
</script>

<template>
  <section class="panel">
    <h2>Examples</h2>
    <p class="section-desc">
      These are real scans processed through our pipeline. First, the raw image is straightened and analyzed to measure the text size. If the text is large enough, it goes straight to the OCR engine. If the text is too small, we use an AI super-resolution model to intelligently upscale the image before extracting the text. Any scans with text too tiny to recover are rejected early to save processing time.
    </p>

    <div class="example-grid">
      <div class="example-card" v-for="ex in examples" :key="ex.id">
        <h3 class="example-title">{{ ex.label }}</h3>
        <div class="pipeline-track">
          <figure v-for="(stage, idx) in STAGES" :key="stage">
            <figcaption>
              {{ idx + 1 }}.
              {{ stage.charAt(0).toUpperCase() + stage.slice(1) }}
            </figcaption>
            <img
              :src="stageSrc(ex, stage)"
              :alt="`${ex.id} ${stage}`"
              class="lightbox-trigger"
              @click="openLightbox(stageSrc(ex, stage))"
            />
          </figure>
        </div>
      </div>
    </div>

    <Lightbox
      :src="lightboxSrc"
      :visible="lightboxVisible"
      @close="lightboxVisible = false"
    />
  </section>
</template>

<style scoped>
.example-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: var(--space-4);
  border-bottom: 1px solid var(--border);
  padding-bottom: var(--space-2);
}

.pipeline-track {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-3);
  align-items: end;
}

.pipeline-track img {
  width: 100%;
  height: auto;
  object-fit: contain;
  background: var(--inset);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
}
</style>
