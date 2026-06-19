<script setup lang="ts">
import { ref } from "vue";
import Lightbox from "./Lightbox.vue";

interface Example {
  id: string;
  label: string;
  originalExt: "jpg" | "png";
}

const examples: Example[] = [
  { id: "note1", label: "Handwritten Note", originalExt: "jpg" },
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
      Real scans of handwritten and printed documents processed through the pipeline.
    </p>

    <div class="example-grid">
      <div class="example-card" v-for="ex in examples" :key="ex.id">
        <h3 class="example-title">{{ ex.label }}</h3>
        <div class="pipeline-track">
          <figure v-for="(stage, idx) in STAGES" :key="stage">
            <figcaption>{{ idx + 1 }}. {{ stage.charAt(0).toUpperCase() + stage.slice(1) }}</figcaption>
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
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-3);
  align-items: end;
}

.pipeline-track img {
  width: 100%;
  height: auto;
  max-height: 200px;
  object-fit: contain;
  background: var(--inset);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
}
</style>
