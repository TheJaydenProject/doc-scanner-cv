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
  const prefix = import.meta.env.DEV ? "/static" : "";
  return `${prefix}/examples/${ex.id}_${stage}.${ext}`;
}

function openLightbox(src: string) {
  lightboxSrc.value = src;
  lightboxVisible.value = true;
}
</script>

<template>
  <section class="panel">
    <h2>Examples</h2>
    
    <div class="disclaimer-alert">
      <strong>Note:</strong> Since I have limited resources, this live demo is heavily rate limited (<strong>20 scans per hour</strong>) and it takes around 2 minutes (120s) for one OCR scan to process. If you want faster, more stable OCR, please visit the <a href="https://github.com/TheJaydenProject/doc-scanner-cv" target="_blank" rel="noopener">GitHub repository</a> to clone and run it locally!
    </div>
    <p class="section-desc">
      Production inputs undergo perspective warping and MSER text-height detection, where scans &lt;8px are rejected outright to conserve compute and those &lt;30px are routed through FSRCNN upscaling prior to OCR.
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

.disclaimer-alert {
  background: rgba(248, 113, 113, 0.08);
  color: var(--ink);
  padding: var(--space-3) var(--space-4);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-sm);
  margin-bottom: var(--space-5);
  font-size: 13px;
  line-height: 1.5;
}

.disclaimer-alert a {
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
}

.disclaimer-alert a:hover {
  text-decoration: underline;
}
</style>
