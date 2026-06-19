<script setup lang="ts">
import { ref } from "vue";
import Lightbox from "./Lightbox.vue";

interface Example {
  id: number;
  originalExt: "jpg" | "png";
}

// Originals come in mixed formats; scanned outputs are always PNG.
const examples: Example[] = [
  { id: 1, originalExt: "jpg" },
  { id: 2, originalExt: "jpg" },
  { id: 3, originalExt: "png" },
  { id: 4, originalExt: "png" },
];

const lightboxSrc = ref("");
const lightboxVisible = ref(false);

function openLightbox(src: string) {
  lightboxSrc.value = src;
  lightboxVisible.value = true;
}
</script>

<template>
  <section class="panel">
    <h2>Examples</h2>
    <p class="section-desc">
      Real scans of handwritten notes processed through the pipeline.
    </p>

    <div class="example-grid">
      <div class="example-card" v-for="ex in examples" :key="ex.id">
        <div class="example-images">
          <figure>
            <figcaption>Original</figcaption>
            <img
              :src="`/examples/note${ex.id}_original.${ex.originalExt}`"
              :alt="`Note ${ex.id} original`"
              class="lightbox-trigger"
              @click="
                openLightbox(
                  `/examples/note${ex.id}_original.${ex.originalExt}`,
                )
              "
            />
          </figure>
          <span class="arrow">&#x2192;</span>
          <figure>
            <figcaption>Scanned</figcaption>
            <img
              :src="`/examples/note${ex.id}_scanned.png`"
              :alt="`Note ${ex.id} scanned`"
              class="lightbox-trigger"
              @click="openLightbox(`/examples/note${ex.id}_scanned.png`)"
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
