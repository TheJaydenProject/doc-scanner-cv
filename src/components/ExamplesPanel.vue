<script setup lang="ts">
import { ref } from "vue";
import Lightbox from "./Lightbox.vue";

const lightboxSrc = ref("");
const lightboxVisible = ref(false);

function openLightbox(src: string) {
  lightboxSrc.value = src;
  lightboxVisible.value = true;
}
</script>

<template>
  <section class="panel" id="examples">
    <h2>Examples</h2>
    <p class="section-desc">Real scans of handwritten notes processed through the pipeline.</p>

    <div class="example-grid">
      <div class="example-card" v-for="n in [1, 2]" :key="n">
        <div class="example-images">
          <figure>
            <figcaption>Original</figcaption>
            <img
              :src="`/examples/note${n}_original.jpg`"
              :alt="`Note ${n} original`"
              class="lightbox-trigger"
              @click="openLightbox(`/examples/note${n}_original.jpg`)"
            />
          </figure>
          <span class="arrow">&#x2192;</span>
          <figure>
            <figcaption>Scanned</figcaption>
            <img
              :src="`/examples/note${n}_scanned.png`"
              :alt="`Note ${n} scanned`"
              class="lightbox-trigger"
              @click="openLightbox(`/examples/note${n}_scanned.png`)"
            />
          </figure>
        </div>
      </div>
    </div>

    <Lightbox :src="lightboxSrc" :visible="lightboxVisible" @close="lightboxVisible = false" />
  </section>
</template>
