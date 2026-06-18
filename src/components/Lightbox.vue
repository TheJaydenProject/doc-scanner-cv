<script setup lang="ts">
import { onMounted, onUnmounted } from "vue";

defineProps<{
  src: string;
  visible: boolean;
}>();

const emit = defineEmits<{ close: [] }>();

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Escape") emit("close");
}

onMounted(() => document.addEventListener("keydown", onKeydown));
onUnmounted(() => document.removeEventListener("keydown", onKeydown));
</script>

<template>
  <div
    v-if="visible"
    id="lightbox"
    @click.self="emit('close')"
  >
    <button id="lightbox-close" @click="emit('close')">&#x2715;</button>
    <img :src="src" alt="Expanded view" />
  </div>
</template>
