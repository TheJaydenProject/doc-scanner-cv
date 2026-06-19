<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from "vue";

const props = defineProps<{ src: string; visible: boolean }>();
const emit = defineEmits<{ close: [] }>();

const MIN_SCALE = 0.2;
const MAX_SCALE = 10;
const ZOOM_STEP = 1.3;

const scale = ref(1);
const tx = ref(0);
const ty = ref(0);
const isDragging = ref(false);
const hasDragged = ref(false);
const dragStart = ref({ x: 0, y: 0, tx: 0, ty: 0 });
const containerRef = ref<HTMLElement | null>(null);

const imgStyle = computed(() => ({
  transform: `translate(${tx.value}px, ${ty.value}px) scale(${scale.value})`,
  cursor: isDragging.value
    ? "grabbing"
    : scale.value > 1.05
      ? "grab"
      : "zoom-in",
}));

function reset() {
  scale.value = 1;
  tx.value = 0;
  ty.value = 0;
  isDragging.value = false;
  hasDragged.value = false;
}

watch(() => props.src, reset);
watch(
  () => props.visible,
  (v) => {
    if (v) reset();
  },
);

function applyZoom(clientX: number, clientY: number, factor: number) {
  const c = containerRef.value;
  if (!c) return;
  const r = c.getBoundingClientRect();
  // Mouse position relative to the container center (image origin point)
  const mx = clientX - r.left - r.width / 2;
  const my = clientY - r.top - r.height / 2;
  const newScale = Math.min(
    MAX_SCALE,
    Math.max(MIN_SCALE, scale.value * factor),
  );
  const ratio = newScale / scale.value;
  // Keep the point under the cursor fixed in screen space
  tx.value = mx + (tx.value - mx) * ratio;
  ty.value = my + (ty.value - my) * ratio;
  scale.value = newScale;
}

function onWheel(e: WheelEvent) {
  e.preventDefault();
  applyZoom(e.clientX, e.clientY, e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
}

function onMousedown(e: MouseEvent) {
  if (e.button !== 0) return;
  e.preventDefault();
  isDragging.value = true;
  hasDragged.value = false;
  dragStart.value = { x: e.clientX, y: e.clientY, tx: tx.value, ty: ty.value };
}

function onMousemove(e: MouseEvent) {
  if (!isDragging.value) return;
  const dx = e.clientX - dragStart.value.x;
  const dy = e.clientY - dragStart.value.y;
  if (Math.abs(dx) > 2 || Math.abs(dy) > 2) hasDragged.value = true;
  tx.value = dragStart.value.tx + dx;
  ty.value = dragStart.value.ty + dy;
}

function onMouseup() {
  isDragging.value = false;
}

function onBackdropClick() {
  if (!hasDragged.value) emit("close");
  hasDragged.value = false;
}

function zoomToCenter(factor: number) {
  const c = containerRef.value;
  if (!c) return;
  const r = c.getBoundingClientRect();
  applyZoom(r.left + r.width / 2, r.top + r.height / 2, factor);
}

function onKeydown(e: KeyboardEvent) {
  if (!props.visible) return;
  if (e.key === "Escape") emit("close");
  if (e.key === "+" || e.key === "=") zoomToCenter(ZOOM_STEP);
  if (e.key === "-") zoomToCenter(1 / ZOOM_STEP);
  if (e.key === "0") reset();
}

onMounted(() => {
  document.addEventListener("keydown", onKeydown);
  document.addEventListener("mousemove", onMousemove);
  document.addEventListener("mouseup", onMouseup);
});
onUnmounted(() => {
  document.removeEventListener("keydown", onKeydown);
  document.removeEventListener("mousemove", onMousemove);
  document.removeEventListener("mouseup", onMouseup);
});
</script>

<template>
  <div v-if="visible" id="lightbox">
    <div
      ref="containerRef"
      class="lightbox-canvas"
      @wheel.prevent="onWheel"
      @click="onBackdropClick"
    >
      <img
        :src="src"
        alt="Expanded view"
        :style="imgStyle"
        @mousedown="onMousedown"
        @click.stop
        @dblclick.stop="reset"
        draggable="false"
      />
    </div>

    <div class="lightbox-toolbar">
      <button @click="zoomToCenter(1 / ZOOM_STEP)" title="Zoom out (-)">
        −
      </button>
      <span class="lightbox-zoom-level">{{ Math.round(scale * 100) }}%</span>
      <button @click="zoomToCenter(ZOOM_STEP)" title="Zoom in (+)">+</button>
      <button @click="reset" title="Reset zoom (0)">Reset</button>
      <button
        class="lightbox-close-btn"
        @click="emit('close')"
        title="Close (Esc)"
      >
        ✕
      </button>
    </div>
  </div>
</template>

<style scoped>
#lightbox {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: flex;
  flex-direction: column;
}

.lightbox-canvas {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.93);
}

.lightbox-canvas img {
  display: block;
  max-width: 90vw;
  max-height: 84vh;
  transform-origin: center center;
  user-select: none;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
}

.lightbox-toolbar {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-5);
  background: rgba(0, 0, 0, 0.88);
  border-top: 1px solid var(--border-strong);
}

.lightbox-toolbar button {
  background: transparent;
  border: 1px solid var(--border-strong);
  color: var(--ink);
  font-size: 14px;
  padding: 4px 12px;
  cursor: pointer;
  border-radius: var(--radius-sm);
  line-height: 1.5;
}

.lightbox-toolbar button:hover {
  background: var(--surface-2);
}

.lightbox-zoom-level {
  font-size: 12px;
  color: var(--ink-secondary);
  min-width: 44px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.lightbox-close-btn {
  margin-left: var(--space-4);
}
</style>
