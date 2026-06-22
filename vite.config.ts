import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: "static",
    // static/ also holds committed source assets (examples/, docs.html) that are
    // NOT build output, so emptying it on every build wipes them. Leave it false
    // and let Vite overwrite only its own index.html + assets/ in place.
    emptyOutDir: false,
  },
});
