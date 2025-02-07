import { defineConfig } from "vite";
import glsl from "vite-plugin-glsl";
import typegpu from "rollup-plugin-typegpu";

export default defineConfig({
  plugins: [glsl(), typegpu({ include: [/\.ts$/] })],
});
