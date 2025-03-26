import { defineConfig } from 'vite';
import glsl from 'vite-plugin-glsl';
import typegpu from 'unplugin-typegpu/vite';

export default defineConfig({
    plugins: [glsl(), typegpu({ include: [/\.ts$/] })],
});
