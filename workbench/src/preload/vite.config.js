import { defineConfig } from 'vite';
import path from 'path';

const PROJECT_ROOT = '../..';

// https://vitejs.dev/config/
export default defineConfig({
  build: {
    mode: process.env.MODE,
    sourcemap: 'inline',
    minify: false,
    outDir: path.join(PROJECT_ROOT, 'build/preload'),
    target: 'chrome98',
    lib: {
      entry: 'preload.js',
      name: 'preload.js',
    },
    rollupOptions: {
      output: {
        entryFileNames: 'preload.js',
      },
    },
    emptyOutDir: true,
  },
});
