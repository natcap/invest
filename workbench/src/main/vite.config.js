import { defineConfig } from 'vite';
// import react from '@vitejs/plugin-react';
import { builtinModules } from 'module';

// https://vitejs.dev/config/
export default defineConfig({
  // root: './src/renderer/',
  // plugins: [react()],
  build: {
    // sourcemap: 'inline',
    outDir: '../../dist/main',
    minify: false,
    target: 'node14',
    lib: {
      mode: process.env.MODE,
      entry: './main.js',
      formats: ['cjs'],
      name: 'Workbench',
    },
    rollupOptions: {
      external: [
        'electron',
        'electron-devtools-installer',
        ...builtinModules,
      ],
      output: {
        entryFileNames: '[name].cjs',
      },
    },
    emptyOutDir: true,
  },
});
