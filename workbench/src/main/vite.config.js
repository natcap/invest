import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { builtinModules } from 'module';

// https://vitejs.dev/config/
export default defineConfig({
  // root: './src/renderer/',
  plugins: [react()],
  build: {
    target: 'node14',
    lib: {
      entry: './main.js',
      formats: ['cjs'],
      name: 'Workbench'
    },
    rollupOptions: {
      external: [
        'electron',
        'electron-devtools-installer',
        ...builtinModules
      ]
    }
  }
});
