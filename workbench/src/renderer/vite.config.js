import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  // root: './src/renderer/',
  plugins: [react()],
  build: {
    target: 'chrome98',
    rollupOptions: {
      input: {
        main: './index.html'
      }
    }
  }
});
