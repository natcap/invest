import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

const PROJECT_ROOT = '.';

// https://vitejs.dev/config/
export default defineConfig({
  root: PROJECT_ROOT,
  // base: empty string creates relative paths for assets.
  // We need this because our files are embedded within an electron asar
  // on the local filesystem, not served from a consistent base url.
  base: '',
  plugins: [react()],
  build: {
    mode: process.env.MODE,
    sourcemap: 'inline',
    minify: process.env.MODE !== 'development',
    outDir: path.join(PROJECT_ROOT, 'build'),
    target: 'chrome98',
    rollupOptions: {
      input: [
        path.resolve(PROJECT_ROOT, 'index.html'),
        path.resolve(PROJECT_ROOT, 'splash.html'),
        path.resolve(PROJECT_ROOT, 'report_a_problem.html'),
        path.resolve(PROJECT_ROOT, 'about.html'),
      ],
    },
    emptyOutDir: true,
  },
});
