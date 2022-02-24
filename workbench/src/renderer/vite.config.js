import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { builtinModules } from 'module';

const PROJECT_ROOT = path.join(__dirname, '../..');

// https://vitejs.dev/config/
export default defineConfig({
  root: PROJECT_ROOT,
  // base: "",
  plugins: [react()],
  // server: {
  //   fs: {
  //     strict: true,
  //   },
  // },
  build: {
    mode: process.env.MODE,
    sourcemap: 'inline',
    minify: process.env.MODE !== 'development',
    outDir: path.join(__dirname, '../../dist/renderer'),
    target: 'chrome98',
    rollupOptions: {
      input: {
        main: path.join(__dirname, 'index.html'),
        splash: path.join(__dirname, 'static/splash.html'),
      },
      external: [
        ...builtinModules,
      ],
    },
    emptyOutDir: true,
  },
});
