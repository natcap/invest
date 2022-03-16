import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { builtinModules } from 'module';

const PROJECT_ROOT = '.';

// https://vitejs.dev/config/
export default defineConfig({
  root: PROJECT_ROOT,
  // publicDir: path.join(PROJECT_ROOT, 'src/static'),
  base: "", // for prod: assets are here rel to index.html
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
    outDir: path.join(PROJECT_ROOT, 'build'),
    target: 'chrome98',
    // rollupOptions: {
    //   input: {
    //     main: path.join(PROJECT_ROOT, 'src/index.html'),
    //     splash: path.join(__dirname, 'static/splash.html'),
    //   },
    //   external: [
    //     ...builtinModules,
    //   ],
    // },
    emptyOutDir: true,
  },
});
