import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
// import { builtinModules } from 'module';

const PROJECT_ROOT = '.';

// https://vitejs.dev/config/
export default defineConfig({
  root: PROJECT_ROOT,
  // publicDir: path.join(PROJECT_ROOT, 'src/static'),
  // base: empty string creates relative paths for assets.
  // We need this because our files are embedded within an electron asar
  // on the local filesystem, not served from a consistent base url.
  base: "",
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
    rollupOptions: {
      // input: {
      //   main: path.resolve(PROJECT_ROOT, 'index.html'),
      //   splash: path.resolve(PROJECT_ROOT, 'src/renderer/static/splash.html'),
      //   report: path.resolve(PROJECT_ROOT, 'src/renderer/static/report_a_problem.html'),
      //   about: path.resolve(PROJECT_ROOT, 'src/renderer/static/about.html'),
      // },
      input: [
        path.resolve(PROJECT_ROOT, 'index.html'),
        path.resolve(PROJECT_ROOT, 'splash.html'),
        path.resolve(PROJECT_ROOT, 'report_a_problem.html'),
        path.resolve(PROJECT_ROOT, 'about.html'),
      ],
      // external: [
      //   ...builtinModules,
      // ],
    },
    emptyOutDir: true,
  },
});
