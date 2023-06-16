import ELECTRON_DEV_MODE from './isDevMode';

export default (ELECTRON_DEV_MODE)
  ? 'http://localhost:5173/' // default port for vite dev server
  : `file://${__dirname}/../`; // resolves to the build/ dir
