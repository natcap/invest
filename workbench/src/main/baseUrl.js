import ELECTRON_DEV_MODE from './isDevMode';

export default (ELECTRON_DEV_MODE)
  ? 'http://127.0.0.1:3000/'
  : `file://${__dirname}/../`; // resolves to the build/ dir
