// eslint-disable-next-line import/no-extraneous-dependencies
const { contextBridge } = require('electron');
// TODO: not sure why but vite does not translate electron correctly for the browser
// when using `import`. it results in a __dirname reference, which is not found in browser.
// using require() works.

import api from './api';

contextBridge.exposeInMainWorld('Workbench', api);
