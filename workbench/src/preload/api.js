// eslint-disable-next-line import/no-extraneous-dependencies
const { ipcRenderer } = require('electron');
// using `import` for electron messes with vite and yields a bad bundle.
// `import`` is okay for local modules though

import { ipcMainChannels } from '../main/ipcMainChannels';

const isDevMode = process.argv.includes('--devMode');

// Most IPC initiates in renderer and main does the listening,
// but these channels are exceptions: renderer listens for them
const ipcRendererChannels = [
  /invest-logging-*/,
  /invest-stdout-*/,
  /invest-exit-*/,
  /download-status/,
];

// In DevMode, local UG is served at the root path
const userguidePath = isDevMode
  ? ''
  : `file:///${process.resourcesPath}/documentation`;

export default {
  PORT: process.env.PORT, // where the flask app is running
  USERGUIDE_PATH: userguidePath,
  logger: {
    debug: (message) => ipcRenderer.send(ipcMainChannels.LOGGER, 'debug', message),
    info: (message) => ipcRenderer.send(ipcMainChannels.LOGGER, 'info', message),
    warning: (message) => ipcRenderer.send(ipcMainChannels.LOGGER, 'warning', message),
    error: (message) => ipcRenderer.send(ipcMainChannels.LOGGER, 'error', message),
  },
  electron: {
    ipcRenderer: {
      invoke: (channel, ...args) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          return ipcRenderer.invoke(channel, ...args);
        }
        return undefined;
      },
      send: (channel, ...args) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          ipcRenderer.send(channel, ...args);
        }
      },
      sendSync: (channel, ...args) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          ipcRenderer.sendSync(channel, ...args);
        }
      },
      on: (channel, func) => {
        if (ipcRendererChannels.some((regex) => regex.test(channel))) {
          ipcRenderer.on(channel, (event, ...args) => func(...args));
        }
      },
      removeListener: (channel, func) => {
        if (ipcRendererChannels.some((regex) => regex.test(channel))) {
          ipcRenderer.removeListener(channel, func);
        }
      },
      removeAllListeners: (channel) => {
        if (ipcRendererChannels.some((regex) => regex.test(channel))) {
          ipcRenderer.removeAllListeners(channel);
        }
      },
    },
  },
};
