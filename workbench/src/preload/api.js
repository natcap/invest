// eslint-disable-next-line import/no-extraneous-dependencies
const { ipcRenderer } = require('electron');
// using `import` for electron messes with vite and yields a bad bundle.
// `import`` is okay for local modules though
import { ipcMainChannels } from '../main/ipcMainChannels';

// Most IPC initiates in renderer and main does the listening,
// but these channels are exceptions: renderer listens for them
const ipcRendererChannels = [
  /invest-logging-*/,
  /invest-stdout-*/,
  /invest-exit-*/,
  /download-status/,
];

// args sent via `additionalArguments` to `webPreferences` for `BroswerWindow`
const portArg = process.argv.filter((arg) => arg.startsWith('--port'))[0];
const PORT = portArg ? portArg.split('=')[1] : '';
const userPaths = ipcRenderer.sendSync(ipcMainChannels.GET_ELECTRON_PATHS);
const isDevMode = process.argv.includes('--devmode');

// As well as being loaded by the electron preload script,
// this module is loaded by the jest setup script in order to
// mock APIs that are normally setup during electron preload.
// In the jest case, we do not have userPaths data.
const userguidePath = (!isDevMode && userPaths)
  ? `file:///${userPaths.resourcesPath}/documentation` // as a URL
  : ''; // In DevMode, local UG is served at the root path
const electronLogPath = (userPaths)
  ? userPaths.logfilePath
  : '';

export default {
  PORT: PORT, // where the flask app is running
  ELECTRON_LOG_PATH: electronLogPath,
  USERGUIDE_PATH: userguidePath,
  LANGUAGE: ipcRenderer.sendSync(ipcMainChannels.GET_LANGUAGE),
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
