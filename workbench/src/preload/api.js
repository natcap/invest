import path from 'path';

// eslint-disable-next-line import/no-extraneous-dependencies
import { ipcRenderer } from 'electron';

import { ipcMainChannels } from '../main/ipcMainChannels';
import { getLogger } from '../main/logger';

const isDevMode = process.argv.includes('--devMode');

const logger = getLogger();

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
  // Port where the flask app is running
  PORT: process.env.PORT,
  USERGUIDE_PATH: userguidePath,
  // Workbench logfile location, so Report window can open to it
  LOGFILE_PATH: logger.transports.file.getFile().path,
  getLogger: getLogger,
  path: {
    resolve: path.resolve,
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
