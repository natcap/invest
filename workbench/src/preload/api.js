import crypto from 'crypto';
import path from 'path';

// eslint-disable-next-line import/no-extraneous-dependencies
import { ipcRenderer } from 'electron';

import { ipcMainChannels } from '../main/ipcMainChannels';
import { getLogger } from '../main/logger';

const logger = getLogger();

// Most IPC initiates in renderer and main does the listening,
// but these channels are exceptions: renderer listens for them
const ipcRendererChannels = [
  /invest-logging-*/,
  /invest-stdout-*/,
  /invest-exit-*/,
  /download-status/,
];

export default {
  // Port where the flask app is running
  PORT: process.env.PORT,
  // Workbench logfile location, so Report window can open to it
  LOGFILE_PATH: logger.transports.file.getFile().path,
  // The gettext callable; a partially applied function
  _: ipcRenderer.sendSync.bind(null, ipcMainChannels.GETTEXT),
  getLogger: getLogger,
  path: {
    resolve: path.resolve,
  },
  crypto: {
    sha1hash: (data) => crypto.createHash('sha1').update(data).digest('hex'),
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
