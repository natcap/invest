const {
  ipcRenderer,
} = require('electron');
const crypto = require('crypto');
const path = require('path');
const { ipcMainChannels } = require('../main/ipcMainChannels');
const { getLogger } = require('../main/logger');

const logger = getLogger();

// Most IPC initiates in renderer and main does the listening,
// but these channels are exceptions: renderer listens for them
const ipcRendererChannels = [
  /invest-logging-*/,
  /invest-stdout-*/,
  /invest-exit-*/,
  /download-status/,
];

const api = {
  // The gettext callable
  _: ipcRenderer.sendSync.bind(null, ipcMainChannels.GETTEXT), // partially applied function
  PORT: process.env.PORT,
  LOGFILE_PATH: logger.transports.file.getFile().path,
  getLogger: getLogger,
  // TODO: this next one feels out of place, just expose crypto.createHash
  // here instead?
  getWorkspaceHash: (modelRunName, workspaceDir, resultsSuffix) => {
    return crypto.createHash('sha1').update(
      `${modelRunName}
       ${JSON.stringify(path.resolve(workspaceDir))}
       ${JSON.stringify(resultsSuffix)}`
    ).digest('hex');
  },
  electron: {
    ipcRenderer: {
      invoke: (channel, ...args) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          return ipcRenderer.invoke(channel, ...args);
        }
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

module.exports.api = api;
