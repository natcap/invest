// Dont' use ES6 features that require babel transpiling here
// because the preload script seems to be outside the chain
// of our "-r @babel/register" dev mode strategy.
const {
  contextBridge,
  ipcRenderer,
  shell,
} = require('electron');
const crypto = require('crypto');

const { ipcMainChannels } = require('../main/ipcMainChannels');
const { getLogger } = require('../main/logger');

contextBridge.exposeInMainWorld('Workbench', {
  // The gettext callable
  _: ipcRenderer.sendSync.bind(null, ipcMainChannels.GETTEXT), // partially applied function
  // Passing data from main to renderer for window.fetch
  PORT: process.env.PORT,
  getLogger: getLogger,
  // TODO: this next one feels out of place, just expose crypto.createHash
  // here instead?
  getWorkspaceHash: (modelRunName, workspaceDir, resultsSuffix) => {
    return crypto.createHash('sha1').update(
      `${modelRunName}
       ${JSON.stringify(workspaceDir)}
       ${JSON.stringify(resultsSuffix)}`
    ).digest('hex');
  },
  electron: {
    ipcRenderer: {
      invoke: (channel, data) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          ipcRenderer.invoke(channel, data);
        }
      },
      send: (channel, data) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          ipcRenderer.send(channel, data);
        }
      },
      sendSync: (channel, data) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          ipcRenderer.sendSync(channel, data);
        }
      },
      on: (channel, func) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          ipcRenderer.on(channel, (event, ...args) => func(...args));
        }
      },
      removeListener: (channel, func) => {
        if (Object.values(ipcMainChannels).includes(channel)) {
          ipcRenderer.removeListener(channel, func);
        }
      },
      removeAllListeners: () => ipcRenderer.removeAllListeners(),
    },
    shell: shell,
  },
});
