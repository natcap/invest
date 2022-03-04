// Dont' use ES6 features that require babel transpiling here
// because the preload script seems to be outside the chain
// of our "-r @babel/register" dev mode strategy.
const {
  contextBridge,
  ipcRenderer,
  shell,
} = require('electron');
const crypto = require('crypto');

const { ipcMainChannels } = require('./main/ipcMainChannels');
const { getLogger } = require('./logger');

contextBridge.exposeInMainWorld('Workbench', {
  PORT: process.env.PORT,
  getLogger: getLogger,
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
    },
    shell: shell,
  },
});
console.log('loaded PRELOAD');

// function preload() {
//   window.Workbench = {
//     getLogger: getLogger,
//     getWorkspaceHash: (modelRunName, workspaceDir, resultsSuffix) => {
//       return crypto.createHash('sha1').update(
//         `${modelRunName}
//          ${JSON.stringify(workspaceDir)}
//          ${JSON.stringify(resultsSuffix)}`
//       ).digest('hex');
//     }
//   };
// }

// preload();
