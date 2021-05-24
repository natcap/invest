// Dont' use ES6 features that require babel transpiling here
// because the preload script seems to be outside the chain
// of our "-r @babel/register" dev mode strategy.
// const { app, contextBridge, ipcRenderer } = require('electron');
// console.log('preload script')
// console.log(app) // undefined here
// console.log(contextBridge)
// console.log(ipcRenderer)

const { getLogger } = require('./logger');

function preload() {
  window.Workbench = {
    isDevMode: !!process.defaultApp,
    // userDataPath: app.getPath('userData'),
    getLogger: getLogger,
  };
}

preload();
