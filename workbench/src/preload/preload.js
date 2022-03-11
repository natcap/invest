// TODO: remove this comment and switch from require to import?
// Dont' use ES6 features that require babel transpiling here
// because the preload script seems to be outside the chain
// of our "-r @babel/register" dev mode strategy.
const {
  contextBridge,
} = require('electron');

const { api } = require('./api');

contextBridge.exposeInMainWorld('Workbench', api);
