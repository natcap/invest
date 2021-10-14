// Dont' use ES6 features that require babel transpiling here
// because the preload script seems to be outside the chain
// of our "-r @babel/register" dev mode strategy.

const { getLogger } = require('./logger');

function preload() {
  window.Workbench = {
    getLogger: getLogger,
  };
}

preload();
