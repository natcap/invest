// Dont' use ES6 features that require babel transpiling here
// because the preload script seems to be outside the chain
// of our "-r @babel/register" dev mode strategy.
const crypto = require('crypto');

const { getLogger } = require('./logger');

function preload() {
  window.Workbench = {
    getLogger: getLogger,
    getWorkspaceHash: (modelRunName, workspaceDir, resultsSuffix) => {
      return crypto.createHash('sha1').update(
        `${modelRunName}
         ${JSON.stringify(workspaceDir)}
         ${JSON.stringify(resultsSuffix)}`
      ).digest('hex');
    }
  };
}

preload();
