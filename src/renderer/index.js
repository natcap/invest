import { ipcRenderer } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import React from 'react';
import ReactDom from 'react-dom';

import App from './app';
import { ipcMainChannels } from '../main/ipcMainChannels';

const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);

async function render() {
  const isFirstRun = await ipcRenderer.invoke(ipcMainChannels.IS_FIRST_RUN);
  const nCPU = await ipcRenderer.invoke(ipcMainChannels.GET_N_CPUS);
  ReactDom.render(
    <App
      isFirstRun={isFirstRun}
      nCPU={nCPU}
    />,
    document.getElementById('App')
  );
}

render();
// ipcRenderer.invoke(ipcMainChannels.IS_FIRST_RUN)
//   .then((response) => {
//     render(response);
//   });
