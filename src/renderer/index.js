import { ipcRenderer } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import React from 'react';
import ReactDom from 'react-dom';

import App from './app';
import { ipcMainChannels } from '../main/ipcMainChannels';

const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);

// Create a right-click menu
let rightClickPosition = null;
if (window.Workbench.isDevMode) {
  window.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    rightClickPosition = { x: e.x, y: e.y };
    ipcRenderer.send(ipcMainChannels.SHOW_CONTEXT_MENU, rightClickPosition);
  });
}

function render(isFirstRun) {
  ReactDom.render(
    <App isFirstRun={isFirstRun} />,
    document.getElementById('App')
  );
}

ipcRenderer.invoke(ipcMainChannels.IS_FIRST_RUN)
  .then((response) => {
    render(response);
  });
