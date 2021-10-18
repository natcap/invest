import { ipcRenderer } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import React from 'react';
import ReactDom from 'react-dom';

import App from './app';
import { ipcMainChannels } from '../main/ipcMainChannels';

const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);

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
