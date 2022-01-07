import { ipcRenderer } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import React from 'react';
import ReactDom from 'react-dom';
import App from './app';
import { ipcMainChannels } from '../main/ipcMainChannels';
import { getSettingsValue } from './components/SettingsModal/SettingsStorage';

const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);

const language = await getSettingsValue('language');

// call this before rendering the app so that _() is defined
ipcRenderer.invoke(ipcMainChannels.SET_LANGUAGE, language);
window._ = ipcRenderer.sendSync.bind(null, ipcMainChannels.GETTEXT);  // partially applied function

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
