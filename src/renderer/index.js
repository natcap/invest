import { ipcRenderer } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import React from 'react';
import ReactDom from 'react-dom';
import App from './app';
import { ipcMainChannels } from '../main/ipcMainChannels';

import { getSettingsValue } from './components/SettingsModal/SettingsStorage';


const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);

// Create a right-click menu
// TODO: Not sure if Inspect Element should be available in production
// very useful in dev though.
let rightClickPosition = null;
window.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  rightClickPosition = { x: e.x, y: e.y };
  ipcRenderer.send(ipcMainChannels.SHOW_CONTEXT_MENU, rightClickPosition);
});

const language = await getSettingsValue('language');
console.log(language)
// call this before rendering the app so that _() is defined
// default to English
ipcRenderer.invoke(ipcMainChannels.SET_LANGUAGE, language);
window._ = ipcRenderer.sendSync.bind(null, ipcMainChannels.GETTEXT);  // partially applied function

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
