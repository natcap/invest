const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);
const isDevMode = process.argv.includes('--dev');
if (isDevMode) {
  require('react-devtools');
}

import { ipcRenderer } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import React from 'react';
import ReactDom from 'react-dom';

import App from './app';

// Create a right-click menu
// TODO: Not sure if Inspect Element should be available in production
// very useful in dev though.
let rightClickPosition = null;
window.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  rightClickPosition = { x: e.x, y: e.y };
  ipcRenderer.send('show-context-menu', rightClickPosition);
});

function render(isFirstRun) {
  ReactDom.render(
    <App isFirstRun/>,
    document.getElementById('App')
  )
};

ipcRenderer.invoke('is-first-run')
  .then((response) => {
    render(response.isFirstRun);
  })
