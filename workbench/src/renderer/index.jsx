import React from 'react';
import { createRoot } from 'react-dom/client';

import App from './app';
import './i18n/i18n';
import ErrorBoundary from './components/ErrorBoundary';
import { ipcMainChannels } from '../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

const isFirstRun = await ipcRenderer.invoke(ipcMainChannels.IS_FIRST_RUN);
const nCPU = await ipcRenderer.invoke(ipcMainChannels.GET_N_CPUS);

const root = createRoot(document.getElementById('App'));
root.render(
  <ErrorBoundary>
    <App
      isFirstRun={isFirstRun}
      nCPU={nCPU}
    />
  </ErrorBoundary>
);
