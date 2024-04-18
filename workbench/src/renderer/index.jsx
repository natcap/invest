import React from 'react';
import { createRoot } from 'react-dom/client';

import App from './app';
import './i18n/i18n';
import ErrorBoundary from './components/ErrorBoundary';
import { ipcMainChannels } from '../main/ipcMainChannels';
import { getInvestModelNames } from './server_requests';

const { ipcRenderer } = window.Workbench.electron;

const isFirstRun = await ipcRenderer.invoke(ipcMainChannels.IS_FIRST_RUN);
const nCPU = await ipcRenderer.invoke(ipcMainChannels.GET_N_CPUS);

if (isFirstRun) {
  const investList = getInvestModelNames();
  await Object.keys(investList).forEach(async (modelName) => {
    const modelId = investList[modelName].model_name;
    await ipcRenderer.invoke(ipcMainChannels.SET_SETTING(`models.${modelId}.model_name`, modelName));
    await ipcRenderer.invoke(ipcMainChannels.SET_SETTING(`models.${modelId}.type`, 'core'));
  });
}

const root = createRoot(document.getElementById('App'));
root.render(
  <ErrorBoundary>
    <App
      isFirstRun={isFirstRun}
      nCPU={nCPU}
    />
  </ErrorBoundary>
);
