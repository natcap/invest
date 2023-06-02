import { ipcMain } from 'electron';
import Store from 'electron-store';

import { ipcMainChannels } from './ipcMainChannels';

const defaults = {
  nWorkers: '-1',
  taskgraphLoggingLevel: 'INFO',
  loggingLevel: 'INFO',
  sampleDataDir: '',
  language: 'en'
};

export const settingsStore = new Store({ defaults: defaults });

export function setupSettingsHandlers() {
  ipcMain.handle(ipcMainChannels.GET_SETTING, (event, key) => settingsStore.get(key));
  ipcMain.on(ipcMainChannels.SET_SETTING, (event, key, value) => {
    settingsStore.set(key, value);
  });
}
