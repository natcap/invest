import { ipcMain } from 'electron';
import Store from 'electron-store';

import { ipcMainChannels } from './ipcMainChannels';

const defaults = {
  nWorkers: '-1',
  taskgraphLoggingLevel: 'INFO',
  loggingLevel: 'INFO',
  language: 'en'
};

const schema = {
  nWorkers: {
    type: 'string',
    default: '-1',
  },
  taskgraphLoggingLevel: {
    enum: ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
    default: 'INFO',
  },
  loggingLevel: {
    enum: ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
    default: 'INFO',
  },
  language: {
    enum: ['en', 'es', 'zh'],
    default: 'en',
  },
};

export const settingsStore = new Store({ schema });

export function setupSettingsHandlers() {
  ipcMain.handle(ipcMainChannels.GET_SETTING, (event, key) => settingsStore.get(key));
  ipcMain.on(ipcMainChannels.SET_SETTING, (event, key, value) => {
    settingsStore.set(key, value);
  });
}
