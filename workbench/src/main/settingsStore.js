import { ipcMain } from 'electron';
import Store from 'electron-store';
import Ajv from 'ajv';

import { ipcMainChannels } from './ipcMainChannels';

export const defaults = {
  nWorkers: '-1',
  taskgraphLoggingLevel: 'INFO',
  loggingLevel: 'INFO',
  language: 'en',
};

export const schema = {
  type: 'object',
  properties: {
    nWorkers: {
      type: 'string',
    },
    taskgraphLoggingLevel: {
      enum: ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
    },
    loggingLevel: {
      enum: ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
    },
    language: {
      enum: ['en', 'es', 'zh'],
    },
  },
  required: ['nWorkers', 'taskgraphLoggingLevel', 'loggingLevel', 'language']
};

/**
 * Open a store and validate against a schema.
 *
 * Required properties missing from the store are initialized with defaults.
 * Invalid properties are reset to defaults.
 *
 * @param  {object} data key-values with which to initialize a store.
 * @returns {Store} an instance of an electron-store Store
 */
export function initStore(data = defaults) {
  const ajv = new Ajv();
  const validate = ajv.compile(schema);
  const store = new Store({ defaults: data });
  const valid = validate(store.store);
  if (!valid) {
    validate.errors.forEach((e) => {
      store.set(e.keyword, defaults[e.keyword]);
    });
  }
  return store;
}

export const settingsStore = initStore();

export function setupSettingsHandlers() {
  ipcMain.handle(ipcMainChannels.GET_SETTING, (event, key) => settingsStore.get(key));
  ipcMain.on(ipcMainChannels.SET_SETTING, (event, key, value) => {
    settingsStore.set(key, value);
  });
}
