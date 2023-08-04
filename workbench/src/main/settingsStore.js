import { app, ipcMain } from 'electron';
import Store from 'electron-store';
import Ajv from 'ajv';

import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export const defaults = {
  nWorkers: -1,
  taskgraphLoggingLevel: 'INFO',
  loggingLevel: 'INFO',
  language: 'en',
};

export const schema = {
  type: 'object',
  properties: {
    nWorkers: {
      type: 'number',
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
  const ajv = new Ajv({ allErrors: true });
  const validate = ajv.compile(schema);
  const store = new Store({ defaults: data });
  const valid = validate(store.store);
  if (!valid) {
    validate.errors.forEach((e) => {
      logger.debug(e);
      let property;
      if (e.instancePath) {
        property = e.instancePath.split('/').pop();
      } else if (e.keyword === 'required') {
        property = e.params.missingProperty;
      } else {
        // something is invalid that we're not prepared to fix
        // so just reset the whole store to defaults.
        logger.debug(e);
        store.reset();
      }
      logger.debug(`resetting value for setting ${property}`);
      store.set(property, defaults[property]);
    });
  }
  return store;
}

export const settingsStore = initStore();

export function setupSettingsHandlers() {
  ipcMain.handle(
    ipcMainChannels.GET_SETTING,
    (event, key) => settingsStore.get(key)
  );

  ipcMain.on(
    ipcMainChannels.SET_SETTING,
    (event, key, value) => settingsStore.set(key, value)
  );

  // language is stored in the same store, but has special
  // needs for getting & setting because we need to get
  // the value synchronously during preload, and trigger
  // an app restart on change.
  ipcMain.on(ipcMainChannels.GET_LANGUAGE, (event) => {
    event.returnValue = settingsStore.get('language');
  });

  ipcMain.handle(
    ipcMainChannels.CHANGE_LANGUAGE,
    (e, languageCode) => {
      logger.debug('changing language to', languageCode);
      settingsStore.set('language', languageCode);
      app.relaunch();
      app.quit();
    }
  );
}
