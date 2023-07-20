import { ipcMain } from 'electron';
import Store from 'electron-store';
import Ajv from 'ajv';

import { ipcMainChannels } from './ipcMainChannels';

const defaults = {
  nWorkers: '-1',
  taskgraphLoggingLevel: 'INFO',
  loggingLevel: 'INFO',
  language: 'en',
  foo: 'bar'
};

const schema = {
  type: 'object',
  properties: {
    nWorkers: {
      type: 'string',
      default: '-1'
    },
    taskgraphLoggingLevel: {
      enum: ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
      default: 'INFO'
    },
    loggingLevel: {
      enum: ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
      default: 'INFO'
    },
    language: {
      enum: ['en', 'es', 'zh'],
      default: 'en'
    },
    foo: {
      type: 'string',
      default: 'bar'
    }
  },
  required: ['nWorkers', 'taskgraphLoggingLevel', 'loggingLevel', 'language', 'foo']
};

// if the schema includes new properties compared to the file,
// we can update the file with default value for the missing prop.
const ajv = new Ajv();
const validate = ajv.compile(schema);

const store = new Store({ defaults: defaults });
console.log(store.store)

// if the schema for a property has changed making it invalid, reset that property to default

// if the schema contains fewer properties than present in file, you're probably
// running an older version of the workbench, should be valid if additionalProperties are okay

const valid = validate(store.store);
console.log(valid)
if (!valid) {
  console.log(validate.errors)
  validate.errors.forEach((e) => {
    store.set(e.keyword, schema.properties[e.keyword].default);
  });
  // store.reset(invalidKeys);
  // store = new Store({ defaults: defaults });
}

export const settingsStore = store;

export function setupSettingsHandlers() {
  ipcMain.handle(ipcMainChannels.GET_SETTING, (event, key) => settingsStore.get(key));
  ipcMain.on(ipcMainChannels.SET_SETTING, (event, key, value) => {
    settingsStore.set(key, value);
  });
}
