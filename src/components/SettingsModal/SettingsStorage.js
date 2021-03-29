import localforage from 'localforage';

import { getLogger } from '../../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const investSettingsStore = localforage.createInstance({
  name: 'InvestSettings',
});

/** Getter function for global default settings.
 *
 * @returns {object} to destructure into:
 *     {String} nWorkers - TaskGraph number of workers
 *     {String} loggingLevel - InVEST model logging level
 *     {String} sampleDataDir - default location for sample datastack downloads
 */
export function getDefaultSettings() {
  const defaultSettings = {
    nWorkers: '-1',
    loggingLevel: 'INFO',
    sampleDataDir: '',
  };
  return defaultSettings;
}

/** Getter function for settings store value.
 *
 * @param {object} obj.argsValues - an invest "args dict" with initial values
 * @param {string} key - setting key to get value
 *
 * @returns {string} - value of the setting key.
 */
export async function getSettingsValue(key) {
  const value = await investSettingsStore.getItem(key);
  console.log(`getSettings ${key}: ${value}`);
  return value;
}

/** Clear the settings store. */
export async function clearSettingsStore() {
  await investSettingsStore.clear();
}

/** Setter function for saving store values.
 *
 * @param {object} settingsObj - object with one or more key:value pairs
 *
 */
export async function saveSettingsStore(settingsObj) {
  try {
    for (const [setting, value] of Object.entries(settingsObj)) {
      await investSettingsStore.setItem(setting, value);
    }
  } catch (err) {
    logger.error(`Error saving settings: ${err}`);
  }
}
