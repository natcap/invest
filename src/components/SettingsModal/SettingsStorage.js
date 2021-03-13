import localforage from 'localforage';

import { getLogger } from '../../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const investSettingsStore = localforage.createInstance({
  name: 'InvestSettings',
});

/** Getter function for global default settings.
 *
 * @returns {object} to destructure into two args:
 *     {String} nWorkers - TaskGraph number of workers
 *     {String} logggingLevel - InVEST model logging level
 */
function getDefaultSettings() {
  const defaultSettings = {
    nWorkers: '-1',
    loggingLevel: 'INFO',
  };
  return defaultSettings;
}

/** Helper function for testing purposes
 *
 * @returns {object} localforage store for invest settings
 */
function getSettingsStore() {
  const postSettings = { ...investSettingsStore };
  return postSettings;
}

async function getSettingsValue(key) {
  return await investSettingsStore.getItem(key);
}

/** Helper function for testing purposes */
async function clearSettingsStore() {
  await investSettingsStore.clear();
}

/** Helper function to save key, value to the store */
async function saveSettings(settingsObj) {
  try {
    for (const [setting, value] of Object.entries(settingsObj)) {
      await investSettingsStore.setItem(setting, value);
    }
  } catch (err) {
    logger.error(`Error saving settings: ${err}`);
  }
}

export const settingsStorage = {
    getDefaultSettings: getDefaultSettings,
    saveSettings: saveSettings,
    getSettingsValue: getSettingsValue,
}
