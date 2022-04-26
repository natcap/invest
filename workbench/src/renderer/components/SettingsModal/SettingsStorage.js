import localforage from 'localforage';

const logger = window.Workbench.getLogger('SettingsStorage.js');

const investSettingsStore = localforage.createInstance({
  name: 'InvestSettings',
});

/** Getter function for global default settings.
 *
 * @returns {object} to destructure into:
 *     {String} nWorkers - TaskGraph number of workers
 *     {String} taskgraphLoggingLevel - InVEST taskgraph logging level
 *     {String} loggingLevel - InVEST model logging level
 *     {String} sampleDataDir - default location for sample datastack downloads
 */
export function getDefaultSettings() {
  const defaultSettings = {
    nWorkers: '-1',
    taskgraphLoggingLevel: 'INFO',
    loggingLevel: 'INFO',
    sampleDataDir: '',
    language: 'en'
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
  if (!value) {
    return getDefaultSettings()[key];
  }
  return value;
}

/** Getter function for entire contents of store.
 *
 * @returns {Object} - key: value pairs of settings
 */
export async function getAllSettings() {
  try {
    const promises = [];
    const keys = Object.keys(getDefaultSettings());
    keys.forEach((key) => {
      promises.push(getSettingsValue(key));
    });
    const values = await Promise.all(promises);
    const settings = Object.fromEntries(keys.map(
      (_, i) => [keys[i], values[i]]
    ));
    return settings;
  } catch (err) {
    logger.error(err.message);
    return getDefaultSettings();
  }
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
