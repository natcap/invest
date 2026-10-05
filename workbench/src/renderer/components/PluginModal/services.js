const { logger } = window.Workbench;

const registryMetadataURL = "https://natcap.github.io/invest-plugin-registry/workbench_metadata.json";
const dataCacheKey = "registryData";
const cacheTimeout = 1000 * 60 * 60 * 24; // 24 hours

function sortByName(a, b) {
  if (a.plugin_name > b.plugin_name) {
    return 1;
  }
  return -1;
}

export async function fetchRegistryData() {
  let cacheJSON = null;
  let cacheStale = true;

  // Check if data is cached in localStorage
  const cachedData = localStorage.getItem(dataCacheKey);

  if (cachedData) {
    cacheJSON = JSON.parse(cachedData);
    if (Date.now() - cacheJSON.cacheDate < cacheTimeout) {
      cacheStale = false;
    }
  }

  if (cacheJSON && !cacheStale) {
    logger.debug('Using cached data');
    return cacheJSON.data;
  } else {
    logger.debug('Cache miss; fetching data...');
    try {
      // Fetch data from the Registry if not cached
      const response = await fetch(registryMetadataURL);
      if (!response.ok) {
        throw new Error(`Response status: ${response.status}`);
      }
      const pluginJSON = await response.json();
      const sortedPlugins = pluginJSON.data.sort(sortByName);

      const cacheData = Object({
        'data': sortedPlugins,
        'cacheDate': Date.now()
      });

      // Cache the data in localStorage
      localStorage.setItem(dataCacheKey, JSON.stringify(cacheData));

      return sortedPlugins;
    } catch (error) {
      logger.error(error.message);
      return null;
    }
  }
}
