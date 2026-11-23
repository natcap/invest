const { logger } = window.Workbench;

const registryMetadataURL = "https://natcap.github.io/invest-plugin-registry/workbench_metadata.json";
const downloadStatsURL = "https://storage.googleapis.com/invest-plugin-registry/plugin_download_stats.json";

function sortByName(a, b) {
  if (a.plugin_name > b.plugin_name) {
    return 1;
  }
  return -1;
}

export async function fetchRegistryData() {
  try {
    const response = await fetch(registryMetadataURL);
    if (!response.ok) {
      throw new Error(`Error fetching Plugin Registry data. Response status: ${response.status}`);
    }
    const pluginJSON = await response.json();
    const sortedPlugins = pluginJSON.data.sort(sortByName);
    return sortedPlugins;
  } catch (error) {
    logger.error(error.message);
    return null;
  }
}

export async function fetchDownloadStats() {
  try {
    const response = await fetch(downloadStatsURL);
    if (!response.ok) {
      throw new Error(`Error fetching download stats. Response status: ${response.status}`);
    }
    const downloadStatsJSON = await response.json();
    return downloadStatsJSON.data;
  } catch (error) {
    logger.error(error.message);
    return null;
  }
}
