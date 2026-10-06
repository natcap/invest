const { logger } = window.Workbench;

const registryMetadataURL = "https://natcap.github.io/invest-plugin-registry/workbench_metadata.json";

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
