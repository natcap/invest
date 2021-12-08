const fs = require('fs');
const path = require('path');
const url = require('url');

const fetch = require('node-fetch');

const pkg = require('../package');

const { bucket, fork } = pkg.invest;
const repo = 'invest';
const VERSION = pkg.invest.version;
let DATA_PREFIX;
// forknames are only in the path on the dev-builds bucket
if (bucket === 'releases.naturalcapitalproject.org') {
  DATA_PREFIX = `${repo}/${VERSION}/data`;
} else if (bucket === 'natcap-dev-build-artifacts') {
  DATA_PREFIX = `${repo}/${fork}/${VERSION}/data`;
}
const STORAGE_URL = 'https://storage.googleapis.com';
const DATA_QUERY_URL = url.resolve(
  'https://www.googleapis.com/storage/v1/b/',
  `${bucket}/o?prefix=${encodeURIComponent(DATA_PREFIX)}`
);

const dataRegistryRelativePath = 'renderer/sampledata_registry.json';
const DATA_REGISTRY_SRC_PATH = path.join(
  __dirname, '../src/', dataRegistryRelativePath
);
const DATA_REGISTRY_BUILD_PATH = path.join(
  __dirname, '../build/', dataRegistryRelativePath
);

/** Update a JSON registry of sampledata zipfiles available on GCS.
 *
 * This function validates that all the Models in the registry have
 * a "filename" that exists in our storage bucket for the invest version
 * configured in package.json. An error is raised if the file is not found.
 * Most likely that indicates a typo in the registry's model name or filename.
 *
 * If the invest version has changed, this function updates the registry
 * with new urls & filesizes. If the invest version hasn't changed,
 * we still query the bucket to confirm the files are still there, but
 * the registry JSON won't be changed.
 *
 * If a new InVEST model is added, or a new sampledata zipfile, a new
 * entry in the registry should be added manually, but the `url` and `filesize`
 * keys may be left out and then this script can be run to populate those keys.
 */
async function updateSampledataRegistry() {
  const template = require(DATA_REGISTRY_SRC_PATH);
  // make a deep copy so we can check if any updates were made and
  // only overwrite the file if necessary.
  const registry = JSON.parse(JSON.stringify(template));

  async function queryStorage(endpoint) {
    let data;
    const response = await fetch(endpoint);
    if (response.status === 200) {
      data = await response.json();
    } else {
      throw new Error(response.status);
    }
    if (!data.items) {
      throw new Error(`no items found at ${DATA_QUERY_URL}`);
    }
    // organize the data so we can index into it by filename
    const dataIndex = {};
    data.items.forEach((item) => {
      dataIndex[item.name] = item;
    });
    return dataIndex;
  }
  const dataItems = await queryStorage(DATA_QUERY_URL);
  Object.keys(registry).forEach((model) => {
    const itemName = `${decodeURIComponent(DATA_PREFIX)}/${registry[model].filename}`;
    try {
      registry[model].url = `${STORAGE_URL}/${bucket}/${dataItems[itemName].name}`;
      registry[model].filesize = dataItems[itemName].size;
    } catch {
      throw new Error(
        `no item found for ${itemName} in ${JSON.stringify(dataItems, null, 2)}`
      );
    }
  });

  if (JSON.stringify(template) === JSON.stringify(registry)) {
    console.log(`sample data registry is already up to date for invest ${VERSION}`);
    return;
  }
  fs.writeFileSync(
    DATA_REGISTRY_SRC_PATH,
    JSON.stringify(registry, null, 2).concat('\n')
  );
  // babel does this copy also, but doing it here too so that
  // it doesn't matter if this script is run before or after npm run build
  fs.copyFileSync(
    DATA_REGISTRY_SRC_PATH,
    DATA_REGISTRY_BUILD_PATH
  );
  console.log('sample data registry was updated. Please review the changes and commit them');
}

updateSampledataRegistry();
