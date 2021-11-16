const os = require('os');
const fs = require('fs');
const https = require('https');
const path = require('path');
const url = require('url');
const { exec, execFileSync } = require('child_process');

const fetch = require('node-fetch');
const fsExtra = require('fs-extra');

const pkg = require('../package');

let filePrefix;
// The invest build process only builds for these OS
// The prefixes on the zip file are defined by invest's Makefile $OSNAME
switch (process.platform) {
  case 'win32':
    filePrefix = 'windows';
    break;
  case 'darwin':
    filePrefix = 'mac';
    break;
  default:
    throw new Error(
      `No prebuilt invest binaries are available for ${process.platform}`
    );
}
const binaryZipName = `${filePrefix}_invest_binaries.zip`;

const { bucket, fork } = pkg.invest;
const repo = 'invest';
const VERSION = pkg.invest.version;
const DESTFILE = path.join(
  __dirname, `../build/version_${VERSION}_${binaryZipName}`
);
const INVEST_BINARIES_DIR = 'build/invest/';
let DATA_PREFIX;
let binaryZipBucketPath;
// forknames are only in the path on the dev-builds bucket
if (bucket === 'releases.naturalcapitalproject.org') {
  binaryZipBucketPath = `${bucket}/${repo}/${VERSION}/${binaryZipName}`;
  DATA_PREFIX = `${repo}/${VERSION}/data`;
} else if (bucket === 'natcap-dev-build-artifacts') {
  binaryZipBucketPath = `${bucket}/${repo}/${fork}/${VERSION}/${binaryZipName}`;
  DATA_PREFIX = `${repo}/${fork}/${VERSION}/data`;
}
const SRC_BINARY_URL = url.resolve(
  'https://storage.googleapis.com',
  binaryZipBucketPath
);
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

/**
 * Download a zip file and unzip it, overwriting all.
 *
 * @param  {string} src - url for a single file
 * @param  {string} dest - local path for saving the file
 */
function downloadAndUnzipBinaries(src, dest) {
  fs.existsSync(path.dirname(dest)) || fs.mkdirSync(path.dirname(dest));
  const fileStream = fs.createWriteStream(dest);
  https.get(src, (response) => {
    console.log(`http status: ${response.statusCode}`);
    if (response.statusCode !== 200) {
      fileStream.close();
      throw new Error(`${response.statusCode} for ${src}`);
    }
    response.pipe(fileStream);
    fileStream.on('finish', () => {
      fileStream.close();
      const unzip = exec(`unzip -o ${dest} -d ${INVEST_BINARIES_DIR}`);
      unzip.stdout.on('data', (data) => {
        console.log(`${data}`);
      });
      unzip.stderr.on('data', (data) => {
        console.log(`${data}`);
      });
      unzip.on('close', (code) => {
        if (code === 0) {
          fs.unlinkSync(dest);
        } else {
          throw new Error(code);
        }
      });
    });
  }).on('error', (e) => {
    console.log(e);
  });
}

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
  // const googleAPI = 'https://www.googleapis.com/storage/v1/b';
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
    const filename = `${decodeURIComponent(DATA_PREFIX)}/${registry[model].filename}`;
    try {
      registry[model].url = dataItems[filename].mediaLink;
      registry[model].filesize = dataItems[filename].size;
    } catch {
      throw new Error(
        `no item found for ${filename} in ${JSON.stringify(dataItems, null, 2)}`
      );
    }
  });

  if (JSON.stringify(template) === JSON.stringify(registry)) {
    console.log(`sample data registry is already up to date for invest ${VERSION}`);
    return;
  }
  fs.writeFileSync(
    DATA_REGISTRY_SRC_PATH,
    JSON.stringify(registry, null, 2)
  );
  // babel does this copy also, but doing it here too so that
  // it doesn't matter if this script is run before or after npm run build
  fs.copyFileSync(
    DATA_REGISTRY_SRC_PATH,
    DATA_REGISTRY_BUILD_PATH
  );
  console.log('sample data registry was updated. Please review the changes and commit them');
}

if (process.argv[2] && process.argv[2] === 'sampledata') {
  updateSampledataRegistry();
} else {
  // Find the local binaries that we'll be packaging, if they already
  // exist and match the version in package.json, we don't need to download.
  let willDownload = true;
  const ext = (process.platform === 'win32') ? '.exe' : '';
  const investExe = `build/invest/invest${ext}`;

  try {
    const investVersion = execFileSync(investExe, ['--version']);
    if (`${investVersion}`.trim(os.EOL) === VERSION) {
      willDownload = false;
      console.log(`binaries already up-to-date with version ${VERSION}`);
    } else {
      console.log(`existing binaries are outdated at version ${investVersion}`);
      console.log(`will download for version ${VERSION}`);
    }
  } catch {
    console.log(`no local binaries in ${INVEST_BINARIES_DIR}`);
    console.log(`will download for version ${VERSION}`);
  } finally {
    if (willDownload) {
      fsExtra.emptyDirSync(INVEST_BINARIES_DIR);
      downloadAndUnzipBinaries(SRC_BINARY_URL, DESTFILE);
    }
    updateSampledataRegistry();
  }
}
