const https = require('https');
const fs = require('fs');
const path = require('path');
const url = require('url');
const fetch = require('node-fetch');
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

const HOSTNAME = pkg.invest.hostname;
const BUCKET = pkg.invest.bucket;
// forknames are only in the path on the dev-builds bucket
const FORK = BUCKET === 'releases.naturalcapitalproject.org'
  ? '' : pkg.invest.fork;
const REPO = 'invest';
const VERSION = pkg.invest.version;
const SRCFILE = `${filePrefix}_invest_binaries.zip`;
const DESTFILE = path.join(__dirname, '../build/binaries.zip');

const urladdress = url.resolve(
  HOSTNAME, path.join(BUCKET, REPO, FORK, VERSION, SRCFILE)
);

/**
 * Download a file from src to dest.
 *
 * @param  {string} src - url for a single publicly hosted file
 * @param  {string} dest - local path for saving the file
 */
function download(src, dest) {
  console.log(`downloading ${src}`);
  fs.existsSync(path.dirname(dest)) || fs.mkdirSync(path.dirname(dest));
  const fileStream = fs.createWriteStream(dest);
  https.get(src, (response) => {
    console.log(`http status: ${response.statusCode}`);
    if (response.statusCode !== 200) {
      fileStream.close();
      return;
    }
    response.pipe(fileStream);
    fileStream.on('finish', () => {
      fileStream.close();
    });
  }).on('error', (e) => {
    console.log(e);
  });
}

async function update_sampledata_registry() {
  const template = require('../src/sampledata_registry.json');
  // make a deep copy so we can check if any updates were made and
  // only overwrite the file if necessary.
  const registry = JSON.parse(JSON.stringify(template));
  const prefix = encodeURIComponent(`invest/${VERSION}/data`);
  const queryURL = `https://www.googleapis.com/storage/v1/b/${BUCKET}/o?prefix=${prefix}`;
  let data;
  const response = await fetch(queryURL);
  if (response.status === 200) {
    data = await response.json();
  } else {
    throw new Error(response.status);
  }
  // organize the data so we can index into it by filename
  const dataIndex = {};
  data.items.forEach((item) => {
    dataIndex[item.name] = item;
  });
  Object.keys(registry).forEach((model) => {
    const filename = `invest/${VERSION}/data/${registry[model].filename}`;
    try {
      registry[model].url = dataIndex[filename].mediaLink;
      registry[model].filesize = dataIndex[filename].size;
      // registry[model].filesize = parseFloat(
      //   `${dataIndex[filename].size / 1000000}`
      // ).toFixed(2) + ' MB';
    } catch {
      throw new Error(`no item found for ${filename} in ${JSON.stringify(dataIndex, null, 2)}`);
    }
  });
  if (JSON.stringify(template) === JSON.stringify(registry)) {
    console.log(`sample data registry is already up to date for invest ${VERSION}`);
    return;
  }
  fs.writeFileSync(
    path.join(__dirname, '../src/sampledata_registry.json'), JSON.stringify(registry, null, 2)
  );
  console.log('sample data registry was updated. Please review the changes and commit them');
  console.log('git diff src/sampledata_registry.json');
}

if (process.argv[2] && process.argv[2] === 'sampledata') {
  update_sampledata_registry();
} else {
  download(urladdress, DESTFILE);
  update_sampledata_registry();
}
