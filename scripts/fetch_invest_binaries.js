const https = require('https');
const fs = require('fs');
const path = require('path');
const url = require('url');
const pkg = require('../package');

let filePrefix;
// The invest build process only builds for these OS
switch (process.platform) {
  case 'win32':
    filePrefix = 'windows';
    break;
  case 'darwin':
    filePrefix = 'macos';
    break;
  // case 'linux':
  //   filePrefix = 'linux';
  //   break;
  default:
    throw new Error(
      `No prebuilt invest binaries are available for ${process.platform}`
    );
}

const HOSTNAME = pkg.invest.hostname;
const BUCKET = pkg.invest.bucket;
const FORK = pkg.invest.fork;
const VERSION = pkg.invest.version;
const SRCFILE = `${filePrefix}_invest_binaries.zip`;
const DESTFILE = path.resolve('build/binaries.zip');

const urladdress = url.resolve(
  HOSTNAME, path.join(BUCKET, FORK, VERSION, SRCFILE)
);

/**
 * Download a file from src to dest.
 *
 * @param  {string} src - url for a single publicly hosted file
 * @param  {string} dest - local path for saving the file
 */
function download(src, dest) {
  console.log(`downloading ${url}`);
  fs.existsSync(path.dirname(dest)) || fs.mkdirSync(path.dirname(dest));
  const fileStream = fs.createWriteStream(dest);
  https.get(src, (response) => {
    console.log(response.statusCode);
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

download(urladdress, DESTFILE);
