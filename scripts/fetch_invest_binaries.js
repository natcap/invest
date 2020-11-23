/**
 * 2020-07-30 - the workbench build process no longer depends on
 * pre-built binaries, instead building them itself, so this script
 * is not being used anymore.
 * 
 * This is a utility script for fetching prebuilt invest binaries
 * that include the server.py module and thus are compatible with the
 * workbench.
 */

const https = require('https')
const fs = require('fs')
const path = require('path')
const url = require('url')
const package = require('../package')

let fileSuffix;
switch (process.platform) {
	case 'win32':
		filePrefix = 'windows'
		break
	case 'darwin':
		filePrefix = 'macos'
		break
	case 'linux':
		filePrefix = 'linux'
		break
	default:
		throw new Error("expected platform to be windows, mac, or linux")
}

const HOSTNAME = package.invest.hostname
const BUCKET = package.invest.bucket
const FORK = package.invest.fork
const VERSION = package.invest.version
const SRCFILE = `${filePrefix}_invest_binaries.zip`
const DESTFILE = path.resolve('build/binaries.zip');

const urladdress = url.resolve(HOSTNAME, path.join(BUCKET, FORK, VERSION, SRCFILE))

const download = function(url, dest) {
	console.log(`fetching ${url}`)
	fs.existsSync(path.dirname(dest)) || fs.mkdirSync(path.dirname(dest));
	const fileStream = fs.createWriteStream(dest)
	const request = https.get(url, function(response) {
		console.log(response.statusCode)
		if (response.statusCode != 200) {
			fileStream.close()
			console.log(url)
			return
		}
		response.pipe(fileStream)
		fileStream.on('finish', function() {
			fileStream.close()
		})
	})
}

download(urladdress, DESTFILE)
