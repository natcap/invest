const https = require('https')
const fs = require('fs')
const path = require('path')
const url = require('url')
const package = require('../package')

const args = process.argv.slice(2)
if (args.length !== 1) {
	throw new Error('expected exactly 1 argument: the current OS');
}
let fileSuffix;
switch (args[0]) {
	case 'windows-latest':
		fileSuffix = 'windows'
		break
	case 'macos-latest':
		fileSuffix = 'Darwin'
		break
	case 'ubuntu-latest':
		fileSuffix = 'Linux'
		break
	default:
		throw new Error("expected argument to be in ['windows-latest, 'macos-latest', 'ubuntu-latest']")
}

const HOSTNAME = package.invest.hostname
const BUCKET = package.invest.bucket
const FORK = package.invest.fork
const VERSION = package.invest.version
const SRCFILE = `invest_binaries_${fileSuffix}.zip`
const DESTFILE = './build/binaries.zip'

const urladdress = url.resolve(HOSTNAME, path.join(BUCKET, FORK, VERSION, SRCFILE))

const download = function(url, dest) {
	const fileStream = fs.createWriteStream(dest)
	const request = https.get(url, function(response) {
		console.log(response.statusCode)
		if (response.statusCode != 200) {
			fileStream.close()
			return
		}
		response.pipe(fileStream)
		fileStream.on('finish', function() {
			fileStream.close()
		})
	})
}

download(urladdress, DESTFILE)
