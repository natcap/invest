const https = require('https')
const fs = require('fs')
const path = require('path')
const url = require('url')

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
console.log(fileSuffix)

// todo, move this config to package.json, other config file,
// and/or command line args, e.g. Actions YAML should set OS suffix
// for SRCFILE
const HOSTNAME = 'https://storage.googleapis.com/'
const BUCKET = 'natcap-dev-build-artifacts'
const FORK = 'invest/davemfish'
const VERSION = '3.8.0.post632+g7a58318e/'
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
