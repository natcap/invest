const https = require('https')
const fs = require('fs')
const path = require('path')
const url = require('url')

// todo, move this config to package.json or other config file
const HOSTNAME = 'https://storage.googleapis.com/'
const BUCKET = 'natcap-dev-build-artifacts'
const FORK = 'invest/davemfish'
const VERSION = '3.8.0.post629+g0e68c01c'
const SRCFILE = 'invest_binaries_Linux.zip'
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