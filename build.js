'use strict';

const spawnSync = require('child_process').spawnSync;
const fs = require('fs-extra');
const path = require('path');
const glob = require('glob');

const SRC_DIR = 'src'
const BUILD_DIR = 'build';

if (!fs.existsSync(BUILD_DIR)) {
	fs.mkdirSync(BUILD_DIR)
}

// transpile jsx and es6 files to javasciprt
const cmdArgs = [SRC_DIR, '-d', BUILD_DIR]
const investRun = spawnSync('npx babel', cmdArgs, {
        shell: true, 
      });

// copy all other files to their same relative location in the build dir
glob(SRC_DIR.concat('/**/*'), (err, files) => {
	files.forEach(file => {
		if (['.css', '.html'].includes(path.extname(file))) {
			const dest = file.replace(SRC_DIR, BUILD_DIR)
			fs.copySync(file, dest)
		}
	})
})

console.log('build directory contains:')
console.log(fs.readdirSync(BUILD_DIR));