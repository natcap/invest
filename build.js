'use strict';

const spawn = require('child_process').spawn;
const fs = require('fs-extra');
// const fs = require('fs');
const path = require('path');
const glob = require('glob');

const SRC_DIR = 'src'
const BUILD_DIR = 'build';

if (!fs.existsSync(BUILD_DIR)) {
	fs.mkdirSync(BUILD_DIR)
}

const cmdArgs = [SRC_DIR, '-d', BUILD_DIR]
const investRun = spawn('npx babel', cmdArgs, {
        shell: true, 
      });

// fs.copySync('src', 'build', {
//     dereference: true,
//     filter: file => filterJS
//   })

// function filterJS(src, dest) {
// 	console.log(src);
// 	return path.extname(src) === 'html'
// }

glob(SRC_DIR.concat('/**/*'), (err, files) => {
	files.forEach(file => {
		if (['.css', '.html'].includes(path.extname(file))) {
			console.log(file)
			const dest = file.replace(SRC_DIR, BUILD_DIR)
			console.log(dest)
			fs.copySync(file, dest)
		}
	})
})
// fs.readdir('src', (err, files) => {
// 	files.forEach(file => {
// 		if (['.css', '.html'].includes(path.extname(file))) {
// 			console.log(file);
// 			// fs.copyFileSync(file)
// 		}
// 	})
// })
// fs.copySync('src', 'build', {
//     dereference: true,
//   })