'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs-extra');
const path = require('path');
const glob = require('glob');
const { DEFAULT_EXTENSIONS } = require('@babel/core');

const SRC_DIR = 'src';
const BUILD_DIR = 'build';
const ELECTRON_BUILDER_ENV = 'electron-builder.env';

if (process.argv[2] && process.argv[2] === 'clean') {
  clean();
} else {
  // clean before build just to remove any files that may
  // have been removed from src/ code but are still in build/
  // from a previous build.
  clean();
  build();
}

/** Remove all the files created during build()
 *
 * Do not remove other things in the build/ folder such as
 * invest binaries, which are not created during build().
 */
function clean() {
  const files = glob.sync(
    BUILD_DIR.concat(path.sep, '**', path.sep, '*'),
    {
      ignore: [
        path.join(BUILD_DIR, 'invest/**'),
        path.join(BUILD_DIR, 'pyi-build/**'),
      ]
    }
  );
  files.forEach((file) => {
    fs.unlinkSync(file);
  });
  try {
    fs.unlinkSync(ELECTRON_BUILDER_ENV);
  } catch {}
}

/** Transpile or copy all src/ files to build folder. */
function build() {
  if (!fs.existsSync(BUILD_DIR)) {
    fs.mkdirSync(BUILD_DIR);
  }

  // transpile all jsx and es6 files to javascript
  // excluding ResultsTab jsx because we've temporarily removed that feature
  const cmdArgs = [SRC_DIR, '-d', BUILD_DIR, '--ignore', '**/ResultsTab/*'];
  const runBabel = spawnSync('npx babel', cmdArgs, {
    shell: true,
  });

  console.log(`${runBabel.stdout}`);
  if (runBabel.stderr) {
    console.log(`${runBabel.stderr}`);
  }

  // copy all other files to their same relative location in the build dir
  const files = glob.sync(SRC_DIR.concat(path.sep, '**', path.sep, '*'));
  files.forEach((file) => {
    if (!DEFAULT_EXTENSIONS.includes(path.extname(file))) {
      const dest = file.replace(SRC_DIR, BUILD_DIR);
      fs.copySync(file, dest);
    }
  });
}
