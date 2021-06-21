'use strict';

const fs = require('fs-extra');
const path = require('path');
const glob = require('glob');

const SRC_DIR = 'src';
const BUILD_DIR = 'build';

if (process.argv[2] === 'clean') {
  clean();
} else {
  // clean before build just to remove any files that may
  // have been removed from src/ code but are still in build/
  // from a previous build.
  clean();
  // TODO: formerly we were doing other stuff here,
  // and we might again in the future with work on
  // https://github.com/natcap/invest-workbench/issues/146
  // So we should clean this up as part of that issue.
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
    try {
      fs.unlinkSync(file);
    } catch {}
  });
}
