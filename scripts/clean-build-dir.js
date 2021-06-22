'use strict';
/* We can't rely on babel to clean it's output directory every time
it compiles because,
1. There's a bug https://github.com/babel/babel/issues/9293
2. We have the special build/invest dir that is outside the scope
   of babel and that should persist.
*/

const fs = require('fs-extra');
const path = require('path');
const glob = require('glob');

const SRC_DIR = 'src';
const BUILD_DIR = 'build';

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
