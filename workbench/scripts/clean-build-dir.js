/* We can't rely on babel to clean it's output directory every time
it compiles because there's a bug
https://github.com/babel/babel/issues/9293
*/

const fs = require('fs-extra');
const path = require('path');
const glob = require('glob');

const BUILD_DIR = 'build';

/** Remove all the files created during build()
 *
 */
function clean() {
  const files = glob.sync(
    BUILD_DIR.concat(path.sep, '**', path.sep, '*')
  );
  files.forEach((file) => {
    try {
      fs.unlinkSync(file);
    } catch {}
  });
}

clean();
