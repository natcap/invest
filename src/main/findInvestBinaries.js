import path from 'path';
import { execFileSync } from 'child_process';

import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
/**
 * Find paths to local invest executeable under dev or production environments.
 *
 * @param {boolean} isDevMode - a boolean designating dev mode or not.
 * @returns {string} invest binary path string.
 */
export default function findInvestBinaries(isDevMode) {
  try {
    // Binding to the invest server binary:
    let investExe;
    const ext = (process.platform === 'win32') ? '.exe' : '';
    const filename = `invest${ext}`;

    if (isDevMode) {
      investExe = path.join('build', 'invest', filename);
    } else {
      const binaryPath = path.join(process.resourcesPath, 'invest');
      // It's likely the path includes spaces because it's composed of the
      // app's Product Name, a user-facing name given to electron-builder.
      // escape spaces because https://github.com/nodejs/node/issues/38490
      investExe = path.join(binaryPath, filename).replace(/(\s+)/g, '\\$1');
    }
    // Checking that we have a functional invest exe by getting version
    const investVersion = execFileSync(investExe, ['--version']);
    logger.info(
      `Found invest binaries ${investExe} for version ${investVersion}`
    );
    return investExe;
  } catch (error) {
    logger.error(error.message);
    logger.error('InVEST binaries are probably missing.');
    throw error;
  }
}
