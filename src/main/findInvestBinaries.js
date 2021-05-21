import path  from 'path';
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
      // If no dotenv vars are set, default to where this project's
      // build process places the binaries.
      // TODO: remove or add back the dotenv functionality?
      investExe = process.env.INVEST
        || path.join('build', 'invest', filename);

    // point to binaries included in this app's installation.
    } else {
      const binaryPath = path.join(process.resourcesPath, 'invest');
      investExe = path.join(binaryPath, filename);
    }
    // Checking that we have a functional invest exe by getting version
    const investVersion = execFileSync(investExe, ['--version']);
    logger.info(`Found invest binaries ${investExe} for version ${investVersion}`);
    return investExe;
  } catch (error) {
    logger.error(error.message);
    logger.error('InVEST binaries are probably missing.');
    throw new Error(error.message);
  }
}
