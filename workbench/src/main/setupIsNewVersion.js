import fs from 'fs';
import path from 'path';

import {
  app,
  ipcMain,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from './logger';
import pkg from '../../package.json';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export const APP_VERSION_TOKEN = 'app-version-token.txt';

/** Determine whether this is the first run of the current running version.
 *
 * @returns {boolean} true if this version has not run before, otherwise false
 */
export async function isNewVersion() {
  // Getting version from package.json is simplest because there is no need to
  // spawn an invest process simply to get the version of the installed binary.
  const currentVersion = pkg.version;
  const userDataPath = app.getPath('userData');
  const tokenPath = path.join(userDataPath, APP_VERSION_TOKEN);
  try {
    if (fs.existsSync(tokenPath)) {
      const tokenString = fs.readFileSync(tokenPath, {encoding: 'utf8'});
      if (tokenString) {
        const installedVersionList = tokenString.split(',');
        if (installedVersionList.includes(currentVersion)) {
          return false;
        }
        // If current version not found, add it
        fs.writeFileSync(tokenPath, `${tokenString},${currentVersion}`);
        return true;
      }
    }
    // If file does not exist, create it
    fs.writeFileSync(tokenPath, currentVersion);
  } catch (error) {
    logger.warn(`Unable to write app-version token: ${error}`);
  }
  return true;
}

export function setupIsNewVersion() {
  ipcMain.handle(
    ipcMainChannels.IS_NEW_VERSION, () => isNewVersion()
  );
}
