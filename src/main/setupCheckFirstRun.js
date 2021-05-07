import fs from 'fs';
import path from 'path';
import {
  app,
  ipcMain,
} from 'electron';

import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

/** Check if the user has run this application before.
 *
 * @returns {boolean}
 */
export function checkFirstRun() {
  const userDataPath = app.getPath('userData');
  const hasRunTokenPath = path.join(userDataPath, 'app-has-run-token');
  try {
    if (fs.existsSync(hasRunTokenPath)) {
      return false;
    }
    fs.writeFileSync(hasRunTokenPath, '');
  } catch (error) {
    logger.warn(`Unable to write first-run token: ${error}`);
  }
  return true;
}

export function setupCheckFirstRun() {
  ipcMain.handle('is-first-run', (event) => checkFirstRun());
}
