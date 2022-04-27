import fs from 'fs';
import path from 'path';

import {
  ipcMain,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export const STORAGE_TOKEN = 'storage_token.txt';

/** Check for a token from the invest build process.
 *
 * It contains the google storage url for sampledata associated
 * with this invest version.
 *
 * @returns {string} a public google storage url.
 */
export function checkStorageToken() {
  const tokenPath = path.join(process.resourcesPath, STORAGE_TOKEN);
  try {
    const data = fs.readFileSync(tokenPath, { encoding: 'utf8' });
    logger.debug(data);
    return data;
  } catch (error) {
    logger.warn(`Unable to check release token: ${error}`);
  }
  return undefined;
}

export function setupCheckStorageToken() {
  ipcMain.handle(
    ipcMainChannels.CHECK_STORAGE_TOKEN, () => checkStorageToken()
  );
}
