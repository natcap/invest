import fs from 'fs';
import path from 'path';

import {
  ipcMain,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function setupCheckFilePermissions() {
  ipcMain.handle(
    ipcMainChannels.CHECK_FILE_PERMISSIONS, (event, folder) => {
      const filepath = path.join(folder, 'foo.txt');
      try {
        // The only reliable way to determine if a folder is writeable
        // is to write to it. https://github.com/nodejs/node/issues/2949
        fs.writeFileSync(filepath, '');
        fs.rm(filepath, (err) => logger.debug(err));
        return true;
      } catch (err) {
        logger.debug(err);
        return false;
      }
    }
  );
}
