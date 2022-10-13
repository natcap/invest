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
