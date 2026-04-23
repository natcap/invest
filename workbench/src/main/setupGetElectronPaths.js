import { ipcMain } from 'electron';

import { ipcMainChannels } from './ipcMainChannels';
import { logger } from './logger';

export default function setupGetElectronPaths() {
  ipcMain.on(ipcMainChannels.GET_ELECTRON_PATHS, (event) => {
    event.returnValue = {
      resourcesPath: process.resourcesPath,
      logfilePath: logger.transports.file.getFile().path
    };
  });
}
