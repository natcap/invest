import path from 'path';

import { app, ipcMain } from 'electron';

import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from './logger';

export default function setupGetElectronPaths() {
  ipcMain.on(ipcMainChannels.GET_ELECTRON_PATHS, (event) => {
    event.returnValue = {
      resourcesPath: process.resourcesPath,
      logfilePath: getLogger().transports.file.getFile().path
    };
  });
}
