import { app, ipcMain } from 'electron';

import { ipcMainChannels } from './ipcMainChannels';

export default function setupGetElectronPaths() {
  ipcMain.on(ipcMainChannels.GET_ELECTRON_PATHS, (event) => {
    event.returnValue = {
      resourcesPath: process.resourcesPath,
      userDataPath: app.getPath('userData')
    };
  });
}
