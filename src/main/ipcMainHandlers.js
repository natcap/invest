import {
  ipcMain,
  BrowserWindow,
  Menu,
  dialog,
  app
} from 'electron';

import { checkFirstRun } from '../main_helpers';
import pkg from '../../package.json';

const ELECTRON_DEV_MODE = !!process.defaultApp;

// TODO: move this to a preload script and make global vars?
const mainProcessVars = {
  investVersion: pkg.invest.version,
  workbenchVersion: pkg.version,
  userDataPath: app.getPath('userData'),
  isFirstRun: checkFirstRun(),
};

export default function setupIpcMainHandlers() {
  ipcMain.handle('show-context-menu', (event, rightClickPos) => {
    const template = [
      {
        label: 'Inspect Element',
        click: () => {
          BrowserWindow.fromWebContents(event.sender)
            .inspectElement(rightClickPos.x, rightClickPos.y);
        }
      },
    ];
    const menu = Menu.buildFromTemplate(template);
    menu.popup(BrowserWindow.fromWebContents(event.sender));
  });

  ipcMain.handle('variable-request', async (event) => {
    return mainProcessVars;
  });

  ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await dialog.showOpenDialog(options);
    return result;
  });

  ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await dialog.showSaveDialog(options);
    return result;
  });

  ipcMain.handle('is-dev-mode', async (event) => {
    const result = ELECTRON_DEV_MODE;
    return result;
  });

  ipcMain.handle('user-data', async (event) => {
    const result = await mainProcessVars.userDataPath;
    return result;
  });
}
