import {
  ipcMain,
  app
} from 'electron';

const ELECTRON_DEV_MODE = !!process.defaultApp;

export default function setupIpcMainHandlers() {
  ipcMain.handle('is-dev-mode', async (event) => {
    const result = ELECTRON_DEV_MODE;
    return result;
  });

  ipcMain.handle('user-data', async (event) => {
    const result = app.getPath('userData');
    return result;
  });
}
