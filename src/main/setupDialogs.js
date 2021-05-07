import {
  ipcMain,
  dialog,
} from 'electron';

export default function setupDialogs() {
  ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await dialog.showOpenDialog(options);
    return result;
  });

  ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await dialog.showSaveDialog(options);
    return result;
  });
}
