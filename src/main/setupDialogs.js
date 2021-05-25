import {
  ipcMain,
  dialog,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';

export default function setupDialogs() {
  ipcMain.handle(
    ipcMainChannels.SHOW_OPEN_DIALOG, async (event, options) => {
      const result = await dialog.showOpenDialog(options);
      return result;
    }
  );

  ipcMain.handle(
    ipcMainChannels.SHOW_SAVE_DIALOG, async (event, options) => {
      const result = await dialog.showSaveDialog(options);
      return result;
    }
  );
}
