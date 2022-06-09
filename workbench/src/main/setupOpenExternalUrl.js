import {
  ipcMain,
  shell,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';

export default function setupOpenExternalUrl() {
  ipcMain.on(
    ipcMainChannels.OPEN_EXTERNAL_URL, (event, url) => {
      shell.openExternal(url);
    }
  );
}
