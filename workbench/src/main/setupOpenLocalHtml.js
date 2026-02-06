import {
  ipcMain,
  BrowserWindow,
} from 'electron';
import contextMenu from 'electron-context-menu';

import { ipcMainChannels } from './ipcMainChannels';

export default function setupOpenLocalHtml(parentWindow, isDevMode) {
  ipcMain.on(
    ipcMainChannels.OPEN_LOCAL_HTML, (_event, url, isFilePath = false) => {
      if (isFilePath) {
        url = require('node:url').pathToFileURL(url).toString();
      }
      const [width, height] = parentWindow.getSize();
      const win = new BrowserWindow({
        width: width > 1300 ? 1300 : width, // accommodate reports and UG
        height: height,
        frame: true,
        webPreferences: {
          partition: url, // do not share the webContents.session of other windows
        },
      });
      contextMenu({
        window: win,
        showSaveImageAs: true,
        showSearchWithGoogle: false,
      });
      win.loadURL(url);
      if (isDevMode) {
        win.webContents.openDevTools();
      }
    }
  );
}
