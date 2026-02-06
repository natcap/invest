import {
  ipcMain,
  BrowserWindow,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';

export default function setupOpenLocalHtml(parentWindow, isDevMode) {
  ipcMain.on(
    ipcMainChannels.OPEN_LOCAL_HTML, (_event, url, isFilePath = false) => {
      if (isFilePath) {
        url = require('node:url').pathToFileURL(url).toString();
      }
      const [width, height] = parentWindow.getSize();
      const child = new BrowserWindow({
        parent: parentWindow,
        width: width > 1300 ? 1300 : width, // accommodate reports and UG
        height: height,
        frame: true,
      });
      child.loadURL(url);
      if (isDevMode) {
        child.webContents.openDevTools();
      }
    }
  );
}
