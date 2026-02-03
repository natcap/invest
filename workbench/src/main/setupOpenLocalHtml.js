import {
  ipcMain,
  BrowserWindow,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';

export default function setupOpenLocalHtml(parentWindow, isDevMode) {
  ipcMain.on(
    ipcMainChannels.OPEN_LOCAL_HTML, (event, url) => {
      const [width, height] = parentWindow.getSize();
      const child = new BrowserWindow({
        parent: parentWindow,
        width: width > 1000 ? 1000 : width, // UG content is never wider
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
