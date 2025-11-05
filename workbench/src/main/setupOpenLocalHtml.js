import {
  ipcMain,
  BrowserWindow,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';
import setupContextMenu from './setupContextMenu';

export default function setupOpenLocalHtml(parentWindow, isDevMode) {
  ipcMain.on(
    ipcMainChannels.OPEN_LOCAL_HTML, (event, url) => {
      const [width, height] = parentWindow.getSize();
      const child = new BrowserWindow({
        parent: parentWindow,
        width: width > 1300 ? 1300 : width, // accomodate reports and UG
        height: height,
        frame: true,
      });
      setupContextMenu(child);
      child.loadURL(url);
      if (isDevMode) {
        child.webContents.openDevTools();
      }
    }
  );
}
