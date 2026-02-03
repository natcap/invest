import {
  ipcMain,
  BrowserWindow,
} from 'electron';

import { pathToFileURL } from 'node:url';

import { ipcMainChannels } from './ipcMainChannels';
import setupContextMenu from './setupContextMenu';

export default function setupOpenLocalHtml(parentWindow, isDevMode) {
  ipcMain.on(
    ipcMainChannels.OPEN_LOCAL_HTML, (event, url) => {
      const fileUrl = pathToFileURL(url);
      const [width, height] = parentWindow.getSize();
      const child = new BrowserWindow({
        parent: parentWindow,
        width: width > 1300 ? 1300 : width, // accomodate reports and UG
        height: height,
        frame: true,
      });
      setupContextMenu(child);
      child.loadURL(fileUrl.toString());
      if (isDevMode) {
        child.webContents.openDevTools();
      }
    }
  );
}
