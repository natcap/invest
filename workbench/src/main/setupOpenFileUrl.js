import {
  ipcMain,
  BrowserWindow,
} from 'electron';

import { pathToFileURL } from 'node:url';

import { ipcMainChannels } from './ipcMainChannels';
import setupContextMenu from './setupContextMenu';

export default function setupOpenFileUrl(parentWindow, isDevMode) {
  ipcMain.on(
    ipcMainChannels.OPEN_FILE_URL, (event, filepath) => {
      const fileUrl = pathToFileURL(filepath);
      const [width, height] = parentWindow.getSize();
      const child = new BrowserWindow({
        parent: parentWindow,
        width: width > 1300 ? 1300 : width,
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
