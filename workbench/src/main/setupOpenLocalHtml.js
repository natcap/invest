import {
  ipcMain,
  BrowserWindow,
  shell,
} from 'electron';
import contextMenu from 'electron-context-menu';
import { download, CancelError } from 'electron-dl';

import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

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
        append: (defaultActions, parameters, browserWindow) => [
          {
            // The default Save As item does not handle rejections from download,
            // So this is a reconstruction of that item, with error handling.
            // https://github.com/sindresorhus/electron-context-menu/issues/195
            label: 'Save Image As...',
            // Only show it when right-clicking an image
            visible: parameters.mediaType === 'image',
            click: async (menuItem) => {
              parameters.srcURL = menuItem.transform ? menuItem.transform(parameters.srcURL) : parameters.srcURL;
              try {
                await download(win, parameters.srcURL, { saveAs: true });
              } catch (error) {
                if (error instanceof CancelError) {
                  logger.info('Download item was cancelled');
                } else {
                  logger.error(error);
                }
              }
            },
          },
        ],
        showSaveImageAs: false,
        showSearchWithGoogle: false,
      });

      win.webContents.setWindowOpenHandler((details) => {
        shell.openExternal(details.url);
        return { action: 'deny' };
      });

      win.loadURL(url);
      if (isDevMode) {
        win.webContents.openDevTools();
      }
    }
  );
}
