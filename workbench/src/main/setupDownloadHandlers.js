import path from 'path';
import fs from 'fs';

import { ipcMain } from 'electron';

import extractZipInplace from './extractZipInplace';
import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

/** Setup listeners and handlers for zipfile downloads initiated by users.
 *
 * Listen for DOWNLOAD_URL events from the renderer, which are passed with
 * an array of URLs for zip files, and a directory where files will be saved.
 *
 * Send messages back to the renderer to indicate progress in terms of number
 * of files completely downloaded and extracted.
 *
 * @param {BrowserWindow} mainWindow - instance of an electron BrowserWindow
 *
 * @returns {undefined}
 */
export default function setupDownloadHandlers(mainWindow) {
  let downloadDir;
  let downloadLength;
  const downloadQueue = [];
  ipcMain.on(ipcMainChannels.DOWNLOAD_URL,
    async (event, urlArray, directory) => {
      logger.debug(`User requesting downloads for: ${urlArray}`);
      downloadDir = directory;
      downloadQueue.push(...urlArray);
      downloadLength = downloadQueue.length;
      mainWindow.webContents.send(
        'download-status',
        [0, downloadLength]
      );
      urlArray.forEach((url) => mainWindow.webContents.downloadURL(url));
    });

  mainWindow.webContents.session.on('will-download', (event, item) => {
    // item is an instance of electron's DownloadItem
    const filename = item.getFilename();
    item.setSavePath(path.join(downloadDir, filename));
    const itemURL = item.getURL();
    item.on('updated', (event, state) => {
      if (state === 'interrupted') {
        logger.info('download interrupted');
      } else if (state === 'progressing') {
        if (item.isPaused()) {
          logger.info('download paused');
        } else {
          logger.info(`Saving: ${item.getSavePath()}`);
          logger.info(`Received bytes: ${item.getReceivedBytes()}`);
        }
      }
    });
    item.once('done', async (event, state) => {
      if (state === 'completed') {
        logger.info(`${itemURL} complete`);
        try {
          await extractZipInplace(item.savePath);
          fs.unlink(item.savePath, (err) => {
            if (err) { logger.error(err); }
          });
        } catch (error) {
          logger.error(`Something went wrong unzipping ${item.savePath}`);
          logger.error(error.stack);
        }
        const idx = downloadQueue.findIndex((item) => item === itemURL);
        downloadQueue.splice(idx, 1);
        mainWindow.webContents.send(
          'download-status',
          [(downloadLength - downloadQueue.length), downloadLength]
        );
      } else {
        logger.info(`download failed: ${state}`);
        mainWindow.webContents.send(
          'download-status',
          ['failed', 'failed'] // ProgressBar expects array length 2
        );
      }
      if (!downloadQueue.length) {
        logger.info('all downloads complete');
      }
    });
  });
}
