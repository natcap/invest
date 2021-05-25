import path from 'path';
import fs from 'fs';

import { ipcMain } from 'electron';

import extractZipInplace from './extractZipInplace';
import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function setupDownloadHandlers(mainWindow) {
  let downloadDir;
  let downloadLength;
  const downloadQueue = [];
  ipcMain.on(ipcMainChannels.DOWNLOAD_URL,
    async (event, urlArray, directory) => {
      logger.debug(`${urlArray}`);
      downloadDir = directory;
      downloadQueue.push(...urlArray);
      downloadLength = downloadQueue.length;
      mainWindow.webContents.send(
        'download-status',
        [(downloadLength - downloadQueue.length), downloadLength]
      );
      urlArray.forEach((url) => mainWindow.webContents.downloadURL(url));
    });

  mainWindow.webContents.session.on('will-download', (event, item) => {
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
          logger.info(`${item.getSavePath()}`);
          logger.info(`Received bytes: ${item.getReceivedBytes()}`);
        }
      }
    });
    item.once('done', async (event, state) => {
      if (state === 'completed') {
        logger.info(`${itemURL} complete`);
        await extractZipInplace(item.savePath);
        fs.unlink(item.savePath, (err) => {
          if (err) { logger.error(err); }
        });
        const idx = downloadQueue.findIndex((item) => item === itemURL);
        downloadQueue.splice(idx, 1);
        mainWindow.webContents.send(
          'download-status',
          [(downloadLength - downloadQueue.length), downloadLength]
        );
      } else {
        logger.info(`download failed: ${state}`);
      }
      if (!downloadQueue.length) {
        logger.info('all downloads complete');
      }
    });
  });
}
