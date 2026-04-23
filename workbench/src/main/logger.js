import log from 'electron-log';
import { ipcMain } from 'electron';

import { ipcMainChannels } from './ipcMainChannels';

const format = '[{h}:{i}:{s}.{ms}] [{label}] {text}';
const logLevel = process.env.ELECTRON_LOG_LEVEL || 'debug';

const mainLogger = log.create({ logId: 'main' });
mainLogger.variables.label = 'main';
mainLogger.transports.console.format = format;
mainLogger.transports.file.format = format;
mainLogger.transports.console.level = logLevel;

/**
 * Setup an IPC handler for log messages from the renderer process.
 *
 * These messages will log to the same console and file as the main process
 * logger.
 */
function setupRendererLogger() {
  const logger = log.create({ logId: 'renderer' });
  logger.variables.label = 'renderer';
  logger.transports.console.format = format;
  logger.transports.file.format = format;
  logger.transports.console.level = logLevel;

  ipcMain.on(ipcMainChannels.LOGGER, (event, level, message) => {
    switch (level) {
      case 'debug':
        logger.debug(message);
        break;
      case 'info':
        logger.info(message);
        break;
      case 'warning':
        logger.warning(message);
        break;
      case 'error':
        logger.error(message);
        break;
      default:
        logger.info(message);
    }
  });
}

export { mainLogger as logger, setupRendererLogger };
