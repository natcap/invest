import { ipcMain } from 'electron';
import log from 'electron-log/main';

import { ipcMainChannels } from './ipcMainChannels';

const logger = log.create({ logId: 'renderer' });
logger.variables.label = 'renderer';
logger.transports.console.format = '[{h}:{i}:{s}.{ms}] [{label}] {text}';
logger.transports.file.format = '[{h}:{i}:{s}.{ms}] [{label}] {text}';
logger.transports.console.level = process.env.ELECTRON_LOG_LEVEL || 'debug';

export default function setupRendererLogger() {
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
