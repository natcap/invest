import i18n from 'i18next';
import { ipcMain } from 'electron';
import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function setupChangeLanguage() {
  ipcMain.handle(
    ipcMainChannels.CHANGE_LANGUAGE,
    (e, languageCode) => {
      logger.debug('changing language to', languageCode);
      i18n.changeLanguage(languageCode);
    }
  );
}
