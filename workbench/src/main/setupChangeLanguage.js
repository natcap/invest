import Store from 'electron-store';
import { ipcMain } from 'electron';
import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const store = new Store();

export default function setupChangeLanguage() {
  ipcMain.handle(
    ipcMainChannels.CHANGE_LANGUAGE,
    (e, languageCode) => {
      logger.debug('changing language to', languageCode);
      store.set('language', languageCode);
    }
  );
}
