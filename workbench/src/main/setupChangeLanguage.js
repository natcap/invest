import Store from 'electron-store';
import { app, ipcMain } from 'electron';
import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const store = new Store();

export default function setupChangeLanguage() {
  ipcMain.on(ipcMainChannels.GET_LANGUAGE, (event) => {
    // default to en if no language setting exists
    event.returnValue = store.get('language', 'en');
  });

  ipcMain.handle(
    ipcMainChannels.CHANGE_LANGUAGE,
    (e, languageCode) => {
      logger.debug('changing language to', languageCode);
      store.set('language', languageCode);
      app.relaunch();
      app.quit();
    }
  );
}
