import Store from 'electron-store';
import { ipcMain } from 'electron';

import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// const store = new Store();

export default function setupAddPlugin() {

  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    (e, url) => {
      logger.info('adding plugin at', url);
      // store.set('language', languageCode);
    }
  );
}
