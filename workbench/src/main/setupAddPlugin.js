import Store from 'electron-store';
import { app, ipcMain } from 'electron';
import path from 'path';
import { execSync } from 'child_process';

import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// const store = new Store();

export default function setupAddPlugin() {

  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    (e, pluginURL) => {
      logger.info('adding plugin at', pluginURL);
      // store.set('language', languageCode);
      const pluginsPath = path.join(app.getPath('userData'), 'plugins');
      const configPath = path.join(app.getPath('userData'), 'config.json');
      logger.info(pluginsPath);
      const stdout = execSync(
        `src/main/addPlugin.sh ${pluginURL} '${pluginsPath}' '${configPath}'`,
        { encoding: 'utf-8' }
      );
      logger.info(stdout);
    }
  );
}


