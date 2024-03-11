import upath from 'upath';
import { execSync } from 'child_process';
import { app, ipcMain } from 'electron';

import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function setupAddPlugin() {
  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    (e, pluginURL) => {
      logger.info('adding plugin at', pluginURL);
      // store.set('language', languageCode);
      const pluginsPath = upath.join(app.getPath('userData'), 'plugins');
      const configPath = upath.join(app.getPath('userData'), 'config.json');
      logger.info(pluginsPath);
      execSync(
        `src/main/addPlugin.sh ${pluginURL} '${pluginsPath}' '${configPath}'`,
        { encoding: 'utf-8', stdio: 'inherit' }
      );
    }
  );
}
