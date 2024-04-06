import upath from 'upath';
import fs from 'fs';
import toml from 'toml';
import { execSync } from 'child_process';
import { app, ipcMain } from 'electron';

import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';
import { settingsStore } from './settingsStore';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function setupAddPlugin() {
  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    (e, pluginURL) => {
      logger.info('adding plugin at', pluginURL);
      const pluginsDir = upath.join(app.getPath('userData'), 'plugins');

      // Download the plugin from its remote git repo
      if (!fs.existsSync(pluginsDir)) {
        fs.mkdirSync(pluginsDir);
      }
      const pluginRepoName = upath.basename(pluginURL, upath.extname(pluginURL));
      fs.rmSync(`${pluginsDir}/${pluginRepoName}`, { recursive: true, force: true });
      execSync(`git clone ${pluginURL} "${pluginsDir}/${pluginRepoName}"`, { stdio: 'inherit' });

      // Read in plugin metadata from the pyproject.toml
      const pyprojectTOML = toml.parse(fs.readFileSync(
        `${pluginsDir}/${pluginRepoName}/pyproject.toml`
      ).toString());
      const pluginID = pyprojectTOML.tool.natcap.invest.model_id;
      const pluginName = pyprojectTOML.tool.natcap.invest.model_name;
      const pluginPyName = pyprojectTOML.tool.natcap.invest.pyname;

      // Create a conda env containing the plugin and its dependencies
      const envName = `invest_plugin_${pluginID}`;
      fs.writeFileSync('tmp_env.txt', 'python<3.12\ngdal<3.6');
      execSync(`micromamba create --yes --name ${envName} -f tmp_env.txt -c conda-forge`);
      execSync(`micromamba run --name ${envName} pip install "git+${pluginURL}"`, { stdio: 'inherit' });

      // Write plugin metadata to the workbench's config.json
      const envInfo = execSync(`micromamba info --name ${envName}`).toString();
      const envPath = envInfo.split('env location : ')[1].split('\n')[0];
      settingsStore.set(
        `models.${pluginID}`,
        {
          model_name: pluginName,
          pyname: pluginPyName,
          type: 'plugin',
          source: pluginURL,
          env: envPath,
        }
      );
      logger.info('successfully added plugin');
    }
  );
}
