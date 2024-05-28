import upath from 'upath';
import fs from 'fs';
import { tmpdir } from 'os';
import toml from 'toml';
import { execSync } from 'child_process';
import { ipcMain } from 'electron';

import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';
import { settingsStore } from './settingsStore';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function setupAddPlugin() {
  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    (e, pluginURL) => {
      logger.info('adding plugin at', pluginURL);

      try {
        // Create a temporary directory and check out the plugin's pyproject.toml
        const tmpPluginDir = fs.mkdtempSync(upath.join(tmpdir(), 'natcap-invest-'));
        execSync(
          `git clone --depth 1 --no-checkout ${pluginURL} "${tmpPluginDir}"`,
          { stdio: 'inherit' }
        );
        execSync('git checkout HEAD pyproject.toml', { cwd: tmpPluginDir, stdio: 'inherit' });

        // Read in the plugin's pyproject.toml, then delete it
        const pyprojectTOML = toml.parse(fs.readFileSync(
          upath.join(tmpPluginDir, 'pyproject.toml')
        ).toString());
        fs.rmSync(tmpPluginDir, { recursive: true, force: true });

        // Access plugin metadata from the pyproject.toml
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
          `plugins.${pluginID}`,
          {
            model_name: pluginName,
            pyname: pluginPyName,
            type: 'plugin',
            source: pluginURL,
            env: envPath,
          }
        );
        logger.info('successfully added plugin');
      } catch (error) {
        return error;
      }
    }
  );
}
