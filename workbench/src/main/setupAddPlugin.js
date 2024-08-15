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
        const mamba = settingsStore.get('mamba');
        execSync(`${mamba} create --yes --name ${envName} -c conda-forge "python<3.12" "gdal<3.6"`,
          { stdio: 'inherit' });
        logger.info('created mamba env for plugin');
        execSync(`${mamba} run --name ${envName} pip install "git+${pluginURL}"`,
          { stdio: 'inherit' });
        logger.info('installed plugin into its env');

        // Write plugin metadata to the workbench's config.json
        const envInfo = execSync(`${mamba} info --envs`).toString();
        logger.info(`env info:\n${envInfo}`);
        const envPath = envInfo.match(`${envName}\\s+(.+)\n`)[1];
        logger.info(`env path: ${envPath}`);
        logger.info('writing plugin info to settings store');
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
