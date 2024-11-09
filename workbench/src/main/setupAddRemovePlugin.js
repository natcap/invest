import upath from 'upath';
import fs from 'fs';
import { tmpdir } from 'os';
import toml from 'toml';
import { execSync, spawn } from 'child_process';
import { ipcMain } from 'electron';

import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';
import { settingsStore } from './settingsStore';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

function customSpawn(cmd, args, options) {
  logger.info(cmd, args);
  const cmdProcess = spawn(cmd, args, options);
  if (cmdProcess.stdout) {
    cmdProcess.stderr.on('data', (data) => logger.info(data.toString()));
    cmdProcess.stdout.on('data', (data) => logger.info(data.toString()));
  }
  return new Promise((resolve, reject) => {
    cmdProcess.on('close', (code) => { resolve(code); });
  });
}

export function setupAddPlugin() {
  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    (e, pluginURL) => {
      logger.info('adding plugin at', pluginURL);

      const mamba = settingsStore.get('mamba');
      let envName;
      let pluginID;
      let pluginName;
      let pluginPyName;

      try {
        // Create a temporary directory and check out the plugin's pyproject.toml
        const tmpPluginDir = fs.mkdtempSync(upath.join(tmpdir(), 'natcap-invest-'));
        customSpawn(
          'git',
          ['clone', '--depth', '1', '--no-checkout', pluginURL, tmpPluginDir],
          { windowsHide: true }
        ).then(async () => {
          await customSpawn(
            'git',
            ['checkout', 'HEAD', 'pyproject.toml'],
            { cwd: tmpPluginDir, windowsHide: true }
          );
        }).then(async () => {
          // Read in the plugin's pyproject.toml, then delete it
          const pyprojectTOML = toml.parse(fs.readFileSync(
            upath.join(tmpPluginDir, 'pyproject.toml')
          ).toString());
          fs.rmSync(tmpPluginDir, { recursive: true, force: true });

          // Access plugin metadata from the pyproject.toml
          pluginID = pyprojectTOML.tool.natcap.invest.model_id;
          pluginName = pyprojectTOML.tool.natcap.invest.model_name;
          pluginPyName = pyprojectTOML.tool.natcap.invest.pyname;

          // Create a conda env containing the plugin and its dependencies
          envName = `invest_plugin_${pluginID}`;
          await customSpawn(
            mamba,
            ['create', '--yes', '--name', envName, '-c', 'conda-forge', '"python<3.12"', '"gdal<3.6"'],
            { windowsHide: true }
          );
          logger.info('created mamba env for plugin');
        }).then(async () => {
          await customSpawn(
            mamba,
            ['run', '--verbose', '--no-capture-output', '--name', envName, 'pip', 'install', `git+${pluginURL}`],
            { windowsHide: true }
          );
          logger.info('installed plugin into its env');
        })
          .then(() => {
            // Write plugin metadata to the workbench's config.json
            const envInfo = execSync(`${mamba} info --envs`, { windowsHide: true }).toString();
            logger.info(`env info:\n${envInfo}`);

            const regex = new RegExp(String.raw`^${envName} +(.+)$`, 'm');
            const envPath = envInfo.match(regex)[1];

            logger.info(`env path:\n${envPath}`);
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
          });
      } catch (error) {
        return error;
      }
    }
  );
}

export function setupRemovePlugin() {
  ipcMain.handle(
    ipcMainChannels.REMOVE_PLUGIN,
    (e, pluginID) => {
      logger.info('removing plugin', pluginID);
      try {
        // Delete the plugin's conda env
        const env = settingsStore.get(`plugins.${pluginID}.env`);
        const mamba = settingsStore.get('mamba');
        execSync(
          `${mamba} remove --yes --prefix ${env} --all`,
          { stdio: 'inherit' }
        );
        // Delete the plugin's data from storage
        settingsStore.delete(`plugins.${pluginID}`);
        logger.info('successfully removed plugin');
      } catch (error) {
        logger.info('Error removing plugin:');
        logger.info(error);
      }
    }
  );
}
