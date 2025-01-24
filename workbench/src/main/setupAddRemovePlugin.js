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

/**
 * Spawn a child process and log its stdout, stderr, and any error in spawning.
 *
 * child_process.spawn is called with the provided cmd, args, and options,
 * and the windowsHide option set to true. The shell option is set to true
 * because spawn by default sets shell to false.
 *
 * Required properties missing from the store are initialized with defaults.
 * Invalid properties are reset to defaults.
 * @param  {string} cmd - command to pass to spawn
 * @param  {Array} args - command arguments to pass to spawn
 * @param  {object} options - options to pass to spawn.
 * @returns {Promise} resolves when the command finishes with exit code 0.
 *                    Rejects with error otherwise.
 */
function spawnWithLogging(cmd, args, options) {
  logger.info(cmd, args);
  const cmdProcess = spawn(
    cmd, args, { ...options, shell: true, windowsHide: true });
  if (cmdProcess.stdout) {
    cmdProcess.stderr.on('data', (data) => logger.info(data.toString()));
    cmdProcess.stdout.on('data', (data) => logger.info(data.toString()));
  }
  return new Promise((resolve, reject) => {
    cmdProcess.on('error', (err) => {
      logger.error(err);
      reject(err);
    });
    cmdProcess.on('close', (code) => {
      if (code === 0) {
        resolve(code);
      } else {
        reject(code);
      }
    });
  });
}

export function setupAddPlugin() {
  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    async (e, url, revision, path) => {
      try {
        let pyprojectTOML;
        let installString;
        if (url) { // install from git URL
          if (revision) {
            installString = `git+${url}@${revision}`;
            logger.info(`adding plugin from ${installString}`);
          } else {
            installString = `git+${url}`;
            logger.info(`adding plugin from ${installString} at default branch`);
          }

          // Create a temporary directory and check out the plugin's pyproject.toml,
          // without downloading any extra files or git history
          const tmpPluginDir = fs.mkdtempSync(upath.join(tmpdir(), 'natcap-invest-'));
          await spawnWithLogging('git', ['clone', '--depth', 1, '--no-checkout', url, tmpPluginDir]);
          let head = 'HEAD';
          if (revision) {
            head = 'FETCH_HEAD';
            await spawnWithLogging(
              'git',
              ['fetch', 'origin', `${revision}`],
              { cwd: tmpPluginDir }
            );
          }
          await spawnWithLogging(
            'git',
            ['checkout', head, '--', 'pyproject.toml'],
            { cwd: tmpPluginDir }
          );
          // Read in the plugin's pyproject.toml, then delete it
          pyprojectTOML = toml.parse(fs.readFileSync(
            upath.join(tmpPluginDir, 'pyproject.toml')
          ).toString());
          fs.rmSync(tmpPluginDir, { recursive: true, force: true });
        } else { // install from local path
          logger.info(`adding plugin from ${path}`);
          installString = path;
          // Read in the plugin's pyproject.toml
          pyprojectTOML = toml.parse(fs.readFileSync(
            upath.join(path, 'pyproject.toml')
          ).toString());
        }

        const micromamba = settingsStore.get('micromamba');
        // Access plugin metadata from the pyproject.toml
        const pluginID = pyprojectTOML.tool.natcap.invest.model_id;
        const pluginName = pyprojectTOML.tool.natcap.invest.model_name;
        const pluginPyName = pyprojectTOML.tool.natcap.invest.pyname;
        // Create a conda env containing the plugin and its dependencies
        const envName = `invest_plugin_${pluginID}`;
        await spawnWithLogging(
          micromamba,
          ['create', '--yes', '--name', envName, '-c', 'conda-forge', '"python<3.12"', '"gdal<3.6"']
        );
        logger.info('created micromamba env for plugin');
        await spawnWithLogging(
          micromamba,
          ['run', '--name', envName, 'pip', 'install', installString]
        );
        logger.info('installed plugin into its env');
        // Write plugin metadata to the workbench's config.json
        const envInfo = execSync(`${micromamba} info --name ${envName}`, { windowsHide: true }).toString();
        logger.info(`env info:\n${envInfo}`);
        const regex = /env location : (.+)/;
        const envPath = envInfo.match(regex)[1];
        logger.info(`env path:\n${envPath}`);
        logger.info('writing plugin info to settings store');
        settingsStore.set(
          `plugins.${pluginID}`,
          {
            model_name: pluginName,
            pyname: pluginPyName,
            type: 'plugin',
            source: installString,
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

export function setupRemovePlugin() {
  ipcMain.handle(
    ipcMainChannels.REMOVE_PLUGIN,
    async (e, pluginID) => {
      logger.info('removing plugin', pluginID);
      try {
        // Delete the plugin's conda env
        const env = settingsStore.get(`plugins.${pluginID}.env`);
        const micromamba = settingsStore.get('micromamba');
        await spawnWithLogging(micromamba, ['remove', '--yes', '--prefix', `"${env}"`, '--all']);
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
