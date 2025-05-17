import upath from 'upath';
import fs from 'fs';
import { tmpdir } from 'os';
import toml from 'toml';
import { execFile, execSync, spawn } from 'child_process';
import { promisify } from 'util';
import { app, ipcMain } from 'electron';
import { Downloader } from 'nodejs-file-downloader';

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
  let errMessage;
  if (cmdProcess.stdout) {
    cmdProcess.stderr.on('data', (data) => {
      errMessage = data.toString();
      logger.info(errMessage);
    });
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
        reject(errMessage);
      }
    });
  });
}

export function setupAddPlugin(i18n) {
  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    async (event, url, revision, path) => {
      try {
        let pyprojectTOML;
        let installString;
        const micromamba = settingsStore.get('micromamba');
        const rootPrefix = upath.join(app.getPath('userData'), 'micromamba_envs');
        if (url) { // install from git URL
          if (revision) {
            installString = `git+${url}@${revision}`;
            logger.info(`adding plugin from ${installString}`);
          } else {
            installString = `git+${url}`;
            logger.info(`adding plugin from ${installString} at default branch`);
          }
          const baseEnvPrefix = upath.join(rootPrefix, 'invest_base');
          // Create invest_base environment, if it doesn't already exist
          // The purpose of this environment is just to ensure that git is available
          if (!fs.existsSync(baseEnvPrefix)) {
            event.sender.send('plugin-install-status', i18n.t('Creating base environment...'));
            await spawnWithLogging(
              micromamba,
              ['create', '--yes', '--prefix', `"${baseEnvPrefix}"`, '-c', 'conda-forge', 'git']
            );
          }

          // Create a temporary directory and check out the plugin's pyproject.toml,
          // without downloading any extra files or git history
          event.sender.send('plugin-install-status', i18n.t('Downloading plugin source code...'));
          const tmpPluginDir = fs.mkdtempSync(upath.join(tmpdir(), 'natcap-invest-'));
          await spawnWithLogging(
            micromamba,
            ['run', '--prefix', `"${baseEnvPrefix}"`,
              'git', 'clone', '--depth', 1, '--no-checkout', url, tmpPluginDir]);
          let head = 'HEAD';
          if (revision) {
            head = 'FETCH_HEAD';
            await spawnWithLogging(
              micromamba,
              ['run', '--prefix', `"${baseEnvPrefix}"`, 'git', 'fetch', 'origin', `${revision}`],
              { cwd: tmpPluginDir }
            );
          }
          await spawnWithLogging(
            micromamba,
            ['run', '--prefix', `"${baseEnvPrefix}"`, 'git', 'checkout', head, '--', 'pyproject.toml'],
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
        // Access plugin metadata from the pyproject.toml
        const condaDeps = pyprojectTOML.tool.natcap.invest.conda_dependencies;
        const packageName = pyprojectTOML.tool.natcap.invest.package_name;
        const version = pyprojectTOML.project.version;

        // Create a conda env containing the plugin and its dependencies
        // use timestamp to ensure a unique path
        // I wanted the env path to match the plugin model_id, but we can't
        // know the model_id until after creating the environment to be able to
        // import metadata from the MODEL_SPEC. And mamba does not support
        // renaming or moving environments after they're created.
        const pluginEnvPrefix = upath.join(rootPrefix, `plugin_${Date.now()}`);
        const createCommand = [
          'create', '--yes', '--prefix', `"${pluginEnvPrefix}"`,
          '-c', 'conda-forge', 'python', 'git'];
        if (condaDeps) { // include dependencies read from pyproject.toml
          condaDeps.forEach((dep) => createCommand.push(`"${dep}"`));
        }
        event.sender.send('plugin-install-status', i18n.t('Creating plugin environment...'));
        await spawnWithLogging(micromamba, createCommand);
        logger.info('created micromamba env for plugin');
        event.sender.send('plugin-install-status', i18n.t('Installing plugin into environment...'));
        await spawnWithLogging(
          micromamba,
          ['run', '--prefix', `"${pluginEnvPrefix}"`,
           'python', '-m', 'pip', 'install', installString]
        );
        logger.info('installed plugin into its env');

        event.sender.send('plugin-install-status', i18n.t('Importing plugin...'));
        // Access plugin metadata from the MODEL_SPEC
        const modelID = execSync(
          `micromamba run --prefix "${pluginEnvPrefix}" ` +
          `python -c "import ${packageName}; print(${packageName}.MODEL_SPEC.model_id)"`
        ).toString().trim();
        const modelTitle= execSync(
          `micromamba run --prefix "${pluginEnvPrefix}" ` +
          `python -c "import ${packageName}; print(${packageName}.MODEL_SPEC.model_title)"`
        ).toString().trim();
        console.log(modelID);
        console.log(modelTitle);

        // Write plugin metadata to the workbench's config.json
        logger.info('writing plugin info to settings store');
        const pluginID = `${modelID}@${version}`;
        settingsStore.set(
          // Escape dots so that electron-store doesn't expand them into separate keys
          `plugins.${pluginID.replace(/\./g, '\\.')}`,
          {
            modelID: modelID,
            modelTitle: modelTitle,
            type: 'plugin',
            source: installString,
            env: pluginEnvPrefix,
            version: version,
          }
        );
        logger.info('successfully added plugin');
      } catch (error) {
        console.log(error);
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
        const env = settingsStore.get(`plugins.${pluginID.replace(/\./g, '\\.')}.env`);
        const micromamba = settingsStore.get('micromamba');
        await spawnWithLogging(micromamba, ['env', 'remove', '--yes', '--prefix', `"${env}"`]);
        // Delete the plugin's data from storage
        settingsStore.delete(`plugins.${pluginID}`);
        logger.info('successfully removed plugin');
      } catch (error) {
        logger.info('Error removing plugin:');
        logger.info(error);
        return error;
      }
    }
  );
}

export function setupWindowsMSVCHandlers() {
  ipcMain.handle(
    ipcMainChannels.HAS_MSVC,
    () => {
      return fs.existsSync(upath.join('C:', 'Windows', 'System32', 'VCRUNTIME140_1.dll'));
    }
  );

  ipcMain.handle(
    ipcMainChannels.DOWNLOAD_MSVC,
    async () => {
      const tmpDir = app.getPath('temp');
      const exeName = 'vc_redist.x64.exe';
      const downloader = new Downloader({
        url: 'https://aka.ms/vs/17/release/vc_redist.x64.exe',
        directory: tmpDir,
        fileName: exeName,
      });
      try {
        await downloader.download();
        logger.info("Download complete");
      } catch (error) {
        logger.error("Download failed", error);
      }
      logger.info('Launching MSVC installer');
      const exePath = upath.join(tmpDir, exeName);
      await promisify(execFile)(exePath, ['/norestart']);
    }
  );
}
