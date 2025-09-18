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
import { shutdownPythonProcess } from './createPythonFlaskProcess';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

/**
 * Spawn a child process and log its stdout, stderr, and any error in spawning.
 *
 * child_process.spawn is called with the provided cmd, args, and options,
 * and the windowsHide option set to true. The shell option is set to true
 * because spawn by default sets shell to false.
 *
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

/**
 * Create a base micromamba environment containing git.
 * @param  {string} baseEnvPrefix location for the environment
 * @param  {string} micromamba    path to the micromamba executable
 * @return {Promise}
 */
async function createBaseEnv(baseEnvPrefix, micromamba) {
  // Create environment from a YML file so that we can specify nodefaults
  // which is needed for licensing reasons. micromamba does not support
  // disabling the default channel in the command line.
  const baseEnvYMLPath = upath.join(app.getPath('temp'), 'baseEnv.yml');
  const baseEnvYMLContents = (
    'channels:\n' +
    '- conda-forge\n' +
    '- nodefaults\n' +
    'dependencies:\n' +
    '- git\n');
  logger.info(`creating base environment from file:\n${baseEnvYMLContents}`);
  fs.writeFileSync(baseEnvYMLPath, baseEnvYMLContents);
  await spawnWithLogging(micromamba, [
    'create', '--yes', '--prefix', `"${baseEnvPrefix}"`,
    '--file', `"${baseEnvYMLPath}"`]);
}

/**
 * Clone a git repository to a temporary directory and check out a revision
 * @param  {string} micromamba    path to the micromamba executable
 * @param  {string} baseEnvPrefix the base environment where git is available
 * @param  {string} url           a git repository
 * @param  {string} revision      a valid git revision (branch, tag, commit hash)
 * @param  {string} tmpPluginDir  a location to clone to
 * @return {Promise}
 */
async function cloneAndCheckoutPlugin(
  micromamba, baseEnvPrefix, url, revision, tmpPluginDir
) {
  await spawnWithLogging(
    micromamba, [
      'run', '--prefix', `"${baseEnvPrefix}"`,
      'git', 'clone', '--depth', 1, '--no-checkout', url, tmpPluginDir,
    ]
  );

  let head = 'HEAD';
  if (revision) {
    head = 'FETCH_HEAD';
    await spawnWithLogging(
      micromamba,
      [
        'run', '--prefix', `"${baseEnvPrefix}"`,
        'git', 'fetch', 'origin', `${revision}`,
      ],
      { cwd: tmpPluginDir }
    );
  }
  await spawnWithLogging(
    micromamba,
    [
      'run', '--prefix', `"${baseEnvPrefix}"`,
      'git', 'checkout', head, '--', 'pyproject.toml',
    ],
    { cwd: tmpPluginDir }
  );
}

/**
 * Install a plugin and its dependencies into a new micromamba environment.
 * @param  {string} micromamba      path to the micromamba executable
 * @param  {string} pluginEnvPrefix location for the new environment
 * @param  {string} condaDeps       dependencies to install from conda-forge
 * @param  {string} installString   pip install argument for the plugin
 * @param  {Object} i18n            i18n instance for translating status messages
 * @param  {Object} event           ipcMain event object so we can send status
 *                                  updates to renderer
 * @return {Promise}
 */
async function installPlugin(
  micromamba, pluginEnvPrefix, condaDeps, installString, i18n, event
) {
  // Create environment from a YML file so that we can specify nodefaults
  // which is needed for licensing reasons. micromamba does not support
  // disabling the default channel in the command line.
  const pluginEnvYMLPath = upath.join(app.getPath('temp'), 'env.yml');
  let pluginEnvYMLContents = (
    'channels:\n' +
    '- conda-forge\n' +
    '- nodefaults\n' +
    'dependencies:\n' +
    '- python\n' +
    '- git\n'
  );
  if (condaDeps) { // include dependencies read from pyproject.toml
    condaDeps.forEach((dep) => pluginEnvYMLContents += `- ${dep}\n`);
  }
  logger.info(`creating plugin environment from file:\n${pluginEnvYMLContents}`);

  fs.writeFileSync(pluginEnvYMLPath, pluginEnvYMLContents);
  event.sender.send('plugin-install-status', i18n.t('Creating plugin environment...'));
  // In case the specific natcap.invest dependency does not have a
  // conda-forge build, setup.py requires this env variable.
  process.env['NATCAP_INVEST_GDAL_LIB_PATH'] = `${pluginEnvPrefix}/Library`;
  await spawnWithLogging(micromamba, [
    'create', '--yes', '--prefix', `"${pluginEnvPrefix}"`,
    '--file', `"${pluginEnvYMLPath}"`]);
  logger.info('created micromamba env for plugin');
  event.sender.send('plugin-install-status', i18n.t('Installing plugin into environment...'));
  await spawnWithLogging(
    micromamba,
    [
      'run', '--prefix', `"${pluginEnvPrefix}"`,
      'python', '-m', 'pip', 'install', installString
    ]
  );
  logger.info('installed plugin into its env');
}

/**
 * Get plugin metadata from the installed Python package,
 * and store that metadata in the Workbench's settingsStore.
 * @param  {string} micromamba      path to the micromamba executable
 * @param  {string} pluginEnvPrefix location of the plugin's micromamba env
 * @param  {string} packageName     the plugin's python package name
 * @param  {string} installString   pip install argument for the plugin
 */
function storePluginMetadataSync(
  micromamba, pluginEnvPrefix, packageName, installString
) {
  const modelID = execSync(
    `${micromamba} run --prefix "${pluginEnvPrefix}" ` +
    `python -c "import ${packageName}; print(${packageName}.MODEL_SPEC.model_id)"`
  ).toString().trim();
  const modelTitle = execSync(
    `${micromamba} run --prefix "${pluginEnvPrefix}" ` +
    `python -c "import ${packageName}; print(${packageName}.MODEL_SPEC.model_title)"`
  ).toString().trim();
  const version = execSync(
    `${micromamba} run --prefix "${pluginEnvPrefix}" ` +
    `python -c "from importlib.metadata import version; ` +
    `print(version('${packageName}'))"`
  ).toString().trim();

  // Write plugin metadata to the workbench's config.json
  logger.info('writing plugin info to settings store');
  // Uniquely identify plugin by a hash of its ID and version
  // Replace dots with underscores in the version, because dots are a
  // special character in keys for electron-store's set and get methods
  const pluginID = `${modelID}@${version}`;
  settingsStore.set(
    `plugins.${pluginID.replaceAll('.', '_')}`,
    {
      modelID: modelID,
      modelTitle: modelTitle,
      type: 'plugin',
      source: installString,
      env: pluginEnvPrefix,
      version: version,
    }
  );
}

export function setupAddPlugin(i18n) {
  ipcMain.handle(
    ipcMainChannels.ADD_PLUGIN,
    async (event, url, revision, localPath) => {
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
            event.sender.send(
              'plugin-install-status',
              i18n.t('Creating base environment...')
            );
            await createBaseEnv(baseEnvPrefix, micromamba);
          }

          // check out the plugin's pyproject.toml,
          // without downloading any extra files or git history
          event.sender.send(
            'plugin-install-status',
            i18n.t('Downloading plugin source code...')
          );
          const tmpPluginDir = fs.mkdtempSync(upath.join(tmpdir(), 'natcap-invest-'));
          try {
            await cloneAndCheckoutPlugin(
              micromamba, baseEnvPrefix, url, revision, tmpPluginDir
            );
            // Read in the plugin's pyproject.toml, then delete it
            pyprojectTOML = toml.parse(fs.readFileSync(
              upath.join(tmpPluginDir, 'pyproject.toml')
            ).toString());
          } catch (error) {
            logger.error('Error downloading plugin source code:');
            throw error;
          } finally {
            // We only needed the pyproject.toml content, the rest can go.
            fs.rmSync(tmpPluginDir, { recursive: true, force: true });
          }
        } else { // install from local path
          logger.info(`adding plugin from ${localPath}`);
          installString = localPath;
          // Read in the plugin's pyproject.toml
          pyprojectTOML = toml.parse(fs.readFileSync(
            upath.join(localPath, 'pyproject.toml')
          ).toString());
        }
        // Access plugin metadata from the pyproject.toml
        const condaDeps = pyprojectTOML.tool.natcap.invest.conda_dependencies;
        const packageName = pyprojectTOML.tool.natcap.invest.package_name;

        // Create a conda env containing the plugin and its dependencies
        // use timestamp to ensure a unique path
        // I wanted the env path to match the plugin model_id, but we can't
        // know the model_id until after creating the environment to be able to
        // import metadata from the MODEL_SPEC. And mamba does not support
        // renaming or moving environments after they're created.
        const pluginEnvPrefix = upath.join(rootPrefix, `plugin_${Date.now()}`);
        try {
          await installPlugin(
            micromamba, pluginEnvPrefix, condaDeps, installString, i18n, event
          );
          event.sender.send('plugin-install-status', i18n.t('Importing plugin...'));
          storePluginMetadataSync(
            micromamba, pluginEnvPrefix, packageName, installString
          );
          logger.info('successfully added plugin');
        } catch (error) {
          logger.info('Cleaning up failed installation environment.');
          await spawnWithLogging(
            micromamba, ['env', 'remove', '--yes', '--prefix', `"${pluginEnvPrefix}"`]
          );
          throw error;
        }
      } catch (error) {
        logger.error(error);
        throw error;
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
        // Shut down the plugin server process
        const pluginPID = settingsStore.get(`plugins.${pluginID}.pid`);
        await shutdownPythonProcess(pluginPID);
        // Delete the plugin's conda env
        const env = settingsStore.get(`plugins.${pluginID}.env`);
        const micromamba = settingsStore.get('micromamba');
        await spawnWithLogging(
          micromamba, ['env', 'remove', '--yes', '--prefix', `"${env}"`]
        );
        // Delete the plugin's data from storage
        settingsStore.delete(`plugins.${pluginID}`);
        logger.info('successfully removed plugin');
      } catch (error) {
        logger.info('Error removing plugin:');
        logger.info(error);
        throw error;
      }
    }
  );
}

export function setupWindowsMSVCHandlers() {
  ipcMain.handle(
    ipcMainChannels.HAS_MSVC,
    () => {
      return fs.existsSync(
        upath.join('C:', 'Windows', 'System32', 'VCRUNTIME140_1.dll')
      );
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
