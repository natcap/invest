const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { app } = require('electron'); // eslint-disable-line import/no-extraneous-dependencies
const { getLogger } = require('./logger');

const logger = getLogger(__filename.split('/').slice(-1)[0]);
/**
 * Find paths to local invest binaries under dev or production environments.
 *
 * @param {boolean} isDevMode - a boolean designating dev mode or not.
 * @returns {Promise} Resolves object with filepaths to invest binaries
 */
export function findInvestBinaries(isDevMode) {
  return new Promise(resolve => {
    // Binding to the invest server binary:
    let serverExe;
    let investExe;
    const ext = (process.platform === 'win32') ? '.exe' : '';

    // A) look for a local registry of available invest installations
    const investRegistryPath = path.join(
      app.getPath('userData'), 'invest_registry.json'
    );
    if (fs.existsSync(investRegistryPath)) {
      const investRegistry = JSON.parse(fs.readFileSync(investRegistryPath));
      const activeVersion = investRegistry.active;
      serverExe = investRegistry.registry[activeVersion].server;
      investExe = investRegistry.registry[activeVersion].invest;

    // B) check for dev mode and an environment variable from dotenv
    } else if (isDevMode) {
      // If no dotenv vars are set, default to where this project's
      // build process places the binaries.
      serverExe = `${process.env.SERVER || 'build/invest/server'}${ext}`;
      investExe = `${process.env.INVEST || 'build/invest/invest'}${ext}`;

    // C) point to binaries included in this app's installation.
    } else {
      const binaryPath = path.join(
        process.resourcesPath, 'app.asar.unpacked', 'build', 'invest'
      );
      serverExe = path.join(binaryPath, `server${ext}`);
      investExe = path.join(binaryPath, `invest${ext}`);
    }
    logger.info(`Found invest binaries ${investExe} and ${serverExe}`);
    resolve({ invest: investExe, server: serverExe });
  });
}

/**
 * Spawn a child process running the Python Flask app.
 *
 * @param  {string} serverExe - path to executeable that launches flask app.
 * @param {boolean} isDevMode - a boolean designating dev mode or not.
 * @returns {undefined}
 */
export function createPythonFlaskProcess(serverExe, isDevMode) {
  if (serverExe) {
    let pythonServerProcess;
    if (isDevMode && process.env.PYTHON && serverExe.endsWith('.py')) {
      // A special devMode case for launching from the source code
      // to facilitate debugging & development of src/server.py
      pythonServerProcess = spawn(process.env.PYTHON, [serverExe]);
    } else {
      // The most reliable, cross-platform way to make sure spawn
      // can find the exe is to pass only the command name while
      // also putting it's location on the PATH:
      pythonServerProcess = spawn(path.basename(serverExe), {
        env: { PATH: path.dirname(serverExe) },
      });
    }

    logger.debug(`Started python process as PID ${pythonServerProcess.pid}`);
    logger.debug(serverExe);
    pythonServerProcess.stdout.on('data', (data) => {
      logger.debug(`${data}`);
    });
    pythonServerProcess.stderr.on('data', (data) => {
      logger.debug(`${data}`);
    });
    pythonServerProcess.on('error', (err) => {
      logger.error(err.stack);
      logger.error(
        `The flask app ${serverExe} crashed or failed to start
         so this application must be restarted`
      );
      throw err;
    });
    pythonServerProcess.on('close', (code, signal) => {
      logger.debug(`Flask process terminated with code ${code} and signal ${signal}`);
    });
  } else {
    logger.error('no existing invest installations found');
  }
}
