import { spawn, execSync } from 'child_process';
import http from 'http';
import fetch from 'node-fetch';

import { logger } from './logger';
import { settingsStore } from './settingsStore';

const HOSTNAME = 'http://127.0.0.1';

const pidToSubprocess = {};

// https://stackoverflow.com/a/71178451
async function getFreePort() {
  return new Promise((resolve) => {
    const srv = http.createServer();
    srv.listen(0, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

/**
 * Wait for a invest server process to start up, handling any errors.
 * @param  {ChildProcess} pythonServerProcess - server process instance.
 * @param {string} url - server status url to retry
 * @param {number} maxRetries - number of retries allowed
 * @returns {number} PID of the started process, or undefined if it fails to launch
 */
export async function handleServerStartup(pythonServerProcess, url, maxRetries=500) {
  let processErrored = false;
  pythonServerProcess.stdout.on('data', (data) => {
    logger.debug(`${data}`);
  });
  pythonServerProcess.stderr.on('data', (data) => {
    logger.debug(`${data}`);
  });
  // The 'error' event happens when the child process fails to spawn.
  // This should be rare.
  pythonServerProcess.on('error', (err) => {
    logger.error(pythonServerProcess.spawnargs);
    logger.error(err.stack);
    logger.error(
      `The invest flask app crashed or failed to start
       so this application must be restarted`
    );
    throw err;
  });
  // The 'exit' event is what happens on routine errors like a
  // typo in the micromamba command or a plugin failing to import
  pythonServerProcess.on('exit', (code) => {
    logger.debug(`Flask process exited with code ${code}`);
    if (code != 0) {
      processErrored = true;
    }
  });
  pythonServerProcess.on('close', (code, signal) => {
    logger.debug(`Flask process closed with code ${code} and signal ${signal}`);
  });
  pythonServerProcess.on('disconnect', () => {
    logger.debug('Flask process disconnected');
  });
  pidToSubprocess[pythonServerProcess.pid] = pythonServerProcess;

  // Wait for the server to start up
  let retries = 0;
  while (retries < 500) {
    if (processErrored) {
      return undefined;
    }
    logger.debug(`retry # ${retries}`);
    try {
      await fetch(url, { method: 'get' });
      logger.info('flask is ready');
      return pythonServerProcess.pid;
    } catch (error) {
      if (error.code === 'ECONNREFUSED') {
        // wait 300ms before retrying
        await new Promise((resolve) => setTimeout(resolve, 300));
        retries++;
      } else {
        logger.error(error);
        throw error;
      }
    }
  }
  logger.error(`Not able to connect to server after ${retries} tries.`);
}

/**
 * Spawn a child process running the Python Flask app for core invest.
 *
 * @param  {integer} _port - if provided, port to launch server on. Otherwise,
 *                         an available port is chosen.
 * @returns { integer } - PID of the process that was launched
 */
export async function createCoreServerProcess(_port = undefined) {
  let port = _port;
  if (port === undefined) {
    port = await getFreePort();
  }
  logger.debug('creating invest core server process');
  const pythonServerProcess = spawn(
    settingsStore.get('investExe'),
    ['--debug', 'serve', '--port', port],
    { shell: true } // necessary in dev mode & relying on a conda env
  );
  settingsStore.set('core.port', port);
  settingsStore.set('core.pid', pythonServerProcess.pid);

  logger.debug(`Started python process as PID ${pythonServerProcess.pid}`);
  const pid = await handleServerStartup(
    pythonServerProcess, `${HOSTNAME}:${port}/api/ready`);
  return pid;
}

/**
 * Spawn a child process running the Python Flask app for a plugin.
 * @param {string} modelID - id of the plugin to launch
 * @param  {integer} _port - if provided, port to launch server on. Otherwise,
 *                         an available port is chosen.
 * @returns { integer } - PID of the process that was launched
 */
export async function createPluginServerProcess(modelID, _port = undefined) {
  let port = _port;
  if (port === undefined) {
    port = await getFreePort();
  }
  logger.debug('creating invest plugin server process');
  const micromamba = settingsStore.get('micromamba');
  const modelEnvPath = settingsStore.get(`plugins.${modelID}.env`);
  const args = [
    'run', '--prefix', `"${modelEnvPath}"`,
    // calling invest with python avoids issues with unescaped
    // spaces in the python path in the conda bin/invest script
    'python -m natcap.invest', '--debug', 'serve', '--port', port];
  logger.debug('spawning command:', micromamba, args);
  // shell mode is necessary in dev mode & relying on a conda env
  const pythonServerProcess = spawn(micromamba, args, { shell: true });
  settingsStore.set(`plugins.${modelID}.port`, port);
  settingsStore.set(`plugins.${modelID}.pid`, pythonServerProcess.pid);

  logger.debug(`Started python process as PID ${pythonServerProcess.pid}`);
  const pid = await handleServerStartup(
    pythonServerProcess, `${HOSTNAME}:${port}/api/ready`);
  return pid;
}

/**
 * Kill the process running the Flask app
 * @param {number} pid - process ID of the child process to shut down
 * @returns {Promise}
 */
export async function shutdownPythonProcess(pid) {
  // builtin kill() method on a nodejs ChildProcess doesn't work on windows.
  try {
    if (process.platform !== 'win32') {
      pidToSubprocess[pid.toString()].kill();
    } else {
      execSync(`taskkill /pid ${pid} /t /f`);
    }
  } catch (error) {
    // if the process was already killed by some other means
    logger.debug(error);
  } finally {
    Promise.resolve();
  }
}
