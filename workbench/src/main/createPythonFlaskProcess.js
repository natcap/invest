import { spawn, execSync } from 'child_process';
import http from 'http';
import fetch from 'node-fetch';

import { getLogger } from './logger';
import { settingsStore } from './settingsStore';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const HOSTNAME = 'http://127.0.0.1';

const pidToSubprocess = {};

// https://stackoverflow.com/a/71178451
export async function getFreePort() {
  return new Promise((resolve) => {
    const srv = http.createServer();
    srv.listen(0, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

/** Find out if the Flask server is online, waiting until it is.
 *
 * @param {number} i - the number or previous tries
 * @param {number} retries - number of recursive calls this function is allowed.
 * @returns { Promise } resolves text indicating success.
 */
export async function getFlaskIsReady(port, i = 0, retries = 41) {
  try {
    await fetch(`${HOSTNAME}:${port}/api/ready`, {
      method: 'get',
    });
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      while (i < retries) {
        i++;
        // Try every X ms, usually takes a couple seconds to startup.
        await new Promise((resolve) => setTimeout(resolve, 300));
        logger.debug(`retry # ${i}`);
        return getFlaskIsReady(port, i, retries);
      }
      logger.error(`Not able to connect to server after ${retries} tries.`);
    }
    logger.error(error);
    throw error;
  }
}

/**
 * Set up handlers for server process events.
 * @param  {ChildProcess} pythonServerProcess - server process instance.
 * @returns {undefined}
 */
export function setupServerProcessHandlers(subprocess) {
  subprocess.stdout.on('data', (data) => {
    logger.debug(`${data}`);
  });
  subprocess.stderr.on('data', (data) => {
    logger.debug(`${data}`);
  });
  subprocess.on('error', (err) => {
    logger.error(subprocess.spawnargs);
    logger.error(err.stack);
    logger.error(
      `The invest flask app crashed or failed to start
       so this application must be restarted`
    );
    throw err;
  });
  subprocess.on('close', (code, signal) => {
    logger.debug(`process closed with code ${code} and signal ${signal}`);
  });
  subprocess.on('exit', (code) => {
    logger.debug(`process exited with code ${code}`);
  });
  subprocess.on('disconnect', () => {
    logger.debug(`process disconnected`);
  });
  pidToSubprocess[subprocess.pid] = subprocess;
}

export async function createJupyterProcess(jupyterExe, notebookDir, _port = undefined) {
  let port = _port;
  if (port === undefined) {
    port = await getFreePort();
    logger.debug(`PORT ${port}`)
  }
  logger.debug('creating jupyterlab server process');
  logger.debug(jupyterExe);
  let notebookPath = `${notebookDir}/ipyleaflet.ipynb`;
  const subprocess = spawn(
    jupyterExe,
    //['lab', '--notebook-dir', notebookDir, '--no-browser', '--port', port],
    [notebookPath, '--debug', '--no-browser', '--port', port],
    { shell: true } // necessary in dev mode & relying on a conda env
  );
  setupServerProcessHandlers(subprocess);
  await getJupyterIsReady(port, 0, 500);
  return [subprocess, port];
}

// /**
//  * Spawn a child process running the Python Flask app.
//  *
//  * @param  {string} investExe - path to executeable that launches flask app.
//  * @returns {ChildProcess} - a reference to the subprocess.
//  */
// export function createPythonFlaskProcess(investExe) {
//   const subprocess = launchSubprocess(
//     investExe,
//     ['--debug', 'serve', '--port', process.env.PORT],
//     { shell: true } // necessary in dev mode & relying on a conda env
//   );
//   return subprocess;
// // =======
// //   pythonServerProcess.on('close', (code, signal) => {
// //     logger.debug(`Flask process closed with code ${code} and signal ${signal}`);
// //   });
// //   pythonServerProcess.on('exit', (code) => {
// //     logger.debug(`Flask process exited with code ${code}`);
// //   });
// //   pythonServerProcess.on('disconnect', () => {
// //     logger.debug('Flask process disconnected');
// //   });
//   pidToSubprocess[subprocess.pid] = subprocess;
// // >>>>>>> upstream/feature/plugins
// }

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

  setupServerProcessHandlers(pythonServerProcess);
  await getFlaskIsReady(port, 0, 500);
  logger.info('flask is ready');
}

/**
 * Spawn a child process running the Python Flask app for a plugin.
 * @param {string} modelName - name of the plugin to launch
 * @param  {integer} _port - if provided, port to launch server on. Otherwise,
 *                         an available port is chosen.
 * @returns { integer } - PID of the process that was launched
 */
export async function createPluginServerProcess(modelName, _port = undefined) {
  let port = _port;
  if (port === undefined) {
    port = await getFreePort();
  }

  logger.debug('creating invest plugin server process');
  const mamba = settingsStore.get('mamba');
  const modelEnvPath = settingsStore.get(`plugins.${modelName}.env`);
  const args = [
    'run', '--prefix', `"${modelEnvPath}"`,
    'invest', '--debug', 'serve', '--port', port];
  logger.debug('spawning command:', mamba, args);
  // shell mode is necessary in dev mode & relying on a conda env
  const pythonServerProcess = spawn(mamba, args, { shell: true });
  settingsStore.set(`plugins.${modelName}.port`, port);
  settingsStore.set(`plugins.${modelName}.pid`, pythonServerProcess.pid);

  logger.debug(`Started python process as PID ${pythonServerProcess.pid}`);

  setupServerProcessHandlers(pythonServerProcess);

  await getFlaskIsReady(port, 0, 500);
  logger.info('flask is ready');
  return pythonServerProcess.pid;
}

/** Find out if the Jupyter server is online, waiting until it is.
 *
 * @param {number} i - the number or previous tries
 * @param {number} retries - number of recursive calls this function is allowed.
 * @returns { Promise } resolves text indicating success.
 */
export async function getJupyterIsReady(port = undefined, { i = 0, retries = 41 } = {}) {
  try {
    logger.debug(`${HOSTNAME}:${port}/?token=${process.env.JUPYTER_TOKEN}`)
    await fetch(`${HOSTNAME}:${port}`, {
      method: 'get',
    });
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      while (i < retries) {
        i++;
        // Try every X ms, usually takes a couple seconds to startup.
        await new Promise((resolve) => setTimeout(resolve, 300));
        logger.debug(`retry # ${i}`);
        return getJupyterIsReady(port, { i: i, retries: retries });
      }
      logger.error(`Not able to connect to server after ${retries} tries.`);
    }
    logger.error(error);
    throw error;
  }
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
