import { spawn, execSync } from 'child_process';

import fetch from 'node-fetch';

import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const HOSTNAME = 'http://127.0.0.1';

function launchSubprocess(exe, args, opts) {
  const subprocess = spawn(exe, args, opts);

  logger.debug(`Started python process as PID ${subprocess.pid}`);
  subprocess.stdout.on('data', (data) => {
    logger.debug(`${data}`);
  });
  subprocess.stderr.on('data', (data) => {
    logger.debug(`${data}`);
  });
  subprocess.on('error', (err) => {
    logger.error(err.stack);
    logger.error(
      `${exe} crashed or failed to start
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

  return subprocess;
}

export function createJupyterProcess(jupyterExe, notebookDir) {
  const subprocess = launchSubprocess(
    jupyterExe,
    ['lab', '--notebook-dir', notebookDir, '--no-browser', '--port', process.env.JUPYTER_PORT],
    { shell: true } // necessary in dev mode & relying on a conda env
  );
  return subprocess;
}

/**
 * Spawn a child process running the Python Flask app.
 *
 * @param  {string} investExe - path to executeable that launches flask app.
 * @returns {ChildProcess} - a reference to the subprocess.
 */
export function createPythonFlaskProcess(investExe) {
  const subprocess = launchSubprocess(
    investExe,
    ['--debug', 'serve', '--port', process.env.PORT],
    { shell: true } // necessary in dev mode & relying on a conda env
  );
  return subprocess;
}

/** Find out if the Flask server is online, waiting until it is.
 *
 * @param {number} i - the number or previous tries
 * @param {number} retries - number of recursive calls this function is allowed.
 * @returns { Promise } resolves text indicating success.
 */
export async function getFlaskIsReady({ i = 0, retries = 41 } = {}) {
  try {
    await fetch(`${HOSTNAME}:${process.env.PORT}/api/ready`, {
      method: 'get',
    });
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      while (i < retries) {
        i++;
        // Try every X ms, usually takes a couple seconds to startup.
        await new Promise((resolve) => setTimeout(resolve, 300));
        logger.debug(`retry # ${i}`);
        return getFlaskIsReady({ i: i, retries: retries });
      }
      logger.error(`Not able to connect to server after ${retries} tries.`);
    }
    logger.error(error);
    throw error;
  }
}

/** Find out if the Jupyter server is online, waiting until it is.
 *
 * @param {number} i - the number or previous tries
 * @param {number} retries - number of recursive calls this function is allowed.
 * @returns { Promise } resolves text indicating success.
 */
export async function getJupyterIsReady({ i = 0, retries = 41 } = {}) {
  try {
    await fetch(`${HOSTNAME}:${process.env.JUPYTER_PORT}/?token=${process.env.JUPYTER_TOKEN}`, {
      method: 'get',
    });
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      while (i < retries) {
        i++;
        // Try every X ms, usually takes a couple seconds to startup.
        await new Promise((resolve) => setTimeout(resolve, 300));
        logger.debug(`retry # ${i}`);
        return getJupyterIsReady({ i: i, retries: retries });
      }
      logger.error(`Not able to connect to server after ${retries} tries.`);
    }
    logger.error(error);
    throw error;
  }
}

/**
 * Kill the process running the Flask app
 *
 * @param {ChildProcess} subprocess - such as created by child_process.spawn
 * @returns {Promise}
 */
export async function shutdownPythonProcess(subprocess) {
  // builtin kill() method on a nodejs ChildProcess doesn't work on windows.
  try {
    if (process.platform !== 'win32') {
      subprocess.kill();
    } else {
      const { pid } = subprocess;
      execSync(`taskkill /pid ${pid} /t /f`);
    }
  } catch (error) {
    // if the process was already killed by some other means
    logger.debug(error);
  } finally {
    Promise.resolve();
  }
}
