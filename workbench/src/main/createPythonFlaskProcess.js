import { spawn, exec } from 'child_process';

import fetch from 'node-fetch';

import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const HOSTNAME = 'http://localhost';

/**
 * Spawn a child process running the Python Flask app.
 *
 * @param  {string} investExe - path to executeable that launches flask app.
 * @returns {ChildProcess} - a reference to the subprocess.
 */
export function createPythonFlaskProcess(investExe) {
  const pythonServerProcess = spawn(
    investExe,
    ['--debug', 'serve', '--port', process.env.PORT],
    { shell: true } // necessary in dev mode & relying on a conda env
  );

  logger.debug(`Started python process as PID ${pythonServerProcess.pid}`);
  pythonServerProcess.stdout.on('data', (data) => {
    logger.debug(`${data}`);
  });
  pythonServerProcess.stderr.on('data', (data) => {
    logger.debug(`${data}`);
  });
  pythonServerProcess.on('error', (err) => {
    logger.error(err.stack);
    logger.error(
      `The flask app ${investExe} crashed or failed to start
       so this application must be restarted`
    );
    throw err;
  });
  pythonServerProcess.on('close', (code, signal) => {
    logger.debug(`Flask process closed with code ${code} and signal ${signal}`);
  });
  pythonServerProcess.on('exit', (code) => {
    logger.debug(`Flask process exited with code ${code}`);
  });
  pythonServerProcess.on('disconnect', () => {
    logger.debug(`Flask process disconnected`);
  });

  return pythonServerProcess;
}

/** Find out if the Flask server is online, waiting until it is.
 *
 * @param {number} i - the number or previous tries
 * @param {number} retries - number of recursive calls this function is allowed.
 * @returns { Promise } resolves text indicating success.
 */
export function getFlaskIsReady({ i = 0, retries = 41 } = {}) {
  return (
    fetch(`${HOSTNAME}:${process.env.PORT}/api/ready`, {
      method: 'get',
    })
      .then((response) => response.text())
      .catch(async (error) => {
        if (error.code === 'ECONNREFUSED') {
          while (i < retries) {
            i++;
            // Try every X ms, usually takes a couple seconds to startup.
            await new Promise((resolve) => setTimeout(resolve, 300));
            logger.debug(`retry # ${i}`);
            return await getFlaskIsReady({ i: i, retries: retries });
          }
          logger.error(`Not able to connect to server after ${retries} tries.`);
        }
        logger.error(error);
        throw error;
      })
  );
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
      exec(`taskkill /pid ${pid} /t /f`);
    }
  } catch (error) {
    // if the process was already killed by some other means
    logger.debug(error);
  }

  // If we return too quickly, it seems the electron app is allowed
  // to quit before the subprocess is killed, and the subprocess remains
  // open. Here we poll a flask endpoint and resolve only when it
  // gives ECONNREFUSED.
  return fetch(`${HOSTNAME}:${process.env.PORT}/ready`, {
    method: 'get',
  })
    .then(async () => {
      await new Promise((resolve) => setTimeout(resolve, 300));
      return shutdownPythonProcess(subprocess);
    })
    .catch(() => {
      logger.debug('flask server is closed');
      return Promise.resolve();
    });
}
