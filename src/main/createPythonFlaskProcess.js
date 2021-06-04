import path from 'path';
import { spawn } from 'child_process';

import fetch from 'node-fetch';

import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const HOSTNAME = 'http://localhost';

/**
 * Spawn a child process running the Python Flask app.
 *
 * @param  {string} investExe - path to executeable that launches flask app.
 * @returns {undefined}
 */
export function createPythonFlaskProcess(investExe) {
  if (investExe) {
    // TODO: starting `invest serve`` without any python logging
    // because of https://github.com/natcap/invest/issues/563
    // & https://github.com/natcap/invest-workbench/issues/144
    // Once those are resolved, we probably want some logging here,
    // maybe --debug if devMode, -vvv if production?
    const pythonServerProcess = spawn(
      path.basename(investExe),
      ['serve', '--port', process.env.PORT],
      {
        env: {
          PATH: path.dirname(investExe),
        },
      }
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
  } else {
    logger.error('no existing invest installations found');
  }
}

/** Find out if the Flask server is online, waiting until it is.
 *
 * Sometimes the app will make a server request before it's ready,
 * so awaiting this response is one way to avoid that.
 *
 * @param {number} i - the number or previous tries
 * @param {number} retries - number of recursive calls this function is allowed.
 * @returns { Promise } resolves text indicating success.
 */
export function getFlaskIsReady({ i = 0, retries = 21 } = {}) {
  return (
    fetch(`${HOSTNAME}:${process.env.PORT}/ready`, {
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
          logger.error(error.stack);
          throw error;
        } else {
          logger.error(error.stack);
          throw error;
        }
      })
  );
}

/**
 * Request the shutdown of the Flask app
 *
 * @returns {Promise} resolves undefined
 */
export function shutdownPythonProcess() {
  return (
    fetch(`http://localhost:${process.env.PORT}/shutdown`, {
      method: 'get',
    })
      .then((response) => response.text())
      .then((text) => { logger.debug(text); })
      .catch((error) => { logger.error(error.stack); })
  );
}
