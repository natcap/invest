import fetch from 'node-fetch';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const PORT = process.env.PORT || '5000';
const HOSTNAME = 'http://localhost';

/** Recursive function to find out if the Flask server is online.
 *
 * Sometimes the app will make a server request before it's ready,
 * so awaiting this response is one way to avoid that.
 *
 * @param {int} retries - number of times to retry a request
 * @returns { Promise } 
 */
export function getFlaskIsReady(retries = 0) {
  return (
    fetch(`${HOSTNAME}:${PORT}/ready`, {
      method: 'get',
    })
      .then((response) => response.text())
      .catch(async (error) => {
        if (error.code === 'ECONNREFUSED') {
          while (retries < 21) {
            retries++;
            // try again after a short pause
            await new Promise((resolve) => setTimeout(resolve, 500));
            logger.debug(`retry # ${retries}`);
            return await getFlaskIsReady(retries)
          }
        } else {
          logger.error(error.stack);
          throw error;
        }
      })
  );
}

export function getInvestList() {
  return (
    fetch(`${HOSTNAME}:${PORT}/models`, {
      method: 'get',
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

export function getSpec(payload) {
  return (
    fetch(`${HOSTNAME}:${PORT}/getspec`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

export function fetchValidation(payload) {
  return (
    fetch(`${HOSTNAME}:${PORT}/validate`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

export function fetchDatastackFromFile(payload) {
  return (
    fetch(`${HOSTNAME}:${PORT}/post_datastack_file`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

export function saveToPython(payload) {
  return (
    fetch(`${HOSTNAME}:${PORT}/save_to_python`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.text())
      .then((text) => logger.debug(text))
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * @param  {object} payload - body expected by write_parameter_set_file endpoint
 * @returns {Promise} - resolves to null
 */
export function writeParametersToFile(payload) {
  // Even though the purpose here is to request a file
  // is written to disk, we want to return a Promise in
  // order to await success.
  return (
    fetch(`${HOSTNAME}:${PORT}/write_parameter_set_file`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.text())
      .then((text) => logger.debug(text))
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Request the shutdown of the Flask app
 *
 * @returns {Promise} resolves string communicating success
 */
export function shutdownPythonProcess() {
  return (
    fetch(`http://localhost:${PORT}/shutdown`, {
      method: 'get',
    })
      .then((response) => response.text())
      .then((text) => { logger.debug(text); })
      .catch((error) => { logger.error(error.stack); })
  );
}
