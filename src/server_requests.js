import fetch from 'node-fetch';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const PORT = process.env.PORT || '5000';
const HOSTNAME = 'http://localhost';

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
    fetch(`${HOSTNAME}:${PORT}/ready`, {
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
 * Get the list of invest model names that can be passed to getSpec.
 *
 * @returns {Promise} resolves object
 */
export function getInvestList() {
  return (
    fetch(`${HOSTNAME}:${PORT}/models`, {
      method: 'get',
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Get the ARGS_SPEC dict from an invest model as a JSON.
 *
 * @param {string} payload - model name as given by `invest list`
 * @returns {Promise} resolves object
 */
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

/**
 * Send invest arguments to a model's validate function.
 *
 * @param {object} payload {
 *   model_module: string (e.g. natcap.invest.carbon)
 *   args: JSON string of InVEST model args keys and values
 * }
 * @returns {Promise} resolves array
 */
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

/**
 * Load invest arguments from a datastack-compliant file.
 *
 * @param {string} payload - path to file
 * @returns {Promise} resolves undefined
 */
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

/**
 * Write invest model arguments to a python script.
 *
 * @param  {object} payload {
 *   filepath: string
 *   modelname: string (e.g. carbon)
 *   pyname: string (e.g. natcap.invest.carbon)
 *   args_dict: JSON string of InVEST model args keys and values
 * }
 * @returns {Promise} resolves undefined
 */
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
 * Write invest model arguments to a JSON file.
 *
 * @param  {object} payload {
 *   parameterSetPath: string
 *   moduleName: string (e.g. natcap.invest.carbon)
 *   args: JSON string of InVEST model args keys and values
 *   relativePaths: boolean
 * }
 * @returns {Promise} resolves undefined
 */
export function writeParametersToFile(payload) {
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
 * @returns {Promise} resolves undefined
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
