import { getSettingsValue } from './components/SettingsModal/SettingsStorage';

const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);
const HOSTNAME = 'http://localhost';

// The Flask server sends UTF-8 encoded responses by default
// response.text() always decodes the response using UTF-8
// https://developer.mozilla.org/en-US/docs/Web/API/Body/text
// response.json() doesn't say but is presumably also UTF-8
// https://developer.mozilla.org/en-US/docs/Web/API/Body/json

/**
 * Get the list of invest model names that can be passed to getSpec.
 *
 * @returns {Promise} resolves object
 */
export async function getInvestModelNames() {
  const language = await getSettingsValue('language');
  return (
    window.fetch(`${HOSTNAME}:${process.env.PORT}/models?language=${language}`, {
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
export async function getSpec(payload) {
  const language = await getSettingsValue('language');
  return (
    window.fetch(`${HOSTNAME}:${process.env.PORT}/getspec?language=${language}`, {
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
export async function fetchValidation(payload) {
  const language = await getSettingsValue('language');
  return (
    window.fetch(`${HOSTNAME}:${process.env.PORT}/validate?language=${language}`, {
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
    window.fetch(`${HOSTNAME}:${process.env.PORT}/post_datastack_file`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Get a list of the column names of a vector file.
 *
 * @param {string} payload - path to file
 * @returns {Promise} resolves array
 */
export function getVectorColumnNames(payload) {
  return (
    window.fetch(`${HOSTNAME}:${process.env.PORT}/colnames`, {
      method: 'post',
      body: JSON.stringify({vector_path: payload}),
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
 *   args_dict: JSON string of InVEST model args keys and values
 * }
 * @returns {Promise} resolves undefined
 */
export function saveToPython(payload) {
  return (
    window.fetch(`${HOSTNAME}:${process.env.PORT}/save_to_python`, {
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
 * Write invest model arguments to a python script.
 *
 * @param  {object} payload {
 *   filepath: string
 *   modelname: string (e.g. carbon)
 *   args_dict: JSON string of InVEST model args keys and values
 * }
 * @returns {Promise} resolves undefined
 */
export function archiveDatastack(payload) {
  return (
    window.fetch(`${HOSTNAME}:${process.env.PORT}/build_datastack_archive`, {
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
    window.fetch(`${HOSTNAME}:${process.env.PORT}/write_parameter_set_file`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.text())
      .then((text) => logger.debug(text))
      .catch((error) => logger.error(error.stack))
  );
}
