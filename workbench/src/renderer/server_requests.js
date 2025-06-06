import { ipcMainChannels } from '../main/ipcMainChannels';

const { logger, LANGUAGE } = window.Workbench;
const { ipcRenderer } = window.Workbench.electron;

const HOSTNAME = 'http://127.0.0.1';
const PREFIX = 'api';

/**
 * Get the port number running the server and model ID to use for the given model.
 *
 * Server must already be started. If the model is a core invest model, the core
 * port is returned. If a plugin, the port for that plugin's server is returned.
 * Model ID is returned unchanged for core models. For plugins, the version is
 * removed from the id string.
 * @param {string} modelID - model name as given by `invest list`
 * @returns {Promise} resolves object
 */
async function getPortAndID(modelID) {
  let port, id;
  const plugins = await ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'plugins');
  if (plugins && Object.keys(plugins).includes(modelID)) {
    port = await ipcRenderer.invoke(ipcMainChannels.GET_SETTING, `plugins.${modelID}.port`);
    id = await ipcRenderer.invoke(ipcMainChannels.GET_SETTING, `plugins.${modelID}.modelID`);
  } else {
    port = await ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'core.port');
    id = modelID
  }
  return { port, id };
}

async function getCorePort() {
  return ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'core.port');
}

// The Flask server sends UTF-8 encoded responses by default
// response.text() always decodes the response using UTF-8
// https://developer.mozilla.org/en-US/docs/Web/API/Body/text
// response.json() doesn't say but is presumably also UTF-8
// https://developer.mozilla.org/en-US/docs/Web/API/Body/json

export async function getInvestModelIDs() {
  const port = await getCorePort();
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/models?language=${LANGUAGE}`, {
      method: 'get',
    })
      .then((response) => response.json())
      .catch((error) => { logger.error(`${error.stack}`); })
  );
}

/**
 * Get the MODEL_SPEC dict from an invest model as a JSON.
 *
 * @param {string} modelID - model name as given by `invest list`
 * @returns {Promise} resolves object
 */
export async function getSpec(modelID) {
  const { port, id } = await getPortAndID(modelID);
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/getspec?language=${LANGUAGE}`, {
      method: 'post',
      body: JSON.stringify(id),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Get the dynamically determined dropdown options for a given model.
 *
 * @param {object} payload {
 *   model_id: string (e.g. carbon)
 *   args: JSON string of InVEST model args keys and values
 * }
 * @returns {Promise} resolves object
 */
export async function getDynamicDropdowns(payload) {
  const { port, id } = await getPortAndID(payload.model_id);
  payload.model_id = id;
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/dynamic_dropdowns`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Get the enabled/disabled status of arg inputs.
 *
 * @param {object} payload {
 *   model_id: string (e.g. carbon)
 *   args: JSON string of InVEST model args keys and values
 * }
 * @returns {Promise} resolves object
 */
export async function fetchArgsEnabled(payload) {
  const { port, id } = await getPortAndID(payload.model_id);
  payload.model_id = id;
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/args_enabled`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .catch((error) => {
        logger.error(error.stack);
        // In practice this function is debounced, so there's a case (tests)
        // where it is not called until after the flask app was killed.
        // So instead of letting it return undefined, return the expected type.
        return [];
      })
  );
}

/**
 * Send invest arguments to a model's validate function.
 *
 * @param {object} payload {
 *   model_id: string (e.g. carbon)
 *   args: JSON string of InVEST model args keys and values
 * }
 * @returns {Promise} resolves array
 */
export async function fetchValidation(payload) {
  const { port, id } = await getPortAndID(payload.model_id);
  payload.model_id = id;
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/validate?language=${LANGUAGE}`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .catch((error) => {
        logger.error(error.stack);
        // In practice this function is debounced, so there's a case (tests)
        // where it is not called until after the flask app was killed.
        // So instead of letting it return undefined, return the expected type.
        return [];
      })
  );
}

/**
 * Load invest arguments from a datastack-compliant file.
 *
 * @param {string} payload - path to file
 * @returns {Promise} resolves undefined
 */
export async function fetchDatastackFromFile(payload) {
  const port = await getCorePort();
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/post_datastack_file`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
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
export async function saveToPython(payload) {
  const { port, id } = await getPortAndID(payload.model_id);
  payload.model_id = id;
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/save_to_python`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.text())
      .then((text) => {
        logger.debug(text);
        return text;
      })
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Archive invest model input data.
 *
 * @param  {object} payload {
 *   filepath: string
 *   model_id: string (e.g. carbon)
 *   args_dict: JSON string of InVEST model args keys and values
 * }
 * @returns {Promise} resolves undefined
 */
export async function archiveDatastack(payload) {
  const { port, id } = await getPortAndID(payload.model_id);
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/build_datastack_archive`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .then(({ message, error }) => {
        if (error) {
          logger.error(message);
        } else {
          logger.debug(message);
        }
        return { message, error };
      })
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Write invest model arguments to a JSON file.
 *
 * @param  {object} payload {
 *   filepath: string
 *   model_id: string (e.g. carbon)
 *   args: JSON string of InVEST model args keys and values
 *   relativePaths: boolean
 * }
 * @returns {Promise} resolves undefined
 */
export async function writeParametersToFile(payload) {
  const { port, id } = await getPortAndID(payload.model_id);
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/write_parameter_set_file`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .then(({ message, error }) => {
        if (error) {
          logger.error(message);
        } else {
          logger.debug(message);
        }
        return { message, error };
      })
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Get the mapping of supported language codes to display names.
 *
 * @returns {Promise} resolves object
 */
export async function getSupportedLanguages() {
  const port = await getCorePort();
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/languages`, {
      method: 'get',
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Get the user-profile from GeoMetaMaker.
 *
 * @returns {Promise} resolves object
 */
export async function getGeoMetaMakerProfile() {
  const port = await getCorePort();
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/get_geometamaker_profile`, {
      method: 'get',
    })
      .then((response) => response.json())
      .catch((error) => logger.error(error.stack))
  );
}

/**
 * Set the user-profile in GeoMetaMaker.
 *
 * @param {object} payload {
 *   contact: {
 *     individual_name: string
 *     email: string
 *     organization: string
 *     position_name: string
 *   }
 *   license: {
 *     title: string
 *     url: string
 *   }
 * }
 * @returns {Promise} resolves object
 */
export async function setGeoMetaMakerProfile(payload) {
  const port = await getCorePort();
  return (
    window.fetch(`${HOSTNAME}:${port}/${PREFIX}/set_geometamaker_profile`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
      .then((response) => response.json())
      .then(({ message, error }) => {
        if (error) {
          logger.error(message);
        } else {
          logger.debug(message);
        }
        return { message, error };
      })
      .catch((error) => logger.error(error.stack))
  );
}
