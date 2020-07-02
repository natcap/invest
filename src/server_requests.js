import fetch from 'node-fetch';
import { getLogger } from './logger'
const logger = getLogger('main')

// TODO: elsewhere I've used async/await instead of 
// .then chaining of callbacks. Consider refactoring
// everything here with async/await.

const PORT = process.env.PORT || '5000';
const HOSTNAME = 'http://localhost';

export function getFlaskIsReady(retries=0) {
  /* Recursive function to find out if the Flask server is online.
  * Sometimes the app will make a server request before it's ready,
  * so awaiting this response is one way to avoid that. 
  */
  return(
    fetch(`${HOSTNAME}:${PORT}/ready`, {
      method: 'get',
    })
    .then((response) => { return response.text() })
    // .then((text) => { logger.debug(text) })
    .catch(async (error) => {
      logger.debug(error)
      if (error.code === 'ECONNREFUSED') {
        while (retries < 21) {
          retries++;
          // try again after a short pause
          await new Promise(resolve => setTimeout(resolve, 50));
          logger.debug('retry # ' + retries);
          return await getFlaskIsReady(retries)
        }
      } else {
        logger.debug(error)
        return error 
      }
   })
  )
}

export function getInvestList() {
  return(
    fetch(`${HOSTNAME}:${PORT}/models`, {
      method: 'get',
    })
    .then((response) => { 
      return response
    })
    .then((response) => { return response.json() })
    .catch((error) => { logger.debug(error) })
  )
}

export function getSpec(payload) {
  return (
    fetch(`${HOSTNAME}:${PORT}/getspec`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.json() })
    .catch((error) => { logger.debug(error) })
  )
}

export function fetchValidation(payload) {
  return (
    fetch(`${HOSTNAME}:${PORT}/validate`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.json() })
    .catch((error) => { logger.debug(error) })
  )
}

export function fetchLogfilename(payload) {
  return (
    fetch(`${HOSTNAME}:${PORT}/get_invest_logfilename`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.text() })
    .catch((error) => { logger.debug(error) })
    )
}

export function fetchDatastackFromFile(payload) {
  return (
    fetch(`${HOSTNAME}:${PORT}/post_datastack_file`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.json() })
    .catch((error) => { logger.debug(error) })
  )
}

export function saveToPython(payload) {
  fetch(`${HOSTNAME}:${PORT}/save_to_python`, {
    method: 'post',
    body: JSON.stringify(payload),
    headers: { 'Content-Type': 'application/json' },
  })
  .then((response) => { return response.text() })
  .then((text) => { logger.debug(text) })
  .catch((error) => { logger.debug(error) })
}

export function writeParametersToFile(payload) {
  // even though we don't need a response sent back
  // from this fetch, we must ``return`` a Promise 
  // in order to ``await writeParametersToFile``.
  return (
    fetch(`${HOSTNAME}:${PORT}/write_parameter_set_file`, {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.text() })
    .then((text) => { logger.debug(text) })
    .catch((error) => { logger.debug(error) })
  );
}