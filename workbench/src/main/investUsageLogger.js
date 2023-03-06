import crypto from 'crypto';

import fetch from 'node-fetch';

import { getLogger } from './logger';
import pkg from '../../package.json';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const WORKBENCH_VERSION = pkg.version;
const HOSTNAME = 'http://127.0.0.1';
const PREFIX = 'api';

export default function investUsageLogger() {
  const sessionId = crypto.randomUUID();

  function start(modelPyName, args) {
    logger.debug('logging model start');
    fetch(`${HOSTNAME}:${process.env.PORT}/${PREFIX}/log_model_start`, {
      method: 'post',
      body: JSON.stringify({
        model_pyname: modelPyName,
        model_args: JSON.stringify(args),
        invest_interface: `Workbench ${WORKBENCH_VERSION}`,
        session_id: sessionId,
      }),
      headers: { 'Content-Type': 'application/json' },
    })
      .then(async (response) => {
        if (!response.ok) { logger.error(await response.text()); }
      })
      .catch((error) => logger.error(error));
  }

  function exit(status) {
    logger.debug('logging model exit');
    fetch(`${HOSTNAME}:${process.env.PORT}/${PREFIX}/log_model_exit`, {
      method: 'post',
      body: JSON.stringify({
        session_id: sessionId,
        status: status,
      }),
      headers: { 'Content-Type': 'application/json' },
    })
      .then(async (response) => {
        if (!response.ok) { logger.error(await response.text()); }
      })
      .catch((error) => logger.error(error));
  }

  return {
    start: start,
    exit: exit,
  };
}
