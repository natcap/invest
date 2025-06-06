import crypto from 'crypto';

import fetch from 'node-fetch';

import { getLogger } from './logger';
import pkg from '../../package.json';
import { settingsStore } from './settingsStore';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const WORKBENCH_VERSION = pkg.version;
const HOSTNAME = 'http://127.0.0.1';
const PREFIX = 'api';

export default function investUsageLogger() {
  const sessionId = crypto.randomUUID();

  function start(modelID, args, port) {
    logger.debug('logging model start');

    const body = {
      model_id: modelID,
      model_args: JSON.stringify(args),
      invest_interface: `Workbench ${WORKBENCH_VERSION}`,
      session_id: sessionId,
    };

    const plugins = settingsStore.get('plugins');
    if (plugins && Object.keys(plugins).includes(modelID)) {
      const source = plugins[modelID].source;
      body.type = 'plugin';
      // don't log the path to a local plugin, just log that it's local
      body.source = source.startsWith('git+') ? source : 'local';
    } else {
      body.type = 'core';
    }
    fetch(`${HOSTNAME}:${port}/${PREFIX}/log_model_start`, {
      method: 'post',
      body: JSON.stringify(body),
      headers: { 'Content-Type': 'application/json' },
    })
      .then(async (response) => {
        if (!response.ok) { logger.error(await response.text()); }
      })
      .catch((error) => logger.error(error));
  }

  function exit(status, port) {
    logger.debug('logging model exit');
    fetch(`${HOSTNAME}:${port}/${PREFIX}/log_model_exit`, {
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
