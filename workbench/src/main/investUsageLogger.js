import fetch from 'node-fetch';

import { logger } from './logger';
import pkg from '../../package.json';
import { settingsStore } from './settingsStore';

const WORKBENCH_VERSION = pkg.version;
const HOSTNAME = 'http://127.0.0.1';
const PREFIX = 'api';

export default function investUsageLogger() {

  function start(modelID, args, port) {
    logger.debug('logging model start');

    const body = {
      model_id: modelID,
      model_args: JSON.stringify(args),
      invest_interface: `Workbench ${WORKBENCH_VERSION}`,
    };

    const plugins = settingsStore.get('plugins');
    if (plugins && Object.keys(plugins).includes(modelID)) {
      body.model_type = 'plugin';
      const plugin_source = plugins[modelID].source;
      // don't log the path to a local plugin, just log that it's local
      body.plugin_source = plugin_source.startsWith('git+') ? plugin_source : 'local';
    } else {
      body.model_type = 'core';
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

  return {
    start: start,
  };
}
