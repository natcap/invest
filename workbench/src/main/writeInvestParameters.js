import fetch from 'node-fetch';

import { getLogger } from './logger';
import { settingsStore } from './settingsStore';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const HOSTNAME = 'http://127.0.0.1';

export default function writeParametersToFile(payload) {
  const port = settingsStore.get('core.port');
  return (
    fetch(`${HOSTNAME}:${port}/api/write_parameter_set_file`, {
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
