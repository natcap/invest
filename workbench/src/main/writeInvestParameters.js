import fetch from 'node-fetch';

import { logger } from './logger';

const HOSTNAME = 'http://127.0.0.1';

export default function writeParametersToFile(payload, port) {
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
