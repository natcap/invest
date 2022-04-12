import fetch from 'node-fetch';

import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const HOSTNAME = 'http://localhost';

export default function writeParametersToFile(payload) {
  return (
    fetch(`${HOSTNAME}:${process.env.PORT}/api/write_parameter_set_file`, {
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
