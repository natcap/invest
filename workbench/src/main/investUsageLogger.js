import crypto from 'crypto';

import fetch from 'node-fetch';

import { getLogger } from './logger';
import pkg from '../../package.json';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const WORKBENCH_VERSION = pkg.version;
const HOSTNAME = 'http://localhost';

export default function investUsageLogger() {
  const sessionId = crypto.randomUUID();

  function start(modelPyName, args) {
    try {
      logger.debug('logging model start');
      fetch(`${HOSTNAME}:${process.env.PORT}/log_model_start`, {
        method: 'post',
        body: JSON.stringify({
          model_pyname: modelPyName,
          model_args: JSON.stringify(args),
          invest_interface: `Workbench ${WORKBENCH_VERSION}`,
          session_id: sessionId,
        }),
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (error) {
      logger.warn('Failed to log model start');
      logger.warn(error.stack);
    }
  }

  function exit(status) {
    try {
      logger.debug('logging model exit');
      fetch(`${HOSTNAME}:${process.env.PORT}/log_model_exit`, {
        method: 'post',
        body: JSON.stringify({
          session_id: sessionId,
          status: status,
        }),
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (error) {
      logger.warn('Failed to log model exit');
      logger.warn(error.stack);
    }
  }

  return {
    start: start,
    exit: exit,
  };
}
