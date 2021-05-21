import path from 'path';
import { spawn } from 'child_process';

import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
/**
 * Spawn a child process running the Python Flask app.
 *
 * @param  {string} investExe - path to executeable that launches flask app.
 * @returns {undefined}
 */
export default function createPythonFlaskProcess(investExe) {
  if (investExe) {
    const pythonServerProcess = spawn(
      path.basename(investExe),
      ['--debug', 'serve', '--port', process.env.PORT],
      {
        env: {
          PATH: path.dirname(investExe),
        },
      }
    );

    logger.debug(`Started python process as PID ${pythonServerProcess.pid}`);
    pythonServerProcess.stdout.on('data', (data) => {
      logger.debug(`${data}`);
    });
    pythonServerProcess.stderr.on('data', (data) => {
      logger.debug(`${data}`);
    });
    pythonServerProcess.on('error', (err) => {
      logger.error(err.stack);
      logger.error(
        `The flask app ${investExe} crashed or failed to start
         so this application must be restarted`
      );
      throw err;
    });
    pythonServerProcess.on('close', (code, signal) => {
      logger.debug(`Flask process terminated with code ${code} and signal ${signal}`);
    });
  } else {
    logger.error('no existing invest installations found');
  }
}
