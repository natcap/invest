const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const yauzl = require('yauzl');

const { getLogger } = require('../logger');

const logger = getLogger(__filename.split('/').slice(-1)[0]);
/**
 * Spawn a child process running the Python Flask app.
 *
 * @param  {string} investExe - path to executeable that launches flask app.
 * @returns {undefined}
 */
export function createPythonFlaskProcess(investExe) {
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

export function extractZipInplace(zipFilePath) {
  return new Promise((resolve, reject) => {
    const extractToDir = path.dirname(zipFilePath);
    logger.info(`extracting ${zipFilePath}`);
    yauzl.open(zipFilePath, { lazyEntries: true }, (err, zipfile) => {
      if (err) throw err;
      zipfile.readEntry();
      zipfile.on('entry', (entry) => {
        if (/\/$/.test(entry.fileName)) {
          // if entry is a directory
          fs.mkdir(path.join(extractToDir, entry.fileName), (err) => {
            if (err) {
              if (err.code === 'EEXIST') { } else logger.error(err);
            }
            zipfile.readEntry();
          });
        } else {
          zipfile.openReadStream(entry, (err, readStream) => {
            if (err) throw err;
            readStream.on('end', () => {
              zipfile.readEntry();
            });
            const writable = fs.createWriteStream(path.join(
              extractToDir, entry.fileName
            ));
            readStream.pipe(writable);
          });
        }
      });
      zipfile.on('close', () => {
        resolve(true);
      });
    });
  });
}
