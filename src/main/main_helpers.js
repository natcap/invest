const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, execFileSync } = require('child_process');
const yauzl = require('yauzl');

const { getLogger } = require('../logger');

const logger = getLogger(__filename.split('/').slice(-1)[0]);
/**
 * Find paths to local invest executeable under dev or production environments.
 *
 * @param {boolean} isDevMode - a boolean designating dev mode or not.
 * @returns {Promise} Resolves array w/ invest binary path & version strings.
 */
export function findInvestBinaries(isDevMode) {
  return new Promise(resolve => {
    // Binding to the invest server binary:
    let investExe;
    const ext = (process.platform === 'win32') ? '.exe' : '';

    if (isDevMode) {
      // If no dotenv vars are set, default to where this project's
      // build process places the binaries.
      investExe = `${process.env.INVEST || 'build/invest/invest'}${ext}`;

    // point to binaries included in this app's installation.
    } else {
      const binaryPath = path.join(process.resourcesPath, 'invest');
      investExe = path.join(binaryPath, `invest${ext}`);
    }
    const investVersion = execFileSync(investExe, ['--version']);
    logger.info(`Found invest binaries ${investExe} for version ${investVersion}`);
    resolve([investExe, `${investVersion}`.trim(os.EOL)]);
    // TODO reject somethign of the same shape as what is resolved?
  }).catch(error => {
    console.log(error.message);
    logger.error(error.message);
    logger.error('InVEST binaries are probably missing.');
  });
}

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
