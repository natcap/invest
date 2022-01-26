import path from 'path';
import fs from 'fs';
import os from 'os';
import { spawn, exec } from 'child_process';

import { app, ipcMain } from 'electron';
import fetch from 'node-fetch';

import { getLogger } from '../logger';
import { ipcMainChannels } from './ipcMainChannels';
import ELECTRON_DEV_MODE from './isDevMode';
import investUsageLogger from './investUsageLogger';
import markupMessage from './investLogMarkup';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  DEBUG: '--debug',
  INFO: '-vv',
  WARNING: '-v',
  ERROR: '',
};
const TEMP_DIR = path.join(app.getPath('userData'), 'tmp');
const HOSTNAME = 'http://localhost';

export function setupInvestRunHandlers(investExe) {
  const runningJobs = {};

  ipcMain.on(ipcMainChannels.INVEST_KILL, (event, jobID) => {
    if (runningJobs[jobID]) {
      const pid = runningJobs[jobID];
      if (process.platform !== 'win32') {
        // the '-' prefix on pid sends signal to children as well
        process.kill(-pid, 'SIGTERM');
      } else {
        exec(`taskkill /pid ${pid} /t /f`);
      }
    }
  });

  ipcMain.on(ipcMainChannels.INVEST_RUN, async (
    event, modelRunName, pyModuleName, args, loggingLevel, jobID
  ) => {
    let investRun;
    let investStarted = false;
    let investStdErr = '';
    const usageLogger = investUsageLogger();

    // Write a temporary datastack json for passing to invest CLI
    try {
      fs.mkdirSync(TEMP_DIR);
    } catch {}
    const tempDatastackDir = fs.mkdtempSync(
      path.join(TEMP_DIR, 'data-')
    );
    const datastackPath = path.join(tempDatastackDir, 'datastack.json');
    const payload = {
      parameterSetPath: datastackPath,
      moduleName: pyModuleName,
      relativePaths: false,
      args: JSON.stringify(args),
    };
    try {
      const response = await fetch(`${HOSTNAME}:${process.env.PORT}/write_parameter_set_file`, {
        method: 'post',
        body: JSON.stringify(payload),
        headers: { 'Content-Type': 'application/json' },
      });
      logger.debug(await response.text());
    } catch (error) {
      logger.error(error.stack);
    }

    const cmdArgs = [
      LOGLEVELMAP[loggingLevel],
      'run',
      modelRunName,
      '--headless',
      `-d "${datastackPath}"`,
    ];
    logger.debug(`set to run ${cmdArgs}`);
    if (process.platform !== 'win32') {
      // It's likely the exe path includes spaces because it's composed of the
      // app's Product Name, a user-facing name given to electron-builder.
      // extra quotes because https://github.com/nodejs/node/issues/38490
      // Seems quotes only needed when shell is /usr/bin/sh
      investRun = spawn(`"${investExe}"`, cmdArgs, {
        shell: true, // without shell, IOError when datastack.py loads json
        detached: true, // counter-intuitive, but w/ true: invest terminates when this shell terminates
      });
    } else { // windows
      investRun = spawn(`"${investExe}"`, cmdArgs, {
        shell: true,
      });
    }

    // There's no general way to know that a spawned process started,
    // so this logic to listen once on stdout seems like the way.
    // We need to do the following only once after the process started:
    // 1. store the PID to enable killing the task.
    // 2. parse the logfile path from stdout.
    // 3. log the model run for invest usage stats.
    const stdOutCallback = async (data) => {
      if (!investStarted) {
        if (`${data}`.match('Writing log messages to')) {
          investStarted = true;
          runningJobs[jobID] = investRun.pid;
          const investLogfile = `${data}`.split(' ').pop().trim();
          event.reply(`invest-logging-${jobID}`, investLogfile);
          if (!ELECTRON_DEV_MODE && !process.env.PUPPETEER) {
            usageLogger.start(pyModuleName, args);
          }
        }
      }
      // python logging flushes with each message, so data here should
      // only be one logger message at a time.
      event.reply(
        `invest-stdout-${jobID}`,
        markupMessage(`${data}`, pyModuleName)
      );
    };
    investRun.stdout.on('data', stdOutCallback);

    const stdErrCallback = (data) => {
      logger.debug(`${data}`);
      investStdErr += `${data}`;
    };
    investRun.stderr.on('data', stdErrCallback);

    investRun.on('close', (code) => {
      logger.debug('invest subprocess stdio streams closed');
    });

    investRun.on('exit', (code) => {
      delete runningJobs[jobID];
      event.reply(`invest-exit-${jobID}`, {
        code: code,
        stdErr: investStdErr,
      });
      logger.debug(code);
      fs.unlink(datastackPath, (err) => {
        if (err) { logger.error(err); }
        fs.rmdir(tempDatastackDir, (e) => {
          if (e) { logger.error(e); }
        });
      });
      if (!ELECTRON_DEV_MODE && !process.env.PUPPETEER) {
        usageLogger.exit(investStdErr);
      }
    });
  });
}

export function setupInvestLogReaderHandler() {
  ipcMain.on(ipcMainChannels.INVEST_READ_LOG,
    (event, logfile, channel) => {
      const fileStream = fs.createReadStream(logfile);
      fileStream.on('error', (err) => {
        logger.error(err);
        event.reply(
          `invest-stdout-${channel}`,
          `Logfile is missing or unreadable: ${os.EOL}${logfile}`
        );
      });

      fileStream.on('data', (data) => {
        event.reply(`invest-stdout-${channel}`, `${data}`);
      });
    });
}
