import path from 'path';
import fs from 'fs';
import os from 'os';
import { spawn, exec } from 'child_process';

import { app, ipcMain } from 'electron';

import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';
import ELECTRON_DEV_MODE from './isDevMode';
import investUsageLogger from './investUsageLogger';
import markupMessage from './investLogMarkup';
import writeInvestParameters from './writeInvestParameters';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  DEBUG: '--debug',
  INFO: '-vv',
  WARNING: '-v',
  ERROR: '',
};
const TGLOGLEVELMAP = {
  DEBUG: '--taskgraph-log-level=DEBUG',
  INFO: '--taskgraph-log-level=INFO',
  WARNING: '--taskgraph-log-level=WARNING',
  ERROR: '--taskgraph-log-level=ERROR',
};
const TEMP_DIR = path.join(app.getPath('userData'), 'tmp');

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
    event, modelRunName, pyModuleName, args, loggingLevel, taskgraphLoggingLevel, language, tabID
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
      filepath: datastackPath,
      moduleName: pyModuleName,
      relativePaths: false,
      args: JSON.stringify(args),
    };
    await writeInvestParameters(payload);

    const cmdArgs = [
      LOGLEVELMAP[loggingLevel],
      TGLOGLEVELMAP[taskgraphLoggingLevel],
      `--language "${language}"`,
      'run',
      modelRunName,
      '--headless',
      `-d "${datastackPath}"`,
    ];
    logger.debug(`set to run ${cmdArgs}`);
    if (process.platform !== 'win32') {
      investRun = spawn(investExe, cmdArgs, {
        shell: true, // without shell, IOError when datastack.py loads json
        detached: true, // counter-intuitive, but w/ true: invest terminates when this shell terminates
      });
    } else { // windows
      const envVars = JSON.parse(JSON.stringify(process.env));
      envVars.PYTHONUTF8 = '1'; // #1167 - force UTF-8 on Windows
      investRun = spawn(investExe, cmdArgs, {
        shell: true,
        env: envVars,
      });
    }

    // There's no general way to know that a spawned process started,
    // so this logic to listen once on stdout seems like the way.
    // We need to do the following only once after the process started:
    // 1. store the PID to enable killing the task.
    // 2. parse the logfile path from stdout.
    // 3. log the model run for invest usage stats.
    const stdOutCallback = async (data) => {
      const strData = `${data}`;
      if (!investStarted) {
        if (strData.match('Writing log messages to')) {
          investStarted = true;
          runningJobs[tabID] = investRun.pid;
          const investLogfile = strData.substring(
            strData.indexOf('[') + 1, strData.indexOf(']')
          );
          event.reply(`invest-logging-${tabID}`, path.resolve(investLogfile));
          if (!ELECTRON_DEV_MODE && !process.env.PUPPETEER) {
            usageLogger.start(pyModuleName, args);
          }
        }
      }
      // python logging flushes with each message, so data here should
      // only be one logger message at a time.
      event.reply(
        `invest-stdout-${tabID}`,
        [strData, markupMessage(strData, pyModuleName)]
      );
    };
    investRun.stdout.on('data', stdOutCallback);

    const stdErrCallback = (data) => {
      logger.debug(`${data}`);
    };
    investRun.stderr.on('data', stdErrCallback);

    investRun.on('close', (code) => {
      logger.debug('invest subprocess stdio streams closed');
    });

    investRun.on('exit', (code) => {
      delete runningJobs[tabID];
      event.reply(`invest-exit-${tabID}`, {
        code: code,
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
        logger.info(err);
        event.reply(
          `invest-stdout-${channel}`,
          [`Logfile is missing or unreadable: ${os.EOL}${logfile}`, '']
        );
      });

      fileStream.on('data', (data) => {
        event.reply(`invest-stdout-${channel}`, [`${data}`, '']);
      });
    });
}
