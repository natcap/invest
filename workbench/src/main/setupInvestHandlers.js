import path from 'path';
import fs from 'fs';
import http from 'http';
import os from 'os';
import { spawn, exec } from 'child_process';

import { app, ipcMain } from 'electron';

import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';
import ELECTRON_DEV_MODE from './isDevMode';
import investUsageLogger from './investUsageLogger';
import markupMessage from './investLogMarkup';
import writeInvestParameters from './writeInvestParameters';
import { settingsStore } from './settingsStore';
import { createPluginServerProcess } from './createPythonFlaskProcess';

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

export function setupLaunchPluginServerHandler() {
  ipcMain.handle(
    ipcMainChannels.LAUNCH_PLUGIN_SERVER,
    async (event, pluginName) => {
      const pid = await createPluginServerProcess(pluginName);
      return pid;
    }
  );
}

export function setupInvestRunHandlers() {
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
    event, modelRunName, pyModuleName, args, tabID
  ) => {
    let investStarted = false;
    const investStdErr = '';
    const usageLogger = investUsageLogger();
    const loggingLevel = settingsStore.get('loggingLevel');
    const taskgraphLoggingLevel = settingsStore.get('taskgraphLoggingLevel');
    const language = settingsStore.get('language');
    const nWorkers = settingsStore.get('nWorkers');

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
      args: JSON.stringify({
        ...args,
        n_workers: nWorkers,
      }),
    };
    await writeInvestParameters(payload);
    let cmd;
    let cmdArgs;
    const plugins = settingsStore.get('plugins');
    if (plugins && Object.keys(plugins).includes(modelRunName)) {
      cmd = 'micromamba'//settingsStore.get('micromamba_path');
      cmdArgs = [
        'run',
        `--prefix ${settingsStore.get(`plugins.${modelRunName}.env`)}`,
        'invest',
        LOGLEVELMAP[loggingLevel],
        TGLOGLEVELMAP[taskgraphLoggingLevel],
        `--language "${language}"`,
        'run',
        modelRunName,
        `-d "${datastackPath}"`,
      ];
    } else {
      cmd = settingsStore.get('investExe');
      cmdArgs = [
        LOGLEVELMAP[loggingLevel],
        TGLOGLEVELMAP[taskgraphLoggingLevel],
        `--language "${language}"`,
        'run',
        modelRunName,
        `-d "${datastackPath}"`];
    }

    logger.debug(`about to run model with command: ${cmd} ${cmdArgs}`);

    // without shell, IOError when datastack.py loads json
    const spawnOptions = { shell: true };
    if (process.platform !== 'win32') {
      // counter-intuitive, but w/ true: invest terminates when this shell terminates
      spawnOptions.detached = true;
    }
    const investRun = spawn(cmd, cmdArgs, spawnOptions);

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

    investRun.on('close', () => {
      logger.debug('invest subprocess stdio streams closed');
    });

    investRun.on('exit', (code, signal) => {
      delete runningJobs[tabID];
      event.reply(`invest-exit-${tabID}`, {
        code: code,
      });
      logger.debug(`invest exited with code: ${code} and signal: ${signal}`);
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
  ipcMain.on(
    ipcMainChannels.INVEST_READ_LOG,
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
    }
  );
}
