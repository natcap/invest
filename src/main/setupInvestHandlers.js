import path from 'path';
import fs from 'fs';
import os from 'os';
import { spawn, exec } from 'child_process';

import { app, ipcMain } from 'electron';
import glob from 'glob';
import fetch from 'node-fetch';

import { getLogger } from '../logger';
import { ipcMainChannels } from './ipcMainChannels';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  DEBUG: '--debug',
  INFO: '-vvv',
  WARNING: '-vv',
  ERROR: '-v',
};
const TEMP_DIR = path.join(app.getPath('userData'), 'tmp');
const LOGFILE_REGEX = /InVEST-natcap\.invest\.[a-zA-Z._]+-log-[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}_[0-9]{2}_[0-9]{2}.txt/g;
const HOSTNAME = 'http://localhost';

/**
 * Given an invest workspace, find the most recently modified invest log.
 *
 * This function is used in order to associate a logfile with an active
 * InVEST run, so the log can be tailed to a UI component.
 *
 * @param {string} directory - the path to an invest workspace directory
 * @returns {Promise} - resolves string path to an invest logfile
 */
export function findMostRecentLogfile(directory) {
  return new Promise((resolve) => {
    const files = glob.sync(path.join(directory, '*.txt'));
    const logfiles = [];
    files.forEach((file) => {
      const match = file.match(LOGFILE_REGEX);
      if (match) {
        logfiles.push(path.join(directory, match[0]));
      }
    });
    if (logfiles.length === 1) {
      // This is the most likely path
      resolve(logfiles[0]);
      return;
    }
    if (logfiles.length > 1) {
      // reverse sort (b - a) based on last-modified time
      const sortedFiles = logfiles.sort(
        (a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs
      );
      resolve(sortedFiles[0]);
    } else {
      logger.error(`No invest logfile found in ${directory}`);
      resolve(undefined);
    }
  });
}

export function setupInvestRunHandlers(investExe) {
  const runningJobs = {};

  ipcMain.on(ipcMainChannels.INVEST_KILL, (event, workspaceDir) => {
    if (runningJobs[workspaceDir]) {
      const pid = runningJobs[workspaceDir];
      if (process.platform !== 'win32') {
        // the '-' prefix on pid sends signal to children as well
        process.kill(-pid, 'SIGTERM');
      } else {
        exec(`taskkill /pid ${pid} /t /f`);
      }
    }
  });

  ipcMain.on(ipcMainChannels.INVEST_RUN, async (
    event, modelRunName, pyModuleName, args, loggingLevel, channel
  ) => {
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
    let investRun;
    if (process.platform !== 'win32') {
      investRun = spawn(path.basename(investExe), cmdArgs, {
        env: { PATH: path.dirname(investExe) },
        shell: true, // without shell, IOError when datastack.py loads json
        detached: true, // counter-intuitive, but w/ true: invest terminates when this shell terminates
      });
    } else { // windows
      investRun = spawn(path.basename(investExe), cmdArgs, {
        env: { PATH: path.dirname(investExe) },
        shell: true,
      });
    }

    // There's no general way to know that a spawned process started,
    // so this logic to listen once on stdout seems like the way.
    investRun.stdout.once('data', async (data) => {
      const logfile = await findMostRecentLogfile(args.workspace_dir);
      // TODO: handle case when logfile is still undefined?
      // Could be if some stdout is emitted before a logfile exists.
      logger.debug(`invest logging to: ${logfile}`);
      runningJobs[args.workspace_dir] = investRun.pid;
      event.reply(`invest-logging-${channel}`, logfile);
    });

    // Capture stderr to a string separate from the invest log
    // so that it can be displayed separately when invest exits.
    // And because it could actually be stderr emitted from the
    // invest CLI or even the shell, rather than the invest model,
    // in which case it's useful to logger.debug too.
    investRun.stderr.on('data', (data) => {
      logger.debug(`${data}`);
      event.reply(`invest-stderr-${channel}`, `${data}${os.EOL}`);
    });

    // Set some state when the invest process exits and update the app's
    // persistent database by calling saveJob.
    investRun.on('exit', (code) => {
      delete runningJobs[args.workspace_dir];
      event.reply(`invest-exit-${channel}`, code);
      logger.debug(code);
      fs.unlink(datastackPath, (err) => {
        if (err) { logger.error(err); }
        fs.rmdir(tempDatastackDir, (e) => {
          if (e) { logger.error(e); }
        });
      });
    });
  });
}
