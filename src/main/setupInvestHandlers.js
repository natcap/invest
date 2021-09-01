import path from 'path';
import fs from 'fs';
import readline from 'readline';
import os from 'os';
import { spawn, exec } from 'child_process';

import { app, ipcMain } from 'electron';
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
const HOSTNAME = 'http://localhost';

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
    let investLogfile;
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
    // We need to store the PID to enable killing the task.
    // And need to parse the logfile path from stdout.
    const stdOutCallback = async (data) => {
      if (!investLogfile) {
        if (`${data}`.match('Writing log messages to')) {
          investLogfile = `${data}`.split(' ').pop();
          runningJobs[args.workspace_dir] = investRun.pid;
          event.reply(`invest-logging-${channel}`, investLogfile);
        }
      }
      event.reply(`invest-stdout-${channel}`, `${data}`);
    };
    investRun.stdout.on('data', stdOutCallback);

    // Capture stderr to a string separate from the invest log
    // so that it can be displayed separately when invest exits.
    // And because it could actually be stderr emitted from the
    // invest CLI or even the shell, rather than the invest model,
    // in which case it's useful to logger.debug too.
    const stdErrCallback = (data) => {
      logger.debug(`${data}`);
      // The PyInstaller exe will always emit a final 'Failed ...' message
      // after an uncaught exception. It is not helpful to display to users
      // so we filter it out and stop listening to stderr when we find it.
      const dataArray = `${data}`.split(/\[[0-9]+\] Failed to execute/);
      if (dataArray.length > 1) {
        investRun.stderr.removeListener('data', stdErrCallback);
      }
      const dat = dataArray[0];
      event.reply(`invest-stderr-${channel}`, `${dat}${os.EOL}`);
    };
    investRun.stderr.on('data', stdErrCallback);

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

export function setupInvestLogReaderHandler() {
  ipcMain.on(ipcMainChannels.INVEST_READ_LOG, async (event, logfile, channel) => {    
    try {
      console.log('trying read stream')
      const fileStream = fs.createReadStream(logfile);
      const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
      });

      for await (const line of rl) {
        // event.reply(channel, line);
        event.reply(`invest-stdout-${channel}`, `${line}`);
      }
    } catch {
      console.log('caught read error')
      event.reply(
        `invest-stdout-${channel}`,
        `Logfile is missing or unreadable: ${os.EOL}${logfile}`
      );
    }
  });
}
