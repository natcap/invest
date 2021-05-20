import path from 'path';
import fs from 'fs';
import os from 'os';
import { spawn, exec } from 'child_process';
import { app, ipcMain } from 'electron';

import { findMostRecentLogfile } from '../utils';
import { writeParametersToFile } from '../server_requests';
import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  DEBUG: '--debug',
  INFO: '-vvv',
  WARNING: '-vv',
  ERROR: '-v',
};

const TEMP_DIR = path.join(app.getPath('userData'), 'tmp');

export default function setupInvestRunHandlers(investExe) {
  const runningJobs = {};

  ipcMain.on('invest-kill', (event, workspaceDir) => {
    console.log('on invest-kill')
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

  ipcMain.on('invest-run', async (
    event, modelRunName, pyModuleName, args, loggingLevel, channel
  ) => {
    // Write a temporary datastack json for passing to invest CLI
    fs.mkdir(TEMP_DIR, (err) => {});
    const tempDatastackDir = fs.mkdtempSync(
      path.join(TEMP_DIR, 'data-')
    );
    const datastackPath = path.join(tempDatastackDir, 'datastack.json');
    // TODO: only need pyModuleName to make a compliant logfile name
    // as the prepare_workspace call in cli.py takes it from the datastack.json
    // It could get it elsewhere, like a lookup based on the run name.
    const payload = {
      parameterSetPath: datastackPath,
      moduleName: pyModuleName,
      relativePaths: false,
      args: JSON.stringify(args),
    };
    await writeParametersToFile(payload);

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
    // let stderr = Object.assign('', this.state.logStdErr);
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
