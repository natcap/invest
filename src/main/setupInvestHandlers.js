import path from 'path';
import fs from 'fs';
import os from 'os';
import { spawn, exec } from 'child_process';
import { ipcMain } from 'electron';

import { findMostRecentLogfile } from '../utils';
import { writeParametersToFile } from '../server_requests';
import { fileRegistry } from '../constants';
import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  DEBUG: '--debug',
  INFO: '-vvv',
  WARNING: '-vv',
  ERROR: '-v',
};

/** Write an invest args JSON file for passing to invest cli.
 *
 * Outsourcing this to natcap.invest.datastack via flask ensures
 * a compliant json with an invest version string.
 *
 * @param {string} datastackPath - path to a JSON file.
 * @param {object} argsValues - the invest "args dictionary"
 *   as a javascript object
 */
export async function argsToJsonFile(datastackPath, argsValues) {
  const payload = {
    parameterSetPath: datastackPath,
    moduleName: this.state.modelSpec.module,
    relativePaths: false,
    args: JSON.stringify(argsValues),
  };
  await writeParametersToFile(payload);
}

export function setupInvestArgsToJsonHandler() {
  ipcMain.handle('invest-args-to-json', (event, datastackPath, args) => {
    argsToJsonFile(datastackPath, args);
  });
}

export function setupInvestRunHandlers(investExe) {
  const runningJobs = {};

  ipcMain.handle('invest-kill', (event, workspaceDir) => {
    if (runningJobs[workspaceDir]) {
      const pid = runningJobs[workspaceDir];
      if (process.platform !== 'win32') {
        // the '-' prefix on pid sends signal to children as well
        process.kill(-pid, 'SIGTERM');
      } else {
        exec(`taskkill /pid ${pid} /t /f`);
      }
      return 'Run Canceled';
    }
  });

  ipcMain.on('invest-run', async (event, modelRunName, args, loggingLevel) => {
    // Write a temporary datastack json for passing to invest CLI
    const tempDir = fs.mkdtempSync(path.join(
      fileRegistry.TEMP_DIR, 'data-'
    ));
    const datastackPath = path.join(tempDir, 'datastack.json');
    await argsToJsonFile(datastackPath, args);

    const cmdArgs = [
      LOGLEVELMAP[loggingLevel],
      'run',
      modelRunName,
      '--headless',
      `-d "${datastackPath}"`,
    ];
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
    investRun.stdout.once('data', async () => {
      const logfile = await findMostRecentLogfile(args.workspaceDir);
      // job.setProperty('logfile', logfile);
      // TODO: handle case when logfile is still undefined?
      // Could be if some stdout is emitted before a logfile exists.
      // logger.debug(`invest logging to: ${job.metadata.logfile}`);
      // job.save();
      runningJobs[args.workspaceDir] = investRun.pid;
      event.reply(`invest-logging-${args.workspaceDir}`, logfile);
      // this.setState(
      //   {
      //     procID: investRun.pid,
      //   }, () => {
      //     this.switchTabs('log');
      //     saveJob(job);
      //   }
      // );
    });

    // Capture stderr to a string separate from the invest log
    // so that it can be displayed separately when invest exits.
    // And because it could actually be stderr emitted from the
    // invest CLI or even the shell, rather than the invest model,
    // in which case it's useful to logger.debug too.
    // let stderr = Object.assign('', this.state.logStdErr);
    investRun.stderr.on('data', (data) => {
      logger.debug(`${data}`);
      // stderr += `${data}${os.EOL}`;
      event.reply(`invest-stderr-${args.workspaceDir}`, `${data}${os.EOL}`);
      // this.setState({
      //   logStdErr: stderr,
      // });
    });

    // Set some state when the invest process exits and update the app's
    // persistent database by calling saveJob.
    investRun.on('exit', (code) => {
      delete runningJobs[args.workspaceDir];
      event.reply(`invest-exit-${args.workspaceDir}`, code);
      logger.debug(code);
      fs.unlink(datastackPath, (err) => {
        if (err) { logger.error(err); }
        fs.rmdir(tempDir, (e) => {
          if (e) { logger.error(e); }
        });
      });
      // if (code === 0) {
      //   job.setProperty('status', 'success');
      // } else {
      //   // Invest CLI exits w/ code 1 when it catches errors,
      //   // Models exit w/ code 255 (on all OS?) when errors raise from execute()
      //   // Windows taskkill yields exit code 1
      //   // Non-windows process.kill yields exit code null
      //   job.setProperty('status', 'error');
      // }
      // this.setState({
      //   jobStatus: job.metadata.status,
      //   procID: null,
      // }, () => {
      //   saveJob(job);
      //   fs.unlink(datastackPath, (err) => {
      //     if (err) { logger.error(err); }
      //     fs.rmdir(tempDir, (e) => {
      //       if (e) { logger.error(e); }
      //     });
      //   });
      // });
    });
  });
}
