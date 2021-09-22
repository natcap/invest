import path from 'path';
import fs from 'fs';
import os from 'os';
import { spawn, exec } from 'child_process';

import { app, ipcMain } from 'electron';
import fetch from 'node-fetch';
import sanitizeHtml from 'sanitize-html';

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

const LOG_TEXT_TAG = 'span';
const ALLOWED_HTML_OPTIONS = {
  allowedTags: [LOG_TEXT_TAG],
  allowedAttributes: { [LOG_TEXT_TAG]: ['class'] },
};
const LOG_ERROR_REGEX = /(Traceback)|(([A-Z]{1}[a-z]*){1,}Error)|(ERROR)/;
export const LOG_PATTERNS = {
  'invest-log-error': LOG_ERROR_REGEX,
  'invest-log-primary': /a^/, // default is regex that will never match
};
/**
 * Encapsulate text in html, assigning class based on text content.
 *
 * @param  {string} message - plaintext string
 * @param  {object} patterns - of shape {string: RegExp}
 * @returns {string} - sanitized html
 */
export function markupMessage(message, patterns) {
  // eslint-disable-next-line
  for (const [cls, pattern] of Object.entries(patterns)) {
    if (pattern.test(message)) {
      const markup = `<${LOG_TEXT_TAG} class="${cls}">${message}</${LOG_TEXT_TAG}>`;
      return sanitizeHtml(markup, ALLOWED_HTML_OPTIONS);
    }
  }
  return sanitizeHtml(message, ALLOWED_HTML_OPTIONS);
}

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
    const env = {
      PATH: path.dirname(investExe),
      PYTHONUNBUFFERED: 'TRUE', // stdio gets python log messages one at a time
    };
    if (process.platform !== 'win32') {
      investRun = spawn(path.basename(investExe), cmdArgs, {
        env: env,
        shell: true, // without shell, IOError when datastack.py loads json
        detached: true, // counter-intuitive, but w/ true: invest terminates when this shell terminates
      });
    } else { // windows
      investRun = spawn(path.basename(investExe), cmdArgs, {
        env: env,
        shell: true,
      });
    }

    const logPatterns = { ...LOG_PATTERNS };
    logPatterns['invest-log-primary'] = new RegExp(pyModuleName);
    // There's no general way to know that a spawned process started,
    // so this logic to listen once on stdout seems like the way.
    // We need to store the PID to enable killing the task.
    // And need to parse the logfile path from stdout.
    const stdOutCallback = async (data) => {
      if (!investLogfile) {
        if (`${data}`.match('Writing log messages to')) {
          investLogfile = `${data}`.split(' ').pop().trim();
          runningJobs[jobID] = investRun.pid;
          event.reply(`invest-logging-${jobID}`, investLogfile);
        }
      }
      // we set python stdio to be unbuffered, so data here should
      // only be one logger message at a time.
      event.reply(
        `invest-stdout-${jobID}`,
        markupMessage(`${data}`, logPatterns)
      );
    };
    investRun.stdout.on('data', stdOutCallback);

    const stdErrCallback = (data) => {
      // The python Traceback for invest comes through stdout & stderr,
      // So no need to merge those streams for the benefit of displaying
      // a complete log.
      logger.debug(`${data}`);
      // The PyInstaller exe will always emit a final 'Failed ...' message
      // after an uncaught exception. It is not helpful to display to users
      // so we filter it out and stop listening to stderr when we find it.
      const dataArray = `${data}`.split(/\[[0-9]+\] Failed to execute/);
      if (dataArray.length > 1) {
        investRun.stderr.removeListener('data', stdErrCallback);
      }
      const dat = dataArray[0];
      event.reply(`invest-stderr-${jobID}`, `${dat}`);
    };
    investRun.stderr.on('data', stdErrCallback);

    investRun.on('exit', (code) => {
      delete runningJobs[jobID];
      event.reply(`invest-exit-${jobID}`, code);
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
  ipcMain.on(ipcMainChannels.INVEST_READ_LOG,
    (event, logfile, pyModuleName, channel) => {
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
