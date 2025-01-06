import upath from 'upath';
import fs from 'fs';
import { execSync, spawnSync } from 'child_process';

import { ipcMain } from 'electron';

import { ipcMainChannels } from './ipcMainChannels';
import { getLogger } from './logger';
import { checkFirstRun } from './setupCheckFirstRun';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

/**
 * Find paths to local invest executeable under dev or production environments.
 *
 * @param {boolean} isDevMode - a boolean designating dev mode or not.
 * @returns {string} invest binary path string.
 */
export function findInvestBinaries(isDevMode) {
  // Binding to the invest server binary:
  let investExe;
  const ext = (process.platform === 'win32') ? '.exe' : '';
  const filename = `invest${ext}`;

  if (isDevMode) {
    investExe = filename; // assume an active python env w/ exe on path
  } else {
    investExe = upath.join(process.resourcesPath, 'invest', filename);
    // It's likely the exe path includes spaces because it's composed of
    // app's Product Name, a user-facing name given to electron-builder.
    // Quoting depends on the shell, ('/bin/sh' or 'cmd.exe') and type of
    // child process. Use `spawn`` because that is how we will launch other
    // invest commands later. https://github.com/nodejs/node/issues/38490
    investExe = `"${investExe}"`;
  }
  // Checking that we have a functional invest exe by getting version
  // shell is necessary in dev mode when relying on an active conda env
  const { stdout, stderr, error } = spawnSync(
    investExe, ['--version'], { shell: true }
  );
  if (error) {
    logger.error(stderr.toString());
    logger.error('InVEST binaries are probably missing.');
    throw error;
  }
  const investVersion = stdout.toString();
  logger.info(
    `Found invest binaries ${investExe} for version ${investVersion}`
  );
  // Allow renderer to ask for the invest version, for About page.
  ipcMain.handle(
    ipcMainChannels.INVEST_VERSION, () => investVersion
  );
  return investExe;
}

/**
 * Return the available micromamba executable.
 *
 * @param {boolean} isDevMode - a boolean designating dev mode or not.
 * @returns {string} micromamba executable.
 */
export function findMicromambaExecutable(isDevMode) {
  let micromambaExe;
  if (isDevMode) {
    micromambaExe = 'micromamba'; // assume that micromamba is available
  } else {
    micromambaExe = `"${upath.join(process.resourcesPath, 'micromamba')}"`;
  }
  // Check that the executable is working
  const { stderr, error } = spawnSync(micromambaExe, ['--help'], { shell: true });
  if (error) {
    logger.error(stderr.toString());
    logger.error('micromamba executable is not where we expected it.');
    throw error;
  }
  logger.info(`using micromamba executable '${micromambaExe}'`);
  return micromambaExe;
}
