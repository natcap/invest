/**
 * @jest-environment node
 */

/* Tests for main process code running in a node env.
 * Tests requiring a DOM do not belong here, they are
 * in tests/renderer/*.test.js.
*/

import fs from 'fs';
import os from 'os';
import upath from 'upath';
import { spawnSync } from 'child_process';

import { app, ipcMain } from 'electron';
import yazl from 'yazl';
import fetch from 'node-fetch';

import {
  createWindow,
  destroyWindow,
  removeIpcMainListeners
} from '../../src/main/main';
import {
  checkFirstRun,
  APP_HAS_RUN_TOKEN
} from '../../src/main/setupCheckFirstRun';
import {
  createCoreServerProcess,
  getFlaskIsReady
} from '../../src/main/createPythonFlaskProcess';
import { findInvestBinaries } from '../../src/main/findBinaries';
import extractZipInplace from '../../src/main/extractZipInplace';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import investUsageLogger from '../../src/main/investUsageLogger';
import { settingsStore } from '../../src/main/settingsStore';

jest.mock('node-fetch');
jest.mock('child_process');
jest.mock('../../src/main/createPythonFlaskProcess');

beforeEach(() => {
  // A mock for the call in findInvestBinaries
  spawnSync.mockReturnValue({
    stdout: Buffer.from('foo', 'utf8'),
    stderr: Buffer.from('', 'utf8'),
    error: null
  });
  createCoreServerProcess.mockImplementation(() => {});
  getFlaskIsReady.mockResolvedValue(true);
  // These vars are only defined in an electron environment and our
  // app expects them to be defined.
  process.defaultApp = 'test'; // imitates dev mode
  process.resourcesPath = 'path/to/electron/package';
});

describe('checkFirstRun', () => {
  const tokenPath = upath.join(app.getPath(), APP_HAS_RUN_TOKEN);
  beforeEach(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  afterEach(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  test('should return true & create token if token does not exist', () => {
    expect(fs.existsSync(tokenPath)).toBe(false);
    expect(checkFirstRun()).toBe(true);
    expect(fs.existsSync(tokenPath)).toBe(true);
  });

  test('should return false if token already exists', () => {
    fs.writeFileSync(tokenPath, '');
    expect(checkFirstRun()).toBe(false);
  });
});

describe('findInvestBinaries', () => {
  const ext = (process.platform === 'win32') ? '.exe' : '';
  const filename = `invest${ext}`;

  test('should be on the PATH in dev mode', () => {
    const isDevMode = true;
    const exePath = findInvestBinaries(isDevMode);
    expect(exePath).toBe(filename);
  });

  test('should point to resourcesPath in production', async () => {
    const isDevMode = false;
    const exePath = findInvestBinaries(isDevMode);
    expect(exePath).toBe(
      `"${upath.join(process.resourcesPath, 'invest', filename)}"`
    );
  });

  test('should throw if the invest exe is bad', async () => {
    spawnSync.mockReturnValue({
      stdout: Buffer.from('', 'utf8'),
      stderr: Buffer.from('', 'utf8'),
      error: new Error('error from invest --version')
    });
    const isDevMode = false;
    expect(() => findInvestBinaries(isDevMode)).toThrow();
  });
});

// For now, skipping on Windows as we've been unable to resolve a race condition
// Fully addressing https://github.com/natcap/invest/issues/1239
// probably means re-writing this test from scratch.
const maybe = process.platform !== 'win32' ? describe : describe.skip;
maybe('extractZipInplace', () => {
  const root = fs.mkdtempSync(upath.join(os.tmpdir(), 'data-'));
  const zipPath = upath.join(root, 'output.zip');
  let level1Dir;
  let level2Dir;
  let file1Path;
  let file2Path;
  let doneZipping = false;

  beforeEach((done) => {
    level1Dir = fs.mkdtempSync(upath.join(root, 'level1'));
    level2Dir = fs.mkdtempSync(upath.join(level1Dir, 'level2'));
    file1Path = upath.join(level1Dir, 'file1');
    file2Path = upath.join(level2Dir, 'file2');
    fs.closeSync(fs.openSync(file1Path, 'w'));
    fs.closeSync(fs.openSync(file2Path, 'w'));

    const zipfile = new yazl.ZipFile();
    // adding the deeper file first, so extract function needs to
    // deal with extracting to non-existent directories.
    zipfile.addFile(file2Path, upath.relative(root, file2Path));
    zipfile.addFile(file1Path, upath.relative(root, file1Path));
    zipfile.outputStream.pipe(
      fs.createWriteStream(zipPath)
    ).on('close', () => {
      fs.unlinkSync(file1Path);
      fs.unlinkSync(file2Path);
      doneZipping = true;
      done();
    });
    zipfile.end();
  });

  afterEach(() => {
    // forced, recursive rm not reliable on Windows
    // so remove specific items one at a time, bottom up.
    fs.unlinkSync(zipPath);
    fs.unlinkSync(file2Path);
    fs.unlinkSync(file1Path);
    fs.rmdirSync(level2Dir);
    fs.rmdirSync(level1Dir);
    fs.rmSync(root, {
      recursive: true,
      force: true,
      maxRetries: 20,
    });
  });

  test('should extract recursively', async () => {
    expect(doneZipping).toBe(true);
    // The expected state after the setup, before extraction
    expect(fs.existsSync(zipPath)).toBe(true);
    expect(fs.existsSync(file1Path)).toBe(false);
    expect(fs.existsSync(file2Path)).toBe(false);

    expect(await extractZipInplace(zipPath)).toBe(true);

    // And the expected state after extraction
    expect(fs.existsSync(file1Path)).toBe(true);
    expect(fs.existsSync(file2Path)).toBe(true);
  });
});

describe('createWindow', () => {
  test('should register various ipcMain listeners', async () => {
    await createWindow();
    const expectedHandleChannels = [
      ipcMainChannels.ADD_PLUGIN,
      ipcMainChannels.REMOVE_PLUGIN,
      ipcMainChannels.BASE_URL,
      ipcMainChannels.CHANGE_LANGUAGE,
      ipcMainChannels.CHECK_STORAGE_TOKEN,
      ipcMainChannels.CHECK_FILE_PERMISSIONS,
      ipcMainChannels.DOWNLOAD_MSVC,
      ipcMainChannels.GET_SETTING,
      ipcMainChannels.GET_N_CPUS,
      ipcMainChannels.HAS_MSVC,
      ipcMainChannels.INVEST_VERSION,
      ipcMainChannels.IS_NEW_VERSION,
      ipcMainChannels.IS_FIRST_RUN,
      ipcMainChannels.LAUNCH_PLUGIN_SERVER,
      ipcMainChannels.OPEN_PATH,
      ipcMainChannels.SHOW_OPEN_DIALOG,
      ipcMainChannels.SHOW_SAVE_DIALOG,
    ];
    const expectedOnChannels = [
      ipcMainChannels.DOWNLOAD_URL,
      ipcMainChannels.GET_ELECTRON_PATHS,
      ipcMainChannels.GET_LANGUAGE,
      ipcMainChannels.INVEST_RUN,
      ipcMainChannels.INVEST_KILL,
      ipcMainChannels.INVEST_READ_LOG,
      ipcMainChannels.LOGGER,
      ipcMainChannels.SHOW_ITEM_IN_FOLDER,
      ipcMainChannels.OPEN_EXTERNAL_URL,
      ipcMainChannels.OPEN_LOCAL_HTML,
      ipcMainChannels.SET_SETTING,
    ];
    // Even with mocking, the 'on' method is a real event handler,
    // so we can get it's registered events from the EventEmitter.
    const registeredOnChannels = ipcMain.eventNames();
    // for 'handle', we query the mock's calls.
    const registeredHandleChannels = ipcMain.handle.mock.calls.map(
      (item) => item[0]
    );
    expect(registeredHandleChannels.sort())
      .toEqual(expectedHandleChannels.sort());
    expect(registeredOnChannels.sort())
      .toEqual(expectedOnChannels.sort());
    removeIpcMainListeners();
    expect(ipcMain.eventNames()).toEqual([]);
    destroyWindow();
  });
});

describe('investUsageLogger', () => {
  // Set default PORT for URL, but it's not used by the test.
  const PORT = 3000;
  const expectedURL = `http://127.0.0.1:${PORT}/api/log_model_start`;
  beforeEach(() => {
    // the expected response
    const response = {
      ok: true,
      text: async () => 'foo',
    };
    fetch.mockResolvedValue(response);
  });

  test('sends requests with correct payload', () => {
    const modelID = 'carbon';
    const args = {
      workspace_dir: 'foo',
      aoi: 'bar',
    };
    const investStdErr = '';
    const usageLogger = investUsageLogger();

    usageLogger.start(modelID, args, PORT);
    expect(fetch.mock.calls).toHaveLength(1);
    expect(fetch.mock.calls[0][0]).toBe(expectedURL);
    const startPayload = JSON.parse(fetch.mock.calls[0][1].body);

    usageLogger.exit(investStdErr, PORT);
    expect(fetch.mock.calls).toHaveLength(2);
    const exitPayload = JSON.parse(fetch.mock.calls[1][1].body);

    expect(startPayload.type).toBe('core');
    expect(startPayload.session_id).toBe(exitPayload.session_id);
    expect(startPayload.model_id).toBe(modelID);
    expect(JSON.parse(startPayload.model_args)).toMatchObject(args);
    expect(startPayload.invest_interface).toContain('Workbench');
    expect(exitPayload.status).toBe(investStdErr);
  });

  test('logs plugin usage correctly', () => {
    const modelID = 'plugin';
    const args = {
      workspace_dir: 'foo',
      aoi: 'bar',
    };
    const investStdErr = '';
    const usageLogger = investUsageLogger();

    settingsStore.set('plugins', {
      pluginA: {
        source: 'git+https://plugin'
      },
      pluginB: {
        source: '/path/to/local/plugin'
      }
    });

    usageLogger.start('pluginA', args, PORT);
    let startPayload = JSON.parse(fetch.mock.calls[0][1].body);
    expect(startPayload.type).toBe('plugin');
    expect(startPayload.model_id).toBe('pluginA');
    expect(startPayload.source).toBe('git+https://plugin');

    usageLogger.start('pluginB', args, PORT);
    startPayload = JSON.parse(fetch.mock.calls[1][1].body);
    expect(startPayload.type).toBe('plugin');
    expect(startPayload.model_id).toBe('pluginB');
    expect(startPayload.source).toBe('local');
  });
});
