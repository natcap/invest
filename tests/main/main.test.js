import fs from 'fs';
import path from 'path';
import { app, ipcMain } from 'electron';
import { execFileSync } from 'child_process';

import { createWindow, destroyWindow } from '../../src/main/main';
import {
  checkFirstRun,
  APP_HAS_RUN_TOKEN
} from '../../src/main/setupCheckFirstRun';
import {
  createPythonFlaskProcess,
} from '../../src/main/main_helpers';
import findInvestBinaries from '../../src/main/findInvestBinaries';
import { getFlaskIsReady } from '../../src/server_requests';

jest.mock('child_process');
execFileSync.mockReturnValue('foo');
jest.mock('../../src/main/main_helpers');
createPythonFlaskProcess.mockImplementation(() => {});
jest.mock('../../src/server_requests');
getFlaskIsReady.mockResolvedValue(true);

// These vars are only defined in an electron environment and our
// app expects them to be defined.
process.defaultApp = 'test'; // imitates dev mode
process.resourcesPath = 'path/to/electron/package';

describe('checkFirstRun', () => {
  const tokenPath = path.join(app.getPath(), APP_HAS_RUN_TOKEN);
  beforeEach(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  afterAll(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  it('should return true & create token if token does not exist', () => {
    expect(fs.existsSync(tokenPath)).toBe(false);
    expect(checkFirstRun()).toBe(true);
    expect(fs.existsSync(tokenPath)).toBe(true);
  });

  it('should return false if token already exists', () => {
    fs.writeFileSync(tokenPath, '');
    expect(checkFirstRun()).toBe(false);
  });
});

describe('findInvestBinaries', () => {
  afterAll(() => {
    execFileSync.mockReset();
  });
  const ext = (process.platform === 'win32') ? '.exe' : '';
  const filename = `invest${ext}`;
  it('should point to build folder in dev mode', () => {
    const isDevMode = true;
    const exePath = findInvestBinaries(isDevMode);
    expect(exePath).toBe(path.join('build', 'invest', filename));
  });
  it('should point to resourcesPath in production', async () => {
    const isDevMode = false;
    const exePath = findInvestBinaries(isDevMode);
    expect(exePath)
      .toBe(path.join(process.resourcesPath, 'invest', filename));
  });
  it('should throw if the invest exe is bad', async () => {
    execFileSync.mockImplementation(() => {
      throw new Error('error from invest --version');
    });
    const isDevMode = false;
    expect(() => findInvestBinaries(isDevMode)).toThrow();
  });
});

describe('createWindow', () => {
  beforeEach(async () => {
    jest.clearAllMocks();
    await createWindow();
  });
  afterEach(() => {
    // TODO: might not need this at all.
    destroyWindow();
  });
  it('should register various ipcMain listeners', () => {
    const expectedHandleChannels = [
      'show-context-menu',
      'show-open-dialog',
      'show-save-dialog',
      'is-dev-mode',
      'user-data',
    ];
    const expectedHandleOnceChannels = ['is-first-run'];
    const expectedOnChannels = ['download-url'];
    const receivedHandleChannels = ipcMain.handle.mock.calls.map(
      (item) => item[0]
    );
    const receivedHandleOnceChannels = ipcMain.handleOnce.mock.calls.map(
      (item) => item[0]
    );
    const receivedOnChannels = ipcMain.on.mock.calls.map(
      (item) => item[0]
    );
    expect(receivedHandleChannels.sort()).toEqual(expectedHandleChannels.sort());
    expect(receivedHandleOnceChannels.sort()).toEqual(expectedHandleOnceChannels.sort());
    expect(receivedOnChannels.sort()).toEqual(expectedOnChannels.sort());
  });
});
