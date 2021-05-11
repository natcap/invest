import fs from 'fs';
import path from 'path';
import { app, ipcMain } from 'electron';

import { createWindow, destroyWindow } from '../../src/main/main';
import {
  checkFirstRun,
  APP_HAS_RUN_TOKEN
} from '../../src/main/setupCheckFirstRun';
import setupContextMenu from '../../src/main/setupContextMenu';

import {
  createPythonFlaskProcess,
  findInvestBinaries
} from '../../src/main/main_helpers';
import { getFlaskIsReady } from '../../src/server_requests';

jest.mock('../../src/main/main_helpers');
findInvestBinaries.mockResolvedValue(['', '']);
createPythonFlaskProcess.mockImplementation(() => {});
jest.mock('../../src/server_requests');
getFlaskIsReady.mockResolvedValue(true);

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

describe('createWindow', () => {
  beforeEach(async () => {
    jest.clearAllMocks();
    await createWindow();
  });
  afterEach(() => {
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
