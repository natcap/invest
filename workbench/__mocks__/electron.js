/* eslint-disable max-classes-per-file */

/* Mocks of all parts of the electron API that we use.

None of the electron APIs are available in a node env, even though
electron is a node module. (Note that node_modules/electron/index.js only
exports the filepath to the electron exe.)

APIs are only available in an electron browser environment, but our tests
run in node where all electron's exports are undefined. So we must mock,
even for APIs that would otherwise seem okay to call during a test.
*/

import os from 'os';
import events from 'events';

class MockIPC extends events.EventEmitter {
  constructor() {
    super();
    this.handle = jest.fn();
    this.invoke = jest.fn().mockImplementation(() => Promise.resolve());
    this.send = (channel, ...args) => {
      const event = {
        reply: (channel, response) => {
          this.emit(channel, {}, response);
        }
      };
      this.emit(channel, event, ...args);
    };
    this.sendSync = jest.fn();
  }
}
const electronMockIPC = new MockIPC();
export const ipcMain = electronMockIPC;
export const ipcRenderer = electronMockIPC;

class MockApp extends events.EventEmitter {
  constructor() {
    super();
    this.getPath = jest.fn().mockImplementation(
      () => os.tmpdir()
    );
    this.setPath = jest.fn();
    this.getName = jest.fn();
    this.getVersion = jest.fn();
  }
}
export const app = new MockApp();

export class BrowserWindow {
  constructor() {
    this.loadURL = jest.fn();
    this.once = jest.fn();
    this.on = jest.fn();
    this.webContents = {
      on: jest.fn(),
      session: {
        on: jest.fn(),
      },
      send: (channel, ...args) => {
        electronMockIPC.emit(channel, {}, ...args);
      },
      downloadURL: jest.fn(),
    };
  }
}

export const dialog = {
  showOpenDialog: jest.fn(),
  showSaveDialog: jest.fn(),
};

export const Menu = {
  buildFromTemplate: jest.fn(),
  setApplicationMenu: jest.fn(),
};

export const nativeTheme = {};

export const screen = {
  getPrimaryDisplay: jest.fn().mockReturnValue({
    workAreaSize: { width: 800, height: 800 }
  })
};

export const shell = {
  showItemInFolder: jest.fn(),
  openExternal: jest.fn(),
};
