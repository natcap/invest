/* Mocks of all parts of the electron API that we use.

None of the electron APIs are available in a node env, even though
electron is a node module. (Note that node_modules/electron/index.js only
exports the filepath to the electron exe.)

APIs are only available in an electron browser environment, but our tests
run in node where all electron's exports are undefined. So we must mock,
even for APIs that would otherwise seem okay to call during a test.
*/

import path from 'path';
import events from 'events';

// TODO - is ReturnThis necessary? What are the implications?
// TODO: given this functional IPC, some tests can probably
// mock return values/implementations at a lower-level
// e.g. electron shell API, etc.
// class mockIPC extends events.EventEmitter {
//   invoke(channel, ...args) {
//     return new Promise((resolve, reject) => {
//       this.once('send-to-main')

//       this.emit('send-to-main', channel, ...args)
//     });
//   }
// }
const mockIPC = new events.EventEmitter();
mockIPC.handle = jest.fn().mockReturnThis();
mockIPC.handleOnce = jest.fn().mockReturnThis();
mockIPC.send = (channel, ...args) => {
  const event = {
    reply: (channel, response) => {
      mockIPC.emit(channel, {}, response);
    }
  };
  mockIPC.emit(channel, event, ...args);
};
mockIPC.invoke = jest.fn().mockImplementation(() => Promise.resolve());
export const ipcMain = mockIPC;
export const ipcRenderer = mockIPC;

const mockApp = new events.EventEmitter();
mockApp.getPath = jest.fn().mockImplementation(
  () => path.join('tests', 'data')
);
export const app = mockApp;

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
        mockIPC.emit(channel, {}, ...args);
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
