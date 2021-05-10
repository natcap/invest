import path from 'path';

export const app = {
  getPath: jest.fn().mockImplementation(
    () => path.resolve('tests/data')
  )
};

export class BrowserWindow {
  constructor() {
    this.loadURL = jest.fn();
  }
};

export const dialog = {
  showOpenDialog: jest.fn(),
  showSaveDialog: jest.fn(),
};

export const ipcMain = {
  on: jest.fn().mockReturnThis(),
  handle: jest.fn().mockReturnThis(),
  handleOnce: jest.fn().mockReturnThis(),
};

export const ipcRenderer = {
  on: jest.fn(),
  send: jest.fn(),
  invoke: jest.fn().mockImplementation(() => Promise.resolve()),
};

export const nativeTheme = {};

export const screen = {
  getPrimaryDisplay: jest.fn().mockReturnValue({
    workAreaSize: { width: 800, height: 800}
  })
};

