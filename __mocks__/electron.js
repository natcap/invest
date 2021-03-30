import path from 'path';

const mockApp = {
  getPath: jest.fn().mockImplementation(
    () => path.resolve('tests/data')
  )
};

export const ipcRenderer = {
  on: jest.fn(),
  send: jest.fn(),
  invoke: jest.fn().mockImplementation(() => Promise.resolve()),
};

export const dialog = {
  showOpenDialog: jest.fn(),
  showSaveDialog: jest.fn(),
}


// mocks for when electron API called from main process
export const app = mockApp;
