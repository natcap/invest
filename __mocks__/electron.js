import path from 'path';

const mockApp = {
  getPath: jest.fn().mockImplementation(() => {
    return path.resolve('tests/data');
  })
};

// mocks for when electron API accessed via remote from renderer process
export const remote = {
  dialog: {
    showOpenDialog: jest.fn(),
    showSaveDialog: jest.fn(),
  },
  app: mockApp,
  // there are checks for '--dev' in process.argv[2],
  // but we don't need '--dev' when running tests.
  process: {
    argv: ['', '', '--foo']
  },
};

// mocks for when electron API called from main process
export const app = mockApp
