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
  // normally we have the command-line args passed to electron
  // via `npm start` available through the remote.process module.
  // The first two are built into npm start (I forget what they are).
  // The third one we use and so must mock.
  process: {
    argv: ['', '', '--dev'],
  },
};

// mocks for when electron API called from main process
export const app = mockApp
