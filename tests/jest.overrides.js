const crypto = require('crypto');

// Cause tests to fail on console.error messages
// Taken from https://stackoverflow.com/questions/28615293/is-there-a-jest-config-that-will-fail-tests-on-console-warn/50584643#50584643
let error = console.error;

console.error = function (message) {
  error.apply(console, arguments); // keep default behaviour
  throw (message instanceof Error ? message : new Error(message));
};

global.window.Workbench = {
  getLogger: jest.fn().mockReturnValue({
    debug: jest.fn(),
    verbose: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    silly: jest.fn(),
  })
};

global.window.crypto = {
  getRandomValues: () => {
    return [crypto.randomBytes(4).toString('hex')];
  }
};

jest.mock('../src/logger');
