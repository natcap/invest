const log = {
  debug: jest.fn(),
  verbose: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  silly: jest.fn(),
};

module.exports.logger = log;
module.exports.setupRendererLogger = jest.fn();
