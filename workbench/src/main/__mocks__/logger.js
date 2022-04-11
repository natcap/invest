const log = {
  debug: jest.fn(),
  verbose: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  silly: jest.fn(),
};

const getLogger = jest.fn().mockReturnValue(log);
module.exports.getLogger = getLogger;
