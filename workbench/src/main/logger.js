// Dont' use ES6 features that require babel transpiling here
// because this module is loaded by the preload script, which seems
// to be outside the chain of our "-r @babel/register" dev mode strategy
// const log = require('electron-log');
import log from 'electron-log';

/**
 * Creates and returns a logger with Console & File transports.
 *
 * @param {string} label - for identifying the origin of the message
 * @returns {logger} - with File and Console transports.
 */
export function getLogger(label) {
  log.variables.label = label;
  log.transports.console.format = '[{h}:{i}:{s}.{ms}] [{label}] {text}';
  log.transports.file.format = '[{h}:{i}:{s}.{ms}] [{label}] {text}';

  return log;
}

// module.exports.getLogger = getLogger;
