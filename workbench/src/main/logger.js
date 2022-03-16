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
  log.transports.console.level = process.env.ELECTRON_LOG_LEVEL || 'debug';
  return log;
}
