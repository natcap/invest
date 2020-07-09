const path = require('path');
const winston = require('winston');
const { app, remote } = require('electron');


let userDataPath;
let isDevMode;
if (remote) {
  // When this module is imported from render process, access via remote
  userDataPath = remote.app.getPath('userData')
  isDevMode = remote.process.argv[2] == '--dev'
} else {
  // But we also import it from the main process
  userDataPath = app.getPath('userData')
  isDevMode = process.argv[2] == '--dev'
}

/**
 * Creates and returns a logger with Console & File transports.
 *
 * @param {string} label - for identifying the origin of the message
 * @returns {logger} - with File and Console transports. 
 */
function getLogger(label, filename='log.txt') {
  // Right now the logging in this app has multiple loggers streaming to the
  // same file. Maybe that's okay? The goal is having only one file on disk.
  // But it feels wrong to create several different loggers just so that
  // each message can have a 'label' that refers to the calling file.
  // The only alternative I know of is to pass extra metadata anytime we log,
  // but that also feels wrong:
  // `logger.debug('foo', {label: filename})`
  if (!winston.loggers.has(label)) {
    const myFormat = winston.format.printf(
      ({ level, message, label, timestamp }) => {
      return `${timestamp} [${label}] ${level}: ${message}`
    })

    const transport = new winston.transports.File({
      level: 'debug',
      filename: path.join(userDataPath, filename),
      handleExceptions: true
    })

    const transportArray = [transport]
    if (isDevMode) {
      transportArray.push(new winston.transports.Console({
        level: 'debug',
        handleExceptions: true
      }))
    }
    winston.loggers.add(label, {
      format: winston.format.combine(
        winston.format.label({ label: label}),
        winston.format.timestamp(),
        myFormat),
      transports: transportArray
    })
  }
  return winston.loggers.get(label)
}

module.exports.getLogger = getLogger
