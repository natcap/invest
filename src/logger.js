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

const format = winston.format.combine(
  winston.format.splat(),
  winston.format.simple())

const transport = new winston.transports.File({
  format: format,
  level: 'debug',
  filename: path.join(userDataPath, 'log.txt')})

const transportArray = [transport]
if (isDevMode) {
  transportArray.push(new winston.transports.Console({
    format: format,
    level: 'debug'
  }))
}

function getLogger(label) {
  if (!winston.loggers.has(label)) {
    winston.loggers.add(label, {
      format: winston.format.label({ label: label}),
      transports: transportArray
    })
  }
  return winston.loggers.get(label)
}

module.exports.getLogger = getLogger