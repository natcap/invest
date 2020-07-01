import path from 'path';
import fs from 'fs';
import glob from 'glob';
import winston from 'winston';
const logger = winston.loggers.get('logger')

const LOGFILE_REGEX = /InVEST-natcap\.invest\.[a-zA-Z._]+-log-[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}_[0-9]{2}_[0-9]{2}.txt/g
logger.debug(LOGFILE_REGEX)
export function loadRecentSessions(jobDatabase) {
  /** Load job data from a persistent file and return the jobs
  * sorted by creation time. Right now the persistent file is a JSON.
  * 
  * @params {string} jobDatabase - path to a json file with jobs metadata.
  * @return {Promise<array>} - 
    [ [ "job1",
      {
        "model": "carbon",
        "workspace": { "directory": null, "suffix": null },
        "statefile": "carbon_setup.json",
        "status": null,
        "humanTime": "3/5/2020, 10:43:14 AM",
        "systemTime": 1583259376573.759,
        "description": null } ],
      [ "job2",
        { ... } ] ]
  */
  return new Promise(function(resolve, reject) {
    if (!fs.existsSync(jobDatabase)) {
      resolve([])
    }
    const db = JSON.parse(fs.readFileSync(jobDatabase, 'utf8'));
    const sortedJobs = Object.entries(db).sort((a, b) => b[1]['systemTime'] - a[1]['systemTime'])
    resolve(sortedJobs)
  })
}

export async function updateRecentSessions(jobdata, jobDatabase) {
  /** Append/overwrite an entry to the persistent jobs file and reload its data.
  * If a job already exists with the same name is jobdata, it is overwritten.
  * 
  * @params {string} jobdata - object with job's metadata
  * @params {string} jobDatabase - path to a json file with jobs metadata.
  * @return {array} - see documentation for loadRecentSessions
  */
  let jsonContent;
  if (fs.existsSync(jobDatabase)) {
    let db = JSON.parse(fs.readFileSync(jobDatabase, 'utf8'));
    Object.keys(jobdata).forEach(job => {
      db[job] = jobdata[job]
    })
    jsonContent = JSON.stringify(db, null, 2);
  } else {
    jsonContent = JSON.stringify(jobdata, null, 2)
  }
  fs.writeFileSync(jobDatabase, jsonContent, 'utf8', function (err) {
    if (err) {
      logger.debug("An error occured while writing JSON Object to File.");
      return logger.debug(err);
    }
    logger.debug("updated" + this.state.sessionID);
  });
  const updated = await loadRecentSessions(jobDatabase);
  return updated
}

export function findMostRecentLogfile(directory) {
  /**
  * Given an invest workspace, find the most recently modified invest log.
  *
  * This function is used in order to associate a logfile with an active
  * InVEST run, so the log can be tailed to a UI component.
  * 
  * @param {string} directory - the path to an invest workspace directory
  * @return {Promise<string>} - the path to an invest logfile
  */
  return new Promise(function(resolve, reject) {
    const files = glob.sync(path.join(directory, '*.txt'));
    let logfiles = [];
    files.forEach(file => {
      const match = file.match(LOGFILE_REGEX)
      if (match) {
        logfiles.push(path.join(directory, match[0]))
      }
    })
    if (logfiles.length === 1) {
      resolve(logfiles[0])
      return
    } else if (logfiles.length > 1) {
      // reverse sort (b - a) based on last-modified time
      const sortedFiles = logfiles.sort(function(a, b) {
        return fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs
      });
      resolve(sortedFiles[0]);
      return
    }
    logger.debug(`No invest logfile found in ${directory}`)
    resolve(undefined)
  });
}

export function boolStringToBoolean(val) {
  /** Convert a string representing a bool to an actual boolean. 

  * HTML inputs in this app must send string values, but for invest args of
  * type boolean, we want to convert that to a real boolean before passing
  * to invest's validate or execute.
  *
  * @param {string} val - such as "true", "True", "false", "False"
  * @returns {boolean} unless the input was not a string, then undefined
  */
  let valBoolean;
  try {
    const valString = val.toLowerCase()
    valBoolean = (valString === 'true') ? true : false
  }
  catch(TypeError) {
    valBoolean = undefined
  }
  return valBoolean
}

export function argsDictFromObject(args) {
  /** Create a JSON string with invest argument keys and values.
  *
  * This is a convenience function to create a JSON string datastack
  * in the form expected by the natcap.invest API
  *
  * @param {object} args - object keyed by invest argument keys and
  *   with each item including a `value` property.
  * @returns {string} - JSON.stringify'd key: value pairs for an
  *   invest model.
  */
  let args_dict = {};
  for (const argname in args) {
    args_dict[argname] = args[argname]['value']
  }
  return(JSON.stringify(args_dict));
}

/* Convenience function, mainly for cleaning up after tests */
export function cleanupDir(dir) {
  fs.readdirSync(dir).forEach(filename => {
    const filepath = path.join(dir, filename)
    if (fs.lstatSync(filepath).isFile()) {
      fs.unlinkSync(filepath)
    } else {
      cleanupDir(filepath)
    }
  })
  fs.rmdirSync(dir)
}
