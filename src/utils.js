import path from 'path';
import fs from 'fs';
import glob from 'glob';

import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const LOGFILE_REGEX = /InVEST-natcap\.invest\.[a-zA-Z._]+-log-[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}_[0-9]{2}_[0-9]{2}.txt/g;

/**
 * Load job metadata from a persistent file and return the jobs
 * sorted by creation time.
 *
 * @param  {string} jobDatabasePath - path to a json file with jobs metadata.
 * @returns {Promise} - Resolves sorted array of jobs with metadata.
 */
export function loadRecentJobs(jobDatabasePath) {
  return new Promise((resolve) => {
    const db = JSON.parse(fs.readFileSync(jobDatabasePath, 'utf8'));
    const sortedJobs = Object.entries(db).sort(
      (a, b) => b[1].systemTime - a[1].systemTime
    );
    resolve(sortedJobs);
  });
}

/** Append/overwrite an entry to the persistent jobs file and reload its data.
 * If a job already exists with the same name is jobdata, it is overwritten.
 *
 * @param  {object} jobdata - object with job's metadata
 * @param  {string} jobDatabase - path to a json file with jobs metadata.
 * @returns {Array} - sorted array of jobs with metadata.
 */
export async function updateRecentJobs(jobdata, jobDatabase) {
  let jsonContent;
  if (fs.existsSync(jobDatabase)) {
    const db = JSON.parse(fs.readFileSync(jobDatabase, 'utf8'));
    Object.keys(jobdata).forEach((job) => {
      db[job] = jobdata[job];
    });
    jsonContent = JSON.stringify(db, null, 2);
  } else {
    jsonContent = JSON.stringify(jobdata, null, 2);
  }
  fs.writeFileSync(jobDatabase, jsonContent, 'utf8');
  const updated = await loadRecentJobs(jobDatabase);
  return updated;
}

/**
 * Given an invest workspace, find the most recently modified invest log.
 *
 * This function is used in order to associate a logfile with an active
 * InVEST run, so the log can be tailed to a UI component.
 *
 * @param {string} directory - the path to an invest workspace directory
 * @returns {Promise} - resolves string path to an invest logfile
 */
export function findMostRecentLogfile(directory) {
  return new Promise((resolve) => {
    const files = glob.sync(path.join(directory, '*.txt'));
    const logfiles = [];
    files.forEach((file) => {
      const match = file.match(LOGFILE_REGEX);
      if (match) {
        logfiles.push(path.join(directory, match[0]));
      }
    });
    if (logfiles.length === 1) {
      // This is the most likely path
      resolve(logfiles[0]);
      return;
    }
    if (logfiles.length > 1) {
      // reverse sort (b - a) based on last-modified time
      const sortedFiles = logfiles.sort(
        (a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs
      );
      resolve(sortedFiles[0]);
    } else {
      logger.error(`No invest logfile found in ${directory}`);
      resolve(undefined);
    }
  });
}

/** Convert a string representing a bool to an actual boolean.
 *
 * HTML inputs in this app must send string values, but for invest args of
 * type boolean, we want to convert that to a real boolean before passing
 * to invest's validate or execute.
 *
 * @param {string} val - such as "true", "True", "false", "False"
 * @returns {boolean} unless the input was not a string, then undefined
 */
export function boolStringToBoolean(val) {
  let valBoolean;
  try {
    const valString = val.toLowerCase();
    valBoolean = valString === 'true';
  } catch (e) {
    if (e instanceof TypeError) {
      valBoolean = undefined;
    } else {
      throw e;
    }
  }
  return valBoolean;
}

/** Create a JSON string with invest argument keys and values.
 *
 * @param {object} args - object keyed by invest argument keys and
 *   with each item including a `value` property, among others.
 * @returns {object} - invest argument key: value pairs as expected
 * by invest model `execute` and `validate` functions
 */
export function argsDictFromObject(args) {
  const argsDict = {};
  Object.keys(args).forEach((argname) => {
    argsDict[argname] = args[argname].value;
  });
  return argsDict;
}

/** Convenience function to recursively remove a directory.
 *
 * @param {string} dir - path to a directory
 */
export function cleanupDir(dir) {
  fs.readdirSync(dir).forEach((filename) => {
    const filepath = path.join(dir, filename);
    if (fs.lstatSync(filepath).isFile()) {
      fs.unlinkSync(filepath);
    } else {
      cleanupDir(filepath);
    }
  });
  fs.rmdirSync(dir);
}
