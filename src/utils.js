import path from 'path';
import fs from 'fs';
import glob from 'glob';

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
      console.log("An error occured while writing JSON Object to File.");
      return console.log(err);
    }
    console.log("updated" + this.state.sessionID);
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
    const regex = /InVEST-[a-zA-Z-]+-log-[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}_[0-9]{2}_[0-9]{2}.txt/g
    const files = glob.sync(path.join(directory, '*.txt'));
    let logfiles = [];
    files.forEach(file => {
      const match = file.match(regex)
      if (match) {
        logfiles.push(path.join(directory, match[0]))
      }
    })
    if (logfiles.length === 1) {
      resolve(logfiles[0])
    }
    // reverse sort (b - a) based on last-modified time
    const sortedFiles = logfiles.sort(function(a, b) {
      return fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs
    });
    resolve(sortedFiles[0]);
  });
}