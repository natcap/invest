import path from 'path';
import fs from 'fs';
import glob from 'glob';

const JOBS_DATA = 'jobdb.json'

export function loadRecentSessions() {
  /*
  Returns: Array:
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
    const db = JSON.parse(fs.readFileSync(JOBS_DATA, 'utf8'));
    const sortedJobs = Object.entries(db).sort((a, b) => b[1]['systemTime'] - a[1]['systemTime'])
    resolve(sortedJobs)
  })
}

export async function updateRecentSessions(jobdata) {
  let db = JSON.parse(fs.readFileSync(JOBS_DATA, 'utf8'));
  Object.keys(jobdata).forEach(job => {
    db[job] = jobdata[job]
  })
  const jsonContent = JSON.stringify(db, null, 2);
  fs.writeFileSync(JOBS_DATA, jsonContent, 'utf8', function (err) {
    if (err) {
      console.log("An error occured while writing JSON Object to File.");
      return console.log(err);
    }
    console.log("updated" + this.state.sessionID);
  });
  const updated = await loadRecentSessions();
  return updated
}

export function findMostRecentLogfile(directory) {
  return new Promise(function(resolve, reject) {
    const files = glob.sync(path.join(directory, '*.txt'));
    console.log(files);

    // reverse sort (b - a) based on last-modified time
    const sortedFiles = files.sort(function(a, b) {
      return fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs
    });
    resolve(sortedFiles[0]);
  });
}