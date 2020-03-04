import path from 'path';
import fs from 'fs';
import glob from 'glob';

export function loadRecentSessions() {
  return new Promise(function(resolve, reject) {
    const db = JSON.parse(fs.readFileSync('jobdb.json', 'utf8'));
    const sortedJobs = Object.entries(db).sort((a, b) => b[1]['mtime'] - a[1]['mtime'])
    resolve(sortedJobs)
  })
}

export function findRecentSessions(cache_dir) {
  // Populate recentSessions from list of files in cache dir
  // sorted by modified time.

  // TODO: check that files are actually state config files
  // before putting them on the array
  return new Promise(function(resolve, reject) {
    const files = fs.readdirSync(cache_dir);

    // reverse sort (b - a) based on last-modified time
    const sortedFiles = files.sort(function(a, b) {
      return fs.statSync(path.join(cache_dir, b)).mtimeMs -
           fs.statSync(path.join(cache_dir, a)).mtimeMs
    });
    // trim off extension, since that is how sessions
    // were named orginally
    resolve(sortedFiles
      .map(f => path.parse(f).name)
      .slice(0, 15) // max 15 items returned
    );
  });
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