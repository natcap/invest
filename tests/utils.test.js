import fs from 'fs';
import path from 'path';
import { findMostRecentLogfile, loadRecentSessions,
         updateRecentSessions, boolStringToBoolean,
         argsValuesFromSpec } from '../src/utils';

function setupDir() {
  return fs.mkdtempSync('tests/data/_');
}

function cleanupDir(dir) {
  fs.readdirSync(dir).forEach(file => {
  	fs.unlinkSync(path.join(dir, file))
  })
  fs.rmdirSync(dir)
}

test('Test findMostRecentLogfile ignores files that are not invest logs', async () => {
  const dir = setupDir()
  const a = path.join(dir, 'InVEST-some-model-log-9999-99-99--99_99_99.txt')
  const b = path.join(dir, 'foo.txt')
  fs.closeSync(fs.openSync(a, 'w'))

  await new Promise(resolve => setTimeout(resolve, 100));
  fs.closeSync(fs.openSync(b, 'w'))
  const recent = await findMostRecentLogfile(dir)

  // File b was created more recently, but it's not an invest log
  expect(recent).toEqual(a)
  cleanupDir(dir)
})

test('Test findMostRecentLogfile regex matcher', async () => {
  const dir = setupDir()
  const a = path.join(dir, 'InVEST-model-log-9999-99-99--99_99_99.txt')
  fs.closeSync(fs.openSync(a, 'w'))
  let recent = await findMostRecentLogfile(dir)
  expect(recent).toEqual(a)

  await new Promise(resolve => setTimeout(resolve, 100));
  const b = path.join(dir, 'InVEST-some-model-log-9999-99-99--99_99_99.txt')
  fs.closeSync(fs.openSync(b, 'w'))
  recent = await findMostRecentLogfile(dir)
  expect(recent).toEqual(b)

  await new Promise(resolve => setTimeout(resolve, 100));
  const c = path.join(dir, 'InVEST-some-really-long-invest-model-name-log-9999-99-99--99_99_99.txt')
  fs.closeSync(fs.openSync(c, 'w'))
  recent = await findMostRecentLogfile(dir)
  expect(recent).toEqual(c)
  cleanupDir(dir)
})

test('Test findMostRecentLogfile returns undefined when no logiles exist', async () => {
  const dir = setupDir()
  let recent = await findMostRecentLogfile(dir)
  expect(recent).toBe(undefined)
  cleanupDir(dir)
})

test('Test loadRecentSessions returns correct order', async() => {
  const jobData = {
    "carbon_setup": {
      "systemTime": 2583259376573.759, // more recent
    },
    "duck": {
      "systemTime": 1243259376573.759,
    }
  }
  const dir = setupDir()
  const jobdbPath = path.join(dir, 'jobdb.json');
  fs.writeFileSync(jobdbPath, JSON.stringify(jobData))
  const jobs = await loadRecentSessions(jobdbPath);
  expect(jobs[0][0]).toEqual('carbon_setup')
  expect(jobs[1][0]).toEqual('duck')
  cleanupDir(dir)
})

test('Test loadRecentSessions: database is missing', async() => {
  const jobdbPath = 'foo.json';
  const jobs = await loadRecentSessions(jobdbPath);
  expect(jobs).toEqual([])
})

test('Test updateRecentSessions returns correct order', async() => {
  const jobData = {
    "carbon_setup": {
      "systemTime": 2583259376573.759
    },
    "duck": {
      "systemTime": 1243259376573.759
    }
  }

  const newJob = {
    "goose": {
      "systemTime": 3243259376573.759 // most recent
    }
  }
  const dir = setupDir()
  const jobdbPath = path.join(dir, 'jobdb.json');
  fs.writeFileSync(jobdbPath, JSON.stringify(jobData))
  const jobs = await updateRecentSessions(newJob, jobdbPath);
  expect(jobs[0][0]).toEqual('goose')
  expect(jobs[1][0]).toEqual('carbon_setup')
  expect(jobs[2][0]).toEqual('duck')
  cleanupDir(dir)
})

test('Test updateRecentSessions: database is missing', async() => {
  const dir = setupDir()
  const jobdbPath = path.join(dir, 'foo.json');
  const newJob = {
    "goose": {
      "systemTime": 3243259376573.759 // most recent
    }
  }
  expect(fs.existsSync(jobdbPath)).toBeFalsy()
  const jobs = await updateRecentSessions(newJob, jobdbPath);
  expect(jobs[0][0]).toEqual('goose')
  expect(fs.existsSync(jobdbPath)).toBeTruthy()
  cleanupDir(dir)
})

test('Test boolStringToBoolean for expected values', () => {
  expect(boolStringToBoolean('true')).toBe(true)
  expect(boolStringToBoolean('True')).toBe(true)
  expect(boolStringToBoolean('false')).toBe(false)
  expect(boolStringToBoolean('False')).toBe(false)
  expect(boolStringToBoolean('foo')).toBe(false)
  expect(boolStringToBoolean(undefined)).toBe(undefined)
  expect(boolStringToBoolean(1)).toBe(undefined)
})