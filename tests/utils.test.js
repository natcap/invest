import fs from 'fs';
import path from 'path';
import { findMostRecentLogfile } from '../src/utils';

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
  expect(recent).toBe(a)
  cleanupDir(dir)
})

test('Test findMostRecentLogfile regex matcher', async () => {
  const dir = setupDir()
  const a = path.join(dir, 'InVEST-model-log-9999-99-99--99_99_99.txt')
  fs.closeSync(fs.openSync(a, 'w'))
  let recent = await findMostRecentLogfile(dir)
  expect(recent).toBe(a)

  await new Promise(resolve => setTimeout(resolve, 100));
  const b = path.join(dir, 'InVEST-some-model-log-9999-99-99--99_99_99.txt')
  fs.closeSync(fs.openSync(b, 'w'))
  recent = await findMostRecentLogfile(dir)
  expect(recent).toBe(b)

  await new Promise(resolve => setTimeout(resolve, 100));
  const c = path.join(dir, 'InVEST-some-really-long-invest-model-name-log-9999-99-99--99_99_99.txt')
  fs.closeSync(fs.openSync(c, 'w'))
  recent = await findMostRecentLogfile(dir)
  expect(recent).toBe(c)
  cleanupDir(dir)
})

test('Test findMostRecentLogfile returns undefined when no logiles exist', async () => {
  const dir = setupDir()
  let recent = await findMostRecentLogfile(dir)
  expect(recent).toBe(undefined)
  cleanupDir(dir)
})