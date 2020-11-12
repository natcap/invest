import fs from 'fs';
import path from 'path';
import {
  findMostRecentLogfile, boolStringToBoolean, cleanupDir
} from '../src/utils';

function setupDir() {
  return fs.mkdtempSync('tests/data/_');
}

test('findMostRecentLogfile ignores files that are not invest logs', async () => {
  const dir = setupDir();
  const a = path.join(
    dir, 'InVEST-natcap.invest.model-log-9999-99-99--99_99_99.txt'
  );
  // write one file, pause, write a more recent file.
  const b = path.join(dir, 'foo.txt');
  fs.closeSync(fs.openSync(a, 'w'));
  await new Promise((resolve) => setTimeout(resolve, 100));
  fs.closeSync(fs.openSync(b, 'w'));
  const recent = await findMostRecentLogfile(dir);

  // File b was created more recently, but it's not an invest log
  expect(recent).toEqual(a);
  cleanupDir(dir);
});

test('findMostRecentLogfile regex matcher', async () => {
  const dir = setupDir();
  const a = path.join(
    dir, 'InVEST-natcap.invest.model-log-9999-99-99--99_99_99.txt'
  );
  fs.closeSync(fs.openSync(a, 'w'));
  let recent = await findMostRecentLogfile(dir);
  expect(recent).toEqual(a);

  await new Promise((resolve) => setTimeout(resolve, 100));
  const b = path.join(
    dir, 'InVEST-natcap.invest.some.model-log-9999-99-99--99_99_99.txt'
  );
  fs.closeSync(fs.openSync(b, 'w'));
  recent = await findMostRecentLogfile(dir);
  expect(recent).toEqual(b);

  await new Promise((resolve) => setTimeout(resolve, 100));
  const c = path.join(
    dir, 'InVEST-natcap.invest.some.really_long_model.name-log-9999-99-99--99_99_99.txt'
  );
  fs.closeSync(fs.openSync(c, 'w'));
  recent = await findMostRecentLogfile(dir);
  expect(recent).toEqual(c);
  cleanupDir(dir);
});

test('findMostRecentLogfile returns undefined when no logiles exist', async () => {
  const dir = setupDir();
  expect(await findMostRecentLogfile(dir))
    .toBeUndefined();
  cleanupDir(dir);
});

test('boolStringToBoolean converts various strings to bools', () => {
  expect(boolStringToBoolean('true')).toBe(true);
  expect(boolStringToBoolean('True')).toBe(true);
  expect(boolStringToBoolean('false')).toBe(false);
  expect(boolStringToBoolean('False')).toBe(false);
  expect(boolStringToBoolean('foo')).toBe(false);
  expect(boolStringToBoolean(undefined)).toBeUndefined();
  expect(boolStringToBoolean(1)).toBeUndefined();
});
