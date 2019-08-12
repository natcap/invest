import tmp from 'tmp';
import path from 'path';

import validate from '../src/validate';

let TMP_DIR;
let TMP_FILE;
let ruleRequired;

beforeAll(() => {
  TMP_DIR = tmp.dirSync();
  TMP_FILE = tmp.fileSync();
  ruleRequired = {'required': true, 'rule': ''}
});

afterAll(() => {
  TMP_DIR.removeCallback();
  TMP_FILE.removeCallback();
});

test('valid integer', () => {
  const rule = Object.assign({}, ruleRequired);
  rule.rule = 'integer'
  expect(validate('1', rule)).toBe(true);
  expect(validate('a', rule)).toBe(false);
});

test('valid filepath', () => {
  const rule = Object.assign({}, ruleRequired);
  rule.rule = 'filepath'
  expect(validate(TMP_FILE.name, rule)).toBe(true);
  expect(validate('a', rule)).toBe(false);
});

test('valid workspace', () => {
  const rule = Object.assign({}, ruleRequired);
  rule.rule = 'workspace'
  expect(validate(path.join(TMP_DIR.name, 'foo'), rule)).toBe(true);
  expect(validate(path.join(TMP_DIR.name, 'foo', 'bar'), rule)).toBe(false);
});

test('valid optional arg', () => {
  const rule = {'required': false, 'rule': 'integer'};
  expect(validate('', rule)).toBe(true);
  expect(validate('a', rule)).toBe(false);
});