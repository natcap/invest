import {
  boolStringToBoolean
} from '../../src/renderer/utils';

test('boolStringToBoolean converts various strings to bools', () => {
  expect(boolStringToBoolean('true')).toBe(true);
  expect(boolStringToBoolean('True')).toBe(true);
  expect(boolStringToBoolean('false')).toBe(false);
  expect(boolStringToBoolean('False')).toBe(false);
  expect(boolStringToBoolean('foo')).toBe(false);
  expect(boolStringToBoolean(undefined)).toBeUndefined();
  expect(boolStringToBoolean(1)).toBeUndefined();
});
