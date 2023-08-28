import {
  defaults,
  settingsStore,
  initStore,
} from '../../src/main/settingsStore';

afterEach(() => {
  settingsStore.reset();
});

test('an empty store initializes to defaults', () => {
  const store = initStore();
  expect(store.store).toEqual(defaults);
});

test('invalid items are reset, valid items are unchanged', () => {
  const data = { ...defaults };
  data.nWorkers = 5; // valid, but not default
  data.taskgraphLoggingLevel = 'ERROR'; // valid, but not default
  data.loggingLevel = 'FOO'; // wrong value
  data.language = 1; // wrong type

  const store = initStore(data);

  // invalid: should be reset to defaults
  expect(store.get('loggingLevel')).toBe(defaults.loggingLevel);
  expect(store.get('language')).toBe(defaults.language);

  // valid: should be not be reset to defaults
  expect(store.get('taskgraphLoggingLevel')).toBe(data.taskgraphLoggingLevel);
  expect(store.get('nWorkers')).toBe(data.nWorkers);
});

test('properties not present in schema are untouched during validation', () => {
  const data = { ...defaults };
  data.foo = 'bar';

  const store = initStore(data);

  expect(store.get('foo')).toEqual(data.foo);
});

test('missing properties are added with default value', () => {
  const data = { ...defaults };
  delete data.loggingLevel;
  delete data.language;

  const store = initStore(data);

  expect(store.get('loggingLevel')).toEqual(defaults.loggingLevel);
  expect(store.get('language')).toEqual(defaults.language);
});
