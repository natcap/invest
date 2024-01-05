This project is a user-interface layer for InVEST (Integrated Valuation of
Ecosystem Services and Tradeoffs).
InVEST can be found at https://github.com/natcap/invest.

The purpose of this project is to provide a single entry-point for all
InVEST models, and to be extensible to future models or common auxiliary
workflows of an InVEST user.

## To develop and launch this Application
1. from invest/:
  - activate a python environment and install `natcap.invest`
2. from invest/workbench/:
	- `yarn install`
	- `yarn start`

## To package this app for distribution
1. from invest/:
  - `make binaries`
  - `make userguide`
2. from invest/workbench/:
  - `yarn run install`
  - `yarn run build`
  - `yarn run dist`  - Configure the packaging in `electron-builder-config.js`.

## Dependency management in package.json
`dependencies` should only include node modules used by the main process.  

Renderer & preload process dependencies (`react`, `bootstrap`, etc) belong in `devDependencies`. 
They are required in production, but we want electron-builder to ignore them
because they are already packaged via the vite bundle.
electron-builder will package everything under `dependencies` and nothing under `devDependencies`.  

## Testing with Jest
#### Configuration
Jest configuration is in `package.json`.
Tests run in a jsdom environment by default, where a browser API is available
but a node API is not. The environment can be toggled to node on a per-file 
basis using the docblock seen at the top of `main.test.js`.

Config also includes global mock resets. These trigger before each individual test, 
so there is no need to cleanup mocks in `afterEach` blocks.
`beforeEach` blocks within a test file will fire _after_ these global resets,
so mock setup can be done in a `beforeEach`, or in a `test` block itself.

the global config:
```
"restoreMocks": false,
```
Restore unmocked implementations. Ideally, this would be `true`. And previously it was.
But as of jest28 or 29 it behaves differently. Now it restores manual mocks in `__mocks__`,
such as the electron API. That is unhelpful, as we always want that API mocked and there
is no way to revert to the original manual mock between tests. Basically, we have this problem:
https://github.com/jestjs/jest/issues/10419. Though for us it seems triggered by `restoreMocks`
instead of `reset`. Setting to `false` allows `__mocks__` to work as expected, but now it
no longer restores things like,
```javascript
const spy = jest.spyOn(ipcRenderer, 'send')
  .mockImplementation(() => Promise.resolve());
...
spy.mockReset(); // now required, and resets to orignal mock defined in __mocks__
```

```
"clearMocks": true,
```
Jest docs suggest `restoreMocks` should do all the work of `clearMocks`, 
but I found this exception and added the `clear` to the global config:
Using `jest.spyOn(module, 'foo-method')` to keep track of number of times
`foo-method` is called. `clearMocks` is needed to reset the calls data.
```
"resetModules": true
```
Needed to restore to an unmocked module when we mocked it like this:
`jest.mock('ui_config.js', () => mockUISpec(mockSpec));`
Possibly because this is outside control of `restoreMocks`, 
which only works on `jest.spyOn` mocks?

## Client logfile locations:
* Windows: "C:\Users\dmf\AppData\Roaming\invest-workbench\logs\"
* Mac: "\~/Library/Logs/invest-workbench/"
* Linux: "\~/.config/invest-workbench/logs/"

## Internationalization

See also the internationalization readme in the invest repo.

We are using the javascript internationalization package `i18next` and its react extension `react-i18next`. `i18next` takes in translation resources as a javascript object. It's convenient to store translations in JSON format. Vite automatically serves JSON files as Javascript modules, so we can directly import translations from JSON.

The translations for each language live in `workbench/src/renderer/i18n/xx.json`. The JSON object in each file maps English messages to translations.

Nothing needs to be done during routine development. As we make changes to the workbench text, it will inevitably get out of sync with the translations, and that's okay. Strings that have no translation will fall back to English. When it's time to update our translations, this is the process:


### Getting a new batch of translations
These instructions assume you have defined the two-letter locale code in an environment variable `$LL`.

1. Extract messages from the source code:
   ```
   i18next "src/main/**/*.{js,jsx}" --output main-messages.json
   i18next "src/renderer/**/*.{js,jsx}" --output renderer-messages.json
   ```
   This command is provided by the `i18next-parser` package and configured by `workbench/i18next-parser.config.mjs`. The output JSON files should contain a JSON object mapping each translated message from the source code to an empty string.

2. Merge into the existing translation files:
   ```
   jq -s add main-messages.json src/main/i18n/$LL.json > tmp.json
   cat tmp.json > src/main/i18n/$LL.json
   jq -s add renderer-messages.json src/renderer/i18n/$LL.json > tmp.json
   cat tmp.json > src/renderer/i18n/$LL.json
   ```
   This will add new keys into the JSON message catalogs and leave those that already have translations:
   ```
   {
      "text that's already been translated": "translation",
      "new text that doesn't have a translation yet": ""
   }
   ```

4. Commit the changes:
   ```
   git add src/main/i18n/$LL.json src/renderer/i18n/$LL.json
   git commit -m "add new messages into $LL translation files"
   ```
3. (if the translator uses PO format) Convert JSON to PO

4. Send `src/main/i18n/$LL.[json,po]` and `src/renderer/i18n/$LL.[json,po]` to the translator and wait to receive a copy with translations added.

5. (if the translator uses PO format) Convert PO to JSON
   If the translator works with PO files, we can convert them to JSON using this tool: https://github.com/i18next/i18next-gettext-converter

6. Replace `src/main/i18n/$LL.[json,po]` and `src/renderer/i18n/$LL.json` with the updated versions received from the translator

7. Commit the changes:
   ```
   git add src/main/i18n/$LL.json src/renderer/i18n/$LL.json
   git commit -m "add new translations for $LL"
   ```

