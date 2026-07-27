# InVEST Workbench
The InVEST Workbench is a desktop application that provides a graphical user interface (GUI) for [InVEST (Integrated Valuation of
Ecosystem Services and Tradeoffs)](https://naturalcapitalalliance.stanford.edu/software/invest).

The Workbench is designed to provide a single entry point for all
InVEST models, and to be extensible to future models or InVEST-relevant auxiliary workflows.

## To develop and launch this app
1. From ``invest/``:
    - activate a Python environment and install `natcap.invest`

2. From `invest/workbench/`:
	- `yarn install`
	- `yarn start`

## To package this app for distribution
1. From `invest/`:
    - `make binaries`
    - `make userguide`

2. From `invest/workbench/`:
    - `yarn install`
    - `yarn build`
    - `yarn dist` (Packaging is configured in `electron-builder-config.js`.)

## Dependency management in package.json
`dependencies` should only include node modules used by the main process.

Renderer & preload process dependencies (`react`, `bootstrap`, etc) belong in `devDependencies`. They are required in production, but we want `electron-builder` to ignore them because they are already packaged via the `vite` bundle. `electron-builder` will package everything under `dependencies` and nothing under `devDependencies`.

## Testing with Jest
### Configuration
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
* Windows: `C:\Users\dmf\AppData\Roaming\invest-workbench\logs\`
* Mac: `\~/Library/Logs/invest-workbench/`
* Linux: `\~/.config/invest-workbench/logs/`

## Internationalization
This section describes the internationalization setup & processes that are specific to the Workbench. Internationalization of core InVEST is handled separately: see the [InVEST Internationalization Readme](../src/natcap/invest/internationalization/README.md).

We are using the JavaScript internationalization package `i18next` and its React extension `react-i18next`. `i18next` takes in translation resources as a JavaScript object. It's convenient to store translations in JSON format. Vite automatically serves JSON files as JavaScript modules, so we can directly import translations from JSON.

The translations for each language `$LL` live in `workbench/src/main/i18n/$LL.json` and `workbench/src/renderer/i18n/$LL.json`. The JSON object in each file maps English-language messages to translations.

Nothing needs to be done during routine development. As we make changes to the Workbench text, it will inevitably get out of sync with the translations, and that's okay. Strings that have no translation will fall back to English. When it's time to update our translations, this is the process:

### Getting a new batch of translations
These instructions assume you have defined the two-letter locale code in an environment variable `$LL`.

#### Before requesting translation
1. You will need two Node packages available at the system (global) level.
    - Check for an existing `i18next-parser` installation by running `i18next --version`. If this does not return a version number, run `npm install -g i18next-parser`.
    - Check for an existing `i18next-gettext-converter` installation by running `i18next-conv --version`. If this does not return a version number, run `npm install -g i18next-conv`.

2. Extract messages from the source code:
   ```
   i18next "src/main/**/*.{js,jsx}" --output main-messages.json
   i18next "src/renderer/**/*.{js,jsx}" --output renderer-messages.json
   ```
   The `i18next` command is provided by the `i18next-parser` package and configured by `workbench/i18next-parser.config.mjs`. Each output JSON file should contain a JSON object mapping each translatable message from the source code to an empty string.

3. Merge extracted messages into the existing message catalog:
   ```
   jq -s add main-messages.json src/main/i18n/$LL.json > tmp.json
   cat tmp.json > src/main/i18n/$LL.json
   jq -s add renderer-messages.json src/renderer/i18n/$LL.json > tmp.json
   cat tmp.json > src/renderer/i18n/$LL.json
   ```
   This will add new keys into the JSON message catalogs and leave existing translations intact:
   ```
   {
      "text that's already been translated": "translation",
      "new text that doesn't have a translation yet": ""
   }
   ```

4. Examine the diff and make a note of how many lines were added to each JSON file. This number is equivalent to the number of messages that have not yet been translated, and it can be useful when gauging the need for updated translations or estimating the scope of work for an upcoming translation update.

5. Commit the updated JSON files. Use a descriptive commit message, for example: "Extract messages from Workbench code and add new messages to es (.json) catalogs."

6. Convert the JSON files to PO files:
   ```
   i18next-conv -l $LL -s src/main/i18n/$LL.json -t src/main/i18n/$LL.po
   i18next-conv -l $LL -s src/renderer/i18n/$LL.json -t src/renderer/i18n/$LL.po
   ```
   The translator will work with the PO files, and we will convert them back to JSON later.

7. Delete `tmp.json`.

8. If you are preparing translation files for multiple languages, repeat steps 2 through 7 for each remaining language.

9. Delete remaining temp files: `tmp.json` (if you generated a new one by following step 8), `main-messages.json`, and `renderer-messages.json`.

#### Request translation
Send `src/main/i18n/$LL.po` and `src/renderer/i18n/$LL.po` to the translator. They will complete the translations, then send us the updated PO files.

#### After receiving translations
1. Convert the translator-updated PO files to JSON files:
    ```
    i18next-conv -l $LL -s new_main_translations.po -t src/main/i18n/$LL.json
    i18next-conv -l $LL -s new_renderer_translations.po -t src/renderer/i18n/$LL.json
    ```
    This will replace `src/main/i18n/$LL.json` and `src/renderer/i18n/$LL.json` with the updated versions.

2. Commit the updated JSON files. Use a descriptive commit message, for example: "Update es message catalog for Workbench with new translations."
