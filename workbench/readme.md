This project is a user-interface layer for InVEST (Integrated Valuation of
Ecosystem Services and Tradeoffs).
InVEST can be found at https://github.com/natcap/invest.

The purpose of this project is to provide a single entry-point for all
InVEST models, and to be extensible to future models or common auxilary
workflows of an InVEST user.

## To develop and launch this Application
* `yarn install`
* `yarn run fetch-invest`
	+ fetches prebuilt invest binaries (see package.json invest property)
	+ Alternatively, build your own local invest binaries:
		* use invest's `make binaries`, then
		* `cp -r invest/dist/invest/ invest-workbench/build/invest/`
* `yarn run dev` (this process stays live, do it in a separate shell)
* `yarn start`

#### To run javascript outside the electron runtime,
but with the same ECMAscript features and babel configurations:  
`node -r @babel/register script.js`.

## To package this app for distribution
* `yarn run install`
* `yarn run fetch-invest`
* `yarn run build`
* `yarn run dist`  - Configure the packaging in `electron-builder-config.js`.

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
"restoreMocks": true,
```
Restore unmocked implementations. But this only works with `jest.spyOn`
and all the syntactic sugar for it (`mockReturnValue`, `mockImplementation`, etc).
So, all mocks are unmocked before each test, except manual mocks such as in
in `__mocks__/electron.js` where `spyOn` is not used and properties are 
manually assigned `jest.fn()`. We only use manual mocks for modules that
should absolutely be mocked in all test cases. Electron is a good example because
its API is never available outside an electron runtime, which jest does not provide.
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
