This project is a user-interface layer for InVEST (Integrated Valuation of
Ecosystem Services and Tradeoffs).
InVEST can be found at https://github.com/natcap/invest.

The purpose of this project is to provide a single entry-point for all
InVEST models, and to be extensible to future models or common auxilary
workflows of an InVEST user.

## To develop and launch this Application

`npm install` from repo directory  

Create `.env` in the project root by copying `.env-example` and modifying the
invest and server executeable paths as needed.
Options for creating these exes:  
* run `make binaries` from the `natcap/invest` repository. Though currently
this is only available on branch `experimental/merge-pyinstaller-invest-gui` 
of `github.com/davemfish/invest`
* download pre-built binaries. see `scripts/get_invest_binaries_gcs.js`
`npm run prebuilt-invest "windows-latest"` (or "macos-latest" "ubuntu-latest")
`unzip ./build/binaries.zip -d ./build`

`npm start`  

There's an intermittent issue with a `fetch` call. If you don't see a long list 
of green invest model buttons, try refreshing.


## To build this application

`npm run prebuilt-invest "windows-latest"` (or "macos-latest" "ubuntu-latest")
`unzip ./build/binaries.zip -d ./build`

`npm run build`  -- calls babel to transpile ES6 and jsx code to commonjs

`npm run dist`  -- packages build source into an electron application using electron-builder


### To run linter or tests
`npm run lint`  
`npm run test`  

To run these or other command-line utils of locally installed packages outside the context of the `package.json scripts`, use `npx` (e.g. `npx eslint ...`) as a shortcut to the executeable. 

### To run a single test file:
`npx jest --coverage=false app.test.js`  (note this is not the path to the test file, rather a pattern for matching)  

To run snippets of code outside the electron runtime, but with the same ECMAscript features and babel configurations, use `node -r @babel/register script.js`.  


### Developing Visualization components
Visualization components (i.e. folders in `src/components/VizTab/Visualization`) should be named with the same model name used in the `invest run <model>` command so that these components can be imported dynamically and lazily. These are the same names returned by `invest list`.