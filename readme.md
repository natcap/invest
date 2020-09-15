This project is a user-interface layer for InVEST (Integrated Valuation of
Ecosystem Services and Tradeoffs).
InVEST can be found at https://github.com/natcap/invest.

The purpose of this project is to provide a single entry-point for all
InVEST models, and to be extensible to future models or common auxilary
workflows of an InVEST user.

## To develop and launch this Application

* `npm install`  
* clone natcap.invest and checkout a recent revision (e.g. `main`)  
* setup a conda* environment with deps for `natcap.invest Flask PyInstaller`  
* build invest binaries  
	`python -m PyInstaller --workpath build/pyi-build --clean --distpath build ./invest-flask.spec`  
* `npm start`  

(* invest-flask.spec script assumes a conda environment)

## To build this application

`npm run build`  -- calls babel to transpile ES6 and jsx code to commonjs

`npm run dist`  -- packages build source into an electron application using electron-builder


### To run linter or tests
`npm run lint`  
`npm run test`  

To run these or other command-line utils of locally installed packages outside the context of the `package.json scripts`, use `npx` (e.g. `npx eslint ...`) as a shortcut to the executeable. 

### To run a single test file:
`npx jest --coverage=false --verbose app.test.js`  

To run snippets of code outside the electron runtime, but with the same ECMAscript features and babel configurations, use `node -r @babel/register script.js`.  

