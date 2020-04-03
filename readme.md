## To develop and launch this Application

`npm install` from repo directory  

activate a python environment that can import `natcap.invest` and `flask`  

(`natcap.invest` latest `master` is recommended. 3.8.0 should work also.)

Create `.env` in the project root by copying `.env-example` and modifying the invest path as needed.

`npm start`  

### To run linter or tests
`npm run lint`  
`npm run test`  

see `package.json` `scripts` object.  

To run these or other command-line utils of locally installed packages outside the context of the `package.json scripts`, use `npx` (e.g. `npx eslint ...`) as a shortcut to the executeable. 

### To run a single test file:
`npx jest app.test.js`  (note this is not the path to the test file, rather a pattern for matching)  

To run snippets of code outside the electron runtime, but with the same ECMAscript features and babel configurations, use `node -r @babel/register script.js`.  


### Developing Visualization components
Visualization components (i.e. folders in `src/components/VizTab/Visualization`) should be named with the same model name used in the `invest run <model>` command so that these components can be imported dynamically and lazily. These are the same names returned by `invest list`.