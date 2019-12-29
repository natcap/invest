## To develop and launch this Application

`npm install` from repo directory  

set system-specific environment variables at launch. 

### windows:  
`set INVEST=C:\\Users\\dmf\\Miniconda3\\envs\\invest-py37\\Scripts\\invest.exe && set GDAL_DATA=C:\\Users\\dmf\\Miniconda3\\envs\\invest-py37\\Library\\share\\gdal && set PYTHON=C:\\Users\\dmf\\Miniconda3\\envs\\invest-py37\\python && npm start`  

GDAL_DATA path here is optional, do it if GDAL ERRORs suggest you need to.  
PYTHON path must be able to import `flask`  

### linux:  
`INVEST=/home/dmf/miniconda3/envs/invest-env-py36/bin/invest PYTHON=/home/dmf/miniconda3/envs/invest-env-py36/bin/python npm start`  


**Temporary way to test various invest models:**  
set this variable in InvestJob.jsx to a name found in `invest list`:  
`const MODEL_NAME = 'carbon'`  


## To run linter or tests
`npm run lint`  
`npm run test`  

see `package.json` `scripts` object.  

To run these or other command-line utils of locally installed packages outside the context of the `package.json scripts`, use `npx` (e.g. `npx eslint ...`) as a shortcut to the executeable. 

**To run a single test file:**  
`set INVEST=C:\\Users\\dmf\\Miniconda3\\envs\\invest-py37\\Scripts\\invest.exe && npx jest main.test.js`  (note this is not the path to the test file, rather a pattern for matching)  

To run snippets of code outside the electron runtime, but with the same ECMAscript features and babel configurations, use `node -r @babel/register script.js`. This is useful for development of isolated modules.


## Visualization components
Visualization components (i.e. folders in `src/components/Visualization`) should be named with the same model name used in the `invest run <model>` command so that these components can be imported dynamically and lazily.