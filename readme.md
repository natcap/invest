To develop and launch this Application
--------------------------

clone repo  

`npm install` from repo directory  

set path to invest executeable at launch:  

# windows:  

This worked for a while, but it's unreliable to call python that depends
on a specific conda environment without activating that env in the shell first.  
set INVEST=C:\\Users\\dmf\\Miniconda3\\envs\\invest-py36\\Scripts\\invest.exe && npm start  

set INVEST=C:\\Users\\dmf\\projects\\invest\\dist\\invest\\invest.exe && npm start 

  
# linux:  
INVEST=/home/dmf/Miniconda3/envs/invest-py36/Scripts/invest.exe npm start  


To run linter or tests
-----------------------------
`npm run lint`  
`npm run test`  

see `package.json` `scripts` object.  

To run these or other command-line utils of locally installed packages outside the context of the `package.json scripts`, use `npx eslint ...`. 

To run a single test file:  
set INVEST=C:\\Users\\dmf\\Miniconda3\\envs\\invest-py36\\Scripts\\invest.exe && npx jest -u investjob.test.js