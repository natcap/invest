This project is a user-interface layer for InVEST (Integrated Valuation of
Ecosystem Services and Tradeoffs).
InVEST can be found at https://github.com/natcap/invest.

The purpose of this project is to provide a single entry-point for all
InVEST models, and to be extensible to future models or common auxilary
workflows of an InVEST user.

## To develop and launch this Application
* `npm install`
* `npm run fetch-invest`
	+ fetches prebuilt invest binaries (see package.json invest property)
	+ Alternatively, build your own local invest binaries:
		* use invest's `make binaries`, then
		* `cp -r invest/dist/invest/ invest-workbench/build/invest/`
* `npm run dev` (this process stays live, do it in a separate shell)
* `npm start`

#### To run javascript outside the electron runtime,
but with the same ECMAscript features and babel configurations:  
`node -r @babel/register script.js`.

## To package this app for distribution
* `npm run install`
* `npm run fetch-invest`
* `npm run build`
* `npm run dist`  - Configure the packaging in `electron-builder-config.js`.


## Client logfile locations:
* Windows: "C:\Users\dmf\AppData\Roaming\logs\invest-workbench\"
* Mac: "\~/Library/Application Support/logs/invest-workbench/"
* Linux: "\~/.config/invest-workbench/logs/"
