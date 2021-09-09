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


## Client logfile locations:
* Windows: "C:\Users\dmf\AppData\Roaming\invest-workbench\logs\"
* Mac: "\~/Library/Logs/invest-workbench/"
* Linux: "\~/.config/invest-workbench/logs/"
