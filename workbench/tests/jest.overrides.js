const crypto = require('crypto');
const fetch = require('node-fetch');

// a real port used to start the flask server for flaskapp.test.js,
// but different from the port used when running the electron app
// Set it before loading preload/api, as we would when running the real app.
if (!process.env.PORT) {
  process.env.PORT = '56788';
}
const { api } = require('../src/preload/api');

if (global.window) {
  // Detected a jsdom env (as opposed to node). This means
  // we're running renderer tests.
  // mock the work of preload.js here:
  global.window.Workbench = api;

  // mock out the global gettext function - avoid setting up translation
  global.window._ = (x) => x;

  // jsdom does not implement these APIs:
  global.window.crypto = {
    getRandomValues: () => [crypto.randomBytes(4).toString('hex')],
  };
  global.window.fetch = fetch;
}

// Cause tests to fail on console.error messages
// Taken from https://stackoverflow.com/questions/28615293/is-there-a-jest-config-that-will-fail-tests-on-console-warn/50584643#50584643
// let error = console.error;
// console.error = function (message) {
//   error.apply(console, arguments); // keep default behaviour
//   throw (message instanceof Error ? message : new Error(message));
// };

// I've found this obfuscates error messages sometimes:
/* With this override:
  ● InVEST subprocess testing › re-run a job - expect new log display

    Error: Uncaught [ReferenceError: sidebarFooterElementId is not defined]

      23 | console.error = function (message) {
      24 |   error.apply(console, arguments); // keep default behaviour
    > 25 |   throw (message instanceof Error ? message : new Error(message));
         |                                               ^
      26 | };
      27 |
      28 | global.window.Workbench = {

Without this override:
● InVEST subprocess testing › re-run a job - expect new log display

    ReferenceError: sidebarFooterElementId is not defined

      301 |                   nWorkers={this.props.investSettings.nWorkers}
      302 |                   sidebarSetupElementId={sidebarSetupElementId}
    > 303 |                   sidebarFooterElementId={sidebarFooterElementId}
          |                                           ^
      304 |                   isRunning={isRunning}
      305 |                 />
      306 |               </TabPane>
*/
