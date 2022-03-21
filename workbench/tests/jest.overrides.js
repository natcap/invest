import crypto from 'crypto';
import fetch from 'node-fetch';

import api from '../src/preload/api';

// debug logging is a bit noisy, not so useful during tests.
if (!process.env.ELECTRON_LOG_LEVEL) {
  process.env.ELECTRON_LOG_LEVEL = 'info';
}

// Test for a jsdom env (as opposed to node), which means renderer tests.
if (global.window) {
  // mock the work of preload.js here:
  global.window.Workbench = api;

  // mock out the global gettext function - avoid setting up translation
  global.window._ = (x) => x;

  // jsdom does not implement these window APIs:
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
