const crypto = require('crypto');
const {
  ipcRenderer,
  shell,
} = require('electron');

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

if (global.window) {
  // Detected a jsdom env (as opposed to node). This means
  // we're running renderer tests, so need to mock the work
  // of preload.js here.
  global.window.Workbench = {
    getLogger: jest.fn().mockReturnValue({
      debug: jest.fn(),
      verbose: jest.fn(),
      info: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
      silly: jest.fn(),
    }),
    getWorkspaceHash: (x, y, z) => {
      return crypto.createHash('sha1').update(
        `${x}
         ${JSON.stringify(y)}
         ${JSON.stringify(z)}`
      ).digest('hex');
    },
    electron: {
      ipcRenderer: ipcRenderer,
      shell: shell,
    }
  };

  // jsdom does not implement window.crypto
  global.window.crypto = {
    getRandomValues: () => [crypto.randomBytes(4).toString('hex')],
  };

  // mock out the global gettext function - avoid setting up translation
  global.window._ = (x) => x;

  jest.mock('../src/logger');
}
