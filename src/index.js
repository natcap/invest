const { ipcRenderer } = require('electron'); // eslint-disable-line import/no-extraneous-dependencies

console.log('index.js 1')
const isDevMode = process.argv.includes('--dev');
console.log('index.js 2')
if (isDevMode) {
  // in dev mode we can have babel transpile modules on import
  require('@babel/register'); // eslint-disable-line import/no-extraneous-dependencies
  // load the '.env' file from the project root
  const dotenv = require('dotenv'); // eslint-disable-line import/no-extraneous-dependencies
  dotenv.config();
  require('react-devtools');
}
console.log('index.js 3')
// const _interopRequireDefault = require('@babel/runtime/helpers/interopRequireDefault');
console.log('index.js 4')
// const react = _interopRequireDefault(require('react'));
const react = require('react');
console.log('index.js 5')
// const reactDom = _interopRequireDefault(require('react-dom'));
const reactDom = require('react-dom');
console.log('index.js 6')
const app = require('./app');
console.log('index.js 7')
const { getLogger } = require('./logger');
console.log('index.js 8')
const logger = getLogger(__filename.split('/').slice(-1)[0]);
console.log('index.js 9')
// Create a right-click menu
// TODO: Not sure if Inspect Element should be available in production
// very useful in dev though.
let rightClickPosition = null;
window.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  rightClickPosition = { x: e.x, y: e.y };
  ipcRenderer.invoke('show-context-menu', rightClickPosition);
});

const render = async function render(investExe) {
  // reactDom.default.render(
  //   react.default.createElement(
  //     app.default, {
  //       investExe: investExe,
  //     }
  //   ),
  //   document.getElementById('App')
  // );
  reactDom.render(
    react.createElement(
      app.default, {
        investExe: investExe,
      }
    ),
    document.getElementById('App')
  );
};

ipcRenderer.invoke('variable-request')
  // render the App after receiving any critical data
  // from the main process
  .then((response) => {
    logger.debug('rendering react front-end');
    render(response.investExe);
  });
