const { remote, ipcRenderer } = require('electron'); // eslint-disable-line import/no-extraneous-dependencies

const isDevMode = remote.process.argv[2] === '--dev';
if (isDevMode) {
  // in dev mode we can have babel transpile modules on import
  require('@babel/register'); // eslint-disable-line import/no-extraneous-dependencies
  // load the '.env' file from the project root
  const dotenv = require('dotenv'); // eslint-disable-line import/no-extraneous-dependencies
  dotenv.config();
}

const _interopRequireDefault = require('@babel/runtime/helpers/interopRequireDefault');
const react = _interopRequireDefault(require('react'));
const reactDom = _interopRequireDefault(require('react-dom'));

const app = require('./app');
const { fileRegistry } = require('./constants');

// Create a right-click menu
// TODO: Not sure if Inspect Element should be available in production
// very useful in dev though.
const { Menu, MenuItem } = remote;
let rightClickPosition = null
const menu = new Menu();
menu.append(new MenuItem({
  label: 'Inspect Element',
  click: () => {
    remote.getCurrentWindow().inspectElement(rightClickPosition.x, rightClickPosition.y)
  }
}));

window.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  rightClickPosition = { x: e.x, y: e.y };
  menu.popup({ window: remote.getCurrentWindow() });
}, false);

const render = async function render(investExe) {
  reactDom.default.render(
    react.default.createElement(
      app.default, {
        jobDatabase: fileRegistry.JOBS_DATABASE,
        investExe: investExe,
      }
    ),
    document.getElementById('App')
  );
};

ipcRenderer.on('variable-reply', (event, arg) => {
  // render the App after receiving any critical data
  // from the main process
  render(arg.investExe);
})
ipcRenderer.send('variable-request', 'ping');
