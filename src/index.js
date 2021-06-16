const { ipcRenderer } = require('electron'); // eslint-disable-line import/no-extraneous-dependencies

const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);
const isDevMode = process.argv.includes('--dev');
if (isDevMode) {
  // in dev mode we can have babel transpile modules on import
  require('@babel/register'); // eslint-disable-line import/no-extraneous-dependencies
  require('react-devtools');
}

const _interopRequireDefault = require('@babel/runtime/helpers/interopRequireDefault');
const react = _interopRequireDefault(require('react'));
const reactDom = _interopRequireDefault(require('react-dom'));
const app = require('./app');

// Create a right-click menu
// TODO: Not sure if Inspect Element should be available in production
// very useful in dev though.
let rightClickPosition = null;
window.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  rightClickPosition = { x: e.x, y: e.y };
  ipcRenderer.send('show-context-menu', rightClickPosition);
});

function render(isFirstRun) {
  reactDom.default.render(
    react.default.createElement(
      app.default, {
        isFirstRun: isFirstRun,
      }
    ),
    document.getElementById('App')
  );
};

ipcRenderer.invoke('is-first-run')
  .then((response) => {
    render(response.isFirstRun);
  });
