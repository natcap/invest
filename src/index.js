"use strict";
const fs = require('fs');
const path = require('path');
const { remote, ipcRenderer } = require('electron');

const isDevMode = remote.process.argv[2] == '--dev'
if (isDevMode) {
  // in dev mode we can have babel transpile modules on import
  require("@babel/register");
  // load the '.env' file from the project root
  const dotenv = require('dotenv');
  dotenv.config();
}

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

var _react = _interopRequireDefault(require("react"));

var _reactDom = _interopRequireDefault(require("react-dom"));

var _reactHotLoader = require("react-hot-loader");

var _app = _interopRequireDefault(require("./app"));
const { fileRegistry } = require("./constants")

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
}))

window.addEventListener('contextmenu', (e) => {
  e.preventDefault()
  rightClickPosition = {x: e.x, y: e.y}
  menu.popup({ window: remote.getCurrentWindow() })
}, false)


var render = async function render(investExe) {

  _reactDom["default"].render(
    _react["default"].createElement(
      _reactHotLoader.AppContainer, null, _react["default"].createElement(
        _app["default"], { 
          jobDatabase: fileRegistry.JOBS_DATABASE,
          investExe: investExe })),
    document.getElementById('App'));
};

ipcRenderer.on('variable-reply', (event, arg) => {
  // render the App after receiving any critical data
  // from the main process
  render(arg.investExe)
})
ipcRenderer.send('variable-request', 'ping')

if (module.hot) {
  console.log('if hot module');
  module.hot.accept(render);
}
