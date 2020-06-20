"use strict";
const fs = require('fs');
const path = require('path');
const { remote } = require('electron');

const isDevMode = function() {
  return remote.process.argv[2] == '--dev'
};
if (isDevMode()) {
  require("@babel/register");
  const dotenv = require('dotenv');
  dotenv.config();  
}

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

var _react = _interopRequireDefault(require("react"));

var _reactDom = _interopRequireDefault(require("react-dom"));

var _reactHotLoader = require("react-hot-loader");

var _app = _interopRequireDefault(require("./app"));
var InvestConfigModal = _interopRequireDefault(require("./components/InvestConfigModal"));


const JOBS_DATABASE = path.join(__dirname, 'jobdb.json')
const INVEST_REGISTRY_PATH = path.join(
      remote.app.getPath('userData'), 'invest_registry.json')


// Binding to the invest binary:
let investExe;

// A) look for a local registry of available invest installations
if (fs.existsSync(INVEST_REGISTRY_PATH)) {
  const investRegistry = JSON.parse(fs.readFileSync(INVEST_REGISTRY_PATH))
  const activeVersion = investRegistry['active']
  investExe = investRegistry['registry'][activeVersion]['invest']

// B) check for dev mode and an environment variable from dotenv
} else if (isDevMode()) {
  investExe = process.env.INVEST

// C) point to binaries included in this app's installation.
} else {
  // TODO: extension on this file is OS-dependent
  investExe = path.join(__dirname, 'invest/invest')
}


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


var render = function render() {

  // let investExe;
  // if (fs.existsSync(INVEST_REGISTRY_PATH)) {
  //   const investRegistry = JSON.parse(fs.readFileSync(INVEST_REGISTRY_PATH))
  //   const activeVersion = investRegistry['active']
  //   investExe = investRegistry['registry'][activeVersion]['invest']
  // } else {
  //   // TODO: extension on this file is OS-dependent
  //   investExe = path.join(__dirname, 'invest/invest')
  // }
  _reactDom["default"].render(
    _react["default"].createElement(
      _reactHotLoader.AppContainer, null, _react["default"].createElement(
        _app["default"], { appdata: JOBS_DATABASE, investExe: investExe })),
    document.getElementById('App'));
  // }
  // } else {
  //   _reactDom["default"].render(
  //     _react["default"].createElement(
  //       InvestConfigModal["default"], {
  //         investVersion: undefined,
  //         investRegistry: { active: undefined, registry: {} }
  //       }),
  //     document.getElementById('App'));
  // }

};

render();

if (module.hot) {
  console.log('if hot module');
  module.hot.accept(render);
}
