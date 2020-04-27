"use strict";

if (process.env.DEVMODE) {
  require("@babel/register");
}

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

var _react = _interopRequireDefault(require("react"));

var _reactDom = _interopRequireDefault(require("react-dom"));

var _reactHotLoader = require("react-hot-loader");

var _app = _interopRequireDefault(require("./app"));

const { remote } = require('electron');
const { Menu, MenuItem } = remote;
const JOBS_DATABASE = 'jobdb.json'

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
  _reactDom["default"].render(
    _react["default"].createElement(
      _reactHotLoader.AppContainer, null, _react["default"].createElement(
        _app["default"], { appdata: JOBS_DATABASE })), document.getElementById('App'));
};

render();

if (module.hot) {
  console.log('if hot module');
  module.hot.accept(render);
}
