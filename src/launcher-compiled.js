"use strict";

require("@babel/register");

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

var _react = _interopRequireDefault(require("react"));

var _reactDom = _interopRequireDefault(require("react-dom"));

var _reactHotLoader = require("react-hot-loader");

var _app = _interopRequireDefault(require("./app.jsx")); // require won't find *.jsx without extension

var render = function render() {
  _reactDom["default"].render(
  	_react["default"].createElement(
  		_reactHotLoader.AppContainer, null, _react["default"].createElement(
  			_app["default"], null)), document.getElementById('App'));
};

render();

if (module.hot) {
  console.log('if hot module');
  module.hot.accept(render);
}
