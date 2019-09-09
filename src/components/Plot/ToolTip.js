// Reference: https://github.com/rma-consulting/react-easy-chart/blob/master/examples/ToolTip.js
import React from 'react';
import PropTypes from 'prop-types';
import { Style } from 'radium';

const toolTipStyles = {
  '.tooltip': {
    border: 'solid silver 0.5px',
    position: 'absolute',
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderRadius: '4px',
    paddingLeft: '5px',
    paddingRight: '5px',
    paddingTop: '1px',
    paddingBottom: '1px',
    display: 'inline-block',
    fontSize: '12px',
    textAlign: 'center'
  }
};

const ToolTip = (props) => (
  <div className="tooltip-container">
    <Style scopeSelector=".tooltip-container" rules={toolTipStyles} />
    <div className="tooltip" style={{ top: props.top, left: props.left }}>
      {props.children}
    </div>
  </div>
);

ToolTip.propTypes = {
  left: PropTypes.string,
  top: PropTypes.string,
  children: PropTypes.node
};

export default ToolTip;
