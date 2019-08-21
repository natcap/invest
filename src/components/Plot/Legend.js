// Reference: https://github.com/rma-consulting/react-easy-chart/blob/master/examples/Legend.js
import React from 'react';
import PropTypes from 'prop-types';
import { Style } from 'radium';


const legendStyles = {
  '.legend': {
    border: 'solid silver 1px',
    backgroundColor: 'rgba(255, 255, 255, 0.55)',
    borderRadius: '3px',
    padding: '5px',
    maxWidth: '200px',
    wordBreak: 'break-all',
    marginLeft: '100px',  // marginRight: '150px',
    float: 'left', // right
    textAlign: 'left'
  },
  '.legend li': {
    width: '80%',
    lineHeight: '18px',
    paddingLeft: '20px',
    paddingBottom: '2px',
    position: 'relative',
    fontSize: '13px',
    float: 'left'
  },
  '.legend .icon': {
    width: '10px',
    height: '10px',
    background: 'white',
    borderRadius: '6px',
    position: 'absolute',
    left: '3px',
    top: '50%',
    marginTop: '-6px',
  }
};


class Legend extends React.Component {

  static get propTypes() {
    return {
      config: PropTypes.array,
    };
  }

  getList() {
    return (
      this.props.config.map(
        (item, index) => (
          <li key={index}>
            <span
              className="icon"
              style={{ backgroundColor: this.getIconColor(index) }}
            />
            {item.type}
          </li>
        )
      )
    );
  }

  getIconColor(index) {
    const {
      config
    } = this.props;

    return config[index].color;
  }

  render() {
    if (this.props.config.length > 0) {
      return (
        <div className="legend-container">
          <Style scopeSelector=".legend-container" rules={legendStyles} />
          <ul className="legend">{this.getList()}</ul>
        </div>
      );
    } else {
      return null;
    }
  }
}

export default Legend;
