import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';

export default class Portal extends React.Component {
  constructor(props) {
    super(props);
    this.el = document.createElement('div');
    this.parent = null;
  }

  componentDidMount() {
    this.parent = document.getElementById(this.props.elId);
    if (this.parent) {
      this.parent.appendChild(this.el);
    }
  }

  componentWillUnmount() {
    if (this.parent) {
      this.parent.removeChild(this.el);
    }
  }

  render() {
    if (this.parent) {
      return ReactDOM.createPortal(
        this.props.children, this.parent
      );
    }
    return (<div />);
  }
}

Portal.propTypes = {
  elId: PropTypes.string.isRequired,
  children: PropTypes.node,
};
