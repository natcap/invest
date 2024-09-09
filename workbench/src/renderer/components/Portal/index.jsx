import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';

export default function Portal(props) {
  const el = document.createElement('div');
  const [parent, setParent] = React.useState(null);

  React.useEffect(() => {
    setParent(document.getElementById(props.elId));
    if (parent) {
      setParent(parent => parent.appendChild(el));
    }
    return () => {
      if (parent) {
        setParent(parent => parent.removeChild(el));
      }
    }
  }, []);

  if (parent) {
    return ReactDOM.createPortal(
      props.children, parent
    );
  }
  return (<div />);
}

Portal.propTypes = {
  elId: PropTypes.string.isRequired,
  children: PropTypes.node,
};
