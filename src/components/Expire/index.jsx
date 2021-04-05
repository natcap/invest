import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

/** A wrapper component that hides it's children after delay. */
export default function Expire(props) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setVisible(false);
    }, props.delay);
  }, [props.delay]);

  return visible ? <div>{props.children}</div> : <div />;
}

Expire.propTypes = {
  delay: PropTypes.number.isRequired,
  children: PropTypes.oneOfType([
    PropTypes.arrayOf(PropTypes.node),
    PropTypes.node,
  ]).isRequired,
};
