import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

/** A wrapper component that hides it's children after delay. */
export default function Expire(props) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const timerId = setTimeout(() => {
      setVisible(false);
    }, props.delay);

    return () => clearTimeout(timerId);
  });

  return (
    visible
      ? <div className={props.className}>{props.children}</div>
      : <div />
  );
}

Expire.propTypes = {
  className: PropTypes.string,
  delay: PropTypes.number.isRequired,
  children: PropTypes.oneOfType([
    PropTypes.arrayOf(PropTypes.node),
    PropTypes.node,
  ]).isRequired,
};
Expire.defaultProps = {
  className: ''
};
