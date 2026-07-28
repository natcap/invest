import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

/** A wrapper component that hides it's children after delay. */
export default function Expire({delay, children, className = ''}) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const timerId = setTimeout(() => {
      setVisible(false);
    }, delay);

    return () => clearTimeout(timerId);
  });

  return (
    visible
      ? <div className={className}>{children}</div>
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
