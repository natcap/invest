import React from 'react';

class VizErrorBoundary extends React.Component {

  render() {
      return (
        <div>{'This model has no visualization'}</div>
      );
    }
  }

export default VizErrorBoundary;
