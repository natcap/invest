import React from 'react';

class VizErrorBoundary extends React.Component {

	constructor(props) {
		super(props);
		this.state = { hasError: false }
	}

	static getDerivedStateFromError(error) {
		return { hasError: true };
	}

  render() {
  	if (this.state.hasError) {
      return (
        <div>{'This model has no visualization'}</div>
      );
  	}
  	return this.props.children;
    }
  }

export default VizErrorBoundary;
