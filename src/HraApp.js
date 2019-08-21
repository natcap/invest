import React, { Component } from 'react';
import HraMap from './components/Map';
import Plot from './components/Plot';

class HraApp extends Component {

  constructor(props) {
    super (props); // Required to call original constructor
    this.state = {
      title: "Habitat Risk Assessment"
    }
  }

  render() {
    return (
      <div>
        <HraMap />
        <Plot />
      </div>
    );
  }
}

export default HraApp;
