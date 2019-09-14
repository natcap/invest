import React, { Component } from 'react';
import HraMap from './components/Map';
import Plot from './components/Plot';

// The tab where this component renders is only enabled
// for jobStatus === 0 (run completed w/o error).
// So here we need only pass a workspace, not check
// for the status of the related Job.

// this.state.workspace is set on invest run subprocess exit,
// until then workspace is null

class HraApp extends Component {

  // constructor(props) {
  //   super (props); // Required to call original constructor
  //   this.state = {
  //     title: "Habitat Risk Assessment"
  //   }
  // }

  render() {
    if (this.props.workspace) {
      return (
        <div>
          <HraMap 
            workspace={this.props.workspace}
            activeTab={this.props.activeTab}/>
          <Plot />
        </div>
      );
    } else {
      return (
        <div>{'Nothing to see here'}</div>);
    }
  }
}

export default HraApp;
