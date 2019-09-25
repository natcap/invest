import React, { Component } from 'react';
// import('./components/Visualization/habitat_risk_assessment');
// const Visualization = React.lazy(() => import('./components/Visualization/habitat_risk_assessment'));
// import Visualization from './components/Visualization/habitat_risk_assessment';
// import Plot from './components/Plot';

// The tab where this component renders is only enabled
// for jobStatus === 0 (run completed w/o error).
// So here we need only pass a workspace, not check
// for the status of the related Job.

// this.state.workspace is set on invest run subprocess exit,
// until then workspace is null

class VizApp extends Component {

  // constructor(props) {
  //   super (props); // Required to call original constructor
  //   this.state = {
  //     title: "Habitat Risk Assessment"
  //   }
  // }

  render() {
    const model = this.props.model;
    const model_import_space = './components/Visualization/' + {model};
    // import Visualization from model_import_space;
    if (this.props.workspace) {
      return (
        <div>
          <Visualization 
            workspace={this.props.workspace}
            activeTab={this.props.activeTab}/>
        </div>
      );
    } else {
      return (
        <div>{'Nothing to see here'}</div>);
    }
  }
}

export default VizApp;
