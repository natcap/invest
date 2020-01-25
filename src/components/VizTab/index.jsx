import React, { Suspense } from 'react';
import VizErrorBoundary from './VizErrorBoundary';

// The tab where this component renders is only enabled
// for sessionProgress === 'viz' (run completed w/o error).
// So here we need only pass a workspace, not check
// for the status of the related job.

// this.state.workspace is set on invest run subprocess exit,
// until then workspace is null
// this.props.model is set on invest getspec response

export class VizTab extends React.Component {

  render() {
    if (this.props.workspace && this.props.model) {
      const model_viz_space = './Visualization/' + this.props.model;
      const Visualization = React.lazy(() => import(model_viz_space));

      return (
        <div>
          <VizErrorBoundary>
            <Suspense fallback={<div>Loading...</div>}>
              <Visualization 
                workspace={this.props.workspace}
                activeTab={this.props.activeTab}/>
            </Suspense>
          </VizErrorBoundary>
        </div>
      );
    } else {
      return (
        <div>{'Nothing to see here'}</div>
      );
    }
  }
}
