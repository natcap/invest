import React, { Suspense } from 'react';
import PropTypes from 'prop-types';

import VizErrorBoundary from './VizErrorBoundary';


export class ResultsTab extends React.Component {

  render() {
    if (this.props.workspace && this.props.model) {
      const model_viz_space = './Visualization/' + this.props.model;
      const Visualization = React.lazy(() => import(model_viz_space));

      // If Visualization component does not exist, the ErrorBoundary
      // should render instead.
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

ResultsTab.propTypes = {
  model: PropTypes.string,
  workspace: PropTypes.shape({
      directory: PropTypes.string,
      suffix: PropTypes.string
    }),
  jobID: PropTypes.string,
  activeTab: PropTypes.string 
}
