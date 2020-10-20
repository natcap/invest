import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';

import SaveFileButton from '../../SaveFileButton';

export class SaveParametersButtons extends React.Component {
  render() {
    return (
      <React.Fragment>
        <SaveFileButton
          title="Save to JSON"
          defaultTargetPath="invest_args.json"
          func={this.props.wrapArgsToJsonFile}
        />
        <SaveFileButton
          title="Save to Python script"
          defaultTargetPath="execute_invest.py"
          func={this.props.savePythonScript}
        />
      </React.Fragment>
    );
  }
}

export class RunButton extends React.Component {
  render() {
    return (
      <Button
        block
        variant="primary"
        size="lg"
        onClick={this.props.wrapInvestExecute}
        disabled={this.props.disabled}
      >
        {this.props.buttonText}
      </Button>
    );
  }
}
