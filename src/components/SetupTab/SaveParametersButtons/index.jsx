import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import DropdownButton from 'react-bootstrap/DropdownButton';

import SaveFileButton from '../../SaveFileButton';

export class SaveParametersButtons extends React.Component {
  render() {
    const siblingNode = document.getElementById('setup-sidebar-children');
    if (siblingNode) {
      return ReactDOM.createPortal(
        (
          <DropdownButton
            id="dropdown-basic-button"
            className="mx-3 float-right"
            title="Save Parameters"
            renderMenuOnMount // w/o this, items inaccessible in jsdom test env
          >
            <SaveFileButton
              title="Save parameters to JSON"
              defaultTargetPath="invest_args.json"
              func={this.props.wrapArgsToJsonFile}
            />
            <SaveFileButton
              title="Save to Python script"
              defaultTargetPath="execute_invest.py"
              func={this.props.savePythonScript}
            />
          </DropdownButton>
        ), siblingNode
      );
    }
    return (<div />);
  }
}

export class ExecuteButton extends React.Component {
  render() {
    const siblingNode = document.getElementById('sidebar-footer');
    if (siblingNode) {
      return ReactDOM.createPortal(
        (
          <Button
            block
            variant="primary"
            size="lg"
            onClick={this.props.wrapInvestExecute}
            disabled={this.props.disabled}
          >
            Execute
          </Button>
        ), siblingNode
      );
    }
    return (<div />);
  }
}
