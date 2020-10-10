import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';

import DropdownButton from 'react-bootstrap/DropdownButton';

import SaveFileButton from '../../SaveFileButton';

export default class SaveParametersButtons extends React.Component {
  render() {
    const sidebarNode = document.getElementById('setup-sidebar-children');
    return ReactDOM.createPortal(
      (
        <DropdownButton
          id="dropdown-basic-button"
          title="Save Parameters"
          renderMenuOnMount // w/o this, items inaccessible in jsdom test env
          className="mx-3 float-right"
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
      ), sidebarNode
    );
  }
}
