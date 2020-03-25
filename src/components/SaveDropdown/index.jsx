import React from 'react';
import Electron from 'electron'
import PropTypes from 'prop-types';

import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Dropdown from 'react-bootstrap/Dropdown';


export class SaveParametersButton extends React.Component {
  /** Render a button that saves current args to a datastack json.
  * Opens an native OS filesystem dialog to browse to a save location.
  * Creates the JSON using datastack.py.
  */

  constructor(props) {
    super(props);
    this.browseSaveFile = this.browseSaveFile.bind(this);
  }

  browseSaveFile(event) {
    Electron.remote.dialog.showSaveDialog(
      { defaultPath: 'invest_args.json' }, (filepath) => {
      this.props.argsToJsonFile(filepath);
    });
  }

  render() {
    // disabled when there's no modelSpec, i.e. before a model is selected
    return(
      <Button 
        onClick={this.browseSaveFile}
        disabled={this.props.disabled}
        variant="link">
        Save parameters to JSON
      </Button>
    );
  }
}

SaveParametersButton.propTypes = {
  argsToJsonFile: PropTypes.func,
  disabled: PropTypes.bool
}

export class SavePythonButton extends React.Component {
  /** Render a button that saves current model and args to a python file.
  * Opens an native OS filesystem dialog to browse to a save location.
  * Creates a python script that call's the model's execute function.
  */
  
  constructor(props) {
    super(props);
    this.browseFile = this.browseFile.bind(this);
  }

  browseFile(event) {
    Electron.remote.dialog.showSaveDialog(
      { defaultPath: 'execute_invest.py' }, (filepath) => {
      this.props.savePythonScript(filepath)
    });
  }

  render() {
    // disabled when there's no modelSpec, i.e. before a model is selected
    return(
      <Button 
        onClick={this.browseFile}
        disabled={this.props.disabled}
        variant="link">
        Save to Python script
      </Button>
    );
  }
}

SavePythonButton.propTypes = {
  savePythonScript: PropTypes.func,
  disabled: PropTypes.bool
}