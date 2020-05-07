import React from 'react';
import { remote } from 'electron'
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';


export class SaveParametersButton extends React.Component {
  /** Render a button that saves current args to a datastack json.
  * Opens an native OS filesystem dialog to browse to a save location.
  * Creates the JSON using datastack.py.
  */

  constructor(props) {
    super(props);
    this.browseSaveFile = this.browseSaveFile.bind(this);
  }

  async browseSaveFile(event) {
    const data = await remote.dialog.showSaveDialog(
      { defaultPath: 'invest_args.json' })
    if (data.filePath) {   
      this.props.wrapArgsToJsonFile(data.filePath);
    } else {
      console.log('save parameters was cancelled')
    }
  }

  render() {
    // disabled when there's no modelSpec, i.e. before a model is selected
    return(
      <Button 
        onClick={this.browseSaveFile}
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

  async browseFile(event) {
    const data = await remote.dialog.showSaveDialog(
      { defaultPath: 'execute_invest.py' })
    if (data.filePath) {
      this.props.savePythonScript(data.filePath)
    } else {
      console.log('save to python was cancelled')
    }
  }

  render() {
    // disabled when there's no modelSpec, i.e. before a model is selected
    return(
      <Button 
        onClick={this.browseFile}
        variant="link">
        Save to Python script
      </Button>
    );
  }
}

SavePythonButton.propTypes = {
  savePythonScript: PropTypes.func,
}