import fs from 'fs';
import React from 'react';
import Electron from 'electron';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';

export class SetupTab extends React.Component {

  render () {

    const status = this.props.jobStatus

    return (
      <div>
        <ArgsForm 
          args={this.props.args}
          modulename={this.props.modulename}
          updateArgs={this.props.updateArgs}
        />
        <Button 
          variant="primary" 
          size="lg"
          onClick={this.props.investExecute}
          disabled={['invalid', 'running'].includes(status)}>
              Execute
        </Button>
      </div>);
  }
}

class ArgsForm extends React.Component {

  constructor(props) {
    super(props);
    this.handleChange = this.handleChange.bind(this);
    this.selectFile = this.selectFile.bind(this);
    this.onJsonDrop = this.onJsonDrop.bind(this);
  }

  handleChange(event) {
    // Handle changes in form text inputs
    const value = event.target.value;
    const name = event.target.name;
    this.props.updateArgs([name], [value]);
  }

  selectFile(event) {
    // Handle clicks on form browse-button inputs
    const dialog = Electron.remote.dialog;
    const argtype = event.target.value;
    const argname = event.target.name;
    const prop = (argtype === 'directory') ? 'openDirectory' : 'openFile'
    // TODO: could add more filters based on argType (e.g. only show .csv)
    dialog.showOpenDialog({
      properties: [prop]
    }, (filepath) => {
      console.log(filepath);
      this.props.updateArgs([argname], [filepath[0]]); // 0 is safe since we only allow 1 selection
    })
  }

  onJsonDrop(event) {
    // Handle drag-drop of datastack JSON files
    event.preventDefault();
    
    const fileList = event.dataTransfer.files;
    if (fileList.length !== 1) {
      throw alert('only drop one file at a time.')
    }
    const filepath = fileList[0].path;
    const modelParams = JSON.parse(fs.readFileSync(filepath, 'utf8'));

    if (this.props.modulename === modelParams.model_name) {
      let keys = [];
      let values = [];
      Object.keys(modelParams.args).forEach(argkey => {
        keys.push(argkey);
        values.push(modelParams.args[argkey]);
      });
      this.props.updateArgs(keys, values);
    } else {
      throw alert('parameter file does not match this model.')
    }
  }

  render() {
    const current_args = Object.assign({}, this.props.args)
    let formItems = [];
    for (const argname in current_args) {
      const argument = current_args[argname];
      let validationMessage = '';
      if (argument.validationMessage) {
        validationMessage = argument.validationMessage ;
      }
      
      // These types need a text input and a file browser button
      if (['csv', 'vector', 'raster', 'directory'].includes(argument.type)) {
        formItems.push(
          <Form.Group key={argname}>
            <Form.Label>{argument.name}</Form.Label>
            <InputGroup>
              <Form.Control
                name={argname}
                type="text" 
                value={argument.value || ''} // empty string is handled better than `undefined`
                onChange={this.handleChange}
                isValid={argument.valid}
                isInvalid={!argument.valid}
              />
              <InputGroup.Append>
                <Button 
                  variant="outline-secondary"
                  value={argument.type}  // dialog will limit options to files or dirs accordingly
                  name={argname}
                  onClick={this.selectFile}>
                  Browse
                </Button>
              </InputGroup.Append>
              <Form.Control.Feedback type='invalid'>
                {argument.type + ' : ' + validationMessage}
              </Form.Control.Feedback>
            </InputGroup>
          </Form.Group>)
      
      // These types need a text input
      } else if (['freestyle_string', 'number'].includes(argument.type)) {
        formItems.push(
          <Form.Group  key={argname}>
            <Form.Label>{argument.name}</Form.Label>
            <Form.Control
              name={argname}
              type="text" 
              value={argument.value || ''} // empty string is handled better than `undefined`
              onChange={this.handleChange}
              isValid={argument.valid}
              isInvalid={!argument.valid}
            />
            <Form.Control.Feedback type='invalid'>
              {argument.type + ' : ' + validationMessage}
            </Form.Control.Feedback>
          </Form.Group>)
      
      // Radio select for boolean args
      } else if (argument.type === 'boolean') {
        formItems.push(
          <Form.Group key={argname}>
            <Form.Label>{argument.name}</Form.Label>
            <Form.Check 
              type="radio"
              label="Yes"
              value={true}//"true"
              checked={argument.value}//{argument.value === "true"}
              onChange={this.handleChange}
              name={argname}
            />
            <Form.Check
              type="radio"
              label="No"
              value={false}//"false"
              checked={!argument.value}//{argument.value === "false"}
              onChange={this.handleChange}
              name={argname}
            />
          </Form.Group>)

      // Dropdown menus for args with options
      } else if (argument.type === 'option_string') {
        formItems.push(
          <Form.Group  key={argname}>
            <Form.Label>{argument.name}</Form.Label>
            <Form.Control
              as='select'
              name={argname}
              value={argument.value}
              onChange={this.handleChange}>
              {argument.validation_options.options.map(opt =>
                <option value={opt} key={opt}>{opt}</option>
              )}
            </Form.Control>
            <Form.Control.Feedback type='invalid'>
              {argument.type + ' : ' + validationMessage}
            </Form.Control.Feedback>
          </Form.Group>)
      }
    }

    return (
      <Form 
        validated={false}
        onDrop={this.onJsonDrop}
        onDragOver={dragover_handler}>
        {formItems}
      </Form>
    );
  }
}

function dragover_handler(event) {
 event.preventDefault();
 event.dataTransfer.dropEffect = "move";
}


