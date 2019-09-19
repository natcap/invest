import React from 'react';
import Electron from 'electron';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';

import validate from '../../validate';

export class SetupTab extends React.Component {

  componentDidMount() {
    // TODO: once invest getspec replaces valid_HRA_args.js
    // there will not be default arg values present on Mount.
    // so we may not need this function.

    // For now, nice to validate on load if it's possible to load with default args.
    // let openingArgs = this.props.args
    // for (const argname in openingArgs) {
    //   const argument = openingArgs[argname];
    //   openingArgs[argname]['valid'] = validate(argument.value, argument.validationRules)
    // }
    // this.props.checkArgsReadyToValidate(openingArgs)
  }

  render () {

    const status = this.props.jobStatus

    return (
      <div>
        <ArgsForm 
          args={this.props.args}
          onDrop={this.props.onDrop}
          handleChange={this.props.handleChange}
          selectFile={this.props.selectFile}
        />
        <Button 
          variant="primary" 
          size="lg"
          onClick={this.props.executeModel}
          disabled={['invalid', 'running'].includes(status)}>
              Execute
        </Button>
      </div>);
  }
}

class ArgsForm extends React.Component {

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
                onChange={this.props.handleChange}
                isValid={argument.valid}
                isInvalid={!argument.valid}
              />
              <InputGroup.Append>
                <Button 
                  variant="outline-secondary"
                  value={argument.type}  // dialog will limit options to files or dirs accordingly
                  name={argname}
                  onClick={this.props.selectFile}>
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
              // required={argument.required}
              onChange={this.props.handleChange}
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
              value="true"
              checked={argument.value === "true"}
              onChange={this.props.handleChange}
              name={argname}
            />
            <Form.Check
              type="radio"
              label="No"
              value="false"
              checked={argument.value === "false"}
              onChange={this.props.handleChange}
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
              // required={argument.required}
              onChange={this.props.handleChange}
            >
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
        onDrop={this.props.onDrop}
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


