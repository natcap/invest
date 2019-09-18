import React from 'react';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';

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
          handleChange={this.props.handleChange}
          onDrop={this.props.onDrop}
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
      if (argument.type !== 'option_string') {
        formItems.push(
          <Form.Group  key={argname}>
            <Form.Label>
              {argument.name}
            </Form.Label>
            <Form.Control
              name={argname}
              type="text" //{argument.type}
              value={argument.value || ''} // empty string is handled better than `undefined`
              required={argument.required}
              onChange={this.props.handleChange}
              isValid={argument.valid}
              isInvalid={!argument.valid}
            />
            <Form.Control.Feedback type='invalid'>
              {argument.type + ' : ' + validationMessage}
            </Form.Control.Feedback>
          </Form.Group>)
      } else {
        formItems.push(
          <Form.Group  key={argname}>
            <Form.Label>
              {argument.name}
            </Form.Label>
            <Form.Control
              as='select'
              name={argname}
              value={argument.value}
              required={argument.required}
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

function dragover_handler(ev) {
 ev.preventDefault();
 ev.dataTransfer.dropEffect = "move";
}