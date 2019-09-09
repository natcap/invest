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
    let openingArgs = this.props.args
    for (const argname in openingArgs) {
      const argument = openingArgs[argname];
      openingArgs[argname]['valid'] = validate(argument.value, argument.validationRules)
    }
    this.props.checkArgsReadyToValidate(openingArgs)
  }

  render () {

    const status = this.props.jobStatus

    return (
      <div>
        <ArgsForm 
          args={this.props.args}
          handleChange={this.props.handleChange} 
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
    let validationMessage = '';
    for (const arg in current_args) {
      const argument = current_args[arg];
      if (argument.validationMessage) {
        validationMessage = argument.validationMessage ;
      }
      if (argument.type !== 'select') {
        formItems.push(
          <Form.Group>
            <Form.Label>
              {argument.argname}
            </Form.Label>
            <Form.Control 
              name={argument.argname}
              type={argument.type}
              value={argument.value}
              required={argument.required}
              onChange={this.props.handleChange}
              isValid={argument.valid}
              isInvalid={!argument.valid}
            />
            <Form.Control.Feedback type='invalid'>
              {argument.validationRules.rule + ' : ' + validationMessage}
            </Form.Control.Feedback>
          </Form.Group>)
      } else {
        formItems.push(
          <Form.Group>
            <Form.Label>
              {argument.argname}
            </Form.Label>
            <Form.Control as='select'
              name={argument.argname}
              value={argument.value}
              required={argument.required}
              onChange={this.props.handleChange}
            >
              {argument.options.map(opt =>
                <option value={opt}>{opt}</option>
              )}
            </Form.Control>
            <Form.Control.Feedback type='invalid'>
              {argument.validationRules.rule + ' : ' + validationMessage}
            </Form.Control.Feedback>
          </Form.Group>)
      }
    }

    return (
      <Form validated={false}>{formItems}</Form>
    );
  }
}