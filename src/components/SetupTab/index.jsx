import React from 'react';
import { remote } from 'electron';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import InputGroup from 'react-bootstrap/InputGroup';

import { fetchDatastackFromFile } from '../../server_requests';
import { argsValuesFromSpec } from '../../utils';

export class SetupTab extends React.Component {
  /** Renders an Arguments form and an Execute button
  */

  render () {
    // Only mount the ArgsForm when there are actually args
    // This lets us have an ArgsForm.componentDidMount() that
    // does useful initialization of args state.
    if (this.props.args) {
      return (
        <div>
          <ArgsForm 
            args={this.props.args}
            modulename={this.props.modulename}
            updateArg={this.props.updateArg}
            batchUpdateArgs={this.props.batchUpdateArgs}
            investValidate={this.props.investValidate}
          />
          <Button 
            variant="primary" 
            size="lg"
            onClick={this.props.investExecute}
            disabled={!this.props.argsValid}>
                Execute
          </Button>
        </div>);
    }
    // The SetupTab remains disabled in this route, so no need
    // to render anything here.
    return(<div>No args to see here</div>)
  }
}

SetupTab.propTypes = {
  args: PropTypes.object,
  argsValid: PropTypes.bool,
  modulename: PropTypes.string,
  updateArg: PropTypes.func,
  batchUpdateArgs: PropTypes.func,
  investValidate: PropTypes.func,
  investExecute: PropTypes.func
}

class ArgsForm extends React.Component {
  /** Renders an HTML input for each invest argument passed in props.args.
  *
  * Values of input fields inherit from parent components state.args, and so 
  * change handlers for the inputs in this component update their values
  * by calling parent component methods that call parent's setState.
  */

  constructor(props) {
    super(props);
    this.handleChange = this.handleChange.bind(this);
    this.selectFile = this.selectFile.bind(this);
    this.onDragDrop = this.onDragDrop.bind(this);
  }

  componentDidMount() {
    // Validate args immediately on mount. Values will be `undefined`
    // but converted to empty strings by `argsValuesFromSpec` 
    // Validation will handle updating the `valid` property
    // of the optional and conditionally required args so that they 
    // can validate without any user-interaction. Validation messages
    // won't appear to the user until an argument has been touched.

    // TODO: could call batchUpdateArgs here instead
    // to avoid passing investValidate to this component at all.
    // this.props.batchUpdateArgs(JSON.parse(args_dict_string));
    this.props.investValidate(argsValuesFromSpec(this.props.args));
  }

  handleChange(event) {
    /** Handle keystroke changes in text inputs */
    const value = event.target.value;
    const name = event.target.name;
    this.props.updateArg(name, value);
  }

  async selectFile(event) {
    /** Handle clicks on browse-button inputs */
    const argtype = event.target.value;
    const argname = event.target.name;
    const prop = (argtype === 'directory') ? 'openDirectory' : 'openFile'
    // TODO: could add more filters based on argType (e.g. only show .csv)
    const data = await remote.dialog.showOpenDialog({ properties: [prop] })
    if (data.filePaths.length) {
      this.props.updateArg(argname, data.filePaths[0]); // dialog defaults allow only 1 selection
    } else {
      console.log('browse dialog was cancelled')
    }
  }

  async onDragDrop(event) {
    /** Handle drag-drop of datastack JSON files and InVEST logfiles */
    event.preventDefault();
    
    const fileList = event.dataTransfer.files;
    if (fileList.length !== 1) {
      throw alert('only drop one file at a time.')
    }
    const payload = { 
      datastack_path: fileList[0].path
    }
    const datastack = await fetchDatastackFromFile(payload)

    if (datastack['module_name'] === this.props.modulename) {
      this.props.batchUpdateArgs(datastack['args']);
    } else {
      console.log('Parameter/Log file for ' + datastack['module_name'] + ' does not match this model: ' + this.props.modulename)
      throw alert('Parameter/Log file for ' + datastack['module_name'] + ' does not match this model: ' + this.props.modulename)
    }
  }

  render() {
    const current_args = Object.assign({}, this.props.args)
    let formItems = [];
    for (const argname in current_args) {
      if (argname === 'n_workers') { continue }
      const argument = current_args[argname];

      // These types need a text input and a file browser button
      if (['csv', 'vector', 'raster', 'directory'].includes(argument.type)) {
        formItems.push(
          <Form.Group as={Row} key={argname}>
            <Form.Label column sm="3"  htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="8">
              <InputGroup>
                <Form.Control
                  id={argname}
                  name={argname}
                  type="text" 
                  value={argument.value || ''} // empty string is handled better than `undefined`
                  onChange={this.handleChange}
                  isValid={argument.touched && argument.valid}
                  isInvalid={argument.touched && argument.validationMessage}
                />
                <InputGroup.Append>
                  <Button
                    id={argname}
                    variant="outline-secondary"
                    value={argument.type}  // dialog will limit options to files or dirs accordingly
                    name={argname}
                    onClick={this.selectFile}>
                    Browse
                  </Button>
                </InputGroup.Append>
                <Form.Control.Feedback type='invalid' id={argname + '-feedback'}>
                  {argument.type + ' : ' + (argument.validationMessage || '')}
                </Form.Control.Feedback>
              </InputGroup>
            </Col>
          </Form.Group>)
      
      // These types need a text input
      } else if (['freestyle_string', 'number'].includes(argument.type)) {
        formItems.push(
          <Form.Group as={Row} key={argname}>
            <Form.Label column sm="3"  htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="4">
              <Form.Control
                id={argname}
                name={argname}
                type="text" 
                value={argument.value || ''} // empty string is handled better than `undefined`
                onChange={this.handleChange}
                isValid={argument.touched && argument.valid}
                isInvalid={argument.touched && argument.validationMessage}
              />
              <Form.Control.Feedback type='invalid' id={argname + '-feedback'}>
                {argument.type + ' : ' + (argument.validationMessage || '')}
              </Form.Control.Feedback>
            </Col>
          </Form.Group>)
      
      // Radio select for boolean args
      } else if (argument.type === 'boolean') {
        // argument.value will be type boolean if it's coming from an args dict
        // generated by natcap.invest. But it will be type string if the value
        // is set from this UI, because html forms always submit strings.
        // So, `checked` property must accomodate both types to determine state.
        formItems.push(
          <Form.Group as={Row} key={argname}>
            <Form.Label column sm="3" htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="8">
              <Form.Check
                id={argname}
                inline
                type="radio"
                label="Yes"
                value={"true"}
                checked={argument.value || argument.value === "true"}
                onChange={this.handleChange}
                name={argname}
              />
              <Form.Check
                id={argname}
                inline
                type="radio"
                label="No"
                value={"false"}
                checked={!argument.value || argument.value === "false"}
                onChange={this.handleChange}
                name={argname}
              />
            </Col>
          </Form.Group>)

      // Dropdown menus for args with options
      } else if (argument.type === 'option_string') {
        formItems.push(
          <Form.Group as={Row} key={argname}>
            <Form.Label column sm="3" htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="4">
              <Form.Control
                id={argname}
                as='select'
                name={argname}
                value={argument.value}
                onChange={this.handleChange}>
                {argument.validation_options.options.map(opt =>
                  <option value={opt} key={opt}>{opt}</option>
                )}
              </Form.Control>
              <Form.Control.Feedback type='invalid' id={argname + '-feedback'}>
                {argument.type + ' : ' + (argument.validationMessage || '')}
              </Form.Control.Feedback>
            </Col>
          </Form.Group>)
      }
    }

    return (
      <Form data-testid='setup-form'
        validated={false}
        onDrop={this.onDragDrop}
        onDragOver={dragover_handler}>
        {formItems}
      </Form>
    );
  }
}

// These props all get passed through SetupTab's props,
// so they are defined dynamically as such
ArgsForm.propTypes = {
  args: SetupTab.propTypes.args,
  modulename: SetupTab.propTypes.modulename,
  updateArg: SetupTab.propTypes.updateArg,
  batchUpdateArgs: SetupTab.propTypes.batchUpdateArgs,
  investValidate: SetupTab.propTypes.investValidate,
}

function dragover_handler(event) {
 event.preventDefault();
 event.dataTransfer.dropEffect = "move";
}