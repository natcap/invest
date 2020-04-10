import React from 'react';
import { remote } from 'electron';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import InputGroup from 'react-bootstrap/InputGroup';
import Modal from 'react-bootstrap/Modal';

import { fetchDatastackFromFile } from '../../server_requests';
import { argsValuesFromSpec, boolStringToBoolean } from '../../utils';

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
    this.state = {
      inputs: {}
    }
    this.handleChange = this.handleChange.bind(this);
    this.handleBoolChange = this.handleBoolChange.bind(this);
    this.selectFile = this.selectFile.bind(this);
    this.onDragDrop = this.onDragDrop.bind(this);
    this.setInputDisplay = this.setInputDisplay.bind(this);
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

    for (const argkey in this.props.args) {
      if (this.props.args[argkey].ui_control) {
        this.setInputDisplay(this.props.args[argkey], this.props.args[argkey].value);
      }
    }
  }

  componentDidUpdate(prevProps) {
    if (this.props.args !== prevProps.args) {
      for (const argkey in this.props.args) {
        if (this.props.args[argkey].ui_control) {
          this.setInputDisplay(this.props.args[argkey], this.props.args[argkey].value);
        }
      }
    }
  }

  setInputDisplay(argController, value) {
    let inputState = this.state.inputs;
    argController.ui_control.forEach(dependentKey => {
      if (!value) {
        // hide the dependent args
        inputState[dependentKey] = this.props.args[dependentKey].ui_option
      } else {
        inputState[dependentKey] = undefined
      }

    })
    this.setState({inputs: inputState})
  }

  handleChange(event) {
    /** Handle keystroke changes in text inputs */
    const value = event.target.value;
    const argkey = event.target.name;
    this.props.updateArg(argkey, value);
  }

  handleBoolChange(event) {
    /** Handle boolean changes that emitted strings */
    const value = event.target.value;
    const argkey = event.target.name;
    const boolVal = boolStringToBoolean(value);
    this.props.updateArg(argkey, boolVal)
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
    let argTree = {}
    for (const argname in current_args) {
      if (argname === 'n_workers') { continue }
      const argument = current_args[argname];
      let ArgInput;

      // It's kinda nice if conditionally required inputs are disabled until
      // their condition is satisfied and they become required. The tricky part though
      // is enabling/disabling based only on the internal state of the argument without
      // relying on the state of other args that are part of the condition. If that's not
      // possible then we need to either reproduce some validation.py functionality on
      // the JS side, or add more UI-relevant responses to the validation.py API.
      // This solution mostly works, but will not disable an input that is no longer required,
      // if it has a value present. 
      // if (typeof argument.required === 'string') {
      //   if (argument.validationMessage && argument.validationMessage.includes('is required but has no value')) {
      //     argument['isDisabled'] = false
      //   } else if (!argument.value && !argument.validationMessage){
      //     argument['isDisabled'] = true
      //   }
      // }

      // These types need a text input and a file browser button
      if (['csv', 'vector', 'raster', 'directory'].includes(argument.type)) {
        ArgInput = 
          <Form.Group as={Row} key={argname}>
            <Form.Label column sm="3"  htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="8">
              <InputGroup>
                <AboutModal argument={argument}/>
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
          </Form.Group>
      
      // These types need a text input
      } else if (['freestyle_string', 'number'].includes(argument.type)) {
        // let argClass = argument.isDisabled ? 'arg-' + argument.ui_option : ''
        ArgInput = 
          <Form.Group as={Row} key={argname} className={'arg-' + this.state.inputs[argname]}>
            <Form.Label column sm="3"  htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="4">
              <InputGroup>
                <AboutModal argument={argument}/>
                <Form.Control
                  id={argname}
                  name={argname}
                  type="text" 
                  value={argument.value || ''} // empty string is handled better than `undefined`
                  onChange={this.handleChange}
                  isValid={argument.touched && argument.valid}
                  isInvalid={argument.touched && argument.validationMessage}
                  disabled={this.state.inputs[argname] === 'disable' || false}
                />
                <Form.Control.Feedback type='invalid' id={argname + '-feedback'}>
                  {argument.type + ' : ' + (argument.validationMessage || '')}
                </Form.Control.Feedback>
              </InputGroup>
            </Col>
          </Form.Group>
      
      // Radio select for boolean args
      } else if (argument.type === 'boolean') {
        // argument.value will be type boolean if it's coming from an args dict
        // generated by natcap.invest. But it will be type string if the value
        // is set from this UI, because html forms always submit strings.
        // So, `checked` property must accomodate both types to determine state.
        ArgInput = 
          <Form.Group as={Row} key={argname}>
            <Form.Label column sm="3" htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="8">
              <AboutModal argument={argument}/>
              <Form.Check
                id={argname}
                inline
                type="radio"
                label="Yes"
                value={"true"}
                checked={argument.value}
                onChange={this.handleBoolChange}
                name={argname}
              />
              <Form.Check
                id={argname}
                inline
                type="radio"
                label="No"
                value={"false"}
                checked={!argument.value}
                onChange={this.handleBoolChange}
                name={argname}
              />
            </Col>
          </Form.Group>

      // Dropdown menus for args with options
      } else if (argument.type === 'option_string') {
        ArgInput = 
          <Form.Group as={Row} key={argname}>
            <Form.Label column sm="3" htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="4">
              <InputGroup>
                <AboutModal argument={argument}/>
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
              </InputGroup>
            </Col>
          </Form.Group>
      }


      // This grouping and sorting does not fail if argument.order is undefined 
      // (i.e. it was missing in ARGS_SPEC) for one or more args. 
      // Nevertheless, feels better to fill in a float here.
      if (!argument.order) { argument.order = 100.0 }

      // Fill in a tree-like object where each item is an array of objects
      // like { orderNumber: InputComponent }
      // that share a Math.floor argument.order number.
      const group = Math.floor(argument.order)
      let subArg = {}
      if (argTree[group]) {
        subArg[argument.order] = ArgInput
        argTree[group].push(subArg)
      } else {
        subArg[argument.order] = ArgInput
        argTree[group] = [subArg]
      }
    }

    const sortedArgs = Object.entries(argTree).sort((a, b) => a[0] - b[0])
    formItems = []
    for (const orderkey in sortedArgs) {
      const group = sortedArgs[orderkey][1] // an array of objects
      if (group.length === 1) {
        formItems.push(
          <div className="arg-group" id={orderkey}>
            {Object.values(group[0])[0]}
          </div>)
      } else {
        // a and b are objects keyed by the args order value (float)
        const sortedGroup = group.sort((a, b) => parseFloat(Object.keys(a)[0]) - parseFloat(Object.keys(b)[0]))
        const groupItems = [];
        for (const item in sortedGroup) {
          groupItems.push(Object.values(sortedGroup[item])[0])
        }
        formItems.push(
          <div className="arg-group" id={orderkey}>
            {groupItems}
          </div>)
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

class AboutModal extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      aboutShow: false
    }
    this.handleAboutOpen = this.handleAboutOpen.bind(this);
    this.handleAboutClose = this.handleAboutClose.bind(this);
  }

  handleAboutClose() {
    this.setState({aboutShow: false});
  }

  handleAboutOpen() {
    this.setState({aboutShow: true});
  }

  render() {
    return(
      <React.Fragment>
        <Button  className="mr-3"
          onClick={this.handleAboutOpen}
          variant="outline-info">
          i
        </Button>
        <Modal show={this.state.aboutShow} onHide={this.handleAboutClose}>
          <Modal.Header>
            <Modal.Title>{this.props.argument.name}</Modal.Title>
            </Modal.Header>
          <Modal.Body>{this.props.argument.about}</Modal.Body>
        </Modal>
      </React.Fragment>
    );
  }
}

AboutModal.propTypes = {
  argument: PropTypes.shape({
    name: PropTypes.string,
    about: PropTypes.string
  })
}

function dragover_handler(event) {
 event.preventDefault();
 event.dataTransfer.dropEffect = "move";
}
