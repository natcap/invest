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
      inputs: argsValuesFromSpec(this.props.args)
    }
    this.updateLocalArg = this.updateLocalArg.bind(this);
    this.handleKeystrokes = this.handleKeystrokes.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleBoolChange = this.handleBoolChange.bind(this);
    this.selectFile = this.selectFile.bind(this);
    this.onDragDrop = this.onDragDrop.bind(this);
    // this.setInputDisplay = this.setInputDisplay.bind(this);
  }

  // TODO: now we have some form values set by the local state
  // and some set by props from the parent, that could be confusing.
  updateLocalArg(key, value) {
    /** Store a text input value in local state. */
    let inputs = Object.assign({}, this.state.inputs)
    inputs[key] = value
    this.setState({inputs: inputs});
  }

  handleChange(event) {
    /** Pass input value up to InvestJob for storage & validation.
    *
    * For text fields, this is on the onKeyUp handler, specifically 
    * so that we don't call investValidate over and over while 
    * laying on the backspace key (or any key).
    */
    const value = event.target.value;
    const argkey = event.target.name;
    this.props.updateArg(argkey, value);
  }

  handleKeystrokes(event) {
    /** Handle all keystroke changes in text inputs.
    * 
    * This updates the value of the input in response to any/all
    * keypresses, but does not trigger validation.
    */
    const value = event.target.value;
    const argkey = event.target.name;
    this.updateLocalArg(argkey, value)
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
      this.updateLocalArg(argname, data.filePaths[0])
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

      // These types need a text input, and some also need a file browse button
      if (['csv', 'vector', 'raster', 'directory', 'freestyle_string', 'number'].includes(argument.type)) {
        ArgInput = 
          <Form.Group as={Row} key={argname} className={'arg-' + argument.active_ui_option} data-testid={'group-' + argname}>
            <Form.Label column sm="3"  htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="8">
              <InputGroup>
                <AboutModal argument={argument}/>
                <Form.Control
                  id={argname}
                  name={argname}
                  type="text" 
                  value={this.state.inputs[argname] || ''} // empty string is handled better than `undefined`
                  onChange={this.handleKeystrokes}
                  onKeyUp={this.handleChange}
                  isValid={argument.touched && argument.valid}
                  isInvalid={argument.touched && argument.validationMessage}
                  disabled={argument.active_ui_option === 'disable' || false}
                />
                {
                  ['csv', 'vector', 'raster', 'directory'].includes(argument.type) ?
                  <InputGroup.Append>
                    <Button
                      id={argname}
                      variant="outline-secondary"
                      value={argument.type}  // dialog will limit options to files or dirs accordingly
                      name={argname}
                      onClick={this.selectFile}>
                      Browse
                    </Button>
                  </InputGroup.Append> : <React.Fragment/>
                }
                <Form.Control.Feedback type='invalid' id={argname + '-feedback'}>
                  {argument.type + ' : ' + (argument.validationMessage || '')}
                </Form.Control.Feedback>
              </InputGroup>
            </Col>
          </Form.Group>
      
      // Radio select for boolean args
      } else if (argument.type === 'boolean') {
        // this.state.inputs[argname] will be type boolean if it's coming from this.props.args. 
        // But html forms must emit strings, so there's a special change handler
        // for boolean inputs that coverts to a real boolean.
        // Also, the `checked` property does not treat 'undefined' the same as false,
        // instead React avoids setting the property altogether. Hence, double !! to
        // cast undefined to false.
        ArgInput = 
          <Form.Group as={Row} key={argname} data-testid={'group-' + argname}>
            <Form.Label column sm="3" htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="8">
              <AboutModal argument={argument}/>
              <Form.Check
                id={argname}
                inline
                type="radio"
                label="Yes"
                value={"true"}
                checked={!!argument.value}  // double bang casts undefined to false
                onChange={this.handleBoolChange}
                name={argname}
              />
              <Form.Check
                id={argname}
                inline
                type="radio"
                label="No"
                value={"false"}
                checked={!argument.value}  // undefined becomes true, that's okay
                onChange={this.handleBoolChange}
                name={argname}
              />
            </Col>
          </Form.Group>

      // Dropdown menus for args with options
      } else if (argument.type === 'option_string') {
        ArgInput = 
          <Form.Group as={Row} key={argname} className={'arg-' + argument.active_ui_option} data-testid={'group-' + argname}>
            <Form.Label column sm="3" htmlFor={argname}>{argument.name}</Form.Label>
            <Col sm="4">
              <InputGroup>
                <AboutModal argument={argument}/>
                <Form.Control 
                  id={argname}
                  as='select'
                  name={argname}
                  value={argument.value}
                  onChange={this.handleChange}
                  disabled={argument.active_ui_option === 'disable' || false}>
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
      // like [ { orderNumber: InputComponent }, ... ]
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
          <div className="arg-group" key={orderkey}>
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
          <div className="arg-group" key={orderkey}>
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
