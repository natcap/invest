import React from 'react';
import { remote } from 'electron';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import InputGroup from 'react-bootstrap/InputGroup';
import Modal from 'react-bootstrap/Modal';
import DropdownButton from 'react-bootstrap/DropdownButton';

import { SaveParametersButton, SavePythonButton } from '../SaveDropdown'
import { fetchDatastackFromFile, fetchValidation } from '../../server_requests';
import { argsValuesFromSpec, argsDictFromObject, boolStringToBoolean } from '../../utils';

export class SetupTab extends React.Component {
  /** Renders an Arguments form and an Execute button
  */
  constructor(props) {
    super(props);
    this.state = {
      argsValues: null,
      argsValidation: {},
      argsValid: false,
      sortedArgTree: null
    }

    this.savePythonScript = this.savePythonScript.bind(this);
    this.wrapArgsToJsonFile = this.wrapArgsToJsonFile.bind(this);
    this.wrapInvestExecute = this.wrapInvestExecute.bind(this);
    this.investValidate = this.investValidate.bind(this);
    this.updateArgValues = this.updateArgValues.bind(this);
    this.batchUpdateArgs = this.batchUpdateArgs.bind(this);
  }

  componentDidMount(props) {
    /* 
    * Including the `key` property on SetupTab tells React to 
    * re-mount (rather than update & re-render) this component when
    * the key changes. This function does some useful initialization
    * that only needs to compute when this.props.argsSpec changes,
    * not on every re-render.
    */
    console.log('setup did mount');
    if (this.props.argsInitValues) {
      let argTree = {}
      let argsValidation = {};
      const argsValues = {};
      Object.keys(this.props.argsSpec).forEach((argkey) => {
        argsValidation[argkey] = {}
        if (argkey === 'n_workers') { return }
        argsValues[argkey] = {
          value: this.props.argsInitValues[argkey],
          touched: false
        }
      });
      Object.keys(this.props.argsSpec).forEach((argkey) => {
        if (argkey === 'n_workers') { return }
        const argSpec = Object.assign({}, this.props.argsSpec[argkey])
        if (argSpec.ui_control) {
          argSpec.ui_control.forEach(dependentKey => {
            if (!argsValues[argkey].value) {
              // hide/disable the dependent args
              argsValues[dependentKey]['active_ui_option'] = this.props.argsSpec[dependentKey].ui_option
            } else {
              argsValues[dependentKey]['active_ui_option'] = undefined
            }
          });
        }

        // This grouping and sorting does not fail if argument.order is undefined 
        // (i.e. it was missing in ARGS_SPEC) for one or more args. 
        // Nevertheless, feels better to fill in a float here.
        if (!argSpec.order) { argSpec.order = 100.0 }

        // Fill in a tree-like object where each item is an array of objects
        // like [ { orderNumber: InputComponent }, ... ]
        // that share a Math.floor argument.order number.
        const group = Math.floor(argSpec.order)
        let subArg = {}
        if (argTree[group]) {
          subArg[argSpec.order] = argkey
          argTree[group].push(subArg)
        } else {
          subArg[argSpec.order] = argkey
          argTree[group] = [subArg]
        }
      });
      const sortedArgs = Object.entries(argTree).sort((a, b) => a[0] - b[0])
      this.setState({
        argsValues: argsValues,
        argsValidation: argsValidation,
        sortedArgTree: sortedArgs },
        () => this.investValidate(this.state.argsValues)
      )
    }
  }

  savePythonScript(filepath) {
    /** Save the current invest arguments to a python script via datastack.py API.
    *
    * @params {string} filepath - desired path to the python script
    */
    const args_dict_string = argsDictFromObject(this.state.argsValues)
    const payload = { 
      filepath: filepath,
      modelname: this.props.modelName,
      pyname: this.props.modelSpec.module,
      args: args_dict_string
    }
    saveToPython(payload);
  }

  wrapArgsToJsonFile(datastackPath) {
    this.props.argsToJsonFile(datastackPath, this.state.argsValues)
  }

  wrapInvestExecute(argsValues) {
    this.props.investExecute(this.state.argsValues)
  }

  updateArgValues(argsValues) {
    this.setState({argsValues: argsValues})
  }

  batchUpdateArgs(argsDict) {
    // TODO: some of this code is repeated in componentDidMount 
    Object.keys(this.props.argsSpec).forEach((argkey) => {
      argsValidation[argkey] = {}
      if (argkey === 'n_workers') { return }      
      argsValues[argkey] = {
        value: argsDict[argkey],
        touched: true
      }
    });
    Object.keys(this.props.argsSpec).forEach((argkey) => {
      let argSpec = this.props.argsSpec[argkey]
      if (argSpec.ui_control) {
        argSpec.ui_control.forEach(dependentKey => {
          if (!argsValues[argkey].value) {
            // hide/disable the dependent args
            argsValues[dependentKey]['active_ui_option'] = argsSpec[dependentKey].ui_option
          } else {
            argsValues[dependentKey]['active_ui_option'] = undefined
          }
        });
      }
    })
    this.setState({
      argsValues: argsValues,
      argsValidation: argsValidation},
      () => this.investValidate(this.state.argsValues)
    )
  }

  async investValidate(argsValues, limit_to) {
    /** Validate an arguments dictionary using the InVEST model's validate function.
    *
    * @param {object} args_dict_string - a JSON.stringify'ed object of model argument
    *    keys and values.
    * @param {string} limit_to - an argument key if validation should be limited only
    *    to that argument.
    */
    let argsSpec = JSON.parse(JSON.stringify(this.props.argsSpec));
    let argsValidation = Object.assign({}, this.state.argsValidation);
    let keyset = new Set(Object.keys(argsSpec));
    let payload = { 
      model_module: this.props.modelSpec.module,
      args: argsDictFromObject(argsValues)
    };

    // TODO: is there a use-case for `limit_to`? 
    // Right now we're never calling validate with a limit_to,
    // but we have an awful lot of logic here to cover it.
    if (limit_to) {
      payload['limit_to'] = limit_to
    }

    const results = await fetchValidation(payload);

    // A) At least one arg was invalid:
    if (results.length) { 

      results.forEach(result => {
        // Each result is an array of two elements
        // 0: array of arg keys
        // 1: string message that pertains to those args
        const argkeys = result[0];
        const message = result[1];
        argkeys.forEach(key => {
          argsValidation[key]['validationMessage'] = message
          argsValidation[key]['valid'] = false
          keyset.delete(key);
        })
      });
      if (!limit_to) {  // validated all, so ones left in keyset are valid
        keyset.forEach(k => {
          argsValidation[k]['valid'] = true
          argsValidation[k]['validationMessage'] = ''
        })
      }
      this.setState({
        argsValidation: argsValidation,
        argsValid: false
      });

    // B) All args were validated and none were invalid:
    } else if (!limit_to) {
      
      keyset.forEach(k => {
        argsValidation[k]['valid'] = true
        argsValidation[k]['validationMessage'] = ''
      })
      // It's possible all args were already valid, in which case
      // it's nice to avoid the re-render that this setState call
      // triggers. Although only the Viz app components re-render 
      // in a noticeable way. Due to use of redux there?
      if (!this.state.argsValid) {
        this.setState({
          argsValidation: argsValidation,
          argsValid: true
        })
      }

    // C) Limited args were validated and none were invalid
    } else if (limit_to) {

      argsValidation[limit_to]['valid'] = true
      // TODO: this defeats the purpose of using limit_to in the first place:
      // This could be the last arg that needed to go valid,
      // in which case we can trigger a full args_dict validation
      // without any `limit_to`, in order to properly set state.argsValid
      this.setState({ argsValidation: argsValidation },
        () => {
          let argIsValidArray = [];
          for (const key in argsValidation) {
            argIsValidArray.push(argsValidation[key]['valid'])
          }
          if (argIsValidArray.every(Boolean)) {
            this.investValidate(argsValues);
          }
        }
      );
    }
  }

  render () {
    console.log('setup did render')
    console.log(this.props.argsSpec)
    console.log(this.state.argsValues)
    console.log(this.state.sortedArgTree)
    if (this.state.argsValues) {
      return (
        <div>
          <ArgsForm
            argsSpec={this.props.argsSpec}
            argsValues={this.state.argsValues}
            argsValidation={this.state.argsValidation}
            sortedArgTree={this.state.sortedArgTree}
            modulename={this.props.modulename}
            updateArgValues={this.updateArgValues}
            batchUpdateArgs={this.batchUpdateArgs}
            investValidate={this.investValidate}
          />
          <Button 
            variant="primary" 
            size="lg"
            onClick={this.wrapInvestExecute}
            disabled={!this.state.argsValid}>
                Execute
          </Button>
          <DropdownButton id="dropdown-basic-button" title="Save " className="mx-3">
            <SaveParametersButton
              wrapArgsToJsonFile={this.wrapArgsToJsonFile}/>
            <SavePythonButton
              savePythonScript={this.savePythonScript}/>
          </DropdownButton>
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

class ArgsForm extends React.PureComponent {
  /** Renders an HTML input for each invest argument passed in props.args.
  *
  * Values of input fields inherit from parent components state.args, and so 
  * change handlers for the inputs in this component update their values
  * by calling parent component methods that call parent's setState.
  */

  constructor(props) {
    super(props);

    this.updateLocalArg = this.updateLocalArg.bind(this);
    // this.handleKeystrokes = this.handleKeystrokes.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleBoolChange = this.handleBoolChange.bind(this);
    this.selectFile = this.selectFile.bind(this);
    this.onDragDrop = this.onDragDrop.bind(this);
  }

  

  updateLocalArg(key, value) {
    /** Update this.state.args and validate the args. 
    *
    * Updating means 
    * 1) setting the value
    * 2) 'touching' the arg - implications for display of validation warnings
    * 3) updating the enabled/disabled/hidden state of any dependent args
    *
    * @param {string} key - the invest argument key
    * @param {string} value - the invest argument value
    * @param {boolean} touch - whether this function should mark arguments as
    * 'touched', which controls whether validation messages display. Usually
    * desireable, except when this function is used for initial render of an
    * input form, when it's better to not display the arguments as 'touched'.
    */

    const argsSpec = JSON.parse(JSON.stringify(this.props.argsSpec));
    const argsValues = Object.assign({}, this.props.argsValues);
    argsValues[key]['value'] = value;
    argsValues[key]['touched'] = true;

    if (argsSpec[key].ui_control) {
      argsSpec[key].ui_control.forEach(dependentKey => {
        if (!value) {
          // hide/disable the dependent args
          argsValues[dependentKey]['active_ui_option'] = argsSpec[dependentKey].ui_option
        } else {
          argsValues[dependentKey]['active_ui_option'] = undefined
        }
      });
    }

    this.props.updateArgValues(argsValues)
    this.props.investValidate(argsValues)
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
    this.updateLocalArg(argkey, value)
  }

  // handleKeystrokes(event) {
  //   * Handle all keystroke changes in text inputs.
  //   * 
  //   * This updates the value of the input in response to any/all
  //   * keypresses, but does not trigger validation.
    
  //   const value = event.target.value;
  //   const argkey = event.target.name;
  //   this.updateLocalArg(argkey, value)
  // }

  handleBoolChange(event) {
    /** Handle boolean changes that emitted strings */
    const value = event.target.value;
    const argkey = event.target.name;
    const boolVal = boolStringToBoolean(value);
    this.updateLocalArg(argkey, boolVal)
  }

  async selectFile(event) {
    /** Handle clicks on browse-button inputs */
    const argtype = event.target.value;
    const argname = event.target.name;
    const prop = (argtype === 'directory') ? 'openDirectory' : 'openFile'
    // TODO: could add more filters based on argType (e.g. only show .csv)
    const data = await remote.dialog.showOpenDialog({ properties: [prop] })
    if (data.filePaths.length) {
      this.updateLocalArg(argname, data.filePaths[0]);  // dialog defaults allow only 1 selection
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
    let formItems = [];
    for (const orderkey in this.props.sortedArgTree) {
      const group = this.props.sortedArgTree[orderkey][1] // an array of objects
      if (group.length === 1) {
        const argkey = Object.values(group[0])[0]
        formItems.push(
          <div className="arg-group" key={orderkey}>
            <ArgInput
              argkey={argkey}
              argSpec={this.props.argsSpec[argkey]}
              argState={this.props.argsValues[argkey]}
              argValidationState={this.props.argsValidation[argkey]}
              handleChange={this.handleChange}
              handleBoolChange={this.handleBoolChange}
              selectFile={this.selectFile}
            />
          </div>)
      } else {
        // a and b are objects keyed by the args order value (float)
        const sortedGroup = group.sort((a, b) => parseFloat(Object.keys(a)[0]) - parseFloat(Object.keys(b)[0]))
        const groupItems = [];
        for (const item in sortedGroup) {
          const argkey = Object.values(sortedGroup[item])[0]
          groupItems.push(
            <ArgInput
              argkey={argkey}
              argSpec={this.props.argsSpec[argkey]}
              value={this.props.argsValues[argkey]['value']}
              touched={this.props.argsValues[argkey]['touched']}
              active_ui_option={this.props.argsValues[argkey]['active_ui_option']}
              isValid={this.props.argsValidation[argkey]['valid']}
              validationMessage={this.props.argsValidation[argkey]['validationMessage']}
              handleChange={this.handleChange}
              handleBoolChange={this.handleBoolChange}
              selectFile={this.selectFile}/>)
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

class ArgInput extends React.PureComponent {
  constructor(props) {
    super(props)
  }

  render() {
    const argkey = this.props.argkey
    const argSpec = this.props.argSpec
    let ArgInput;

    // These types need a text input, and some also need a file browse button
    if (['csv', 'vector', 'raster', 'directory', 'freestyle_string', 'number'].includes(argSpec.type)) {
      ArgInput = 
        <Form.Group as={Row} key={argkey} className={'arg-' + this.props.active_ui_option} data-testid={'group-' + argkey}>
          <Form.Label column sm="3"  htmlFor={argkey}>{argSpec.name}</Form.Label>
          <Col sm="8">
            <InputGroup>
              <AboutModal argument={argSpec}/>
              <Form.Control
                id={argkey}
                name={argkey}
                type="text" 
                value={this.props.value || ''} // empty string is handled better than `undefined`
                onChange={this.props.handleChange}
                // onKeyUp={this.handleChange}
                isValid={this.props.touched && this.props.isValid}
                isInvalid={this.props.touched && this.props.validationMessage}
                disabled={this.props.active_ui_option === 'disable' || false}
              />
              {
                ['csv', 'vector', 'raster', 'directory'].includes(argSpec.type) ?
                <InputGroup.Append>
                  <Button
                    id={argkey}
                    variant="outline-secondary"
                    value={argSpec.type}  // dialog will limit options to files or dirs accordingly
                    name={argkey}
                    onClick={this.props.selectFile}>
                    Browse
                  </Button>
                </InputGroup.Append> : <React.Fragment/>
              }
              <Form.Control.Feedback type='invalid' id={argkey + '-feedback'}>
                {argSpec.type + ' : ' + (this.props.validationMessage || '')}
              </Form.Control.Feedback>
            </InputGroup>
          </Col>
        </Form.Group>
    
    // Radio select for boolean args
    } else if (argSpec.type === 'boolean') {
      // this.state[argname] will be type boolean if it's coming from this.props.args. 
      // But html forms must emit strings, so there's a special change handler
      // for boolean inputs that coverts to a real boolean.
      // Also, the `checked` property does not treat 'undefined' the same as false,
      // instead React avoids setting the property altogether. Hence, double !! to
      // cast undefined to false.
      ArgInput = 
        <Form.Group as={Row} key={argkey} data-testid={'group-' + argkey}>
          <Form.Label column sm="3" htmlFor={argkey}>{argSpec.name}</Form.Label>
          <Col sm="8">
            <AboutModal argument={argSpec}/>
            <Form.Check
              id={argkey}
              inline
              type="radio"
              label="Yes"
              value={"true"}
              checked={!!this.props.value}  // double bang casts undefined to false
              onChange={this.props.handleBoolChange}
              name={argkey}
            />
            <Form.Check
              id={argkey}
              inline
              type="radio"
              label="No"
              value={"false"}
              checked={!this.props.value}  // undefined becomes true, that's okay
              onChange={this.props.handleBoolChange}
              name={argkey}
            />
          </Col>
        </Form.Group>

    // Dropdown menus for args with options
    } else if (argSpec.type === 'option_string') {
      ArgInput = 
        <Form.Group as={Row} key={argkey} className={'arg-' + this.props.active_ui_option} data-testid={'group-' + argkey}>
          <Form.Label column sm="3" htmlFor={argkey}>{argSpec.name}</Form.Label>
          <Col sm="4">
            <InputGroup>
              <AboutModal argument={argSpec}/>
              <Form.Control 
                id={argkey}
                as='select'
                name={argkey}
                value={this.props.value}
                onChange={this.props.handleChange}
                disabled={this.props.active_ui_option === 'disable' || false}>
                {argSpec.validation_options.options.map(opt =>
                  <option value={opt} key={opt}>{opt}</option>
                )}
              </Form.Control>
              <Form.Control.Feedback type='invalid' id={argkey + '-feedback'}>
                {argSpec.type + ' : ' + (this.props.validationMessage || '')}
              </Form.Control.Feedback>
            </InputGroup>
          </Col>
        </Form.Group>
    }
    return(ArgInput)
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

class AboutModal extends React.PureComponent {
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
