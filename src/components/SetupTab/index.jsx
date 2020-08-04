import React from 'react';
import { remote } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import PropTypes from 'prop-types';

import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import DropdownButton from 'react-bootstrap/DropdownButton';

import { ArgInput } from './ArgInput';
import { SaveParametersButton, SavePythonButton } from '../SaveDropdown'
import { fetchDatastackFromFile, fetchValidation, 
         saveToPython } from '../../server_requests';
import { argsDictFromObject, boolStringToBoolean } from '../../utils';


function toggleDependentInputs(argsSpec, argsValues, argkey) {
  /** Toggle properties that control the display of argument inputs.
  *
  * This function returns a copy of the SetupTab argsValues object
  * after updating the 'ui_option' property of any arguments listed
  * in the 'ui_control' array of `argsSpec[argkey]`.
  *
  * @params {object} argsSpec - merge of an InVEST model's ARGS_SPEC and UI Spec.
  * @params {object} argsValues - of the shape returned by `initializeArgValues`.
  * @params {string} argkey - a key of the argsSpec and argsValues objects
  *    that contains a 'ui_control' property.
  * 
  * @return {object} a copy of `argsValues`
  */
  let updatedValues = Object.assign({}, argsValues)
  argsSpec[argkey].ui_control.forEach(dependentKey => {
    if (!updatedValues[argkey].value) {
      // apply the display option specified in the UI spec
      updatedValues[dependentKey]['ui_option'] = argsSpec[dependentKey].ui_option
    } else {
      updatedValues[dependentKey]['ui_option'] = undefined
    }
  });
  return(updatedValues)
}

function initializeArgValues(argsSpec, argsDict) {
  /** Setup the objects that store InVEST argument values in SetupTab state.
  *
  * One object will store input form values and track if the input has been
  * touched. The other object stores data returned by invest validation.
  * 
  * 
  * @params {object} argsSpec - merge of an InVEST model's ARGS_SPEC.args and UI Spec.
  * @params {object} argsDict - key: value pairs of InVEST model arguments, or {}.
  * 
  * @return {object} to destructure into two args, 
  *   each with the same keys as argsSpec:
  *     {object} argsValues - stores properties that update in response to
  *       user interaction
  *     {object} argsValidation - stores properties that update in response to
  *       validation.
  */
  const initIsEmpty = Object.keys(argsDict).length === 0
  let argsValidation = {};
  let argsValues = {};
  Object.keys(argsSpec).forEach((argkey) => {
    argsValidation[argkey] = {}
    if (argkey === 'n_workers') { return }
    argsValues[argkey] = {
      value: argsDict[argkey],
      touched: !initIsEmpty  // touch them only if initializing with values
    }
  });
  return({ argsValues: argsValues, argsValidation: argsValidation })
}

export class SetupTab extends React.Component {
  /** Renders an arguments form, execute button, and save button.
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

  componentDidMount() {
    /* 
    * Including the `key` property on SetupTab tells React to 
    * re-mount (rather than update & re-render) this component when
    * the key changes. This function does some useful initialization
    * that only needs to compute when this.props.argsSpec changes,
    * not on every re-render.
    */
    if (this.props.argsInitValues) {
      let argTree = {}
      let { argsValues, argsValidation } = initializeArgValues(
        this.props.argsSpec, this.props.argsInitValues)
      
      Object.keys(this.props.argsSpec).forEach((argkey) => {
        if (argkey === 'n_workers' || 
          this.props.argsSpec[argkey].order === 'hidden') { return }
        const argSpec = Object.assign({}, this.props.argsSpec[argkey])
        if (argSpec.ui_control) {
          argsValues = toggleDependentInputs(this.props.argsSpec, argsValues, argkey)
        }

        // This grouping and sorting does not fail if order is undefined 
        // (i.e. it is missing from the UI spec) for one or more args. 
        // Nevertheless, feels better to fill in a float here.
        if (typeof argSpec.order !== 'number') { argSpec.order = 100.0 }

        // Fill in a tree-like object where each item is an array of the args
        // that share a major (Math.floor) order number and should be grouped.
        // Within each array representing a group, items are objects like:
        //  { <precise order number> : <argkey> }
        const group = Math.floor(argSpec.order)
        let subArg = {}
        if (argTree[group]) {  // already have arg(s) in this group
          subArg[argSpec.order] = argkey
          argTree[group].push(subArg)
        } else {
          subArg[argSpec.order] = argkey
          argTree[group] = [subArg]
        }
      });
      // sort the groups by the group number
      const sortedArgs = Object.entries(argTree).sort((a, b) => a[0] - b[0])
      // sort items within the groups
      for (const group in sortedArgs) {
        if (sortedArgs[group].length > 1) {
          // In a group array, first element is the group number
          // Second element is the array of objects keyed by their order number
          sortedArgs[group][1] = sortedArgs[group][1].sort(
            (a, b) => parseFloat(Object.keys(a)[0]) - parseFloat(Object.keys(b)[0]))
        }
      }

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
      pyname: this.props.pyModuleName,
      args: args_dict_string
    }
    saveToPython(payload);
  }

  wrapArgsToJsonFile(datastackPath) {
    this.props.argsToJsonFile(datastackPath, this.state.argsValues)
  }

  wrapInvestExecute() {
    this.props.investExecute(this.state.argsValues)
  }

  updateArgValues(key, value) {
    /** Update state with arg values as they change, and validate the args. 
    *
    * Updating means 
    * 1) setting the value
    * 2) 'touching' the arg - implications for display of validation warnings
    * 3) toggling the enabled/disabled/hidden state of any dependent args
    *
    * @param {string} key - the invest argument key
    * @param {string} value - the invest argument value
    */

    let argsValues = Object.assign({}, this.state.argsValues);
    argsValues[key]['value'] = value;
    argsValues[key]['touched'] = true;

    if (this.props.argsSpec[key].ui_control) {
      const updatedArgsValues = toggleDependentInputs(
        this.props.argsSpec, argsValues, key)
      argsValues = updatedArgsValues;
    }
    this.setState({argsValues: argsValues})
    this.investValidate(argsValues)
  }

  batchUpdateArgs(argsDict) {
    /** Update state with values and validate a batch of InVEST arguments.
    *
    * @params {object} argsDict - key: value pairs of InVEST arguments.
    */
    let { argsValues, argsValidation } = initializeArgValues(
      this.props.argsSpec, argsDict)
    Object.keys(this.props.argsSpec).forEach((argkey) => {
      if (argkey === 'n_workers') { return }
      const argSpec = Object.assign({}, this.props.argsSpec[argkey])
      if (argSpec.ui_control) {
        argsValues = toggleDependentInputs(this.props.argsSpec, argsValues, argkey)
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
      model_module: this.props.pyModuleName,
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
    if (this.state.argsValues) {
      return (
        <Container fluid={true}>
          <ArgsForm
            argsSpec={this.props.argsSpec}
            argsValues={this.state.argsValues}
            argsValidation={this.state.argsValidation}
            sortedArgTree={this.state.sortedArgTree}
            pyModuleName={this.props.pyModuleName}
            updateArgValues={this.updateArgValues}
            batchUpdateArgs={this.batchUpdateArgs}
            investValidate={this.investValidate}
          />
          <Row className="fixed-bottom setup-tab-footer">
            <Col sm="3">
              <Button 
                variant="primary" 
                size="lg"
                onClick={this.wrapInvestExecute}
                disabled={!this.state.argsValid}>
                    Execute
              </Button>
            </Col>
            <Col cm="8">
              <DropdownButton
                id="dropdown-basic-button"
                title="Save Parameters"
                renderMenuOnMount={true}  // w/o this, items inaccessible in jsdom test env
                className="mx-3 float-right">
                <SaveParametersButton
                  wrapArgsToJsonFile={this.wrapArgsToJsonFile}/>
                <SavePythonButton
                  savePythonScript={this.savePythonScript}/>
              </DropdownButton>
            </Col>
          </Row>
        </Container>);
    }
    // The SetupTab remains disabled in this route, so no need
    // to render anything here.
    return(<div>No args to see here</div>)
  }
}

SetupTab.propTypes = {
  pyModuleName: PropTypes.string,
  modelName: PropTypes.string,
  argsSpec: PropTypes.object,
  argsInitValues: PropTypes.object,
  argsToJsonFile: PropTypes.func,
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

    this.handleChange = this.handleChange.bind(this);
    this.handleBoolChange = this.handleBoolChange.bind(this);
    this.selectFile = this.selectFile.bind(this);
    this.onDragDrop = this.onDragDrop.bind(this);
  }

  handleChange(event) {
    /** Pass input value up to SetupTab for storage & validation.
    */
    const value = event.target.value;
    const argkey = event.target.name;
    this.props.updateArgValues(argkey, value)
  }

  handleBoolChange(event) {
    /** Handle changes from boolean inputs that submit strings */
    const value = event.target.value;
    const argkey = event.target.name;
    const boolVal = boolStringToBoolean(value);
    this.props.updateArgValues(argkey, boolVal)
  }

  async selectFile(event) {
    /** Handle clicks on browse-button inputs */
    const argtype = event.target.value;
    const argname = event.target.name;
    const prop = (argtype === 'directory') ? 'openDirectory' : 'openFile'
    // TODO: could add more filters based on argType (e.g. only show .csv)
    const data = await remote.dialog.showOpenDialog({ properties: [prop] })
    if (data.filePaths.length) {
      this.props.updateArgValues(argname, data.filePaths[0]);  // dialog defaults allow only 1 selection
    }
  }

  async onDragDrop(event) {
    /** Handle drag-drop of datastack JSON files and InVEST logfiles */
    event.preventDefault();
    
    const fileList = event.dataTransfer.files;
    if (fileList.length !== 1) {
      throw alert('only drop one file at a time.')
    }
    const datastack = await fetchDatastackFromFile(fileList[0].path);

    if (datastack['module_name'] === this.props.pyModuleName) {
      this.props.batchUpdateArgs(datastack['args']);
    } else {
      throw alert('Parameter/Log file for ' + datastack['module_name'] + ' does not match this model: ' + this.props.pyModuleName)
    }
  }

  render() {
    let formItems = [];
    for (const group of this.props.sortedArgTree) {
      const groupKey = group[0]
      const groupArray = group[1] // the array of argkey objects
      const groupItems = [];
      for (const item of groupArray) {
        const argkey = Object.values(item)[0]
        groupItems.push(
          <ArgInput key={argkey}
            argkey={argkey}
            argSpec={this.props.argsSpec[argkey]}
            value={this.props.argsValues[argkey]['value']}
            touched={this.props.argsValues[argkey]['touched']}
            ui_option={this.props.argsValues[argkey]['ui_option']}
            isValid={this.props.argsValidation[argkey]['valid']}
            validationMessage={this.props.argsValidation[argkey]['validationMessage']}
            handleChange={this.handleChange}
            handleBoolChange={this.handleBoolChange}
            selectFile={this.selectFile}/>)
      }
      formItems.push(
        <div className="arg-group" key={groupKey}>
          {groupItems}
        </div>)
    }

    return (
      <Form data-testid='setup-form' className="args-form"
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
  argsValues:PropTypes.object,
  argsValidation:PropTypes.object,
  sortedArgTree:PropTypes.array,
  updateArgValues:PropTypes.func,
  batchUpdateArgs:PropTypes.func,
  investValidate:PropTypes.func,
  argsSpec: SetupTab.propTypes.argsSpec,
  pyModuleName: SetupTab.propTypes.pyModuleName,
}

function dragover_handler(event) {
 event.preventDefault();
 event.dataTransfer.dropEffect = "move";
}
