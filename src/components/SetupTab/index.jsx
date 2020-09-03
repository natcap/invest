import React from 'react';
import PropTypes from 'prop-types';

import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import DropdownButton from 'react-bootstrap/DropdownButton';

import ArgsForm from './ArgsForm';
import SaveFileButton from '../SaveFileButton';
import { fetchValidation, saveToPython } from '../../server_requests';
import { argsDictFromObject } from '../../utils';

/** Toggle properties that control the display of argument inputs.
 *
 * This function returns a copy of the SetupTab argsValues object
 * after updating the 'ui_option' property of any arguments listed
 * in the 'ui_control' array of `argsSpec[argkey]`.
 *
 * @param {object} argsSpec - merge of an InVEST model's ARGS_SPEC and UI Spec.
 * @param {object} argsValues - of the shape returned by `initializeArgValues`.
 * @param {string} argkey - a key of the argsSpec and argsValues objects
 *    that contains a 'ui_control' property.
 *
 * @returns {object} a copy of `argsValues`
 */
function toggleDependentInputs(argsSpec, argsValues, argkey) {
  const updatedValues = { ...argsValues };
  argsSpec[argkey].ui_control.forEach((dependentKey) => {
    if (!updatedValues[argkey].value) {
      // apply the display option specified in the UI spec
      updatedValues[dependentKey].ui_option = argsSpec[dependentKey].ui_option;
    } else {
      updatedValues[dependentKey].ui_option = undefined;
    }
  });
  return updatedValues;
}

/** Setup the objects that store InVEST argument values in SetupTab state.
 *
 * One object will store input form values and track if the input has been
 * touched. The other object stores data returned by invest validation.
 *
 * @param {object} argsSpec - merge of an InVEST model's ARGS_SPEC.args and UI Spec.
 * @param {object} argsDict - key: value pairs of InVEST model arguments, or {}.
 *
 * @returns {object} to destructure into two args,
 *   each with the same keys as argsSpec:
 *     {object} argsValues - stores properties that update in response to
 *       user interaction
 *     {object} argsValidation - stores properties that update in response to
 *       validation.
 */
function initializeArgValues(argsSpec, argsDict) {
  const initIsEmpty = Object.keys(argsDict).length === 0;
  const argsValidation = {};
  const argsValues = {};
  Object.keys(argsSpec).forEach((argkey) => {
    argsValidation[argkey] = {};
    if (argkey === 'n_workers') { return; }
    argsValues[argkey] = {
      value: argsDict[argkey],
      touched: !initIsEmpty, // touch them only if initializing with values
    };
  });
  return ({ argsValues: argsValues, argsValidation: argsValidation });
}

/** Renders an arguments form, execute button, and save buttons. */
export default class SetupTab extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      argsValues: null,
      argsValidation: {},
      argsValid: false,
      sortedArgKeys: null,
    };

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
    const { argsInitValues, argsSpec, uiSpec } = this.props;
    // extend the args spec with the UI spec
    Object.keys(argsSpec).forEach((key) => {
      Object.assign(argsSpec[key], uiSpec[key]);
    });
    // if (argsInitValues) {
    const argGroups = {};
    let {
      argsValues, argsValidation
    } = initializeArgValues(argsSpec, argsInitValues || {});

    Object.keys(argsSpec).forEach((argkey) => {
      // these argkeys do not get rendered inputs
      if (argkey === 'n_workers'
        || argsSpec[argkey].order === 'hidden') { return; }

      // Update any dependent args in response to this arg's value
      const argumentSpec = { ...argsSpec[argkey] };
      if (argumentSpec.ui_control) {
        argsValues = toggleDependentInputs(argsSpec, argsValues, argkey);
      }

      /** Sort the arg into it's input group */
      // order is optional in the spec, but to be exlplicit about
      // what happens with sorting, defaulting to 100.0.
      if (typeof argumentSpec.order !== 'number') {
        argumentSpec.order = 100.0;
      }
      // Groups are defined by the whole number digits
      const g = Math.floor(argumentSpec.order);
      const orderArgPair = { [argumentSpec.order]: argkey };
      if (argGroups[g]) {
        argGroups[g].push(orderArgPair);
      } else {
        argGroups[g] = [orderArgPair];
      }
    });

    // sort the groups by the group number (keys of argGroups)
    const sortedGroups = Object.entries(argGroups).sort(
      (a, b) => a[0] - b[0]
    );
    // sort args within the groups
    const sortedArgKeys = [];
    sortedGroups.forEach((groupArray) => {
      if (groupArray.length > 1) {
        // [1] is the array of objects keyed by their order number
        const sortedGroup = groupArray[1].sort(
          (a, b) => parseFloat(Object.keys(a)[0]) - parseFloat(Object.keys(b)[0])
        );
        sortedArgKeys.push(
          // drop the order numbers now that argkeys are sorted
          sortedGroup.map((item) => Object.values(item)[0])
        );
      }
    });

    this.setState({
      argsValues: argsValues,
      argsValidation: argsValidation,
      sortedArgKeys: sortedArgKeys,
    }, () => this.investValidate(this.state.argsValues));
    // }
  }

  /** Save the current invest arguments to a python script via datastack.py API.
   *
   * @param {string} filepath - desired path to the python script
   * @returns {undefined}
   */
  savePythonScript(filepath) {
    const argsDict = argsDictFromObject(this.state.argsValues);
    const payload = {
      filepath: filepath,
      modelname: this.props.modelName,
      pyname: this.props.pyModuleName,
      args: JSON.stringify(argsDict),
    };
    saveToPython(payload);
  }

  wrapArgsToJsonFile(datastackPath) {
    this.props.argsToJsonFile(datastackPath, this.state.argsValues);
  }

  wrapInvestExecute() {
    this.props.investExecute(argsDictFromObject(this.state.argsValues));
  }

  /** Update state with arg values as they change. And validate the args.
   *
   * Updating means:
   * 1) setting the value
   * 2) 'touching' the arg - implications for display of validation warnings
   * 3) toggling the enabled/disabled/hidden state of any dependent args
   *
   * @param {string} key - the invest argument key
   * @param {string} value - the invest argument value
   * @returns {undefined}
   */
  updateArgValues(key, value) {
    let { argsValues } = this.state;
    argsValues[key].value = value;
    argsValues[key].touched = true;
    if (this.props.argsSpec[key].ui_control) {
      const updatedArgsValues = toggleDependentInputs(
        this.props.argsSpec, argsValues, key
      );
      argsValues = updatedArgsValues;
    }
    this.setState({ argsValues: argsValues });
    this.investValidate(argsValues);
  }

  batchUpdateArgs(argsDict) {
    /** Update state with values and validate a batch of InVEST arguments.
    *
    * @params {object} argsDict - key: value pairs of InVEST arguments.
    */
    const { argsSpec } = this.props;
    let { argsValues, argsValidation } = initializeArgValues(argsSpec, argsDict);
    Object.keys(argsSpec).forEach((argkey) => {
      if (argkey === 'n_workers') { return; }
      const argumentSpec = Object.assign({}, argsSpec[argkey]);
      if (argumentSpec.ui_control) {
        argsValues = toggleDependentInputs(argsSpec, argsValues, argkey);
      }
    });

    this.setState({
      argsValues: argsValues,
      argsValidation: argsValidation,
    }, () => this.investValidate(this.state.argsValues));
  }

  async investValidate(argsValues) {
    /** Validate an arguments dictionary using the InVEST model's validate function.
     *
     * @param {object} args_dict_string - a JSON.stringify'ed object of model argument
     *    keys and values.
     */
    const { argsSpec, pyModuleName } = this.props;
    // const { argsValidation } = this.state;
    const argsValidation = Object.assign({}, this.state.argsValidation);
    // const argsValid = Object.assign({}, this.state.argsValid);
    const keyset = new Set(Object.keys(argsSpec));
    const payload = {
      model_module: pyModuleName,
      args: JSON.stringify(argsDictFromObject(argsValues)),
    };
    const results = await fetchValidation(payload);

    // A) At least one arg was invalid:
    if (results.length) {
      results.forEach((result) => {
        // Each result is an array of two elements
        // 0: array of arg keys
        // 1: string message that pertains to those args
        const argkeys = result[0];
        const message = result[1];
        argkeys.forEach((key) => {
          argsValidation[key].validationMessage = message;
          argsValidation[key].valid = false;
          keyset.delete(key);
        });
      });
      // validated all, so ones left in keyset are valid
      keyset.forEach((k) => {
        argsValidation[k].valid = true;
        argsValidation[k].validationMessage = '';
      });
      this.setState({
        argsValidation: argsValidation,
        argsValid: false,
      });

    // B) All args were validated and none were invalid:
    } else {
      keyset.forEach((k) => {
        argsValidation[k].valid = true;
        argsValidation[k].validationMessage = '';
      });
      // It's possible all args were already valid, in which case
      // no validation state has changed and this setState call can
      // be avoided entirely.
      if (!this.state.argsValid) {
        this.setState({
          argsValidation: argsValidation,
          argsValid: true,
        });
      }
    }
  }

  render() {
    if (this.state.argsValues) {
      const { argsSpec, pyModuleName } = this.props;
      return (
        <Container fluid>
          <ArgsForm
            argsSpec={argsSpec}
            argsValues={this.state.argsValues}
            argsValidation={this.state.argsValidation}
            sortedArgKeys={this.state.sortedArgKeys}
            pyModuleName={pyModuleName}
            updateArgValues={this.updateArgValues}
            batchUpdateArgs={this.batchUpdateArgs}
          />
          <Row className="fixed-bottom setup-tab-footer">
            <Col sm="3">
              <Button
                variant="primary"
                size="lg"
                onClick={this.wrapInvestExecute}
                disabled={!this.state.argsValid}
              >
                Execute
              </Button>
            </Col>
            <Col cm="8">
              <DropdownButton
                id="dropdown-basic-button"
                title="Save Parameters"
                renderMenuOnMount // w/o this, items inaccessible in jsdom test env
                className="mx-3 float-right"
              >
                <SaveFileButton
                  title="Save parameters to JSON"
                  defaultTargetPath="invest_args.json"
                  func={this.wrapArgsToJsonFile}
                />
                <SaveFileButton
                  title="Save to Python script"
                  defaultTargetPath="execute_invest.python"
                  func={this.savePythonScript}
                />
              </DropdownButton>
            </Col>
          </Row>
        </Container>
      );
    }
    // The SetupTab remains disabled in this route, so no need
    // to render anything here.
    return (<div>No args to see here</div>);
  }
}

SetupTab.propTypes = {
  pyModuleName: PropTypes.string,
  modelName: PropTypes.string,
  argsSpec: PropTypes.objectOf(
    PropTypes.shape({
      name: PropTypes.string,
      type: PropTypes.string,
    }),
  ).isRequired,
  uiSpec: PropTypes.object,
  argsInitValues: PropTypes.object,
  argsToJsonFile: PropTypes.func.isRequired,
  investExecute: PropTypes.func.isRequired,
};
