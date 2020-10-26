import React from 'react';
import PropTypes from 'prop-types';

import Container from 'react-bootstrap/Container';
import Spinner from 'react-bootstrap/Spinner';
import Row from 'react-bootstrap/Row';

import Portal from '../Portal';
import ArgsForm from './ArgsForm';
import {
  RunButton, SaveParametersButtons
} from './SetupButtons';
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
    // 'hidden' args should not be assigned default values by the UI.
    // They are hidden because they rarely need to be parameterized
    // by the user, and the default is probably hardcoded into the
    // invest model (e.g. Rec model's 'port' & 'hostname' args).
    if (argkey === 'n_workers'
      || argsSpec[argkey].order === 'hidden') { return; }
    // When initializing with undefined values, assign defaults so that,
    // a) values are handled well by the html inputs and
    // b) the object exported to JSON on "Save" or "Execute" includes defaults.
    let value;
    if (argsSpec[argkey].type === 'boolean') {
      value = argsDict[argkey] || false;
    } else if (argsSpec[argkey].type === 'option_string') {
      value = argsDict[argkey]
        || argsSpec[argkey].validation_options.options[0]; // default to first
    } else {
      value = argsDict[argkey] || '';
    }
    argsValues[argkey] = {
      value: value,
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
    this.insertNWorkers = this.insertNWorkers.bind(this);
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

      // Sort the arg into it's input group
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

  /**
   * n_workers is a special invest arg stored in global settings
   * 
   * @param  {object} argsValues - of the shape returned by `initializeArgValues`.
   * @returns {object} copy of original argsValues with an n_workers property.
   */
  insertNWorkers(argsValues) {
    return {
      ...argsValues,
      n_workers: { value: this.props.nWorkers }
    };
  }

  /** Save the current invest arguments to a python script via datastack.py API.
   *
   * @param {string} filepath - desired path to the python script
   * @returns {undefined}
   */
  savePythonScript(filepath) {
    const { modelName, pyModuleName } = this.props;
    const argsValues = this.insertNWorkers(this.state.argsValues);
    const argsDict = argsDictFromObject(argsValues);
    const payload = {
      filepath: filepath,
      modelname: modelName,
      pyname: pyModuleName,
      args: JSON.stringify(argsDict),
    };
    saveToPython(payload);
  }

  wrapArgsToJsonFile(datastackPath) {
    const argsValues = this.insertNWorkers(this.state.argsValues);
    this.props.argsToJsonFile(
      datastackPath, argsDictFromObject(argsValues)
    );
  }

  wrapInvestExecute() {
    const argsValues = this.insertNWorkers(this.state.argsValues);
    this.props.investExecute(argsDictFromObject(argsValues));
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

  /** Validate an arguments dictionary using the InVEST model's validate function.
   *
   * @param {object} argsValues - of the shape returned by `initializeArgValues`.
   * @returns undefined
   */
  async investValidate(argsValues) {
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
    const {
      argsValues,
      argsValid,
      argsValidation,
      sortedArgKeys,
    } = this.state;
    if (argsValues) {
      const {
        argsSpec,
        pyModuleName,
        sidebarSetupElementId,
        sidebarFooterElementId,
        isRunning,
      } = this.props;

      const buttonText = (
        isRunning
          ? (
            <span>
              Running
              <Spinner
                animation="border"
                size="sm"
                role="status"
                aria-hidden="true"
              />
            </span>
          )
          : <span>Run</span>
      );

      return (
        <Container fluid>
          <Row>
            <ArgsForm
              argsSpec={argsSpec}
              argsValues={argsValues}
              argsValidation={argsValidation}
              sortedArgKeys={sortedArgKeys}
              pyModuleName={pyModuleName}
              updateArgValues={this.updateArgValues}
              batchUpdateArgs={this.batchUpdateArgs}
            />
          </Row>
          <Portal elId={sidebarSetupElementId}>
            <SaveParametersButtons
              savePythonScript={this.savePythonScript}
              wrapArgsToJsonFile={this.wrapArgsToJsonFile}
            />
          </Portal>
          <Portal elId={sidebarFooterElementId}>
            <RunButton
              disabled={!argsValid || isRunning}
              wrapInvestExecute={this.wrapInvestExecute}
              buttonText={buttonText}
            />
          </Portal>
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
  nWorkers: PropTypes.string.isRequired,
};
