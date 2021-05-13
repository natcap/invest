import React from 'react';
import PropTypes from 'prop-types';
import { ipcRenderer } from 'electron';

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

/** Setup the objects that store InVEST argument values in SetupTab state.
 *
 * One object will store input form values and track if the input has been
 * touched. The other object stores data returned by invest validation.
 *
 * @param {object} argsSpec - an InVEST model's ARGS_SPEC.args 
 * @param {object} uiSpec - the model's UI Spec.
 * @param {object} argsDict - key: value pairs of InVEST model arguments, or {}.
 *
 * @returns {object} to destructure into two args,
 *   each with the same keys as argsSpec:
 *     {object} argsValues - stores properties that update in response to
 *       user interaction
 *     {object} argsDropdownOptions - stores lists of dropdown options for
 *       args of type 'option_string'.
 */
function initializeArgValues(argsSpec, uiSpec, argsDict) {
  const initIsEmpty = Object.keys(argsDict).length === 0;
  const argsValues = {};
  const argsDropdownOptions = {};
  uiSpec.order.flat().forEach((argkey) => {
    // When initializing with undefined values, assign defaults so that,
    // a) values are handled well by the html inputs and
    // b) the object exported to JSON on "Save" or "Execute" includes defaults.
    let value;
    if (argsSpec[argkey].type === 'boolean') {
      value = argsDict[argkey] || false;
    } else if (argsSpec[argkey].type === 'option_string') {
      value = argsDict[argkey]
        || argsSpec[argkey].validation_options.options[0]; // default to first
      argsDropdownOptions[argkey] = argsSpec[argkey].validation_options.options;
    } else {
      value = argsDict[argkey] || '';
    }
    argsValues[argkey] = {
      value: value,
      touched: !initIsEmpty, // touch them only if initializing with values
    };
  });
  return ({
    argsValues: argsValues,
    argsDropdownOptions: argsDropdownOptions,
  });
}

/** Renders an arguments form, execute button, and save buttons. */
export default class SetupTab extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      argsValues: null,
      argsValidation: null,
      argsValid: false,
      argsEnabled: null,
      argsDropdownOptions: null
    };

    this.savePythonScript = this.savePythonScript.bind(this);
    this.wrapArgsToJsonFile = this.wrapArgsToJsonFile.bind(this);
    this.wrapInvestExecute = this.wrapInvestExecute.bind(this);
    this.investValidate = this.investValidate.bind(this);
    this.updateArgValues = this.updateArgValues.bind(this);
    this.batchUpdateArgs = this.batchUpdateArgs.bind(this);
    this.insertNWorkers = this.insertNWorkers.bind(this);
    this.callUISpecFunctions = this.callUISpecFunctions.bind(this);
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

    const {
      argsValues,
      argsDropdownOptions,
    } = initializeArgValues(argsSpec, uiSpec, argsInitValues || {});

    // map each arg to an empty object, to fill in later
    // here we use the argsSpec because it includes all args, even ones like
    // n_workers, which do get validated.
    const argsValidation = Object.keys(argsSpec).reduce((acc, argkey) => {
      acc[argkey] = {};
      return acc;
    }, {});
    // here we only use the keys in uiSpec.order because args that
    // aren't displayed in the form don't need an enabled/disabled state.
    // all args default to being enabled
    const argsEnabled = uiSpec.order.flat().reduce((acc, argkey) => {
      acc[argkey] = true;
      return acc;
    }, {});

    this.setState({
      argsValues: argsValues,
      argsValidation: argsValidation,
      argsEnabled: argsEnabled,
      argsDropdownOptions: argsDropdownOptions
    }, () => {
      this.investValidate();
      this.callUISpecFunctions();
    });
  }

  /**
   * Call functions from the UI spec to determine the enabled/disabled 
   * state and dropdown options for each input, if applicable.
   *
   * @returns {undefined}
   */
  async callUISpecFunctions() {
    const { enabledFunctions, dropdownFunctions } = this.props.uiSpec;

    if (enabledFunctions) {
      // this model has some fields that are conditionally enabled
      const { argsEnabled } = this.state;
      for (const key in enabledFunctions) {
        // evaluate the function to determine if it should be enabled
        argsEnabled[key] = await enabledFunctions[key](this.state);
      }
      this.setState({ argsEnabled: argsEnabled });
    }

    if (dropdownFunctions) {
      // this model has a dropdown that's dynamically populated
      const { argsDropdownOptions } = this.state;
      for (const key in dropdownFunctions) {
        // evaluate the function to get a list of dropdown options
        argsDropdownOptions[key] = await dropdownFunctions[key](this.state);
      }
      this.setState({argsDropdownOptions: argsDropdownOptions});
    }
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
    const args = argsDictFromObject(argsValues);
    // TODO: why did I change this to use IPC? why not call argsToJsonFile?
    ipcRenderer.invoke(
      'invest-args-to-json',
      datastackPath,
      this.props.pyModuleName,
      args
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
    const { argsValues } = this.state;
    argsValues[key].value = value;
    argsValues[key].touched = true;

    this.setState({
      argsValues: argsValues
    }, () => {
      this.investValidate();
      this.callUISpecFunctions();
    });
  }

  /** Update state with values and validate a batch of InVEST arguments.
   *
   * @params {object} argsDict - key: value pairs of InVEST arguments.
   */
  batchUpdateArgs(argsDict) {
    const { argsSpec, uiSpec } = this.props;
    let {
      argsValues,
      argsDropdownOptions,
    } = initializeArgValues(argsSpec, uiSpec, argsDict);

    this.setState({
      argsValues: argsValues,
      argsDropdownOptions: argsDropdownOptions,
    }, () => {
      this.investValidate();
      this.callUISpecFunctions();
    });
  }

  /** Validate an arguments dictionary using the InVEST model's validate function.
   *
   * @returns undefined
   */
  async investValidate() {
    const { argsSpec, pyModuleName } = this.props;
    const { argsValues, argsValidation } = this.state;
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
        const argkeys = result[0]; // array of arg keys
        const message = result[1]; // string that describes those args
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
      argsEnabled,
      argsDropdownOptions
    } = this.state;
    if (argsValues) {
      const {
        argsSpec,
        pyModuleName,
        sidebarSetupElementId,
        sidebarFooterElementId,
        isRunning,
        uiSpec
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
              argsEnabled={argsEnabled}
              argsDropdownOptions={argsDropdownOptions}
              argsOrder={uiSpec.order}
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
  pyModuleName: PropTypes.string.isRequired,
  modelName: PropTypes.string.isRequired,
  argsSpec: PropTypes.objectOf(
    PropTypes.shape({
      name: PropTypes.string,
      type: PropTypes.string,
    }),
  ).isRequired,
  uiSpec: PropTypes.object,
  argsInitValues: PropTypes.object,
  investExecute: PropTypes.func.isRequired,
  nWorkers: PropTypes.string.isRequired,
  sidebarSetupElementId: PropTypes.string.isRequired,
  sidebarFooterElementId: PropTypes.string.isRequired,
  isRunning: PropTypes.bool.isRequired,
};
