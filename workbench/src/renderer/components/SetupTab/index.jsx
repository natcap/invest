import React from 'react';
import PropTypes from 'prop-types';

import Alert from 'react-bootstrap/Alert';
import Container from 'react-bootstrap/Container';
import Spinner from 'react-bootstrap/Spinner';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';
import { MdFolderOpen } from 'react-icons/md';

import Expire from '../Expire';
import Portal from '../Portal';
import ArgsForm from './ArgsForm';
import {
  RunButton, SaveParametersButtons
} from './SetupButtons';
import {
  archiveDatastack,
  fetchDatastackFromFile,
  fetchValidation,
  saveToPython,
  writeParametersToFile
} from '../../server_requests';
import { argsDictFromObject } from '../../utils';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

/** Initialize values of InVEST args based on the model's UI Spec.
 *
 * Values initialize with either a complete args dict, or with empty/default values.
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
      const optionsArray = Array.isArray(argsSpec[argkey].options)
        ? argsSpec[argkey].options
        : Object.keys(argsSpec[argkey].options);
      value = argsDict[argkey]
        || optionsArray[0]; // default to first
      argsDropdownOptions[argkey] = optionsArray;
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
    this._isMounted = false;
    this.validationTimer = null;

    this.state = {
      argsValues: null,
      argsValidation: null,
      argsValid: false,
      argsEnabled: null,
      argsDropdownOptions: null,
      saveAlerts: {},
    };

    this.saveDatastack = this.saveDatastack.bind(this);
    this.savePythonScript = this.savePythonScript.bind(this);
    this.saveJsonFile = this.saveJsonFile.bind(this);
    this.setSaveAlert = this.setSaveAlert.bind(this);
    this.wrapInvestExecute = this.wrapInvestExecute.bind(this);
    this.investValidate = this.investValidate.bind(this);
    this.debouncedValidate = this.debouncedValidate.bind(this);
    this.updateArgTouched = this.updateArgTouched.bind(this);
    this.updateArgValues = this.updateArgValues.bind(this);
    this.batchUpdateArgs = this.batchUpdateArgs.bind(this);
    this.insertNWorkers = this.insertNWorkers.bind(this);
    this.callUISpecFunctions = this.callUISpecFunctions.bind(this);
    this.browseForDatastack = this.browseForDatastack.bind(this);
    this.loadParametersFromFile = this.loadParametersFromFile.bind(this);
  }

  componentDidMount() {
    /*
    * Including the `key` property on SetupTab tells React to
    * re-mount (rather than update & re-render) this component when
    * the key changes. This function does some useful initialization
    * that only needs to compute when this.props.argsSpec changes,
    * not on every re-render.
    */
    this._isMounted = true;
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
      argsDropdownOptions: argsDropdownOptions,
    }, () => {
      this.investValidate();
      this.callUISpecFunctions();
    });
  }

  componentWillUnmount() {
    this._isMounted = false;
    clearTimeout(this.validationTimer);
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
      Object.keys(enabledFunctions).forEach((key) => {
        argsEnabled[key] = enabledFunctions[key](this.state);
      });
      if (this._isMounted) {
        this.setState({ argsEnabled: argsEnabled });
      }
    }

    if (dropdownFunctions) {
      // this model has a dropdown that's dynamically populated
      const { argsDropdownOptions } = this.state;
      await Promise.all(Object.keys(dropdownFunctions).map(async (key) => {
        argsDropdownOptions[key] = await dropdownFunctions[key](this.state);
      }));
      if (this._isMounted) {
        this.setState({ argsDropdownOptions: argsDropdownOptions });
      }
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
      n_workers: { value: this.props.nWorkers },
    };
  }

  /** Save the current invest arguments to a python script via datastack.py API.
   *
   * @param {string} filepath - desired path to the python script
   * @returns {undefined}
   */
  async savePythonScript(filepath) {
    const {
      modelName,
    } = this.props;
    const args = argsDictFromObject(
      this.insertNWorkers(this.state.argsValues)
    );
    const payload = {
      filepath: filepath,
      modelname: modelName,
      args: JSON.stringify(args),
    };
    const response = await saveToPython(payload);
    this.setSaveAlert(response);
  }

  async saveJsonFile(datastackPath) {
    const {
      pyModuleName,
    } = this.props;
    const args = argsDictFromObject(
      this.insertNWorkers(this.state.argsValues)
    );
    const payload = {
      filepath: datastackPath,
      moduleName: pyModuleName,
      relativePaths: false,
      args: JSON.stringify(args),
    };
    const response = await writeParametersToFile(payload);
    this.setSaveAlert(response);
  }

  async saveDatastack(datastackPath) {
    const {
      pyModuleName,
    } = this.props;
    const args = argsDictFromObject(this.state.argsValues);
    const payload = {
      filepath: datastackPath,
      moduleName: pyModuleName,
      args: JSON.stringify(args),
    };
    const key = window.crypto.getRandomValues(new Uint16Array(1))[0].toString();
    this.setSaveAlert('archiving...', key);
    const response = await archiveDatastack(payload);
    this.setSaveAlert(response, key);
  }

  /** State updater for alert messages from various save buttons.
   *
   * @param {string} message - the message to display
   * @param {string} key - a key to uniquely identify each save action,
   *        passed as prop to `Expire` so that it can be aware of whether to,
   *        1. display: because a new save occurred, or
   *        2. not display: on a re-render after `Expire` expired, or
   *        3. update: because 'archiving...' alert changes to final message
   *
   * @returns {undefined}
   */
  setSaveAlert(
    message,
    key = window.crypto.getRandomValues(new Uint16Array(1))[0].toString()
  ) {
    this.setState({
      saveAlerts: { ...this.state.saveAlerts, ...{ [key]: message } }
    });
  }

  async loadParametersFromFile(filepath) {
    const datastack = await fetchDatastackFromFile(filepath);

    if (datastack.module_name === this.props.pyModuleName) {
      this.batchUpdateArgs(datastack.args);
      this.props.switchTabs('setup');
    } else {
      alert( // eslint-disable-line no-alert
        _(`Datastack/Logfile for ${datastack.model_human_name} does not match this model.`)
      );
    }
  }

  async browseForDatastack() {
    const data = await ipcRenderer.invoke(ipcMainChannels.SHOW_OPEN_DIALOG);
    if (data.filePaths.length) {
      this.loadParametersFromFile(data.filePaths[0]);
    }
  }

  wrapInvestExecute() {
    this.props.investExecute(
      argsDictFromObject(this.insertNWorkers(this.state.argsValues))
    );
  }

  /** Update state to indicate that an input was touched.
   *
   * Validation messages only display on inputs that have been touched.
   * Validation state is always displayed, but suppressing the message
   * until an input is touched reduces noise & clutter on a default form.
   *
   * @param {string} key - the invest argument key
   * @returns {undefined}
   */
  updateArgTouched(key) {
    const { argsValues } = this.state;
    if (!argsValues[key].touched) {
      argsValues[key].touched = true;
      this.setState({
        argsValues: argsValues,
      });
    }
  }

  /** Update state with arg values as they change. And validate the args.
   *
   * Updating means:
   * 1) setting the value
   * 2) toggling the enabled/disabled/hidden state of any dependent args
   *
   * @param {string} key - the invest argument key
   * @param {string} value - the invest argument value
   * @returns {undefined}
   */
  updateArgValues(key, value) {
    const { argsValues } = this.state;
    argsValues[key].value = value;
    this.setState({
      argsValues: argsValues,
    }, () => {
      this.debouncedValidate();
      this.callUISpecFunctions();
    });
  }

  /** Update state with values and validate a batch of InVEST arguments.
   *
   * @param {object} argsDict - key: value pairs of InVEST arguments.
   */
  batchUpdateArgs(argsDict) {
    const { argsSpec, uiSpec } = this.props;
    const {
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

  /** Get a debounced version of investValidate.
   *
   * The original validate function will not be called until after the
   * debounced version stops being invoked for N milliseconds.
   *
   * @returns {undefined}
   */
  debouncedValidate() {
    if (this.validationTimer) {
      clearTimeout(this.validationTimer);
    }
    // we want validation to be very responsive,
    // but also to wait for a pause in data entry.
    this.validationTimer = setTimeout(this.investValidate, 200);
  }

  /** Validate an arguments dictionary using the InVEST model's validate function.
   *
   * @returns {undefined}
   */
  async investValidate() {
    const { argsSpec, pyModuleName } = this.props;
    const { argsValues, argsValidation, argsValid } = this.state;
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
      // validated all, so ones left in keyset are either valid
      // or their "required" condition was unmet and so they were
      // not validated and will appear disabled in the UI. Disabled
      // inputs will not display a validation state, so it's okay
      // to simply set all these as valid here.
      keyset.forEach((k) => {
        argsValidation[k].valid = true;
        argsValidation[k].validationMessage = '';
      });
      if (this._isMounted) {
        this.setState({
          argsValidation: argsValidation,
          argsValid: false,
        });
      }

    // B) All args were validated and none were invalid:
    } else {
      keyset.forEach((k) => {
        argsValidation[k].valid = true;
        argsValidation[k].validationMessage = '';
      });
      // It's possible all args were already valid, in which case
      // no validation state has changed and this setState call can
      // be avoided entirely.
      if (!argsValid && this._isMounted) {
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
      argsDropdownOptions,
      saveAlerts,
    } = this.state;
    if (argsValues) {
      const {
        argsSpec,
        userguide,
        sidebarSetupElementId,
        sidebarFooterElementId,
        executeClicked,
        uiSpec,
      } = this.props;

      const SaveAlerts = [];
      Object.keys(saveAlerts).forEach((key) => {
        const message = saveAlerts[key];
        if (message) {
          // Alert won't expire during archiving; will expire 2s after completion
          const alertExpires = (message === 'archiving...') ? 1e7 : 2000;
          SaveAlerts.push(
            <Expire
              key={key}
              className="d-inline"
              delay={alertExpires}
            >
              <Alert variant="success">
                {message}
              </Alert>
            </Expire>
          );
        }
      });

      const buttonText = (
        executeClicked
          ? (
            <span>
              {_('Running')}
              <Spinner
                animation="border"
                size="sm"
                role="status"
                aria-hidden="true"
              />
            </span>
          )
          : <span>{_('Run')}</span>
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
              userguide={userguide}
              updateArgValues={this.updateArgValues}
              updateArgTouched={this.updateArgTouched}
              loadParametersFromFile={this.loadParametersFromFile}
            />
          </Row>
          <Portal elId={sidebarSetupElementId}>
            <OverlayTrigger
              placement="right"
              delay={{ show: 250, hide: 400 }}
              overlay={(
                <Tooltip>
                  {_('Browse to a datastack (.json, .tgz) or InVEST logfile (.txt)')}
                </Tooltip>
              )}
            >
              <Button
                onClick={this.browseForDatastack}
                variant="link"
              >
                <MdFolderOpen className="mr-1" />
                {_('Load parameters from file')}
              </Button>
            </OverlayTrigger>
            <SaveParametersButtons
              savePythonScript={this.savePythonScript}
              saveJsonFile={this.saveJsonFile}
              saveDatastack={this.saveDatastack}
            />
            <React.Fragment>
              {SaveAlerts}
            </React.Fragment>
          </Portal>
          <Portal elId={sidebarFooterElementId}>
            <RunButton
              disabled={!argsValid || executeClicked}
              wrapInvestExecute={this.wrapInvestExecute}
              buttonText={buttonText}
            />
          </Portal>
        </Container>
      );
    }
    // The SetupTab remains disabled in this route, so no need
    // to render anything here.
    return (<div>{_('No args to see here')}</div>);
  }
}

SetupTab.propTypes = {
  pyModuleName: PropTypes.string.isRequired,
  userguide: PropTypes.string.isRequired,
  modelName: PropTypes.string.isRequired,
  argsSpec: PropTypes.objectOf(
    PropTypes.shape({
      name: PropTypes.string,
      type: PropTypes.string,
    })
  ).isRequired,
  uiSpec: PropTypes.shape({
    order: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.string)).isRequired,
    enabledFunctions: PropTypes.objectOf(PropTypes.func),
    dropdownFunctions: PropTypes.objectOf(PropTypes.func),
  }).isRequired,
  argsInitValues: PropTypes.objectOf(PropTypes.string),
  investExecute: PropTypes.func.isRequired,
  nWorkers: PropTypes.string.isRequired,
  sidebarSetupElementId: PropTypes.string.isRequired,
  sidebarFooterElementId: PropTypes.string.isRequired,
  executeClicked: PropTypes.bool.isRequired,
  switchTabs: PropTypes.func.isRequired,
};
