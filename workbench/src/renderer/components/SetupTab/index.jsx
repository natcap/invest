import React from 'react';
import PropTypes from 'prop-types';
import { withTranslation } from 'react-i18next';

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
import SaveAsModal from '../SaveAsModal';
import {
  archiveDatastack,
  fetchDatastackFromFile,
  fetchValidation,
  fetchArgsEnabled,
  getDynamicDropdowns,
  saveToPython,
  writeParametersToFile
} from '../../server_requests';
import { argsDictFromObject } from '../../utils';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;
const { logger } = window.Workbench;

/** Initialize values of InVEST args based on the model's UI Spec.
 *
 * Values initialize with either a complete args dict, or with empty/default values.
 *
 * @param {object} argsSpec - an InVEST model's MODEL_SPEC.args
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
  console.log('argsvalues');
  const initIsEmpty = Object.keys(argsDict).length === 0;
  const argsValues = {};
  const argsDropdownOptions = {};
  console.log(argsSpec);
  const argkeys = [];
  uiSpec.order.flat().forEach((group) => {
    console.log(group);
    if (typeof group === 'string') {
      console.log('string')
      argkeys.push(group);
    } else {
      console.log('object')
      argkeys.push(...Object.values(group)[0]);
    }
  });
  console.log(argkeys);
  argkeys.forEach((argkey) => {
    console.log(argkey);
    // When initializing with undefined values, assign defaults so that,
    // a) values are handled well by the html inputs and
    // b) the object exported to JSON on "Save" or "Execute" includes defaults.
    let value;
    if (argsSpec[argkey].type === 'boolean') {
      value = argsDict[argkey] || false;
    } else if (argsSpec[argkey].type === 'option_string') {
      if (argsDict[argkey]) {
        value = argsDict[argkey];
      } else { // default to first
        if (Array.isArray(argsSpec[argkey].options)) {
          value = argsSpec[argkey].options[0];
        } else {
          value = Object.keys(argsSpec[argkey].options)[0];
        }
      }
      argsDropdownOptions[argkey] = argsSpec[argkey].options;
    // } else if (uiSpec.form_tables.includes(argkey)) {
    //   value = {};
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
class SetupTab extends React.Component {
  constructor(props) {
    super(props);
    this._isMounted = false;
    this.validationTimer = null;
    this.enabledTimer = null;
    this.dropdownTimer = null;

    this.state = {
      argsValues: null,
      argsValidation: null,
      argsValid: false,
      argsEnabled: null,
      argsDropdownOptions: null,
      saveAlerts: {},
      scrollEventCount: 0,
    };

    this.saveDatastack = this.saveDatastack.bind(this);
    this.savePythonScript = this.savePythonScript.bind(this);
    this.saveJsonFile = this.saveJsonFile.bind(this);
    this.setSaveAlert = this.setSaveAlert.bind(this);
    this.wrapInvestExecute = this.wrapInvestExecute.bind(this);
    this.investValidate = this.investValidate.bind(this);
    this.debouncedValidate = this.debouncedValidate.bind(this);
    this.investArgsEnabled = this.investArgsEnabled.bind(this);
    this.debouncedArgsEnabled = this.debouncedArgsEnabled.bind(this);
    this.updateArgTouched = this.updateArgTouched.bind(this);
    this.updateArgValues = this.updateArgValues.bind(this);
    this.batchUpdateArgs = this.batchUpdateArgs.bind(this);
    this.browseForDatastack = this.browseForDatastack.bind(this);
    this.loadParametersFromFile = this.loadParametersFromFile.bind(this);
    this.triggerScrollEvent = this.triggerScrollEvent.bind(this);
    this.callDropdownFunctions = this.callDropdownFunctions.bind(this);
    this.debouncedDropdownFunctions = this.debouncedDropdownFunctions.bind(this);
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
      this.investArgsEnabled();
    });
  }

  componentWillUnmount() {
    this._isMounted = false;
    clearTimeout(this.validationTimer);
    clearTimeout(this.enabledTimer);
    clearTimeout(this.dropdownTimer);
  }

  /**
   * Update scrollEventCount, which is being observed by a useEffect
   * in ArgInput in order to trigger horizontal scrolls in text boxes.
   * Call this during events that fill text boxes by means other than typing.
   *
   * @returns {undefined}
   */
  triggerScrollEvent() {
    this.setState((prevState, props) => ({
      scrollEventCount: prevState.updateEvent + 1
    }));
  }

  /** Save the current invest arguments to a python script via datastack.py API.
   *
   * @param {string} filepath - desired path to the python script
   * @returns {undefined}
   */
  async savePythonScript(filepath) {
    const {
      modelId,
    } = this.props;
    const args = argsDictFromObject(this.state.argsValues);
    const payload = {
      filepath: filepath,
      modelname: modelId,
      args: JSON.stringify(args),
    };
    const response = await saveToPython(payload);
    this.setSaveAlert(response);
  }

  async saveJsonFile(datastackPath, relativePaths) {
    const {
      pyModuleName,
    } = this.props;
    const args = argsDictFromObject(this.state.argsValues);
    const payload = {
      filepath: datastackPath,
      moduleName: pyModuleName,
      relativePaths: relativePaths,
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
    const { pyModuleName, switchTabs, t } = this.props;
    let datastack;
    try {
      if (filepath.endsWith('gz')) {  // .tar.gz, .tgz
        const extractLocation = await ipcRenderer.invoke(
          ipcMainChannels.SHOW_SAVE_DIALOG,
          { title: t('Choose location to extract archive') }
        );
        if (extractLocation.filePath) {
          datastack = await fetchDatastackFromFile({
            filepath: filepath,
            extractPath: extractLocation.filePath});
        } else {
          return;
        }
      } else {
          datastack = await fetchDatastackFromFile({ filepath: filepath });
      }
    } catch (error) {
      logger.error(error);
      alert( // eslint-disable-line no-alert
        t(
          'No InVEST model data can be parsed from the file:\n {{filepath}}',
          { filepath: filepath }
        )
      );
      return;
    }
    if (datastack.module_name === pyModuleName) {
      this.batchUpdateArgs(datastack.args);
      switchTabs('setup');
      this.triggerScrollEvent();
    } else {
      alert( // eslint-disable-line no-alert
        t(
          'Datastack/Logfile for {{modelName}} does not match this model.',
          { modelName: datastack.model_human_name }
        )
      );
    }
  }

  async browseForDatastack() {
    const data = await ipcRenderer.invoke(ipcMainChannels.SHOW_OPEN_DIALOG);
    if (!data.canceled) {
      this.loadParametersFromFile(data.filePaths[0]);
    }
  }

  wrapInvestExecute() {
    this.props.investExecute(
      argsDictFromObject(this.state.argsValues)
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
    console.log('update touched', key);
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
    const { uiSpec } = this.props;
    const { argsValues } = this.state;
    console.log(argsValues, key);
    argsValues[key].value = value;
    this.setState({
      argsValues: argsValues,
    }, () => {
      this.debouncedValidate();
      this.debouncedArgsEnabled();
      if (uiSpec.dropdown_functions) {
        this.debouncedDropdownFunctions();
      }
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
      this.investArgsEnabled();
    });
  }

  /** Get a debounced version of investArgsEnabled.
   *
   * The original function will not be called until after the
   * debounced version stops being invoked for 200 milliseconds.
   *
   * @returns {undefined}
   */
  debouncedArgsEnabled() {
    if (this.enabledTimer) {
      clearTimeout(this.enabledTimer);
    }
    // we want this check to be very responsive,
    // but also to wait for a pause in data entry.
    this.enabledTimer = setTimeout(this.investArgsEnabled, 200);
  }

  /** Set the enabled/disabled status of args.
   *
   * @returns {undefined}
   */
  async investArgsEnabled() {
    const { pyModuleName, modelId } = this.props;
    const { argsValues } = this.state;

    if (this._isMounted) {
      this.setState({
        argsEnabled: await fetchArgsEnabled({
          modelId: modelId,
          model_module: pyModuleName,
          args: JSON.stringify(argsDictFromObject(argsValues)),
        }),
      });
    }
  }

  debouncedDropdownFunctions() {
    if (this.dropdownTimer) {
      clearTimeout(this.dropdownTimer);
    }
    // we want this check to be very responsive,
    // but also to wait for a pause in data entry.
    this.dropdownTimer = setTimeout(this.callDropdownFunctions, 200);
  }

  /** Call endpoint to get dynamically populated dropdown options.
   *
   * @returns {undefined}
   */
  async callDropdownFunctions() {
    const { pyModuleName } = this.props;
    const { argsValues, argsDropdownOptions } = this.state;
    const payload = {
      model_module: pyModuleName,
      args: JSON.stringify(argsDictFromObject(argsValues)),
    };
    const results = await getDynamicDropdowns(payload);
    Object.keys(results).forEach((argkey) => {
      argsDropdownOptions[argkey] = results[argkey];
    });
    this.setState({ argsDropdownOptions: argsDropdownOptions });
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
    const { argsSpec, pyModuleName, modelId } = this.props;
    const { argsValues, argsValidation, argsValid } = this.state;
    const keyset = new Set(Object.keys(argsSpec));
    const payload = {
      modelId: modelId,
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
      scrollEventCount,
    } = this.state;
    const { t } = this.props;
    if (argsValues) {
      const {
        argsSpec,
        userguide,
        sidebarSetupElementId,
        sidebarFooterElementId,
        executeClicked,
        uiSpec,
        modelId,
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
              {t('Running')}
              <Spinner
                animation="border"
                size="sm"
                role="status"
                aria-hidden="true"
              />
            </span>
          )
          : <span>{t('Run')}</span>
      );
      return (
        <Container fluid>
          <Row>
            <ArgsForm
              argsSpec={argsSpec}
              uiSpec={uiSpec}
              argsValues={argsValues}
              argsValidation={argsValidation}
              argsEnabled={argsEnabled}
              argsDropdownOptions={argsDropdownOptions}
              argsOrder={uiSpec.order}
              userguide={userguide}
              updateArgValues={this.updateArgValues}
              updateArgTouched={this.updateArgTouched}
              loadParametersFromFile={this.loadParametersFromFile}
              scrollEventCount={scrollEventCount}
              triggerScrollEvent={this.triggerScrollEvent}
            />
          </Row>
          <Portal elId={sidebarSetupElementId}>
            <OverlayTrigger
              placement="right"
              delay={{ show: 250, hide: 400 }}
              overlay={(
                <Tooltip>
                  {t('Browse to a datastack (.json, .tgz) or InVEST logfile (.txt)')}
                </Tooltip>
              )}
            >
              <Button
                onClick={this.browseForDatastack}
                variant="link"
              >
                <MdFolderOpen className="mr-1" />
                {t('Load parameters from file')}
              </Button>
            </OverlayTrigger>
            <SaveAsModal
              modelName={modelId}
              savePythonScript={this.savePythonScript}
              saveJsonFile={this.saveJsonFile}
              saveDatastack={this.saveDatastack}
            />
            <React.Fragment>
              {SaveAlerts}
            </React.Fragment>
          </Portal>
          <Portal elId={sidebarFooterElementId}>
            <Button
              block
              variant="primary"
              size="lg"
              onClick={this.wrapInvestExecute}
              disabled={!argsValid || executeClicked}
            >
              {buttonText}
            </Button>
          </Portal>
        </Container>
      );
    }
    // The SetupTab remains disabled in this route, so no need
    // to render anything here.
    return (<div>{t('No args to see here')}</div>);
  }
}
export default withTranslation()(SetupTab);

SetupTab.propTypes = {
  pyModuleName: PropTypes.string.isRequired,
  userguide: PropTypes.string.isRequired,
  modelId: PropTypes.string.isRequired,
  argsSpec: PropTypes.objectOf(
    PropTypes.shape({
      name: PropTypes.string,
      type: PropTypes.string,
    })
  ).isRequired,
  uiSpec: PropTypes.shape({
    order: PropTypes.arrayOf(
      PropTypes.arrayOf(
        PropTypes.oneOfType([
          PropTypes.string, PropTypes.object
        ])
      )
    ).isRequired,
  }).isRequired,
  argsInitValues: PropTypes.objectOf(PropTypes.oneOfType(
    [PropTypes.string, PropTypes.bool, PropTypes.number])),
  investExecute: PropTypes.func.isRequired,
  sidebarSetupElementId: PropTypes.string.isRequired,
  sidebarFooterElementId: PropTypes.string.isRequired,
  executeClicked: PropTypes.bool.isRequired,
  switchTabs: PropTypes.func.isRequired,
};
