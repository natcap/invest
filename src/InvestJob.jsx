import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import React from 'react';
import PropTypes from 'prop-types';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import Spinner from 'react-bootstrap/Spinner';
import DropdownButton from 'react-bootstrap/DropdownButton';

import { HomeTab } from './components/HomeTab';
import { SetupTab } from './components/SetupTab';
import { LogTab } from './components/LogTab';
import { ResultsTab } from './components/ResultsTab'
import { ResourcesTab } from './components/ResourcesTab';
import { SaveParametersButton, SavePythonButton } from './components/SaveDropdown'
import { LoadButton } from './components/LoadButton';
import { SettingsModal } from './components/SettingsModal';
import { getSpec, saveToPython, writeParametersToFile, fetchValidation } from './server_requests';
import { argsValuesFromSpec, argsDictFromObject, findMostRecentLogfile } from './utils';

// TODO see issue #12
import { createStore } from 'redux';
import { Provider } from 'react-redux';
import rootReducer from './components/ResultsTab/Visualization/habitat_risk_assessment/reducers';
const store = createStore(rootReducer)

let INVEST_EXE = 'invest'
if (process.env.INVEST) {  // if it was set, override
  INVEST_EXE = process.env.INVEST
}

let gdalEnv = null;
if (process.env.GDAL_DATA) {
  gdalEnv = { GDAL_DATA: process.env.GDAL_DATA }
}

// TODO: some of these 'global' vars are defined in multiple files
// const CACHE_DIR = 'cache' //  for storing state snapshot files
// const TEMP_DIR = 'tmp'  // for saving datastack json files prior to investExecute
// const INVEST_UI_DATA = 'ui_data'

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  'DEBUG':   '--debug',
  'INFO':    '-vvv',
  'WARNING': '-vv',
  'ERROR':   '-v',
}

export class InvestJob extends React.Component {
  /** This component and it's children render all the visible parts of the app.
  *
  * This component's state includes all the data needed to represent one invest
  * job.
  */

  constructor(props) {
    super(props);

    this.state = {
      setupHash: crypto.randomBytes(10).toString('hex'),
      sessionID: null,                 // modelName + workspace.directory + workspace.suffix
      modelName: '',                   // as appearing in `invest list`
      modelSpec: {},                   // ARGS_SPEC dict with all keys except ARGS_SPEC.args
      argsSpec: null,                  // ARGS_SPEC.args, the immutable args stuff
      argsValues: null,                // to hold args keys and values, ui_options, i.e. the mutable args stuff
      argsValidation: null,            // to hold validation state for each arg, set by investValidate.
      argsValid: false,                // are all the args valid? set on investValidate exit
      workspace: { 
        directory: null, suffix: null
      },                               // only set values when execute starts the subprocess
      logfile: null,                   // path to the invest logfile associated with invest job
      logStdErr: null,                 // stderr data from the invest subprocess
      sessionProgress: 'home',         // 'home', 'setup', 'log' - used on loadState to decide which tab to activate
      jobStatus: null,                 // 'running', 'error', 'success'
      activeTab: 'home',               // controls which tab is currently visible
    };
    
    this.argsToJsonFile = this.argsToJsonFile.bind(this);
    this.investGetSpec = this.investGetSpec.bind(this);
    this.investValidate = this.investValidate.bind(this);
    this.investExecute = this.investExecute.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.updateArg = this.updateArg.bind(this);
    this.batchUpdateArgs = this.batchUpdateArgs.bind(this);
    this.saveState = this.saveState.bind(this);
    this.savePythonScript = this.savePythonScript.bind(this);
    this.loadState = this.loadState.bind(this);
    this.setSessionID = this.setSessionID.bind(this);
  }

  saveState() {
    /** Save the state of the application (1) and the current InVEST job (2).
    * 1. Save the state object of this component to a JSON file .
    * 2. Append metadata of the invest job to a persistent database/file.
    * This triggers automatically when the invest subprocess starts and again
    * when it exits.
    */
    const jobName = this.state.sessionID;
    const jsonContent = JSON.stringify(this.state, null, 2);
    const filepath = path.join(this.props.directoryConstants.CACHE_DIR, jobName + '.json');
    fs.writeFile(filepath, jsonContent, 'utf8', function (err) {
      if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
      }
      console.log("saved: " + jobName);
    });
    let job = {};
    job[jobName] = {
      model: this.state.modelName,
      workspace: this.state.workspace,
      statefile: filepath,
      status: this.state.jobStatus,
      humanTime: new Date().toLocaleString(),
      systemTime: new Date().getTime(),
    }
    this.props.updateRecentSessions(job, this.props.appdata);
  }

  savePythonScript(filepath) {
    /** Save the current invest arguments to a python script via datastack.py API.
    *
    * @params {string} filepath - desired path to the python script
    */
    // const args_dict_string = argsValuesFromSpec(this.state.args)
    const args_dict_string = JSON.stringify(this.state.argsValues)
    const payload = { 
      filepath: filepath,
      modelname: this.state.modelName,
      pyname: this.state.modelSpec.module,
      args: args_dict_string
    }
    saveToPython(payload);
  }
  
  setSessionID(event) {
    // TODO: this functionality might be deprecated - probably no need to set custom
    // session names. But the same function could be repurposed for a job description.
    event.preventDefault();
    const value = event.target.value;
    this.setState(
      {sessionID: value});
  }

  loadState(sessionFilename) {
    /** Set this component's state to the object parsed from a JSON file.
    *
    * @params {string} sessionFilename - path to a JSON file.
    */

    // const filename = path.join(this.props.directoryConstants.CACHE_DIR, sessionFilename);
    if (fs.existsSync(sessionFilename)) {
      const loadedState = JSON.parse(fs.readFileSync(sessionFilename, 'utf8'));
      this.setState(loadedState,
        () => {
          this.switchTabs(loadedState.sessionProgress);
          // Validate args on load because referenced files may have moved
          // this.investValidate(argsValuesFromSpec(this.state.args));
          // this.batchUpdateArgs(JSON.parse(argsValuesFromSpec(this.state.args)));
          this.batchUpdateArgs(this.state.argsValues);
          // batchUpdateArgs does validation and also sets inputs to 'touched'
          // which controls whether the validation messages appear or not.
        });
    } else {
      console.log('state file not found: ' + sessionFilename);
    }
  }

  async argsToJsonFile(datastackPath) {
    /** Write an invest args JSON file for passing to invest cli.
    *
    * Outsourcing this to natcap.invest.datastack via flask ensures
    * a compliant json with an invest version string.
    *
    * @params {string} datastackPath - path to a JSON file.
    */

    // The n_workers value always needs to be inserted into args
    // let args_dict = JSON.parse(argsValuesFromSpec(this.state.args));
    let args_dict = Object.assign({}, this.state.args);
    args_dict['n_workers'] = this.props.investSettings.nWorkers;
    
    const payload = {
      parameterSetPath: datastackPath, 
      moduleName: this.state.modelSpec.module,
      relativePaths: false,
      args: JSON.stringify(args_dict)
    }
    await writeParametersToFile(payload);
  }

  async investExecute() {
    /** Spawn a child process to run an invest model via the invest CLI:
    * `invest -vvv run <model> --headless -d <datastack path>`
    *
    * When the process starts (on first stdout callback), job metadata is saved
    * and local state is updated to display the log.

    * When the process exits, job metadata is saved again (overwriting previous)
    * with the final status of the invest run.
    */
    const workspace = {
      directory: this.state.argsValues.workspace_dir.value,
      suffix: this.state.argsValues.results_suffix.value
    }
    // model name, workspace, and suffix are suitable for a unique job identifier
    const sessionName = [
      this.state.modelName, workspace.directory, workspace.suffix].join('-')

    // Write a temporary datastack json for passing as a command-line arg
    const temp_dir = fs.mkdtempSync(path.join(process.cwd(), this.props.directoryConstants.TEMP_DIR, 'data-'))
    const datastackPath = path.join(temp_dir, 'datastack.json')
    const _ = await this.argsToJsonFile(datastackPath);

    // Get verbosity level from the app's settings
    const verbosity = LOGLEVELMAP[this.props.investSettings.loggingLevel]
    
    const cmdArgs = [verbosity, 'run', this.state.modelName, '--headless', '-d ' + datastackPath]
    const investRun = spawn(INVEST_EXE, cmdArgs, {
        cwd: process.cwd(),
        shell: true, // without true, IOError when datastack.py loads json
        env: gdalEnv
      });

    
    // There's no general way to know that a spawned process started,
    // so this logic when listening for stdout seems like the way.
    let logfilename = ''
    investRun.stdout.on('data', async (data) => {
      if (!logfilename) {
        logfilename = await findMostRecentLogfile(workspace.directory)
        console.log(logfilename)
        // TODO: handle case when logfilename is undefined? It seems like
        // sometimes there is some stdout emitted before a logfile exists.
        this.setState(
          {
            logfile: logfilename,
            sessionID: sessionName,
            sessionProgress: 'log',
            workspace: workspace,
            jobStatus: 'running'
          }, () => {
            this.switchTabs('log')
            this.saveState()
          }
        );
      }
    });

    // Capture stderr to a string separate from the invest log
    // so that it can be displayed separately when invest exits,
    // and because it could actually be stderr emitted from the 
    // invest CLI or even the shell, rather than the invest model,
    // in which case it's useful to console.log too.
    let stderr = Object.assign('', this.state.logStdErr);
    investRun.stderr.on('data', (data) => {
      console.log(`${data}`)
      stderr += `${data}`
      this.setState({
        logStdErr: stderr,
      });
    });

    // Set some state when the invest process exits and update the app's
    // persistent database by calling saveState.
    investRun.on('close', (code) => {
      // TODO: there are non-zero exit cases that should be handled
      // differently from one-another, but right now they are all exit code 1.
      // E.g. this callback is designed with a model crash in mind, but not a fail to 
      // launch, in which case the saveState call will probably crash.
      const status = (code === 0 ? 'success' : 'error')
      this.setState({
        jobStatus: status,
      }, () => {
        this.saveState();
      });
    });
  }

  async investValidate(argsValues, limit_to) {
    /** Validate an arguments dictionary using the InVEST model's validate function.
    *
    * @param {object} args_dict_string - a JSON.stringify'ed object of model argument
    *    keys and values.
    * @param {string} limit_to - an argument key if validation should be limited only
    *    to that argument.
    */
    let argsSpec = JSON.parse(JSON.stringify(this.state.argsSpec));
    let argsValidation = Object.assign({}, this.state.argsValidation);
    let keyset = new Set(Object.keys(argsSpec));
    let payload = { 
      model_module: this.state.modelSpec.module,
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
          args: argsValidation,
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

  async investGetSpec(modelName) {
    /** Get an invest model's ARGS_SPEC when a model button is clicked.
    *  
    * Also get the model's ui_spec if it exists.
    * Then reset much of this component's state in case a prior job's 
    * state exists. 
    * Finally, call batchUpdateArgs to properly initialize enabled/disabled
    * state of optional inputs, and trigger invest validate.
    *
    * @param {string} - 
    */

    const payload = { 
        model: modelName
    };
    const spec = await getSpec(payload);
    if (spec) {
      // This "destructuring" captures spec.args into args and leaves 
      // the rest of spec in modelSpec.
      const {args, ...modelSpec} = spec;
      
      // Even if UI spec doesn't exist for a model, a minimum viable input
      // form can still be generated from ARGS_SPEC alone, so don't crash here.
      let uiSpec = {};
      try {
        uiSpec = JSON.parse(fs.readFileSync(
          path.join(this.props.directoryConstants.INVEST_UI_DATA, spec.module + '.json')))
      } catch (err) {
        if (err.code !== 'ENOENT') {
          throw err
        }
      }

      let argsValues = {};  // the object that will store all the mutable values
      let argsValidation = {};
      for (const key in args) {
        Object.assign(args[key], uiSpec[key])
        argsValues[key] = {}
        argsValidation[key] = {}
      }
      console.log(args)


      // This event represents a user selecting a model,
      // and so some existing state should be reset.
      this.setState({
        modelName: modelName,
        modelSpec: modelSpec,
        argsSpec: args,
        argsValues: argsValues,
        argsValidation: argsValidation,
        argsValid: false,
        sessionProgress: 'setup',
        jobStatus: null,
        logStdErr: '',
        logStdOut: '',
        sessionID: null,
        workspace: null,
      }, () => {
        // arg values are yet `undefined` but will be converted to either 
        // empty strings or `false` by `argsValuesFromSpec`.
        // batchUpdateArgs sets the initial ui_active_options for each arg,
        // and triggers validation, allowing optional args to validate 
        // without any user-interaction.
        // `false` argument prevents 'touching' these inputs so that we don't 
        //see scary validation warnings yet.
        // this.batchUpdateArgs(argsValuesFromSpec(args), false)
        this.batchUpdateArgs(argsDictFromObject(argsValues), false)
        this.switchTabs('setup')
      });
    } else {
      console.log('no spec found')
      return new Promise((resolve) => resolve(false))
    }
    return new Promise((resolve) => resolve(true))
  }

  batchUpdateArgs(args_dict, touch=true) {
    /** Update this.state.args in response to batch argument loading events,
    * and then validate the loaded args.
    *
    * @param {object} args_dict - the args dictionay object that comes from datastack.py
    * after parsing args from logfile or datastack file.
    * @param {boolean} touch - whether this function should mark arguments as 'touched'.
    * 'touched' controls whether validation messages display. Usually this is 
    * desireable, except when this function is used for initial render of the input form,
    * when it's better to not display the arguments as 'touched'.
    */

    // TODO: could this now just call through to updateArg? Maybe by adding a touch param there.
    // But that would add a bunch more calls to validate, which here waits until after
    // all the arg values are set, which is better.
    // TODO: this is all wrapped up in the issue with sluggish response-time on keystroke changes
    // which is due to the validate call triggering on every updateArg call.
    const argsSpec = JSON.parse(JSON.stringify(this.state.argsSpec));
    const argsValues = Object.assign({}, this.state.argsValues);
    Object.keys(argsSpec).forEach(argkey => {
      // Loop over argsMeta in order to:
        // 1) clear values for args that are absent from the input
        // 2) skip over items from the input that have incorrect keys, otherwise
        //    investValidate will crash on them.

      argsValues[argkey]['value'] = args_dict[argkey]
      argsValues[argkey]['touched'] = touch;

      if (argsSpec[argkey].ui_control) {
        argsSpec[argkey].ui_control.forEach(dependentKey => {
          if (!args_dict[argkey]) {
            // hide/disable the dependent args
            argsValues[dependentKey]['active_ui_option'] = argsSpec[dependentKey].ui_option
          } else {
            argsValues[dependentKey]['active_ui_option'] = undefined
          }
        });
      }
    });
    
    this.setState({
      // args: argsMeta,
      setupHash: crypto.randomBytes(10).toString('hex')},
      () => { this.investValidate(argsValues) }
    );
  }

  updateArg(key, value, touch=true) {
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

    const argsSpec = JSON.parse(JSON.stringify(this.state.argsSpec));
    const argsValues = Object.assign({}, this.state.argsValues);
    argsValues[key]['value'] = value;
    argsValues[key]['touched'] = touch;

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

    this.investValidate(argsValues)
    // this.setState({args: argsMeta}, 
    //   () => {
    //     // TODO: consolodistae the state setting - maybe just let validate set it.
    //     // this.investValidate(argsValuesFromSpec(argsMeta));
    //   });
  }

  switchTabs(key) {
    /** Change the tab that is currently visible.
    * @param {string} key - the value of one of the Nav.Link eventKey.
    */
    this.setState(
      {activeTab: key}
    );
  }

  render () {
    // console.log('InvestJob Render')
    const activeTab = this.state.activeTab;
    const setupDisabled = !(this.state.argsSpec); // enable once modelSpec has loaded
    const logDisabled = (this.state.jobStatus == null);  // enable during and after execution
    const resultsDisabled = (this.state.jobStatus !== 'success');  // enable only on complete execute with no errors
    const dropdownsDisabled = (this.state.argsSpec == null);
    
    return(
      <TabContainer activeKey={activeTab}>
        <Navbar bg="light" expand="lg">
          <Nav variant="tabs" id="controlled-tab-example" className="mr-auto"
            activeKey={activeTab}
            onSelect={this.switchTabs}>
            <Nav.Item>
              <Nav.Link eventKey="home">Home</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="setup" disabled={setupDisabled}>Setup</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="log" disabled={logDisabled}>
                {this.state.jobStatus === 'running' && 
                 <Spinner animation='border' size='sm' role='status' aria-hidden='true'/>
                } Log
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="results" disabled={resultsDisabled}>Results</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="resources">Resources</Nav.Link>
            </Nav.Item>
          </Nav>
          <Navbar.Brand>{this.state.modelSpec.model_name}</Navbar.Brand>
          <DropdownButton id="dropdown-basic-button" title="Save " className="mx-3">
            <SaveParametersButton
              argsToJsonFile={this.argsToJsonFile}
              disabled={dropdownsDisabled}/>
            <SavePythonButton
              savePythonScript={this.savePythonScript}
              disabled={dropdownsDisabled}/>
          </DropdownButton>
          <LoadButton
            investGetSpec={this.investGetSpec}
            batchUpdateArgs={this.batchUpdateArgs}
          />
          <SettingsModal className="mx-3"
            saveSettings={this.props.saveSettings}
            investSettings={this.props.investSettings}
          />
        </Navbar>
        <TabContent className="mt-3">
          <TabPane eventKey="home" title="Home">
            <HomeTab
              investList={this.props.investList}
              investGetSpec={this.investGetSpec}
              saveState={this.saveState}
              loadState={this.loadState}
              recentSessions={this.props.recentSessions}
            />
          </TabPane>
          <TabPane eventKey="setup" title="Setup">
            <SetupTab
              setupHash={this.state.setupHash}
              argsSpec={this.state.argsSpec}
              argsValues={this.state.argsValues}
              argsValidation={this.state.argsValidation}
              argsValid={this.state.argsValid}
              modulename={this.state.modelSpec.module}
              updateArg={this.updateArg}
              batchUpdateArgs={this.batchUpdateArgs}
              investValidate={this.investValidate}
              investExecute={this.investExecute}
            />
          </TabPane>
          <TabPane eventKey="log" title="Log">
            <LogTab
              jobStatus={this.state.jobStatus}
              logfile={this.state.logfile}
              logStdErr={this.state.logStdErr}
            />
          </TabPane>
          <TabPane eventKey="results" title="Results">
          <Provider store={store}>
            <ResultsTab
              model={this.state.modelName}
              workspace={this.state.workspace}
              sessionID={this.state.sessionID}
              activeTab={activeTab}/> 
          </Provider>
          </TabPane>
          <TabPane eventKey="resources" title="Resources">
            <ResourcesTab 
              modelName={this.state.modelSpec.model_name}
              docs={this.state.modelSpec.userguide_html}
            />
          </TabPane>
        </TabContent>
      </TabContainer>
    );
  }
}

InvestJob.propTypes = {
  investList: PropTypes.object,
  investSettings: PropTypes.object,
  recentSessions: PropTypes.array,
  appdata: PropTypes.string,
  updateRecentSessions: PropTypes.func,
  saveSettings: PropTypes.func
}


