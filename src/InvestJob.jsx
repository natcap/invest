import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import React from 'react';

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
import { SaveSessionButtonModal, SaveParametersButton,
         SavePythonButton } from './components/SaveDropdown'
import { SettingsModal } from './components/SettingsModal';
import { getSpec, saveToPython, writeParametersToFile,
         fetchValidation, fetchLogfilename } from './server_requests';
import { findMostRecentLogfile } from './utils';

// TODO see issue #12
import { createStore } from 'redux';
import { Provider } from 'react-redux';
import rootReducer from './components/ResultsTab/Visualization/habitat_risk_assessment/reducers';
const store = createStore(rootReducer)

let INVEST_EXE = 'invest.exe'
if (process.env.INVEST) {  // if it was set, override
  INVEST_EXE = process.env.INVEST
}

let gdalEnv = null;
if (process.env.GDAL_DATA) {
  gdalEnv = { GDAL_DATA: process.env.GDAL_DATA }
}

// TODO: some of these 'global' vars are defined in multiple files
const CACHE_DIR = 'cache' //  for storing state snapshot files
const TEMP_DIR = 'tmp'  // for saving datastack json files prior to investExecute

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  'DEBUG':   '--debug',
  'INFO':    '-vvv',
  'WARNING': '-vv',
  'ERROR':   '-v',
}

export class InvestJob extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      sessionID: null,
      modelName: '',                   // as appearing in `invest list`
      modelSpec: {},                   // ARGS_SPEC dict with all keys except ARGS_SPEC.args
      args: null,                      // ARGS_SPEC.args, to hold values on user-interaction
      argsValid: false,                // set on investValidate exit
      workspace: { 
        directory: null, suffix: null
      },                               // only set values when execute completes
      logfile: null,
      logStdErr: null,
      sessionProgress: 'home',         // 'home', 'setup', 'log', 'results' (i.e. one of the tabs)
      activeTab: 'home',
      jobStatus: null                  // 'running', 'error', 'success'
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
    // Save a snapshot of this component's state to a JSON file.
    const sessionID = this.state.sessionID;
    const jsonContent = JSON.stringify(this.state, null, 2);
    const filepath = path.join(CACHE_DIR, sessionID + '.json');
    fs.writeFile(filepath, jsonContent, 'utf8', function (err) {
      if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
      }
      console.log("saved: " + sessionID);
    });
    const jobName = this.state.sessionID;
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
    const args_dict_string = argsValuesFromSpec(this.state.args)
    const payload = { 
      filepath: filepath,
      modelname: this.state.modelName,
      pyname: this.state.modelSpec.module,
      args: args_dict_string
    }
    saveToPython(payload);
  }
  
  setSessionID(event) {
    // Handle keystroke events to store a name for current session
    event.preventDefault();
    const value = event.target.value;
    this.setState(
      {sessionID: value});
  }

  loadState(sessionFilename) {
    // Set this component's state to the object parsed from a JSON file.
    // sessionID (string) : the name, without extension, of a saved JSON.

    const filename = path.join(CACHE_DIR, sessionFilename);
    if (fs.existsSync(filename)) {
      const loadedState = JSON.parse(fs.readFileSync(filename, 'utf8'));
      this.setState(loadedState,
        () => {
          this.switchTabs(loadedState.sessionProgress);
          // Validate args on load because referenced files may have moved
          // this.investValidate(argsValuesFromSpec(this.state.args));
          this.batchUpdateArgs(JSON.parse(argsValuesFromSpec(this.state.args)));
          // batchUpdateArgs does validation and also sets inputs to 'touched'
          // which controls whether the validation messages appear or not.
        });
    } else {
      console.log('state file not found: ' + filename);
    }
  }

  async argsToJsonFile(datastackPath) {
    // make simple args json for passing to invest cli

    // parsing it just to make it easy to insert the n_workers value
    let args_dict = JSON.parse(argsValuesFromSpec(this.state.args));
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

    const temp_dir = fs.mkdtempSync(path.join(process.cwd(), TEMP_DIR, 'data-'))
    const datastackPath = path.join(temp_dir, 'datastack.json')
    const _ = await this.argsToJsonFile(datastackPath);

    const verbosity = LOGLEVELMAP[this.props.investSettings.loggingLevel]
    const cmdArgs = [verbosity, 'run', this.state.modelName, '--headless', '-d ' + datastackPath]
    const investRun = spawn(INVEST_EXE, cmdArgs, {
        cwd: process.cwd(),
        shell: true, // without true, IOError when datastack.py loads json
        env: gdalEnv
      });

    const workspace = {
      directory: this.state.args.workspace_dir.value,
      suffix: this.state.args.results_suffix.value
    }
    const sessionName = [
      this.state.modelName, workspace.directory, workspace.suffix].join('-')
    
    // There's no general way to know that a spawned process started,
    // so this logic when listening for stdout seems like the way.
    let logfilename = ''
    investRun.stdout.on('data', async () => {
      if (!logfilename) {
        logfilename = await findMostRecentLogfile(workspace.directory)
        this.setState(
          {
            procID: investRun.pid,
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

    let stderr = Object.assign('', this.state.logStdErr);
    investRun.stderr.on('data', (data) => {
      stderr += `${data}`
      this.setState({
        logStdErr: stderr,
      });
    });

    // Reset the procID when the process finishes because the OS
    // can recycle the pid, and we don't want the kill Button killing
    // another random process.
    investRun.on('close', (code) => {
      const progress = (code === 0 ? 'results' : 'log')
      const status = (code === 0 ? 'success' : 'error')
      this.setState({
        sessionProgress: progress,
        jobStatus: status,
        procID: null,  // see above comment
      }, () => {
        this.saveState();
      });
      console.log(this.state)
    });
  }

  async investValidate(args_dict_string, limit_to) {
    /*
    Validate an arguments dictionary using the InVEST model's validate func.

    Parameters:
      args_dict_string (object) : a JSON.stringify'ed object of model argument
        keys and values.
      limit_to (string) : an argument key if validation should be limited only
        to that argument.

    */
    let argsMeta = JSON.parse(JSON.stringify(this.state.args));
    let keyset = new Set(Object.keys(JSON.parse(args_dict_string)));
    let payload = { 
      model_module: this.state.modelSpec.module,
      args: args_dict_string
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
          argsMeta[key]['validationMessage'] = message
          argsMeta[key]['valid'] = false
          keyset.delete(key);
        })
      });
      if (!limit_to) {  // validated all, so ones left in keyset are valid
        keyset.forEach(k => {
          argsMeta[k]['valid'] = true
          argsMeta[k]['validationMessage'] = ''
        })
      }
      this.setState({
        args: argsMeta,
        argsValid: false
      });

    // B) All args were validated and none were invalid:
    } else if (!limit_to) {
      
      keyset.forEach(k => {
        argsMeta[k]['valid'] = true
        argsMeta[k]['validationMessage'] = ''
      })
      // It's possible all args were already valid, in which case
      // it's nice to avoid the re-render that this setState call
      // triggers. Although only the Viz app components re-render 
      // in a noticeable way. Due to use of redux there?
      if (!this.state.argsValid) {
        this.setState({
          args: argsMeta,
          argsValid: true
        })
      }

    // C) Limited args were validated and none were invalid
    } else if (limit_to) {

      argsMeta[limit_to]['valid'] = true
      // this could be the last arg that needed to go valid,
      // in which case we should trigger a full args_dict validation
      // in order to properly set state.argsValid
      this.setState({ args: argsMeta },
        () => {
          let argIsValidArray = [];
          for (const key in argsMeta) {
            argIsValidArray.push(argsMeta[key]['valid'])
          }
          if (argIsValidArray.every(Boolean)) {
            this.investValidate(argsValuesFromSpec(argsMeta));
          }
        }
      );
    }
  }

  async investGetSpec(event) {
    const modelName = event.target.value;
    const payload = { 
        model: modelName
    };
    const spec = await getSpec(payload);
    if (spec) {
      // for clarity, state has a dedicated args property separte from spec
      // const args = JSON.parse(JSON.stringify(spec.args));
      // delete spec.args // bad idea to mutate var like this.
      const {args, ...modelSpec} = spec;

      // This event represents a user selecting a model,
      // and so some existing state should be reset.
      this.setState({
        modelName: modelName,
        modelSpec: modelSpec,
        args: args,
        argsValid: false,
        sessionProgress: 'setup',
        jobStatus: null,
        logStdErr: '',
        logStdOut: '',
        sessionID: null,
        workspace: null,
      }, () => this.switchTabs('setup'));
    } else {
      console.log('no spec returned');
    }
  }

  batchUpdateArgs(args_dict) {
    // Update this.state.args in response to batch argument loading events

    const argsMeta = JSON.parse(JSON.stringify(this.state.args));
    Object.keys(argsMeta).forEach(argkey => {
      // Loop over argsMeta instead of the args_dict extracted from the input
      // in order to:
        // 1) clear values for args that are absent from the input
        // 2) skip over items from the input that have incorrect keys, otherwise
        //    investValidate will crash on them.
      argsMeta[argkey]['value'] = args_dict[argkey] || '';
      // label as touched even if the argkey was absent, since it's a batch load
      argsMeta[argkey]['touched'] = true;
    });
    
    this.setState({args: argsMeta},
      () => { this.investValidate(argsValuesFromSpec(argsMeta)) }
    );
  }

  updateArg(key, value) {
    // Update this.state.args in response to change events on a single ArgsForm input

    // Parameters:
      // key (string)
      // value (string or number)

    const argsMeta = JSON.parse(JSON.stringify(this.state.args));
    argsMeta[key]['value'] = value;
    argsMeta[key]['touched'] = true;

    this.setState({args: argsMeta}, 
      () => {
        this.investValidate(argsValuesFromSpec(argsMeta));
      });
  }

  switchTabs(key) {
    this.setState(
      {activeTab: key}
    );
  }

  render () {
    const activeTab = this.state.activeTab;
    const sessionProgress = this.state.sessionProgress;
    const setupDisabled = !(this.state.args); // enable once modelSpec has loaded
    const logDisabled = ['home', 'setup'].includes(sessionProgress);  // enable during and after execution
    const resultsDisabled = (sessionProgress !== 'results');  // enable only on complete execute with no errors
    const dropdownsDisabled = (this.state.args == null);
    
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
            <SaveSessionButtonModal 
              saveState={this.saveState}
              sessionID={this.state.sessionID}
              setSessionID={this.setSessionID}/>
            <SaveParametersButton
              argsToJsonFile={this.argsToJsonFile}
              disabled={dropdownsDisabled}/>
            <SavePythonButton
              savePythonScript={this.savePythonScript}
              disabled={dropdownsDisabled}/>
          </DropdownButton>
          <SettingsModal
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
              args={this.state.args}
              argsValid={this.state.argsValid}
              modulename={this.state.modelSpec.module}
              updateArg={this.updateArg}
              batchUpdateArgs={this.batchUpdateArgs}
              investValidate={this.investValidate}
              argsValuesFromSpec={argsValuesFromSpec}
              investExecute={this.investExecute}
            />
          </TabPane>
          <TabPane eventKey="log" title="Log">
            <LogTab
              sessionProgress={this.state.sessionProgress}
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

function boolStringToBoolean(val) {
  let valBoolean;
  try {
    const valString = val.toLowerCase()
    valBoolean = (valString === 'true') ? true : false
  }
  catch(TypeError) {
    valBoolean = val || ''
  }
  return valBoolean
}

// TODO: move this (and boolStringToBoolean) to a module for import instead of passing around in props?
function argsValuesFromSpec(args) {
  /* Given a complete InVEST ARGS_SPEC.args, return just the key:value pairs

  Parameters: 
    args: JSON representation of an InVEST model's ARGS_SPEC.args dictionary.

  Returns:
    JSON.stringify'd args dict

  */
  let args_dict = {};
  for (const argname in args) {
    if (args[argname]['type'] === 'boolean') {
      args_dict[argname] = boolStringToBoolean(args[argname]['value'])
    } else {
      args_dict[argname] = args[argname]['value'] || ''
    }
  }
  return(JSON.stringify(args_dict));
}
