import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import request from 'request';

import React from 'react';
import { createStore } from 'redux';
import { Provider } from 'react-redux';

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
import { SaveSessionDropdownItem, SaveParametersDropdownItem,
         SavePythonDropdownItem } from './components/SaveDropdown'
import { SettingsModal } from './components/SettingsModal';
import { getSpec } from './server_requests';

// TODO see issue #12
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

const CACHE_DIR = 'cache' //  for storing state snapshot files
const TEMP_DIR = 'tmp'  // for saving datastack json files prior to investExecute

export class InvestJob extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      sessionID: defaultSessionID(''),
      modelName: '',                   // as appearing in `invest list`
      modelSpec: {},                   // ARGS_SPEC dict with all keys except ARGS_SPEC.args
      args: null,                      // ARGS_SPEC.args, to hold values on user-interaction
      argsValid: false,                // set on investValidate exit
      workspace: {                     // only set values when execute completes
        directory: null, suffix: null},
      logStdErr: '', 
      logStdOut: '',
      sessionProgress: 'home',       // 'home', 'setup', 'log', 'results' (i.e. one of the tabs)
      activeTab: 'home',
    };
    
    this.argsToJsonFile = this.argsToJsonFile.bind(this);
    this.investGetSpec = this.investGetSpec.bind(this);
    this.investValidate = this.investValidate.bind(this);
    this.investExecute = this.investExecute.bind(this);
    this.investKill = this.investKill.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.updateArg = this.updateArg.bind(this);
    this.batchUpdateArgs = this.batchUpdateArgs.bind(this);
    this.saveState = this.saveState.bind(this);
    this.savePythonScript = this.savePythonScript.bind(this);
    this.loadState = this.loadState.bind(this);
    this.setSessionID = this.setSessionID.bind(this);
  }

  saveState(event) {
    // Save a snapshot of this component's state to a JSON file.
    const jsonContent = JSON.stringify(this.state, null, 2);
    const sessionID = this.state.sessionID;
    const filepath = path.join(CACHE_DIR, sessionID + '.json');
    fs.writeFile(filepath, jsonContent, 'utf8', function (err) {
      if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
      }
      console.log("saved" + sessionID);
    });
    this.props.updateRecentSessions(sessionID);
  }

  savePythonScript(filepath) {
    const args_dict_string = argsValuesFromSpec(this.state.args)

    request.post(
      'http://localhost:5000/save_to_python',
      { json: { 
          filepath: filepath,
          modelname: this.state.modelName,
          pyname: this.state.modelSpec.module,
          args: args_dict_string
        } 
      },
      (error, response, body) => {
        if (!error && response.statusCode == 200) {
         
        } else {
          console.log('Status: ' + response.statusCode)
          console.log('Error: ' + error.message)
        }
      }
    );
  }
  
  setSessionID(event) {
    // Handle keystroke events to store a name for current session
    event.preventDefault();
    const value = event.target.value;
    this.setState(
      {sessionID: value});
  }

  loadState(sessionID) {
    // Set this component's state to the object parsed from a JSON file.
    // sessionID (string) : the name, without extension, of a saved JSON.

    const filename = path.join(CACHE_DIR, sessionID + '.json');
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

  argsToJsonFile(datastackPath) {
    // make simple args json for passing to python cli

    // parsing it just to make it easy to insert the n_workers value
    let args_dict = JSON.parse(argsValuesFromSpec(this.state.args));
    args_dict['n_workers'] = this.props.investSettings.nWorkers;

    request.post(
      'http://localhost:5000/write_parameter_set_file',
      { json: {
          parameterSetPath: datastackPath, 
          moduleName: this.state.modelSpec.module,
          relativePaths: true,
          args: JSON.stringify(args_dict)
        }
      },
      (error, response, body) => {
        if (!error && response.statusCode == 200) {
          console.log("JSON file was saved.");
        } else {
          console.log('Status: ' + response.statusCode);
          console.log('Error: ' + error.message);
        }
      }
    );
  }

  investExecute() {
    const datastackPath = path.join(
      TEMP_DIR, this.state.sessionID + '.json')

    // to translate to the invest CLI's verbosity flag:
    const loggingLevelLookup = {
      'DEBUG':   '--debug',
      'INFO':    '-vvv',
      'WARNING': '-vv',
      'ERROR':   '-v',
    }
    const verbosity = loggingLevelLookup[this.props.investSettings.loggingLevel]

    this.argsToJsonFile(datastackPath);
    
    this.setState(
      {
        sessionProgress: 'log',
        activeTab: 'log',
        logStdErr: '',
        logStdOut: ''
      }
    );

    const cmdArgs = [verbosity, 'run', this.state.modelName, '--headless', '-d ' + datastackPath]
    const investRun = spawn(INVEST_EXE, cmdArgs, {
        cwd: '.',
        shell: true, // without true, IOError when datastack.py loads json
        env: gdalEnv
      });

    // TODO: Find a nicer way to stream a log to the page than
    // passing all the text through this.state
    let stdout = Object.assign('', this.state.logStdOut);
    investRun.stdout.on('data', (data) => {
      stdout += `${data}`
      this.setState({
        logStdOut: stdout,
        procID: investRun.pid
      });
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
      const workspace = {
        directory: this.state.args.workspace_dir.value,
        suffix: this.state.args.results_suffix.value
      }
      this.setState({
        sessionProgress: progress,
        workspace: workspace,
        procID: null,  // see above comment
      });
      console.log(this.state)
    });
  }

  investKill() {
    // TODO: this never worked properly. I think the pid here is from the node subprocess,
    // when we actually want the pid from the python process it launched.
    if (this.state.procID){
      console.log(this.state.procID);
      process.kill(this.state.procID)
    }
  }

  investValidate(args_dict_string, limit_to) {
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
    if (limit_to) {
      payload['limit_to'] = limit_to
    }

    request.post(
      'http://localhost:5000/validate',
      { json: payload},
      (error, response, body) => {
        if (!error && response.statusCode == 200) {
          const results = body;

          // A) At least one arg was invalid:
          if (results.length) { 

            results.forEach(result => {
              // Each result is an array of two elements
              // 0: array of arg keys
              // 1: string message that pertains to those args
              // TODO: test this indexing against all sorts of results
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

        } else {
          console.log('Status: ' + response.statusCode);
          console.log(body);
          if (error) {
            console.error(error);
          }
        }
      }
    );
  }

  async investGetSpec(event) {
    const modelName = event.target.value;
    const payload = { 
      json: { 
        model: modelName
      } 
    };
    const spec = await getSpec(payload);
    if (spec) {
      // for clarity, state has a dedicated args property separte from spec
      const args = JSON.parse(JSON.stringify(spec.args));
      delete spec.args

      // This event represents a user selecting a model,
      // and so some existing state should be reset.
      this.setState({
        modelName: modelName,
        modelSpec: spec,
        args: args,
        argsValid: false,
        sessionProgress: 'setup',
        logStdErr: '',
        logStdOut: '',
        sessionID: defaultSessionID(modelName),
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
    
    // state.procID only has a value during invest execution
    let spinner;
    if (this.state.procID) {
      spinner = <Spinner
                  animation='border'
                  size='sm'
                  role='status'
                  aria-hidden='true'
                />
    } else {
      spinner = <div></div>
    }

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
              <Nav.Link eventKey="log" disabled={logDisabled}>{spinner} Log</Nav.Link>
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
            <SaveSessionDropdownItem 
              saveState={this.saveState}
              sessionID={this.state.sessionID}
              setSessionID={this.setSessionID}/>
            <SaveParametersDropdownItem
              argsToJsonFile={this.argsToJsonFile}/>
            <SavePythonDropdownItem
              savePythonScript={this.savePythonScript}/>
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
              logStdOut={this.state.logStdOut}
              logStdErr={this.state.logStdErr}
              investKill={this.investKill}
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

function defaultSessionID(modelName) {
  const datetime = new Date()
      .toISOString()
      .replace(/:/g, '-').replace('T', '_').slice(0, -5)
  return(modelName + '_' + datetime);
}