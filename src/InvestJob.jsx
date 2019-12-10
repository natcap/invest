import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import request from 'request';

import React from 'react';
import { createStore } from 'redux';
import { Provider } from 'react-redux';

import Tabs from 'react-bootstrap/Tabs';
import Tab from 'react-bootstrap/Tab';
import Navbar from 'react-bootstrap/Navbar';

import { ModelsTab } from './components/ModelsTab';
import { SetupTab } from './components/SetupTab';
import { LogDisplay } from './components/LogDisplay';
import { DocsTab } from './components/DocsTab';
import VizApp from './VizApp'

// Only the HraApp uses this redux store
// TODO refactor HraApp to not depend on redux.
import rootReducer from './components/Visualization/habitat_risk_assessment/reducers';
const store = createStore(rootReducer)

const INVEST_EXE = process.env.INVEST.trim() // sometimes trailing whitespace when set from command-line
const TEMP_DIR = './'

if (process.env.GDAL_DATA) {
  var GDAL_DATA = process.env.GDAL_DATA.trim()
}
// these options are passed to child_process spawn calls
const PYTHON_OPTIONS = {
  cwd: TEMP_DIR,
  shell: true, // without true, IOError when datastack.py loads json
  env: {GDAL_DATA: GDAL_DATA}
};
const DATASTACK_JSON = 'datastack.json'
const CACHE_DIR = 'cache' //  for storing state snapshot files

// TODO: these ought to be dynamic, from invest getspec or a similar lookup
// for now, change here as needed to test other models
const MODEL_NAME = 'carbon'
const MODEL_DOCS = 'C:/InVEST_3.7.0_x86/documentation/userguide/habitat_risk_assessment.html'

function defaultSessionID(modelName) {
  const datetime = new Date()
      .toISOString()
      .replace(/:/g, '-').replace('T', '_').slice(0, -5)
  return(modelName + '_' + datetime);
}

export class InvestJob extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      sessionID: defaultSessionID(''),
      modelSpec: {},
      args: null,
      argsValid: false,  // set on invest validate exit
      workspace: null,
      logStdErr: '', 
      logStdOut: '',
      sessionProgress: 'models', // 'models', 'setup', 'log', 'viz' (i.e. one of the tabs)
      activeTab: 'models',
      docs: MODEL_DOCS
    };
    
    this.investGetSpec = this.investGetSpec.bind(this);
    this.investValidate = this.investValidate.bind(this);
    this.investExecute = this.investExecute.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.updateArg = this.updateArg.bind(this);
    this.batchUpdateArgs = this.batchUpdateArgs.bind(this);
    this.saveState = this.saveState.bind(this);
    this.loadState = this.loadState.bind(this);
    this.setSessionID = this.setSessionID.bind(this);
  }

  saveState() {
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
  }
  
  loadState(sessionID) {
    // Set this component's state to the object parsed from a JSON file.
    // sessionID (string) : the name, without extension, of a saved JSON.

    console.log(sessionID)
    const filename = path.join(CACHE_DIR, sessionID + '.json');
    if (fs.existsSync(filename)) {
      const loadedState = JSON.parse(fs.readFileSync(filename, 'utf8'));
      console.log(loadedState)
      this.setState(loadedState,
        () => this.switchTabs(loadedState.sessionProgress));
    } else {
      console.log('state file not found: ' + filename);
    }
  }

  setSessionID(event) {
    // Handle keystroke events to store a name for current session
    const value = event.target.value;
    console.log(value);
    this.setState(
      {sessionID: value});
  }

  investExecute() {
    argsToJsonFile(this.state.args, this.state.modelSpec.module);
    
    this.setState(
      {
        sessionProgress: 'log',
        activeTab: 'log',
        logStdErr: '',
        logStdOut: ''
      }
    );

    const datastackPath = path.join(TEMP_DIR, 'datastack.json')
    const cmdArgs = ['-vvv', 'run', MODEL_NAME, '--headless', '-d ' + datastackPath]
    const python = spawn(INVEST_EXE, cmdArgs, PYTHON_OPTIONS);

    // TODO: These setState calls on stdout and stderr trigger
    // a re-render of this component (and it's children -- so everything).
    // So far I see no performance penalty, but we may want a different
    // solution for getting a streaming log. 
    // Or we can suppress some re-renders manually, perhaps.
    let stdout = Object.assign('', this.state.logStdOut);
    python.stdout.on('data', (data) => {
      stdout += `${data}`
      this.setState({
        logStdOut: stdout,
      });
    });

    let stderr = Object.assign('', this.state.logStdErr);
    python.stderr.on('data', (data) => {
      stderr += `${data}`
      this.setState({
        logStdErr: stderr,
      });
    });

    python.on('close', (code) => {
      const progress = (code === 0 ? 'viz' : 'log')
      this.setState({
        sessionProgress: progress,
        workspace: this.state.args.workspace_dir.value
      });
      console.log(this.state)
    });
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
    console.log(args_dict_string);
    let warningsIssued = false;
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

          if (results.length) {  // at least one invalid arg
            warningsIssued = true
            results.forEach(result => {
              // Each result is an array of two elements
              // 0: array of arg keys
              // 1: string message that pertains to those args
              console.log(result);
              // TODO: test this indexing against all sorts of results
              const argkeys = result[0];
              const message = result[1];
              console.log(argkeys);
              console.log(argsMeta);
              argkeys.forEach(key => {
                argsMeta[key]['validationMessage'] = message
                argsMeta[key]['valid'] = false
                keyset.delete(key);
              })
            });
            if (!limit_to){
              console.log(keyset);
              // checked all args, so ones left in keyset are valid
              keyset.forEach(k => {
                argsMeta[k]['valid'] = true
              })
            }

            this.setState({
              args: argsMeta,
              argsValid: false
            });

          } else if (!limit_to) { // checked all args and all were valid
            
            keyset.forEach(k => {
              argsMeta[k]['valid'] = true
            })
            // It's possible all args were already valid, in which case
            // it's nice to avoid the re-render that this setState call
            // triggers. Although only the Viz app components re-render in a noticeable way.
            // I wonder if that could be due to use of redux there?
            if (!this.state.argsValid) {
              this.setState({
                args: argsMeta,
                argsValid: true
              })
            }

          } else if (limit_to) {  // checked single arg, was valid

            argsMeta[limit_to]['valid'] = true
            // this could be the last arg that needed to go valid,
            // in which case we should trigger a full args_dict validation
            this.setState({
              args: argsMeta},
              () => {
                let argIsValidArray = [];
                for (const key in argsMeta) {
                  argIsValidArray.push(argsMeta[key]['valid'])
                }
                if (argIsValidArray.every(Boolean)) {
                  const new_args_dict_string = argsValuesFromSpec(argsMeta);
                  this.investValidate(new_args_dict_string);
                }
              }
            );
          }
        } else {
          console.log('Status: ' + response.statusCode)
          if (error.message) {
            console.log('Error: ' + error.message)
          }
        }
      }
    );
  }

  investGetSpec(event) {
    const modelName = event.target.value;

    request.post(
      'http://localhost:5000/getspec',
      { json: { 
        model: modelName} 
      },
      (error, response, body) => {
        if (!error && response.statusCode == 200) {
          const spec = body;
          spec['model_temp_vizname'] = modelName // TODO: later this will be builtin to spec
          // for clarity, state has a dedicated args property separte from spec
          const args = JSON.parse(JSON.stringify(spec.args));
          delete spec.args

          // This event represents a user selecting a model,
          // and so some existing state should be reset.
          this.setState({
            modelSpec: spec,
            args: args,
            sessionProgress: 'setup',
            logStdErr: '',
            logStdOut: '',
            sessionID: defaultSessionID(spec.model_temp_vizname), // see TODO above
            workspace: null,
          }, () => this.switchTabs('setup'));
        } else {
          console.log('Status: ' + response.statusCode)
          console.log('Error: ' + error.message)
        }
      }
    );
  }

  batchUpdateArgs(args_dict) {
    // Update this.state.args in response to batch argument loading events
    const argsMeta = JSON.parse(JSON.stringify(this.state.args));
    Object.keys(args_dict).forEach(argkey => {
      argsMeta[argkey]['value'] = args_dict[argkey]
    });
    this.setState({args: argsMeta},
      () => { this.investValidate(JSON.stringify(args_dict)) }
    );
  }

  updateArg(key, value) {
    // Update this.state.args in response to change events on a single ArgsForm input

    // Parameters:
      // key (string)
      // value (string or number)

    const argsMeta = JSON.parse(JSON.stringify(this.state.args));
    argsMeta[key]['value'] = value;

    this.setState({args: argsMeta}, 
      () => {
        const args_dict_string = argsValuesFromSpec(argsMeta);
        // this.investValidate(args_dict_string, key);
        this.investValidate(args_dict_string);
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
    const logDisabled = ['models', 'setup'].includes(sessionProgress);  // enable during and after execution
    const vizDisabled = (sessionProgress !== 'viz');  // enable only on complete execute with no errors

    return(
      <Tabs id="controlled-tab-example" activeKey={activeTab} onSelect={this.switchTabs}>
        <Tab eventKey="models" title="Models">
          <ModelsTab
            investList={this.props.investList}
            investGetSpec={this.investGetSpec}
            saveState={this.saveState}
            loadState={this.loadState}
            setSessionID={this.setSessionID}
            sessionID={this.state.sessionID}
          />
        </Tab>
        <Tab eventKey="setup" title="Setup" disabled={setupDisabled}>
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
        </Tab>
        <Tab eventKey="log" title="Log" disabled={logDisabled}>
          <LogDisplay
            sessionProgress={this.state.sessionProgress}
            logStdOut={this.state.logStdOut}
            logStdErr={this.state.logStdErr}
          />
        </Tab>
        <Tab eventKey="viz" title="Viz" disabled={vizDisabled}>
        <Provider store={store}>
          <VizApp
            model={this.state.modelSpec.model_temp_vizname} // TODO: later this name will change
            workspace={this.state.workspace}
            activeTab={activeTab}/> 
        </Provider>
        </Tab>
        <Tab eventKey="docs" title="Docs">
          <DocsTab 
            docs={this.state.docs}
          />
        </Tab>
      </Tabs>
    );
  }
}

function argsToJsonFile(currentArgs, modulename) {
  // TODO: should we use the datastack.py API to create the json? 
  // make simple args json for passing to python cli
  let args_dict = {};
  for (const argname in currentArgs) {
    args_dict[argname] = currentArgs[argname]['value']
  }
  const datastack = { // keys expected by datastack.py
    args: args_dict,
    model_name: modulename,
    invest_version: '3.7.0',
  };

  const jsonContent = JSON.stringify(datastack, null, 2);
  fs.writeFile(TEMP_DIR + DATASTACK_JSON, jsonContent, 'utf8', function (err) {
    if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
    }
    console.log("JSON file was saved.");
  });
}

// TODO: move this to a module for import instead of passing around in props.
function argsValuesFromSpec(args) {
  let args_dict = {};
  for (const argname in args) {
    args_dict[argname] = args[argname]['value'] || ''
  }

  const args_dict_string = JSON.stringify(args_dict);
  return(args_dict_string)
}