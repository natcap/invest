import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

import React from 'react';
import { createStore } from 'redux';
import { Provider } from 'react-redux';

import Tabs from 'react-bootstrap/Tabs';
import Tab from 'react-bootstrap/Tab';
import Navbar from 'react-bootstrap/Navbar';

import { MODEL_NAME, MODEL_DOCS } from './valid_HRA_args';
import validate from './validate';
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
// these options are passed to child_process spawn calls
const PYTHON_OPTIONS = {
  cwd: TEMP_DIR,
  shell: true, // without true, IOError when datastack.py loads json
  env: {GDAL_DATA: process.env.GDAL_DATA.trim()}
};
const DATASTACK_JSON = 'datastack.json' // TODO: save this to the workspace, or treat it as temp and delete it.
const CACHE_DIR = 'cache' //  for storing state config files


export class InvestJob extends React.Component {
    constructor(props) {
        super(props);
        const timestamp = new Date().toISOString()
        this.state = {
            sessionID: MODEL_NAME + '_' + timestamp,
            modelSpec: {},
            args: null,
            workspace: null,
            jobStatus: 'invalid', // (invalid, ready, running, then whatever exit code returned by cli.py)
            logStdErr: '', 
            logStdOut: '',
            activeTab: 'models',
            docs: MODEL_DOCS
        };
        
        this.checkArgsReadyToValidate = this.checkArgsReadyToValidate.bind(this);
        this.investGetSpec = this.investGetSpec.bind(this);
        this.investValidate = this.investValidate.bind(this);
        this.investExecute = this.investExecute.bind(this);
        this.switchTabs = this.switchTabs.bind(this);
        this.updateArgs = this.updateArgs.bind(this);
        this.saveState = this.saveState.bind(this);
        this.loadState = this.loadState.bind(this);
        this.setSessionID = this.setSessionID.bind(this);
    }

    saveState() {
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
      console.log(sessionID)
      const filename = path.join(CACHE_DIR, sessionID + '.json');
      if (fs.existsSync(filename)) {
        const loadedState = JSON.parse(fs.readFileSync(filename, 'utf8'));
        console.log(loadedState)
        this.setState(loadedState);
      } else {
        console.log('state file not found: ' + filename);
      }
    }

    setSessionID(event) {
      const value = event.target.value;
      console.log(value);
      this.setState(
        {sessionID: value});
    }

    investExecute() {
      // first write args to a datastack file
      argsToJSON(this.state.args, this.state.modelSpec.module);
      
      this.setState(
        {
          jobStatus: 'running',
          activeTab: 'log',
          logStdErr: '',
          logStdOut: ''
        }
      );

      const datastackPath = path.join(TEMP_DIR, 'datastack.json')
      const cmdArgs = ['-vvv', 'run', MODEL_NAME, '--headless', '-d ' + datastackPath]
      const python = spawn(INVEST_EXE, cmdArgs, PYTHON_OPTIONS);

      let stdout = this.state.logStdOut
      python.stdout.on('data', (data) => {
        stdout += `${data}`
        this.setState({
          logStdOut: stdout,
        });
      });

      let stderr = this.state.logStdErr
      python.stderr.on('data', (data) => {
        stderr += `${data}`
        this.setState({
          logStdErr: stderr,
        });
      });

      python.on('close', (code) => {
        this.setState({
          jobStatus: code,
          workspace: this.state.args.workspace_dir.value
        });
        console.log(this.state)
      });
    }

    investValidate(args) {
      // TODO: this funcs logic does not handle exceptions from invest validate
      // in which case there are no warnings, but jobStatus still updates
      // to 'ready'. Maybe that is desireable though.

      argsToJSON(args, this.state.modelSpec.module);  // first write args to datastack file

      const datastackPath = path.join(TEMP_DIR, DATASTACK_JSON)
      // if we add -vvv flags, we risk getting more stdout 
      // than expected by the results parser below.
      const cmdArgs = ['validate', '--json', datastackPath]
      const validator = spawn(INVEST_EXE, cmdArgs, PYTHON_OPTIONS);

      let warningsIssued = false;
      validator.stdout.on('data', (data) => {
        let results = JSON.parse(data.toString());
        if (results.validation_results.length) {
          warningsIssued = true
          results.validation_results.forEach(x => {
            // TODO: test this indexing against all sorts of validation results
            const argkey = x[0][0];
            const message = x[1];
            args[argkey]['validationMessage'] = message
            args[argkey]['valid'] = false
          });
        }
      });

      validator.stderr.on('data', (data) => {
        console.log(`${data}`);
      });

      validator.on('close', (code) => {
        console.log(code);

        if (warningsIssued) {
          this.setState({
            args: args,
            jobStatus: 'invalid'
          })
        } else {
          this.setState({
            jobStatus: 'ready'
          })
        }
        console.log(this.state);
      });
    }

    investGetSpec(event) {
      const modulename = event.target.value;
      const options = {
        shell: true, // without true, IOError when datastack.py loads json
      };
      const cmdArgs = ['getspec', '--json', modulename]
      const proc = spawn(INVEST_EXE, cmdArgs, options);

      proc.stdout.on('data', (data) => {
        const spec = JSON.parse(data.toString());
        spec['model_temp_vizname'] = MODEL_NAME // TODO: later this will be builtin to spec
        
        // for clarity, state has a dedicated args property separte from spec
        const args = JSON.parse(JSON.stringify(spec.args));
        delete spec.args

        this.setState({
          modelSpec: spec,
          args: args
        }, () => this.switchTabs('setup'));
      });

      proc.stderr.on('data', (data) => {
        console.log(`${data}`);
      });

      proc.on('close', (code) => {
        console.log(code);
      });
    }

    updateArgs(keys, values) {
      // Update this.state.args in response to various events

      // Allow multiple keys and values so this function can be
      // shared by events that modify a single arg (keystroke)
      // and events that modify many args (drag-drop json file).

      // Parameters:
        // keys (Array)
        // valyes (Array)

      let args = JSON.parse(JSON.stringify(this.state.args));

      for (let i = 0; i < keys.length; i++) {
        let key = keys[i];
        let value = values[i];

        const isValid = validate(
          value, args[key].type, args[key].required);

        args[key]['value'] = value;
        args[key]['valid'] = isValid;
      }

      this.setState({args: args}, 
        this.checkArgsReadyToValidate);
    }

    checkArgsReadyToValidate() {
      // Pass args to invest validate if have valid values 
      // according to validate.js
      const args = JSON.parse(JSON.stringify(this.state.args));
      let argIsValidArray = [];
      for (const key in args) {
        argIsValidArray.push(args[key]['valid'])
      }
      console.log('check Args called');

      if (argIsValidArray.every(Boolean)) {
          this.investValidate(args);
          return
      }
      
      // TODO: move this setState to investValidate?
      // then this checkArgs could live in Args component
      // but investValidate would be passed as prop to Args instead
      this.setState(
          {jobStatus: 'invalid'}
      );
    }

    switchTabs(key) {
      this.setState(
        {activeTab: key}
      );
    }

    render () {
        const activeTab = this.state.activeTab;
        const jobStatus = this.state.jobStatus;
        const setupDisabled = !(this.state.args); // enable once modelSpec has loaded
        const logDisabled = ['invalid', 'ready'].includes(jobStatus);  // enable during and after execution
        const vizDisabled = !(jobStatus === 0);  // enable only on complete execute with no errors

        return(
            <Tabs id="controlled-tab-example" activeKey={activeTab} onSelect={this.switchTabs}>
              <Tab eventKey="models" title="Models">
                <ModelsTab
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
                  modulename={this.state.modelSpec.module}
                  jobStatus={this.state.jobStatus}
                  updateArgs={this.updateArgs}
                  investExecute={this.investExecute}
                />
              </Tab>
              <Tab eventKey="log" title="Log" disabled={logDisabled}>
                <LogDisplay 
                  jobStatus={this.state.jobStatus}
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

function argsToJSON(currentArgs, modulename) {
  // TODO: should this use the datastack.py API to create the json? 
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