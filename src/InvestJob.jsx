import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

import React from 'react';
import { createStore } from 'redux';
import { Provider } from 'react-redux';

import Tabs from 'react-bootstrap/Tabs';
import Tab from 'react-bootstrap/Tab';

import { MODEL_ARGS, MODULE_NAME, MODEL_NAME, MODEL_DOCS } from './valid_HRA_args';
import validate from './validate';
import { SetupTab } from './components/SetupTab';
import { LogDisplay } from './components/LogDisplay';
import { DocsTab } from './components/DocsTab';
import HraApp from './HraApp'
import rootReducer from './reducers';
// need the HraApp's index.css?

// const INVEST_EXE = 'C:/InVEST_3.7.0_x86/invest-3-x86/invest.exe'
const INVEST_EXE = process.env.INVEST
const TEMP_DIR = './'
const DATASTACK_JSON = 'datastack.json'

// Only the HraApp uses this redux store
// TODO refactor HraApp to not depend on redux.
const store = createStore(rootReducer)

function argsToJSON(currentArgs) {
  // TODO: should this use the datastack.py API to create the json? 
  // make simple args json for passing to python cli
  let args_dict = {};
  for (const argname in currentArgs) {
    args_dict[argname] = currentArgs[argname]['value']
  }
  const datastack = { // keys expected by datastack.py
    args: args_dict,
    model_name: MODULE_NAME,
    invest_version: '3.7.0',
  };

  const jsonContent = JSON.stringify(datastack, null, 2);
  fs.writeFile(TEMP_DIR + DATASTACK_JSON, jsonContent, 'utf8', function (err) {
    if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
    }
    console.log("JSON file has been saved.");
  });
}

export class InvestJob extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            args: MODEL_ARGS,
            workspace: null,
            jobid: null,
            jobStatus: 'invalid', // (invalid, ready, running, then whatever exit code returned by cli.py)
            logStdErr: '', 
            logStdOut: '',
            activeTab: 'setup',
            docs: MODEL_DOCS
        };
        this.handleChange = this.handleChange.bind(this);
        this.checkArgsReadyToValidate = this.checkArgsReadyToValidate.bind(this);
        this.investValidate = this.investValidate.bind(this);
        this.executeModel = this.executeModel.bind(this);
        this.switchTabs = this.switchTabs.bind(this);
    }

    executeModel() {
      argsToJSON(this.state.args);  // first write args to datastack file
      
      this.setState(
        {
          jobStatus: 'running',
          activeTab: 'log',
          logStdErr: '',
          logStdOut: ''
        }
      );

      const options = {
        cwd: TEMP_DIR,
        shell: true, // without true, IOError when datastack.py loads json
      };
      const datastackPath = path.join(TEMP_DIR, 'datastack.json')
      const cmdArgs = ['-vvv', 'run', MODEL_NAME, '--headless', '-d ' + datastackPath]
      const python = spawn(INVEST_EXE, cmdArgs, options);

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
        });
        console.log(this.state)
      });
    }

    investValidate(args) {
      argsToJSON(args);  // first write args to datastack file

      const options = {
        cwd: TEMP_DIR,
        shell: true, // without true, IOError when datastack.py loads json
      };
      const datastackPath = path.join(TEMP_DIR, DATASTACK_JSON)
      const cmdArgs = ['-vvv', 'validate', '--json', datastackPath]
      const validator = spawn(INVEST_EXE, cmdArgs, options);

      let outgoingArgs = Object.assign({}, args);
      let argsModified = false;
      validator.stdout.on('data', (data) => {
        let results = JSON.parse(data.toString());
        console.log(results);
        if (Boolean(results.validation_results.length)) {
          argsModified = true
          results.validation_results.forEach(x => {
            const argkey = x[0][0];
            const message = x[1];
            outgoingArgs[argkey]['validationMessage'] = message
            outgoingArgs[argkey]['valid'] = false
            console.log(outgoingArgs);
          });
        }
      });

      validator.stderr.on('data', (data) => {
        console.log(`${data}`);
      });

      validator.on('close', (code) => {
        console.log(code);

        if (argsModified) {
          this.setState({
            args: outgoingArgs,
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

    handleChange(event) {
      const target = event.target;
      const value = target.value;
      const name = target.name;
      const required = target.required;

      let current_args = Object.assign({}, this.state.args);
      current_args[name]['value'] = value
      current_args[name]['valid'] = validate(value, current_args[name]['validationRules'])

      this.setState(
          {args: current_args}
      );      

      this.checkArgsReadyToValidate(this.state.args)
    }

    checkArgsReadyToValidate(args) {
      let argIsValidArray = [];
      for (const key in args) {
        argIsValidArray.push(args[key]['valid'])
      }

      if (argIsValidArray.every(Boolean)) {
          this.investValidate(args);
          return
      }
      
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
        const logDisabled = ['invalid', 'ready'].includes(jobStatus);  // enable during and after execution
        const vizDisabled = !Boolean(jobStatus === 0);  // enable only on complete execute with no errors

        return(
          <Tabs id="controlled-tab-example" activeKey={activeTab} onSelect={this.switchTabs}>
            <Tab eventKey="setup" title="Setup">
              <SetupTab
                args={this.state.args}
                jobStatus={this.state.jobStatus}
                checkArgsReadyToValidate={this.checkArgsReadyToValidate}
                handleChange={this.handleChange}
                executeModel={this.executeModel}
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
              <HraApp />
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

