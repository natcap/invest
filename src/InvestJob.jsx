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
import { ModelsTab } from './components/ModelsTab';
import { SetupTab } from './components/SetupTab';
import { LogDisplay } from './components/LogDisplay';
import { DocsTab } from './components/DocsTab';
import HraApp from './HraApp'
import rootReducer from './reducers';

const INVEST_EXE = process.env.INVEST
const TEMP_DIR = './'
const DATASTACK_JSON = 'datastack.json'

// Only the HraApp uses this redux store
// TODO refactor HraApp to not depend on redux.
const store = createStore(rootReducer)

export class InvestJob extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            modelSpec: {},
            // args: MODEL_ARGS,
            workspace: null,
            jobid: null,
            jobStatus: 'invalid', // (invalid, ready, running, then whatever exit code returned by cli.py)
            logStdErr: '', 
            logStdOut: '',
            activeTab: 'models',
            docs: MODEL_DOCS
        };
        this.handleChange = this.handleChange.bind(this);
        this.checkArgsReadyToValidate = this.checkArgsReadyToValidate.bind(this);
        this.loadModelSpec = this.loadModelSpec.bind(this);
        this.investValidate = this.investValidate.bind(this);
        this.executeModel = this.executeModel.bind(this);
        this.switchTabs = this.switchTabs.bind(this);
        this.argsFromJsonDrop = this.argsFromJsonDrop.bind(this);
    }

    executeModel() {
      argsToJSON(this.state.modelSpec.args);  // first write args to datastack file
      
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
          workspace: this.state.modelSpec.args.workspace_dir.value
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

    loadModelSpec(event) {
      const modulename = event.target.value;
      const options = {
        shell: true, // without true, IOError when datastack.py loads json
      };
      const cmdArgs = ['getspec', '--json', modulename]
      const proc = spawn(INVEST_EXE, cmdArgs, options);

      proc.stdout.on('data', (data) => {
        const results = JSON.parse(data.toString());
        console.log(results);
        this.setState({
          modelSpec:results,
        }, () => this.switchTabs('setup'));
      });

      proc.stderr.on('data', (data) => {
        console.log(`${data}`);
      });

      proc.on('close', (code) => {
        console.log(code);
      });
    }

    argsFromJsonDrop(event) {
      event.preventDefault();
      
      const fileList = event.dataTransfer.files;
      if (fileList.length !== 1) {
        throw alert('only drop one file at a time.')
      }
      const filepath = fileList[0].path;
      const modelParams = JSON.parse(fs.readFileSync(filepath, 'utf8'));

      let modelSpec = this.state.modelSpec
      const model_name = this.state.modelSpec.module
      if (model_name === modelParams.model_name) {
        Object.keys(modelSpec.args).forEach(argkey => {
          modelSpec.args[argkey].value = modelParams.args[argkey]
        });
        this.setState({
          modelSpec: modelSpec
        });
      } else {
        throw alert('parameter file does not match this model.')
      }
    }

    handleChange(event) {
      const target = event.target;
      const value = target.value;
      const name = target.name;
      const required = target.required;

      let current_args = Object.assign({}, this.state.modelSpec.args);
      current_args[name]['value'] = value
      current_args[name]['valid'] = validate(value, current_args[name]['validationRules'])

      this.setState(
          {args: current_args}
      );      

      this.checkArgsReadyToValidate(this.state.modelSpec.args)
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
        const setupDisabled = !Boolean(this.state.modelSpec.args); // enable once modelSpec has loaded
        const logDisabled = ['invalid', 'ready'].includes(jobStatus);  // enable during and after execution
        const vizDisabled = !Boolean(jobStatus === 0);  // enable only on complete execute with no errors

        return(
          <Tabs id="controlled-tab-example" activeKey={activeTab} onSelect={this.switchTabs}>
            <Tab eventKey="models" title="Models">
              <ModelsTab
                loadModelSpec={this.loadModelSpec}
              />
            </Tab>
            <Tab eventKey="setup" title="Setup" disabled={setupDisabled}>
              <SetupTab
                args={this.state.modelSpec.args}
                jobStatus={this.state.jobStatus}
                checkArgsReadyToValidate={this.checkArgsReadyToValidate}
                handleChange={this.handleChange}
                executeModel={this.executeModel}
                onDrop={this.argsFromJsonDrop}
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
              <HraApp
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
    console.log("JSON file was saved.");
  });
}