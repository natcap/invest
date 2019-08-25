import fs from 'fs';
import path from 'path';
import {spawn} from 'child_process';

import React from 'react';
import { createStore } from 'redux';
import { Provider } from 'react-redux';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Tabs from 'react-bootstrap/Tabs';
import Tab from 'react-bootstrap/Tab';

import { MODEL_ARGS, MODULE_NAME, MODEL_NAME, MODEL_DOCS } from './valid_HRA_args';
import validate from './validate';
// import { LogDisplay } from './components/LogDisplay';
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
            argStatus: 'invalid', // (invalid, valid)
            jobStatus: 'incomplete', // (incomplete, running, then whatever exit code returned by cli.py)
            logStdErr: '', 
            logStdOut: '',
            activeTab: 'setup',
            docs: MODEL_DOCS
        };
        this.handleChange = this.handleChange.bind(this);
        this.checkArgStatus = this.checkArgStatus.bind(this);
        this.investValidate = this.investValidate.bind(this);
        this.executeModel = this.executeModel.bind(this);
        this.updateArgs = this.updateArgs.bind(this);
        this.switchTabs = this.switchTabs.bind(this);
    }

    updateArgs(args) {
      this.setState(
        {args: args}
      );      
      this.checkArgStatus(this.state.args);
    }

    investValidate(args) {
      argsToJSON(args);  // first write args to datastack file

      const options = {
        cwd: TEMP_DIR,
        shell: true, // without true, IOError when datastack.py loads json
      };
      const datastackPath = path.join(TEMP_DIR, DATASTACK_JSON)
      const cmdArgs = ['-vvv', 'validate', '--json', datastackPath]
      const python = spawn(INVEST_EXE, cmdArgs, options);

      let validationResult = this.state.investValidation
      python.stdout.on('data', (data) => {
        let results = JSON.parse(data.toString());
        console.log(results);
        if (Boolean(results.validation_results.length)) {
          results.validation_results.forEach(x => {
            const argkey = x[0][0];
            const message = x[1];
            args[argkey]['validationMessage'] = message
            args[argkey]['valid'] = false
            console.log(args);
          });
          // console.log(results);
          // validationResult = `${data}`
          this.setState({
            args: args,
            argStatus: 'invalid'
          });
        }
      });

      // let stderr = this.state.logStdErr
      python.stderr.on('data', (data) => {
        console.log(`${data}`);
        // stderr += `${data}`
        // this.setState({
        //   logStdErr: stderr,
        // });
      });

      python.on('close', (code) => {
        console.log(code);
        // this.setState({
        //   jobStatus: code,
        // });
        console.log(this.state.args);
      });

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

      this.checkArgStatus(this.state.args)
    }

    checkArgStatus(args) {
      let valids = [];
      let argStatus = '';

      for (const key in args) {
        valids.push(args[key]['valid'])
      }

      if (valids.every(Boolean)) {
          argStatus = 'valid';
          this.investValidate(args);
      } else {
          argStatus = 'invalid';
      }    
      this.setState(
          {argStatus: argStatus}
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
        const logDisabled = Boolean(jobStatus === 'incomplete');
        const vizDisabled = !Boolean(jobStatus === 0);

        // todo: get this outta here
        const viztabStyle = {
          height: '500px',
          width: '500px'
        };

        return(
          <Tabs id="controlled-tab-example" activeKey={activeTab} onSelect={this.switchTabs}>
            <Tab eventKey="setup" title="Setup">
              <SetupArguments
                args={this.state.args}
                argStatus={this.state.argStatus}
                updateArgs={this.updateArgs}
                handleChange={this.handleChange}
                executeModel={this.executeModel}
              />
            </Tab>
            <Tab eventKey="log" title="Log" disabled={logDisabled}>
              // <div 
              //   jobStatus={this.state.jobStatus}
              //   logStdOut={this.state.logStdOut}
              //   logStdErr={this.state.logStdErr}
              />
            </Tab>
            <Tab eventKey="viz" title="Viz" disabled={vizDisabled}>
            <Provider store={store}>
              <HraApp />
            </Provider>
            </Tab>
            <Tab eventKey="docs" title="Docs">
              <UserGuide 
                docs={this.state.docs}
              />
            </Tab>
          </Tabs>
        );
    }
}

class UserGuide extends React.Component {

  render () {
    let html = 'Local docs not found';
    if (fs.existsSync(this.props.docs)) {
      html = fs.readFileSync(this.props.docs, 'utf8');
    }
    const docStyle = {
      whiteSpace: 'pre-line'
    };
    return(
        <div><p dangerouslySetInnerHTML={{__html: html}}/></div>
      );
  }
}

class SetupArguments extends React.Component {

  componentDidMount() {
    // nice to validate on load, if it's possible to load with default args.
    let openingArgs = this.props.args
    for (const argname in openingArgs) {
      const argument = openingArgs[argname];
      openingArgs[argname]['valid'] = validate(argument.value, argument.validationRules)
    }

    this.props.updateArgs(openingArgs)
  }

  render () {

    let submitButton = <Button 
            onClick={this.props.executeModel}
            disabled>
                execute
            </Button>
        
    if (this.props.argStatus === 'valid') {
        submitButton = <Button 
        onClick={this.props.executeModel}>
            execute
        </Button>
    }

    return (
      <div>
        <ArgsForm 
          args={this.props.args}
          handleChange={this.props.handleChange} 
        />
        {submitButton}
      </div>);
  }
}

class ArgsForm extends React.Component {

  render() {
    const current_args = Object.assign({}, this.props.args)
    let formItems = [];
    let validationMessage = '';
    for (const arg in current_args) {
      const argument = current_args[arg];
      if (argument.validationMessage) {
        validationMessage = argument.validationMessage ;
      }
      if (argument.type !== 'select') {
        formItems.push(
          <Form>
            <Form.Label>
              {argument.argname}
            </Form.Label>
            <Form.Control 
              name={argument.argname}
              type={argument.type}
              value={argument.value}
              required={argument.required}
              onChange={this.props.handleChange}
              isValid={argument.valid}
              isInvalid={!argument.valid}
            />
            <Form.Control.Feedback type='invalid'>
              {argument.validationRules.rule + ' : ' + validationMessage}
            </Form.Control.Feedback>
          </Form>)
      } else {
        formItems.push(
          <Form>
            <Form.Label>
              {argument.argname}
            </Form.Label>
            <Form.Control as='select'
              name={argument.argname}
              value={argument.value}
              required={argument.required}
              onChange={this.props.handleChange}
            >
              {argument.options.map(opt =>
                <option value={opt}>{opt}</option>
              )}
            </Form.Control>
            <Form.Control.Feedback type='invalid'>
              {argument.validationRules.rule + ' : ' + validationMessage}
            </Form.Control.Feedback>
          </Form>)
      }
    }

    return (
      <div>{formItems}</div>
    );
  }
}

