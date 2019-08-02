// {} needed when module does not export default, I think.
// Uncaught TypeError: ... is not a function
import React from 'react';
import fs from 'fs';
import {spawn} from 'child_process';
// import HRA_ARGS from './HRA_args';
import HRA_ARGS from './valid_HRA_args'; // just for testing

function validate(value, rule) {
  // func to validate a single input value
  // returns boolean

  if (rule === 'filepath') {
    return fs.existsSync(value);
  }

  if (rule === 'directory') {
    return (fs.existsSync(value) && fs.lstatSync(value).isDirectory());
  }

  if (rule === 'integer') {
    return Number.isInteger(parseInt(value));
  }

  if (rule === 'string') {
    return true; // for the results_suffix, anything goes?
  }

  if (['select', 'checkbox'].includes(rule)) {
    return true;  // dropdowns and checkboxes are always valid
  }

  throw 'Validation rule is not defined';
}

export class InvestJob extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            args: HRA_ARGS,
            workspace: null,
            jobid: null,
            argStatus: 'invalid', // (invalid, valid)
            jobStatus: 'incomplete' // (incomplete, running, success, error)
        };
        this.handleChange = this.handleChange.bind(this);
        this.checkArgStatus = this.checkArgStatus.bind(this);
        this.executeModel = this.executeModel.bind(this);
    }

    componentDidMount() {
      // nice to validate on load, if it's possible to load with default args.
      // todo, a lot of this is duplicated in handleChange.
      let openingArgs = this.state.args
      for (const arg in openingArgs) {
        const argument = openingArgs[arg];
        if (!argument.required && argument.value !== '') {
          openingArgs[arg]['valid'] = true  
        } else {
          openingArgs[arg]['valid'] = validate(argument.value, argument.validationRules)
        }
      }

      this.setState(
          {args: openingArgs}
      );      
      this.checkArgStatus(this.state.args)
      console.log('from DidMount:')
      console.log(JSON.stringify(this.state, null, 2));
    }


    // todo: validate all args on ComponentDidMount?
    // right now validation only happens on onChange

    executeModel() {
      // todo, execute from callback? child_process async
      // set job state as running
      // on exit, set state as exit code
      const python = spawn('python', ['foo.py']);

      python.stdout.on('data', (data) => {
        console.log(`stdout: ${data}`);
      });

      python.stderr.on('data', (data) => {
        console.log(`stderr: ${data}`);
      });

      python.on('close', (code) => {
        this.setState({
          jobStatus: code,
        })
        console.log('from execute close:')
        console.log(JSON.stringify(this.state, null, 2));
      });
    }

    handleChange(event) {
      const target = event.target;
      const value = target.value;
      const name = target.name;
      const required = target.required;

      let current_args = Object.assign({}, this.state.args);
      current_args[name]['value'] = value
      // todo, maybe handle this logic withing validate?
      if (!required && value !== '') {
        current_args[name]['valid'] = true  
      } else {
        current_args[name]['valid'] = validate(value, current_args[name]['validationRules'])
      }

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
      } else {
          argStatus = 'invalid';
      }    
      this.setState(
          {argStatus: argStatus}
      );
    }

    renderForm() {
        console.log('from InvestJob:')
        console.log(JSON.stringify(this.state, null, 2));
        return(
            <ArgsForm 
                args={this.state.args}
                handleChange={this.handleChange} 
            />
        );
    }

    render () {
        const argStatus = this.state.argStatus;
        const args = this.state.args;
        let submitButton = <button 
            onClick={this.executeModel}
            disabled>
                execute
            </button>
        
        if (argStatus === 'valid') {
            submitButton = <button 
            onClick={this.executeModel}>
                execute
            </button>
        }

        return(
            <div>
              {this.renderForm()}
              {submitButton}
            </div>
        );
    }
}

class ArgsForm extends React.Component {

  render() {
    const current_args = Object.assign({}, this.props.args)
    let formItems = [];
    for (const arg in current_args) {
      const argument = current_args[arg];
      if (argument.type !== 'select') {
        formItems.push(
          <form>
            <label>
              {argument.argname}
              <input 
                name={argument.argname}
                type={argument.type}
                value={argument.value}
                required={argument.required}
                onChange={this.props.handleChange}/>
            </label>
          </form>)
      } else {
        formItems.push(
          <form>
            <label>
              {argument.argname}
              <select 
                name={argument.argname}
                value={argument.value}
                required={argument.required}
                onChange={this.props.handleChange}>
                  {argument.options.map(opt =>
                    <option value={opt}>{opt}</option>
                  )}
              </select>
            </label>
          </form>)
      }
    }

    return (
      <div>{formItems}</div>
    );
  }
}

