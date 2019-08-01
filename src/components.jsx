import React from 'react';
import fs from 'fs';

const HRA_args = {
  aoi_vector_path: {
    argname: 'aoi_vector_path',
    value:'something.shp',
    valid:false,
    touched:false,
    validationRules:'filepath',
    required:true
  },
  workspace_dir: {
    argname: 'workspace_dir',
    value: 'workspace',
    valid:false,
    touched:false,
    validationRules:'directory',
    required:true
  },
}

function validate(value, rules) {
  // func to validate a single input value
  // returns boolean
    console.log(value);
    console.log(rules);

  if (rules === 'filepath') {
    console.log(fs.existsSync(value));
    return fs.existsSync(value)
  }

  if (rules === 'directory') {
    console.log(fs.existsSync(value));
    return fs.existsSync(value)
  }

  // err here because all rules should get caught


}

export class InvestJob extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            args: HRA_args,
            workspace: null,
            jobid: null,
            status: 'invalid',
        };
        this.handleChange = this.handleChange.bind(this);
        this.checkArgs = this.checkArgs.bind(this);
    }

    handleChange(event) {
      const target = event.target;
      const value = target.value;
      const name = target.name;
      const required = target.required;
      console.log(required);

      let current_args = Object.assign({}, this.state.args);
      current_args[name]['value'] = value
      console.log(!required & value !== '');
      if (!required & value !== '') {
        current_args[name]['valid'] = true  
      } else {
        current_args[name]['valid'] = validate(value, current_args[name]['validationRules'])
      }

      this.setState(
          {args: current_args}
      );      

      this.checkArgs(this.state.args)
    }

    checkArgs(args) {
      let valids = [];
      for (const key in args) {
        valids.push(args[key]['valid'])
      }
      if (valids.every(Boolean)) {
          this.setState(
              {status: 'valid'}
          );
      }
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
        return(
            <div>
              {this.renderForm()}
            </div>   
        );
    }
}

export class ArgsForm extends React.Component {

  render() {
    // console.log(JSON.stringify({this.props}));
    // const args = this.props.args
    const current_args = Object.assign({}, this.props.args)
    let formItems = [];
    for (const arg in current_args) {
      const argument = current_args[arg];
      // console.log(argument);
      formItems.push(
        <form>
            <label>
                {argument.argname}
                <input 
                    name={argument.argname}
                    type="text"
                    value={argument.value}
                    required={argument.required}
                    // valid={argument.valid}
                    onChange={this.props.handleChange} />
            </label>
        </form>)
    }

    return (
      <div>{formItems}</div>
    );
  }
}


// const datastack = {
//     "aoi_vector_path": "HabitatRiskAssess/Input/subregions.shp",
//     "criteria_table_path": "HabitatRiskAssess/Input/exposure_consequence_criteria.csv",
//     "decay_eq": "Linear",
//     "info_table_path": "HabitatRiskAssess/Input/habitat_stressor_info.csv",
//     "max_rating": "3",
//     "resolution": "500",
//     "results_suffix": "",
//     "risk_eq": "Euclidean",
//     "visualize_outputs": true
// }



    // const parameter_set = [
    //     {key: "aoi_vector_path", type:"text", required:true},
    //     {key: "criteria_table_path", type:"text", required:true},
    //     {key: "decay_eq", type:"select", required:true},
    //     {key: "info_table_path", type:"text", required:true},
    //     {key: "max_rating", type:"number", required:true},
    //     {key: "resolution", type:"number", required:true},
    //     {key: "results_suffix", type:"text", required:false},
    //     {key: "risk_eq", type:"select", required:true},
    //     {key: "visualize_outputs", type:"checkbox", required:true}
    // ]
    
    
//     render () {    
//         return (
//             <div>
//                 <Workspace args_key='workspace_dir' required='true' />
//                 <File args_key='aoi_vector_path' required='true' />
//                 <File args_key='criteria_table_path' required='true' />
//                 <Dropdown args_key='decay_eq'
//                     options={['euclidean', 'exponential']}
//                     required='true' />
//                 <File args_key='info_table_path' required='true' />
//                 <Integer args_key='max_rating' required='true' />
//                 <Integer args_key='resolution' required='true' />
//                 <Text args_key='results_suffix' required='false' />
//                 <Dropdown args_key='risk_eq'
//                     options={['linear', 'multiplicative']}
//                     required='true' />
//                 <Checkbox args_key='visualize_outputs' required='true' />
//             </div>
//         );
//     }
// }