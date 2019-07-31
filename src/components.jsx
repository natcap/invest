import React from 'react';

const HRA_args = {
  aoi_vector_path: {
    argname: 'aoi_vector_path',
    value:'something.shp',
    valid:false,
    touched:false,
    validationRules:'filepath'
  },
  workspace_dir: {
    argname: 'workspace_dir',
    value: 'workspace',
    valid:false,
    touched:false,
    validationRules:'directory'
  },
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
      console.log(event);

      let current_args = Object.assign({}, this.state.args);
      current_args[name]['value'] = value

      this.setState(
          {args: current_args}
      );
      // do validation. if valid, check entire args obj
      if (required) {

      }

      this.checkArgs(this.state.args)
    }

    checkArgs(args) {
        if (args) {
          console.log('args are checked');
            this.setState(
                {status: 'valid'}
            );
        }
    }

    renderForm() {
        console.log('from InvestJob:')
        console.log(JSON.stringify(this.state));
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
      let argument = current_args[arg];
      console.log(argument);
      formItems.push(
        <form>
            <label>
                {argument.argname}
                <input 
                    name={argument.argname}
                    type="text"
                    value={argument.value}
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