import React from 'react';


export class InvestJob extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            args: {
                aoi_vector_path: 'something.shp',
                workspace_dir: 'workspace'
            },
            workspace: null,
            jobid: null,
            status: 'invalid',
        };
    }

    checkArgs(event) {
        // const target = event.target;
        console.log('from checkArgs');
        console.log(event);
        // if all required args have values:
        // if (target) {
        //     this.setState(
        //         {status: 'valid'}
        //     );
        // }
    }

    renderForm() {
        console.log('from InvestJob:')
        console.log(JSON.stringify(this.state));
        return(
            <ArgsForm 
                args={this.state.args}
                onChange={this.checkArgs} />
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
  // hold state of args here

  // handleChange(event) {
  //   const target = event.target;
  //   const value = target.value;
  //   const name = target.name;

  //   this.setState(
  //       {[name]: value}
  //   );
  // }

  render() {
    // console.log(JSON.stringify({this.props}));
    // const args = this.props.args
    return (
      <form>
        <label>
          AOI:
          <input 
            name="aoi_vector_path"
            required={true}
            type="text"
            value={this.props.args.aoi_vector_path}
            // onChange={this.handleChange}
          />
        </label>
        <br />
        <label>
          Workspace:
          <input 
            name="workspace_dir"
            type="text"
            value={this.props.args.workspace_dir}
            // onChange={this.handleChange} 
          />
        </label>
      </form>
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

class FormStuff extends React.Component {
	// state = {
	// 	values = this.props.initialValues || {},
	// 	touched = {},
	// 	errors = {},
	// };

	// handleChange() {

 //    }
 //    render() {
 //    	// return(
 //    	console.log(this.props.children);
 //    	console.log(this.state);
	// //     return this.props.children({
	// // 		...this.state,
	// // 		handleChange: this.handleChange,
	// // });
	// }
}


// class ParameterForm extends React.Component {
//     constructor(props) {
//         super(props);
//         this.state = {
//             isReady: false,
//             args: {},
//         }
//     }
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
    
    // const formItems = parameter_set.map((parameter) =>
    //     <form>
    //         <label>
    //             {parameter.key}
    //             <input 
    //                 name={parameter.key}
    //                 type={parameter.type}
    //                 onChange={this.handleInputChange} />
    //         </label>

    //     </form>
    // );
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