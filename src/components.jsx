import React from 'react';

// export function ParameterList(props) {
//     const numbers = ['1', '2', '3', '4', '5'];
//     const listItems = numbers.map((number) =>
//         <li>{number}</li>
//     );
//     return (
//         <ul>{listItems}</ul>
//     );
// }

// export class InvestModel extends React.Component {
//     constructor(props) {
//         super(props);
//         this.state = {
//             valid: false,
//             executing: null, // todo
//             complete: null, // todo
//             parameters: {},//Array(nParameters).fill(null),
//         }
//     }

//     handleArgs(args) {  // validation on the inputs happened already
//         const current_parameters = this.state.parameters
//         let parameters = Object.assign(args, current_parameters);
//         this.setState({
//             parameters: parameters
//         })
//     }

//     render() {
//         if (isModelValid(this.state.parameters)) {
//             status = 'Ready to execute'
//         } else {
//             status = 'not ready to execute'
//         }
//         return (
//             <ArgsForm
//                 onChange={(args) => this.handleArgs(args)}
//             />
//             <div>{status}</div>
//         );
//     }
// }

// export class ArgsForm extends React.Component {
//     constructor(props) {
//         super(props);
//         this.state = {};
//     }

//     render() {
//         const arguments = this

//         return(
//             <FileInput
//                 argument={}
//             />
//         );
//     }
// }


// export class FileInput extends React.Component {
//     constructor(props) {
//         super(props);
//         this.state = {
//             key: '',
//             value: '',
//         };
//         this.handleChange = this.handleChange.bind(this);
//     }

//     handleChange(event) {
//         this.setState({value: event.target.value})
//     }

//     render() {
//         return(
//             <input type='text' value={this.state.value}
//             onChange={this.handleChange} />
//         );
//     }
// }

// function isModelValid(parameters){
//     let status = false;
//     if (parameters.every(true)) {
//         status = true
//     }
//     return(status)
// }

// // todo, maybe just store keys here, not sample values.
// function nParameters(datastack) {
//     return(Object.keys(datastack).length)
// }

const datastack = {
    "aoi_vector_path": "HabitatRiskAssess/Input/subregions.shp",
    "criteria_table_path": "HabitatRiskAssess/Input/exposure_consequence_criteria.csv",
    "decay_eq": "Linear",
    "info_table_path": "HabitatRiskAssess/Input/habitat_stressor_info.csv",
    "max_rating": "3",
    "resolution": "500",
    "results_suffix": "",
    "risk_eq": "Euclidean",
    "visualize_outputs": true
}

class ParameterForm extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isReady: false,
            args: {},
        }
    }

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
    render () {    
        return (
            <div>
                <Workspace args_key='workspace_dir' required='true' />
                <File args_key='aoi_vector_path' required='true' />
                <File args_key='criteria_table_path' required='true' />
                <Dropdown args_key='decay_eq'
                    options={['euclidean', 'exponential']}
                    required='true' />
                <File args_key='info_table_path' required='true' />
                <Integer args_key='max_rating' required='true' />
                <Integer args_key='resolution' required='true' />
                <Text args_key='results_suffix' required='false' />
                <Dropdown args_key='risk_eq'
                    options={['linear', 'multiplicative']}
                    required='true' />
                <Checkbox args_key='visualize_outputs' required='true' />
            </div>
        );
    }
}