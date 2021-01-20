import { getVectorColumnNames } from '../server_requests';

// Some input fields are rendered differently conditional upon the state of other input fields.
// Right now, the conditional properties are:
//  - enabled/disabled state
//  - options for a dropdown menu
// and state properties they depend on are:
//  - enabled/disabled state
//  - value
// 
// We want to leave open the possibility of adding more conditional properties
// or dependencies, so I tried to generalize this UI spec implementation so that
// any function of the `SetupTab.state` can determine any prop of the `ArgsForm`.

// const uiSpec = {
//     modelName: {  
//        enabledConditions: {
//            arg: {
//                 function: f,
//                 args: ['arg1', 'arg2']
//             }
//         }
//     }
// }
// where
// - modelName equals ARGS_SPEC.model_name
// - `args` is a list of the args needed to determine whether `arg` is enabled
// - `f` is a function that takes `args.length + 1` arguments.
//     - in the `enabledConditions` section, `f` returns a boolean where true = enabled, false = disabled
//     - in the `dropdownOptions` section, `f` returns a list of dropdown options.

// When the SetupTab component renders, it calls `f(...args, this.state)` to get
// the enabled state of each input, and dropdown options if any.



function isSufficient(argkey, state) {
  return (state.argsEnabled[argkey] && state.argsValues.[argkey].value);
}


const newUiSpec = {
    'Coastal Blue Carbon': {
        'enabledConditions': {
            use_price_table: {isSufficient.bind(null, 'do_valuation'),
            price: (state => !isSufficient('use_price_table', state)),
            inflation_rate: (state => !isSufficient('use_price_table', state)),
            price_table: isSufficient.bind(null, 'use_price_table'),
            discount_rate: isSufficient.bind(null, 'do_valuation'),
        }
    },
    'Finfish Aquaculture': {
        dropdownOptions: {
            farm_ID: (async (state) => {
                    const result = await getVectorColumnNames(state.argsValues['ff_farm_loc'].value);
                    return result.colnames || [];
                })
            }
        },
        enabledConditions: {
            farm_ID: isSufficient.bind(null, 'ff_farm_loc'),
            g_param_a_sd: isSufficient.bind(null, 'use_uncertainty'),
            g_param_b_sd: isSufficient.bind(null, 'use_uncertainty'),
            num_monte_carlo_runs: isSufficient.bind(null, 'use_uncertainty'),
            p_per_kg: isSufficient.bind(null, 'do_valuation'),
            frac_p: isSufficient.bind(null, 'do_valuation'),
            discount: isSufficient.bind(null, 'do_valuation')
        }
    },

    Fisheries: {
        enabledConditions: {
            population_csv_path: (state => !isSufficient('do_batch', state)),
            population_csv_dir: isSufficient.bind(null, 'do_batch'), 
            spawn_units: (state => isSufficient('recruitment_type', state) && 
                ['Beverton-Holt', 'Ricker'].includes(state.argsValues['recruitment_type'].value)),
            alpha: (state => isSufficient('recruitment_type', state) && 
                ['Beverton-Holt', 'Ricker'].includes(state.argsValues['recruitment_type'].value)),
            beta: (state => isSufficient('recruitment_type', state) && 
                ['Beverton-Holt', 'Ricker'].includes(state.argsValues['recruitment_type'].value)),
            total_recur_recruits: (state => isSufficient('recruitment_type', state) && 
                state.argsValues['recruitment_type'].value === 'Fixed'), 
            migration_dir: isSufficient.bind(null, 'migr_cont'),
            frac_post_process: isSufficient.bind(null, 'val_cont'),
            unit_price: isSufficient.bind(null, 'val_cont'),
        }
    }
}


module.exports = { newUiSpec };
