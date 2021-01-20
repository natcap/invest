import { getVectorColumnNames } from '../server_requests';

// Some input fields are rendered differently conditional upon the state of other input fields.
// This file describes these dependencies between fields.
//
// Format:
// const uiSpec = {
//     modelName: {  
//        category: {
//            arg: f
//         }
//     }
// }
// where
// - `modelName` equals `ARGS_SPEC.model_name`
// - `category` is a category that the SetupTab component looks for
//    (currently `enabledFunctions` or `dropdownFunctions`)
// - `f` is a function that accepts `SetupTab.state` as its one argument 
//     - in the `enabledFunctions` section, `f` returns a boolean where true = enabled, false = disabled
//     - in the `dropdownFunctions` section, `f` returns a list of dropdown options.

// When the SetupTab component renders, it calls `f(this.state)` to get
// the enabled state of each input, and dropdown options if any.



function isSufficient(argkey, state) {
    return state.argsEnabled[argkey] && !!state.argsValues.[argkey].value;
}


const uiConfig = {
    'Coastal Blue Carbon': {
        enabledFunctions: {
            use_price_table: isSufficient.bind(null, 'do_economic_analysis'),
            price: (state => isSufficient('do_economic_analysis', state) && 
                !isSufficient('use_price_table', state)),
            inflation_rate: (state => isSufficient('do_economic_analysis', state) && 
                !isSufficient('use_price_table', state)),
            price_table_path: isSufficient.bind(null, 'use_price_table'),
            discount_rate: isSufficient.bind(null, 'do_economic_analysis'),
        }
    },
    'Finfish Aquaculture': {
        dropdownFunctions: {
            farm_ID: (async (state) => {
                    const result = await getVectorColumnNames(state.argsValues['ff_farm_loc'].value);
                    return result.colnames || [];
                }
            )
        },
        enabledFunctions: {
            farm_ID: isSufficient.bind(null, 'ff_farm_loc'),
            g_param_a_sd: isSufficient.bind(null, 'use_uncertainty'),
            g_param_b_sd: isSufficient.bind(null, 'use_uncertainty'),
            num_monte_carlo_runs: isSufficient.bind(null, 'use_uncertainty'),
            p_per_kg: isSufficient.bind(null, 'do_valuation'),
            frac_p: isSufficient.bind(null, 'do_valuation'),
            discount: isSufficient.bind(null, 'do_valuation')
        }
    },
    'Fisheries': {
        enabledFunctions: {
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


module.exports = { uiConfig };
