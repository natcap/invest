
function isSufficient(argkey, state) {
  console.log(argkey, 'is sufficient:', (state.argsEnabled[argkey] && state.argsValues.[argkey].value));
  return (state.argsEnabled[argkey] && state.argsValues.[argkey].value);
}



// carbon
// argsEnabled['cur_lulc_year'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['fut_lulc_raster'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['fut_lulc_year'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['do_redd'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['do_valuation'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['redd_lulc_raster'] = (argsEnabled['do_redd'] && argsValues['do_redd'])




const enabledConditions = {
  'cbc': {
    'use_price_table': {
      'function': isSufficient, 
      'args': ['do_valuation']
    },
    'price': {
      'function': ((x, y) => isSufficent(x) && !isSufficient(y)),
      'args': ['do_valuation', 'use_price_table']
    },
    'inflation_rate': {
      'function': ((x, y) => isSufficent(x) && !isSufficient(y)), 
      'args': ['do_valuation', 'use_price_table']
    },
    'price_table': {
      'function': ((x, y) => isSufficent(x) && !isSufficient(y)), 
      'args': ['do_valuation', 'use_price_table']
    },
    'discount_rate': {
      'function': isSufficient, 
      'args': ['do_valuation']
    }
  },
  'finfish': {
    'farm_ID': {
      'function': isSufficient, 
      'args': ['ff_farm_loc']
    },
    'g_param_a_sd': {
      'function': isSufficient, 
      'args': ['use_uncertainty']
    },
    'g_param_b_sd': {
      'function': isSufficient, 
      'args': ['use_uncertainty']
    },
    'num_monte_carlo_runs': {
      'function': isSufficient, 
      'args': ['use_uncertainty']
    },
    'p_per_kg': {
      'function': isSufficient, 
      'args': ['do_valuation']
    },
    'frac_p': {
      'function': isSufficient, 
      'args': ['do_valuation']
    },
    'discount': {
      'function': isSufficient, 
      'args': ['do_valuation']
    }
  },
  'Fisheries': {
    'population_csv_path': {
      'function': ((x, state) => !isSufficient(x, state)),
      'args': ['do_batch']
    },
    'population_csv_dir': {
      'function': isSufficient, 
      'args': ['do_batch']
    },
    'spawn_units': {
      'function': ((x, state) => isSufficient(x, state) && ['Beverton-Holt', 'Ricker'].includes(state.argsValues[x].value)), 
      'args': ['recruitment_type']
    },
    'alpha': {
      'function': ((x, state) => isSufficient(x, state) && ['Beverton-Holt', 'Ricker'].includes(state.argsValues[x].value)), 
      'args': ['recruitment_type']
    },
    'beta': {
      'function': ((x, state) => isSufficient(x, state) && ['Beverton-Holt', 'Ricker'].includes(state.argsValues[x].value)), 
      'args': ['recruitment_type']
    },
    'total_recur_recruits': {
      'function': ((x, state) => isSufficient(x, state) && state.argsValues[x].value === 'Fixed'), 
      'args': ['recruitment_type']
    },
    'migration_dir': {
      'function': isSufficient, 
      'args': ['migr_cont']
    },
    'frac_post_process': {
      'function': isSufficient, 
      'args': ['val_cont']
    },
    'unit_price': {
      'function': isSufficient, 
      'args': ['val_cont']
    }
  }
}

// argsDropDownOptions = {
//   'farm_ID': [getColNames, 'ff_farm_loc']
// }

module.exports = { enabledConditions};
