
function isSufficient(argkey) {
  return (argsEnabled[argkey] && argsValues[argkey]);
}

function isNotSufficient(argkey) {
  return !isSufficient(argkey);
}



// carbon
// argsEnabled['cur_lulc_year'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['fut_lulc_raster'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['fut_lulc_year'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['do_redd'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['do_valuation'] = (argsEnabled['calc_sequestration'] && argsValues['calc_sequestration'])
// argsEnabled['redd_lulc_raster'] = (argsEnabled['do_redd'] && argsValues['do_redd'])


// cbc
const cbcArgsEnabled = {
  'landcover_snapshot_csv': true,
  'biophysical_table_path': true,
  'landcover_transitions_table': true,
  'analysis_year': true,
  'do_valuation': true,
  'use_price_table': [isSufficient, 'do_valuation'],
  'price': [(x, y) => isSufficent(x) && !isSufficient(y), 'do_valuation', 'use_price_table'],
  'inflation_rate': [(x, y) => isSufficent(x) && !isSufficient(y), 'do_valuation', 'use_price_table'],
  'price_table': [(x, y) => isSufficent(x) && isSufficient(y), 'do_valuation', 'use_price_table'],
  'discount_rate': [isSufficient, 'do_valuation']
}


// fisheries

const fisheriesArgsEnabled = {
  'aoi_vector_path': true,
  'total_timesteps': true,
  'population_type': true,
  'sexsp': true,
  'harvest_units': true,
  'do_batch': true,
  'population_csv_path': [isNotSufficient, 'do_batch'],
  'population_csv_dir': [isSufficient, 'do_batch'],
  'total_init_recruits': true,
  'recruitment_type': true,
  'spawn_units': [x => isSufficient(x) && ['Beverton-Holt', 'Ricker'].includes(x), 'recruitment_type'],
  'alpha': [x => isSufficient(x) && ['Beverton-Holt', 'Ricker'].includes(x), 'recruitment_type'],
  'beta': [x => isSufficient(x) && ['Beverton-Holt', 'Ricker'].includes(x), 'recruitment_type'],
  'total_recur_recruits': [x => isSufficient(x) && x === 'Fixed', 'recruitment_type'],
  'migr_cont': true,
  'migration_dir': [isSufficient, 'migr_cont'],
  'val_cont': true,
  'frac_post_process': [isSufficient, 'val_cont'],
  'unit_price': [isSufficient, 'val_cont']
}


// finfish
const finfishArgsEnabled = {
  'ff_farm_loc': true,
  'farm_ID': [isSufficient, 'ff_farm_loc'],
  'g_param_a': true,
  'g_param_b': true,
  'g_param_tau': true,
  'use_uncertainty': true,
  'g_param_a_sd': [isSufficient, 'use_uncertainty'],
  'g_param_b_sd': [isSufficient, 'use_uncertainty'],
  'num_monte_carlo_runs': [isSufficient, 'use_uncertainty'],
  'water_temp_tbl': true,
  'farm_op_tbl': true,
  'outplant_buffer': true,
  'do_valuation': true,
  'p_per_kg': [isSufficient, 'do_valuation'],
  'frac_p': [isSufficient, 'do_valuation'],
  'discount': [isSufficient, 'do_valuation']
}

// argsDropDownOptions = {
//   'farm_ID': [getColNames, 'ff_farm_loc']
// }

module.exports = fisheriesArgsEnabled;
