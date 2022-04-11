import { getVectorColumnNames } from './server_requests';

/*
Some input fields are rendered conditionally on the state of other inputs.
This file describes these dependencies between fields.

const uiSpec = {
  modelName: {
    order: [['arg']],
    hidden: ['arg_to_hide'],
    category: {
      arg: f
    }
  }
}

where
- `order` is a 2D array of args in the order that they should be rendered.
   Args within each nested array are visually grouped together.
- `hidden` (optional) a 1D array of args that should not be displayed in a GUI.
   Use this for model-specific args, no need to include 'n_workers'.
   All args in ARGS_SPEC (except n_workers) must be contained in `order`+`hidden`.
   `hidden` is only used in tests, to catch args that should be in `order`,
   but are missing.
- `modelName` as passed to `invest getspec <modelName>`
- `category` is a category that the SetupTab component looks for
   (currently `enabledFunctions` or `dropdownFunctions`)
- `f` is a function that accepts `SetupTab.state` as its one argument
    - in the `enabledFunctions` section, `f` returns a boolean where true = enabled, false = disabled
    - in the `dropdownFunctions` section, `f` returns a list of dropdown options.
      Note: Most dropdown inputs will have a static list of options defined in the ARGS_SPEC.
      This is only for dynamically populating a dropdown.

When the SetupTab component renders, it calls `f(this.state)` to get
the enabled state of each input, and dropdown options if any.
*/

/** Check whether an input is sufficient given the state.
 *
 * @param {string} argkey - arg key of the input to check sufficiency for
 * @param {object} state - representation of the state of all inputs
 * @returns {boolean} true if the input is sufficient, false if not
 */
function isSufficient(argkey, state) {
  return state.argsEnabled[argkey] && !!state.argsValues[argkey].value;
}

/** Check whether an input is insufficient given the state.
 *
 * @param {string} argkey - arg key of the input to check insufficiency for
 * @param {object} state - representation of the state of all inputs
 * @returns {boolean} false if the input is sufficient, true if not
 */
function isNotSufficient(argkey, state) {
  return !isSufficient(argkey, state);
}

const UI_SPEC = {
  annual_water_yield: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['precipitation_path', 'eto_path', 'depth_to_root_rest_layer_path', 'pawc_path'],
      ['lulc_path', 'biophysical_table_path', 'seasonality_constant'],
      ['watersheds_path', 'sub_watersheds_path'],
      ['demand_table_path', 'valuation_table_path'],
    ],
  },
  carbon: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['lulc_cur_path', 'carbon_pools_path'],
      ['calc_sequestration', 'lulc_fut_path'],
      ['do_redd', 'lulc_redd_path'],
      [
        'do_valuation',
        'lulc_cur_year',
        'lulc_fut_year',
        'price_per_metric_ton_of_c',
        'discount_rate',
        'rate_change',
      ],
    ],
    enabledFunctions: {
      lulc_fut_path: isSufficient.bind(null, 'calc_sequestration'),

      do_redd: isSufficient.bind(null, 'calc_sequestration'),
      lulc_redd_path: isSufficient.bind(null, 'do_redd'),

      do_valuation: isSufficient.bind(null, 'calc_sequestration'),
      lulc_cur_year: isSufficient.bind(null, 'do_valuation'),
      lulc_fut_year: isSufficient.bind(null, 'do_valuation'),
      price_per_metric_ton_of_c: isSufficient.bind(null, 'do_valuation'),
      discount_rate: isSufficient.bind(null, 'do_valuation'),
      rate_change: isSufficient.bind(null, 'do_valuation'),
    },
  },
  coastal_blue_carbon: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['landcover_snapshot_csv', 'biophysical_table_path', 'landcover_transitions_table', 'analysis_year'],
      ['do_economic_analysis', 'use_price_table', 'price', 'inflation_rate', 'price_table_path', 'discount_rate'],
    ],
    enabledFunctions: {
      use_price_table: isSufficient.bind(null, 'do_economic_analysis'),
      price: ((state) => isSufficient('do_economic_analysis', state)
        && isNotSufficient('use_price_table', state)),
      inflation_rate: ((state) => isSufficient('do_economic_analysis', state)
        && isNotSufficient('use_price_table', state)),
      price_table_path: isSufficient.bind(null, 'use_price_table'),
      discount_rate: isSufficient.bind(null, 'do_economic_analysis'),
    },
  },
  coastal_blue_carbon_preprocessor: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['lulc_lookup_table_path', 'landcover_snapshot_csv'],
    ],
  },
  coastal_vulnerability: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['aoi_vector_path', 'model_resolution', 'landmass_vector_path'],
      ['bathymetry_raster_path', 'wwiii_vector_path', 'max_fetch_distance'],
      [
        'habitat_table_path',
        'shelf_contour_vector_path',
        'dem_path',
        'dem_averaging_radius',
      ],
      ['geomorphology_vector_path', 'geomorphology_fill_value'],
      ['population_raster_path', 'population_radius'],
      ['slr_vector_path', 'slr_field'],
    ],
    dropdownFunctions: {
      slr_field: ((state) => getVectorColumnNames(state.argsValues.slr_vector_path.value)),
    },
    enabledFunctions: {
      slr_field: isSufficient.bind(null, 'slr_vector_path'),
      geomorphology_fill_value: isSufficient.bind(null, 'geomorphology_vector_path'),
      population_radius: isSufficient.bind(null, 'population_raster_path'),
    },
  },
  crop_production_percentile: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['model_data_path', 'landcover_raster_path', 'landcover_to_crop_table_path', 'aggregate_polygon_path'],
    ],
  },
  crop_production_regression: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['model_data_path', 'landcover_raster_path', 'landcover_to_crop_table_path', 'fertilization_rate_table_path', 'aggregate_polygon_path'],
    ],
  },
  delineateit: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['dem_path', 'detect_pour_points', 'outlet_vector_path', 'skip_invalid_geometry'],
      ['snap_points', 'flow_threshold', 'snap_distance'],
    ],
    enabledFunctions: {
      outlet_vector_path: isNotSufficient.bind(null, 'detect_pour_points'),
      skip_invalid_geometry: isNotSufficient.bind(null, 'detect_pour_points'),
      flow_threshold: isSufficient.bind(null, 'snap_points'),
      snap_distance: isSufficient.bind(null, 'snap_points'),
    },
  },
  forest_carbon_edge_effect: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['lulc_raster_path', 'biophysical_table_path', 'pools_to_calculate'],
      ['compute_forest_edge_effects', 'tropical_forest_edge_carbon_model_vector_path', 'n_nearest_model_points', 'biomass_to_carbon_conversion_factor'],
      ['aoi_vector_path'],
    ],
    enabledFunctions: {
      tropical_forest_edge_carbon_model_vector_path: isSufficient.bind(null, 'compute_forest_edge_effects'),
      n_nearest_model_points: isSufficient.bind(null, 'compute_forest_edge_effects'),
      biomass_to_carbon_conversion_factor: isSufficient.bind(null, 'compute_forest_edge_effects'),
    },
  },
  globio: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['predefined_globio', 'globio_lulc_path'],
      ['lulc_to_globio_table_path', 'lulc_path', 'pasture_path', 'potential_vegetation_path', 'primary_threshold', 'pasture_threshold'],
      ['aoi_path', 'infrastructure_dir', 'intensification_fraction', 'msa_parameters_path'],
    ],
    enabledFunctions: {
      globio_lulc_path: isSufficient.bind(null, 'predefined_globio'),
      lulc_to_globio_table_path: isNotSufficient.bind(null, 'predefined_globio'),
      lulc_path: isNotSufficient.bind(null, 'predefined_globio'),
      pasture_path: isNotSufficient.bind(null, 'predefined_globio'),
      potential_vegetation_path: isNotSufficient.bind(null, 'predefined_globio'),
      primary_threshold: isNotSufficient.bind(null, 'predefined_globio'),
      pasture_threshold: isNotSufficient.bind(null, 'predefined_globio'),
    },
  },
  habitat_quality: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['lulc_cur_path', 'lulc_fut_path', 'lulc_bas_path'],
      ['threats_table_path', 'access_vector_path', 'sensitivity_table_path', 'half_saturation_constant'],
    ],
  },
  habitat_risk_assessment: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['info_table_path', 'criteria_table_path'],
      ['resolution', 'max_rating'],
      ['risk_eq', 'decay_eq'],
      ['aoi_vector_path'],
      ['visualize_outputs'],
    ],
  },
  ndr: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['dem_path', 'lulc_path', 'runoff_proxy_path', 'watersheds_path', 'biophysical_table_path'],
      ['calc_p'],
      ['calc_n', 'subsurface_critical_length_n', 'subsurface_eff_n'],
      ['threshold_flow_accumulation', 'k_param'],
    ],
    enabledFunctions: {
      subsurface_critical_length_n: isSufficient.bind(null, 'calc_n'),
      subsurface_eff_n: isSufficient.bind(null, 'calc_n'),
    },
  },
  pollination: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['landcover_raster_path', 'landcover_biophysical_table_path'],
      ['guild_table_path', 'farm_vector_path'],
    ],
  },
  recreation: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['aoi_path', 'start_year', 'end_year'],
      ['compute_regression', 'predictor_table_path', 'scenario_predictor_table_path'],
      ['grid_aoi', 'grid_type', 'cell_size'],
    ],
    hidden: ['hostname', 'port'],
    enabledFunctions: {
      predictor_table_path: isSufficient.bind(null, 'compute_regression'),
      scenario_predictor_table_path: isSufficient.bind(null, 'compute_regression'),
      grid_type: isSufficient.bind(null, 'grid_aoi'),
      cell_size: isSufficient.bind(null, 'grid_aoi'),
    },
  },
  routedem: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['dem_path', 'dem_band_index'],
      ['calculate_slope'],
      ['algorithm'],
      ['calculate_flow_direction'],
      ['calculate_flow_accumulation'],
      ['calculate_stream_threshold', 'threshold_flow_accumulation', 'calculate_downslope_distance'],
    ],
    enabledFunctions: {
      calculate_flow_accumulation: isSufficient.bind(null, 'calculate_flow_direction'),
      calculate_stream_threshold: isSufficient.bind(null, 'calculate_flow_accumulation'),
      threshold_flow_accumulation: isSufficient.bind(null, 'calculate_stream_threshold'),
      calculate_downslope_distance: isSufficient.bind(null, 'calculate_stream_threshold'),
    },
  },
  scenario_generator_proximity: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['base_lulc_path', 'aoi_path'],
      ['area_to_convert', 'focal_landcover_codes', 'convertible_landcover_codes', 'replacement_lucode'],
      ['convert_farthest_from_edge', 'convert_nearest_to_edge', 'n_fragmentation_steps'],
    ],
  },
  scenic_quality: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['aoi_path', 'structure_path', 'dem_path', 'refraction'],
      ['do_valuation', 'valuation_function', 'a_coef', 'b_coef', 'max_valuation_radius'],
    ],
    enabledFunctions: {
      valuation_function: isSufficient.bind(null, 'do_valuation'),
      a_coef: isSufficient.bind(null, 'do_valuation'),
      b_coef: isSufficient.bind(null, 'do_valuation'),
      max_valuation_radius: isSufficient.bind(null, 'do_valuation'),
    },
  },
  sdr: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['dem_path', 'erosivity_path', 'erodibility_path'],
      ['lulc_path', 'biophysical_table_path'],
      ['watersheds_path', 'drainage_path'],
      ['threshold_flow_accumulation', 'k_param', 'sdr_max', 'ic_0_param', 'l_max'],
    ],
  },
  seasonal_water_yield: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['lulc_raster_path', 'biophysical_table_path'],
      ['dem_raster_path', 'aoi_path'],
      ['threshold_flow_accumulation', 'beta_i', 'gamma'],
      ['user_defined_local_recharge', 'l_path', 'et0_dir', 'precip_dir', 'soil_group_path'],
      ['monthly_alpha', 'alpha_m', 'monthly_alpha_path'],
      ['user_defined_climate_zones', 'rain_events_table_path', 'climate_zone_table_path', 'climate_zone_raster_path'],
    ],
    enabledFunctions: {
      l_path: isSufficient.bind(null, 'user_defined_local_recharge'),
      et0_dir: isNotSufficient.bind(null, 'user_defined_local_recharge'),
      precip_dir: isNotSufficient.bind(null, 'user_defined_local_recharge'),
      soil_group_path: isNotSufficient.bind(null, 'user_defined_local_recharge'),
      rain_events_table_path: isNotSufficient.bind(null, 'user_defined_climate_zones'),
      climate_zone_table_path: isSufficient.bind(null, 'user_defined_climate_zones'),
      climate_zone_raster_path: isSufficient.bind(null, 'user_defined_climate_zones'),
      monthly_alpha_path: isSufficient.bind(null, 'monthly_alpha'),
      alpha_m: isNotSufficient.bind(null, 'monthly_alpha'),
    },
  },
  stormwater: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['lulc_path', 'soil_group_path', 'precipitation_path', 'biophysical_table'],
      ['adjust_retention_ratios', 'retention_radius', 'road_centerlines_path'],
      ['aggregate_areas_path', 'replacement_cost'],
    ],
    enabledFunctions: {
      retention_radius: isSufficient.bind(null, 'adjust_retention_ratios'),
      road_centerlines_path: isSufficient.bind(null, 'adjust_retention_ratios'),
    },
  },
  urban_cooling_model: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['lulc_raster_path', 'ref_eto_raster_path', 'aoi_vector_path', 'biophysical_table_path'],
      ['t_ref', 'uhi_max', 't_air_average_radius', 'green_area_cooling_distance', 'cc_method'],
      ['do_energy_valuation', 'building_vector_path', 'energy_consumption_table_path'],
      ['do_productivity_valuation', 'avg_rel_humidity'],
      ['cc_weight_shade', 'cc_weight_albedo', 'cc_weight_eti'],
    ],
    enabledFunctions: {
      building_vector_path: isSufficient.bind(null, 'do_energy_valuation'),
      energy_consumption_table_path: isSufficient.bind(null, 'do_energy_valuation'),
      avg_rel_humidity: isSufficient.bind(null, 'do_productivity_valuation'),
    },
  },
  urban_flood_risk_mitigation: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['aoi_watersheds_path', 'rainfall_depth'],
      ['lulc_path', 'curve_number_table_path', 'soils_hydrological_group_raster_path'],
      ['built_infrastructure_vector_path', 'infrastructure_damage_loss_table_path'],
    ],
  },
  wave_energy: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['wave_base_data_path', 'analysis_area', 'aoi_path', 'dem_path'],
      ['machine_perf_path', 'machine_param_path'],
      ['valuation_container', 'land_gridPts_path', 'machine_econ_path', 'number_of_machines'],
    ],
    enabledFunctions: {
      land_gridPts_path: isSufficient.bind(null, 'valuation_container'),
      machine_econ_path: isSufficient.bind(null, 'valuation_container'),
      number_of_machines: isSufficient.bind(null, 'valuation_container'),
    },
  },
  wind_energy: {
    order: [
      ['workspace_dir', 'results_suffix'],
      ['wind_data_path', 'aoi_vector_path', 'bathymetry_path', 'land_polygon_vector_path', 'global_wind_parameters_path'],
      ['turbine_parameters_path', 'number_of_turbines', 'min_depth', 'max_depth', 'min_distance', 'max_distance'],
      ['valuation_container', 'foundation_cost', 'discount_rate', 'grid_points_path', 'avg_grid_distance', 'price_table', 'wind_schedule', 'wind_price', 'rate_change'],
    ],
    enabledFunctions: {
      land_polygon_vector_path: isSufficient.bind(null, 'aoi_vector_path'),
      min_distance: isSufficient.bind(null, 'land_polygon_vector_path'),
      max_distance: isSufficient.bind(null, 'land_polygon_vector_path'),
      foundation_cost: isSufficient.bind(null, 'valuation_container'),
      discount_rate: isSufficient.bind(null, 'valuation_container'),
      grid_points_path: isSufficient.bind(null, 'valuation_container'),
      avg_grid_distance: isSufficient.bind(null, 'valuation_container'),
      price_table: isSufficient.bind(null, 'valuation_container'),
      wind_schedule: isSufficient.bind(null, 'price_table'),
      wind_price: ((state) => (
        isSufficient('valuation_container', state)
        && isNotSufficient('price_table', state)
      )),
      rate_change: ((state) => (
        isSufficient('valuation_container', state)
        && isNotSufficient('price_table', state)
      )),
    },
  },
};

export { UI_SPEC };
