'''
The Crop Production module contains the high-level code for excuting the Crop
Production model
'''

import logging
import pprint as pp

import crop_production_io as io
import crop_production_model as model

LOGGER = logging.getLogger('natcap.invest.crop_production.crop_production')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')


def execute(args):
    '''
    Entry point into the Crop Production Model

    :param str args['workspace_dir']: location into which all intermediate
        and output files should be placed.

    :param str args['results_suffix']: a string to append to output filenames

    :param str args['crop_lookup_table_uri']: filepath to a CSV table used to
        convert the crop code provided in the Crop Map to the crop name that
        can be used for searching through inputs and formatting outputs.

    :param str args['lulc_map_uri']: a GDAL-supported raster representing a
        crop management scenario.

    :param str args['fertilizer_maps_dir']: filepath to folder that
        should contain a set of GDAL-supported rasters representing the amount
        of Nitrogen (N), Phosphorous (P2O5), and Potash (K2O) applied to each
        area of land (kg/ha).

    :param str args['spatial_dataset_dir']: the provided folder should contain
        a set of folders and data specified in the 'Running the Model' section
        of the model's User Guide.

    :param boolean args['create_crop_production_maps']: If True, a set of crop
        production maps is saved within the folder of each yield function.

    :param boolean args['do_yield_observed']: if True, calculates yield based
        on observed yields within region and creates associated outputs.

    :param boolean args['do_yield_percentile']: if True, calculates yield based
        on climate-specific distribution of observed yields and creates
        associated outputs

    :param boolean args['do_yield_regression']: if True, calculates yield based on
        yield regression model with climate-specific parameters and creates
        associated outputs

    :param str args['modeled_irrigation_map_uri']: filepath to a GDAL-supported
        raster representing whether irrigation occurs or not. A zero value
        indicates that no irrigation occurs.  A one value indicates that
        irrigation occurs.  If any other values are provided, irrigation is
        assumed to occur within that cell area.

    :param boolean args['do_nutrition']: if true, calculates nutrition from
        crop production and creates associated outputs.

    :param str args['nutrition_table_uri']: filepath to a CSV table containing
        information about the nutrient contents of each crop.

    :param boolean args['do_economic_returns']: if true, calculates economic
        returns from crop production and creates associated outputs.

    :param str args['economics_table_uri']: filepath to a CSV table containing
        information related to market price of a given crop and the expenses
        involved with producing that crop.

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'results_suffix': 'scenario_name',
            'crop_lookup_table_uri': 'path/to/crop_lookup_table_uri',
            'lulc_map_uri': 'path/to/lulc_map_uri',
            'do_fertilizer_maps': True,
            'fertilizer_maps_dir': 'path/to/fertilizer_maps_dir/',
            'spatial_dataset_dir': 'path/to/spatial_dataset_dir/',
            'create_crop_production_maps': True,
            'do_yield_observed': True,
            'do_yield_percentile': True,
            'do_yield_regression': True,
            'modeled_irrigation_map_uri': 'path/to/modeled_irrigation_map_uri/',
            'do_nutrition': True,
            'nutrition_table_uri': 'path/to/nutrition_table_uri',
            'do_economic_returns': True,
            'economics_table_uri': 'path/to/economics_table_uri',
        }

    Example Returns::

        results_dict = {
            'observed_vars_dict': {
                # ...
                'crop_production_dict': {...},
                'economics_table_dict': {...},
                'crop_total_nutrition_dict': {...}
            },
            'percentile_vars_dict': {
                # ...
                'crop_production_dict': {...},
                'economics_table_dict': {...},
                'crop_total_nutrition_dict': {...}
            },
            'regression_vars_dict': {
                # ...
                'crop_production_dict': {...},
                'economics_table_dict': {...},
                'crop_total_nutrition_dict': {...}
            }
        }
    '''

    if any([args['do_yield_observed'],
            args['do_yield_percentile'],
            args['do_yield_regression']]) is False:
        LOGGER.error('No Yield Function Selected.  Cannot Run Model.')

    LOGGER.info("Beginning Model Run...")

    # Fetch and Parse Inputs
    LOGGER.info("Fetching Inputs...")
    vars_dict = io.get_inputs(args)

    # Run Model ...
    results_dict = {}

    if vars_dict['do_yield_observed']:
        LOGGER.info("Calculating Yield from Observed Regional Data...")
        observed_vars_dict = model.calc_observed_yield(vars_dict)
        results_dict['observed_vars_dict'] = observed_vars_dict

    if vars_dict['do_yield_percentile']:
        LOGGER.info("Calculating Yield from Climate-based Distribution of "
                    "Observed Yields...")
        percentile_vars_dict = model.calc_percentile_yield(vars_dict)
        results_dict['percentile_vars_dict'] = percentile_vars_dict

    if vars_dict['do_yield_regression']:
        LOGGER.info("Calculating Yield from Regression Model with "
                    "Climate-based Parameters...")
        regression_vars_dict = model.calc_regression_yield(vars_dict)
        results_dict['regression_vars_dict'] = regression_vars_dict

    LOGGER.info("...Model Run Complete.")
    return results_dict
