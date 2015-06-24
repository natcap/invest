'''
The Crop Production IO module contains functions for handling inputs and
outputs
'''

import logging
import os
import csv
import pprint as pp
import collections

import pygeoprocessing.geoprocessing as pygeo
from raster import Raster

LOGGER = logging.getLogger('natcap.invest.crop_production.io')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')


class MissingParameter(StandardError):
    '''
    An exception class that may be raised when a necessary parameter is not
    provided by the user.
    '''
    def __init__(self, msg):
        self.msg = msg


# Fetch and Verify Arguments
def get_inputs(args):
    '''
    Fetches inputs from the user, verifies for correctness and
    completeness, and returns a list of variables dictionaries

    Args:
        args (dictionary): arguments from the user

    Returns:
        vars_dict (dictionary): dictionary of variables to be used in the model

    Example Returns::

        vars_dict = {
            # ... original args ...

            # Workspace
            'intermediate_dir': 'path/to/intermediate_dir',
            'output_dir': 'path/to/output_dir',

            # Crop Lookup Table
            'crop_lookup_dict': {
                'code': 'crop_name',
                ...
            },
            'crops_in_aoi_list': ['crop1', 'crop2', 'crop3'],

            'fertilizer_maps_dict': {
                'nitrogen': 'path/to/nitrogen_fertilizer_map',
                'phosphorous': 'path/to/phosphorous_fertilizer_map',
                'potash': 'path/to/potash_fertilizer_map'
            },

            # From spatial_dataset_dir
            'observed_yield_maps_dir': 'path/to/observed_yield_maps_dir/',
            'observed_yields_maps_dict': {
                'crop': 'path/to/crop_yield_map',
                ...
            },
            'climate_bin_maps_dir': 'path/to/climate_bin_maps_dir/',
            'climate_bin_maps_dict': {
                'crop': 'path/to/crop_climate_bin_map',
                ...
            },
            'percentile_table_uri': 'path/to/percentile_table_uri',
            'percentile_yield_dict': {
                'crop': {
                    <climate_bin>: {
                        'yield_25th': <float>,
                        'yield_50th': <float>,
                        'yield_75th': <float>,
                        'yield_95th': <float>,
                        ...
                    },
                }
                ...
            },
            'modeled_yield_tables_dir': 'path/to/modeled_yield_tables_dir',
            'modeled_yield_dict': {
                'crop': {
                    <climate_bin>: {
                        'yield_ceiling': '<float>',
                        'yield_ceiling_rf': '<float>',
                        'b_nut': '<float>',
                        'b_K2O': '<float>',
                        'c_N': '<float>',
                        'c_P2O5': '<float>',
                        'c_K2O': '<float>',
                    },
                },
                ...
            },

            # For Nutrition
            'nutrition_table_dict': {
                'crop': {
                    'fraction_refuse': <float>,
                    'protein': <float>,
                    'lipid': <float>,
                    'energy': <float>,
                    'ca': <float>,
                    'fe': <float>,
                    'mg': <float>,
                    'ph': <float>,
                    ...
                },
                ...
            },

            # For Economic Returns
            'economics_table_dict': {
                'crop': {
                    'price': <float>,
                    'cost_nitrogen': <float>,
                    'cost_phosphorous': <float>,
                    'cost_potash': <float>,
                    'cost_labor': <float>,
                    'cost_mach': <float>,
                    'cost_seed': <float>,
                    'cost_irrigation': <float>
                }
            },
        }

    '''
    vars_dict = dict(args.items())

    vars_dict = read_crop_lookup_table(vars_dict)
    vars_dict = create_crops_in_aoi_list(vars_dict)
    vars_dict = fetch_spatial_dataset(vars_dict)

    if vars_dict['do_yield_observed']:
        assert_crops_in_list(vars_dict, 'observed_yields_maps_dict')

    if vars_dict['do_yield_percentile']:
        assert_crops_in_list(vars_dict, 'percentile_yield_dict')

    if vars_dict['do_fertilizer_maps']:
        vars_dict = fetch_fertilizer_maps(vars_dict)
    else:
        vars_dict['fertilizer_maps_dict'] = {}

    if vars_dict['do_yield_regression']:
        assert_crops_in_list(vars_dict, 'modeled_yield_dict')
        if vars_dict['do_fertilizer_maps'] == False:
            LOGGER.error("Fertilizer maps must be provided to run the yield "
                         "regression model")

    if vars_dict['do_nutrition']:
        vars_dict = read_nutrition_table(vars_dict)

    if vars_dict['do_economic_returns']:
        vars_dict = read_economics_table(vars_dict)
        assert_crops_in_list(vars_dict, 'economics_table_dict')

    if not os.path.isdir(args['workspace_dir']):
        try:
            os.makedirs(args['workspace_dir'])
        except:
            LOGGER.error("Cannot create Workspace Directory")
            raise OSError

    # Validation
    try:
        vars_dict['results_suffix']
    except:
        vars_dict['results_suffix'] = ''

    # Create output directory
    output_dir = os.path.join(args['workspace_dir'], 'output')
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            LOGGER.error("Cannot create Output Directory")
            raise OSError
    vars_dict['output_dir'] = output_dir

    return vars_dict


def assert_crops_in_list(vars_dict, key):
    crops_in_aoi_list = vars_dict['crops_in_aoi_list']
    key_dict = vars_dict[key]
    key_list = key_dict.keys()
    defined_list = [crop in key_list for crop in crops_in_aoi_list]
    undefined_crops = []
    for i in range(len(crops_in_aoi_list)):
        if defined_list[i] is False:
            undefined_crops.append(crops_in_aoi_list[i])
    if len(undefined_crops) > 0:
        LOGGER.error('%s not in %s' % (undefined_crops, key))
        raise ValueError


def read_crop_lookup_table(vars_dict):
    '''
    Reads in the Crop Lookup Table and returns a dictionary

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'crop_lookup_table_uri': '/path/to/crop_lookup_table_uri'
            'crop_lookup_dict': {
                'code': 'crop_name',
                ...
            }
        }
    '''
    input_dict = pygeo.get_lookup_from_csv(
        vars_dict['crop_lookup_table_uri'], 'code')

    crop_lookup_dict = {}
    for i in input_dict:
        crop_lookup_dict[i] = input_dict[i]['crop'].lower()

    # assert codes are non-negative integers?
    keys = crop_lookup_dict.keys()
    assert(all(map(lambda x: (type(x) is int), keys)))
    assert(all(map(lambda x: (x >= 0), keys)))

    vars_dict['crop_lookup_dict'] = convert_dict_to_unicode(crop_lookup_dict)
    return vars_dict


def create_crops_in_aoi_list(vars_dict):
    '''
    Example Returns::

        vars_dict = {
            # ...
            'crops_in_aoi_list': ['corn', 'rice', 'soy']
        }
    '''
    lulc_raster = Raster.from_file(vars_dict['lulc_map_uri'])
    crop_lookup_dict = vars_dict['crop_lookup_dict']
    # array = np.unique(lulc_raster.get_band(1).data)
    array = lulc_raster.unique()

    crops_in_aoi_list = []
    for crop_num in array:
        try:
            crops_in_aoi_list.append(crop_lookup_dict[crop_num])
        except KeyError:
            LOGGER.warning("Land Use Map contains values not listed in the "
                           "Crop Lookup Table")

    vars_dict['crops_in_aoi_list'] = convert_dict_to_unicode(
        crops_in_aoi_list)
    return vars_dict


def fetch_spatial_dataset(vars_dict):
    '''
    Fetches necessary variables from provided spatial dataset folder

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'observed_yield_maps_dir': 'path/to/observed_yield_maps_dir/',
            'observed_yields_maps_dict': {
                'crop': 'path/to/crop_yield_map',
                ...
            },
            'climate_bin_maps_dir': 'path/to/climate_bin_maps_dir/',
            'climate_bin_maps_dict': {
                'crop': 'path/to/crop_climate_bin_map',
                ...
            },
            'percentile_table_uri': 'path/to/percentile_table_uri',
            'percentile_yield_dict': {
                'crop': {
                    <climate_bin>: {
                        'yield_25th': <float>,
                        'yield_50th': <float>,
                        'yield_75th': <float>,
                        'yield_95th': <float>,
                        ...
                    },
                }
                ...
            },
            'modeled_yield_tables_uri': 'path/to/modeled_yield_tables_uri',
            'modeled_yield_dict': {
                'crop': {
                    'climate_bin': {
                        'yield_ceiling': '<float>',
                        'yield_ceiling_rf': '<float>',
                        'b_nut': '<float>',
                        'b_K2O': '<float>',
                        'c_N': '<float>',
                        'c_P2O5': '<float>',
                        'c_K2O': '<float>',
                    },
                },
                ...
            },
        }
    '''
    # Dictionary in case folder structure changes during development
    spatial_dataset_dict = {
        'observed_yield_maps_dir': 'observed_yield/',
        'climate_bin_maps_dir': 'climate_bin_maps/',
        'percentile_yield_tables_dir': 'climate_percentile_yield/',
        'modeled_yield_tables_dir': 'climate_regression_yield/'
    }

    if vars_dict['do_yield_observed']:
        vars_dict['observed_yield_maps_dir'] = os.path.join(
            vars_dict['spatial_dataset_dir'],
            spatial_dataset_dict['observed_yield_maps_dir'])

        vars_dict = fetch_observed_yield_maps(vars_dict)

    if vars_dict['do_yield_percentile'] or vars_dict['do_yield_regression']:
        vars_dict['climate_bin_maps_dir'] = os.path.join(
            vars_dict['spatial_dataset_dir'],
            spatial_dataset_dict['climate_bin_maps_dir'])

        vars_dict = fetch_climate_bin_maps(vars_dict)

    if vars_dict['do_yield_percentile']:
        vars_dict['percentile_yield_tables_dir'] = os.path.join(
            vars_dict['spatial_dataset_dir'],
            spatial_dataset_dict['percentile_yield_tables_dir'])

        vars_dict = read_percentile_yield_tables(vars_dict)

    if vars_dict['do_yield_regression']:
        vars_dict['modeled_yield_tables_dir'] = os.path.join(
            vars_dict['spatial_dataset_dir'],
            spatial_dataset_dict['modeled_yield_tables_dir'])

        vars_dict = read_regression_model_yield_tables(vars_dict)
    else:
        vars_dict['fertilizer_maps_dir'] = None

    return vars_dict


def fetch_observed_yield_maps(vars_dict):
    '''
    Fetches a dictionary of URIs to observed yield maps with crop names as keys

    Args:
        observed_yield_maps_dir (str): descr

    Returns:
        observed_yields_maps_dict (dict): descr

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'observed_yield_maps_dir': 'path/to/observed_yield_maps_dir/',
            'observed_yields_maps_dict': {
                'crop': 'path/to/crop_yield_map',
                ...
            },
        }
    '''
    map_uris = _listdir(vars_dict['observed_yield_maps_dir'])

    observed_yields_maps_dict = {}
    for map_uri in map_uris:
        # Checks to make sure it's not a QGIS metadata file
        if not map_uri.endswith('.aux.xml'):
            basename = os.path.basename(map_uri)
            cropname = basename.split('_')[0]
            if cropname != '':
                observed_yields_maps_dict[cropname.lower()] = map_uri

    vars_dict['observed_yields_maps_dict'] = convert_dict_to_unicode(
        observed_yields_maps_dict)

    return vars_dict


def fetch_climate_bin_maps(vars_dict):
    '''
    Fetches a dictionary of URIs to climate bin maps with crop names as keys

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'climate_bin_maps_dir': 'path/to/climate_bin_maps_dir/',
            'climate_bin_maps_dict': {
                'crop': 'path/to/crop_climate_bin_map',
                ...
            },
        }
    '''
    map_uris = _listdir(vars_dict['climate_bin_maps_dir'])

    climate_bin_maps_dict = {}
    for map_uri in map_uris:
        # Checks to make sure it's not a QGIS metadata file
        if not map_uri.endswith('.aux.xml'):
            basename = os.path.basename(map_uri)
            cropname = basename.split('_')[0]
            if cropname != '':
                climate_bin_maps_dict[cropname.lower()] = map_uri

    vars_dict['climate_bin_maps_dict'] = convert_dict_to_unicode(
        climate_bin_maps_dict)

    return vars_dict


def read_percentile_yield_tables(vars_dict):
    '''
    Reads in the Percentile Yield Table and returns a dictionary

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'percentile_yield_tables_dir': 'path/to/percentile_yield_tables_dir/',
            'percentile_yield_dict': {
                'crop': {
                    <climate_bin>: {
                        'yield_25th': <float>,
                        'yield_50th': <float>,
                        'yield_75th': <float>,
                        'yield_95th': <float>,
                        ...
                    },
                }
                ...
            },
        }
    '''
    try:
        assert(os.path.exists(vars_dict['percentile_yield_tables_dir']))
    except:
        LOGGER.error("A filepath to the directory containing percentile yield "
                     "tables must be provided to run the percentile yield "
                     "model.")
        raise KeyError

    table_uris = _listdir(vars_dict['percentile_yield_tables_dir'])

    percentile_yield_dict = {}
    for table_uri in table_uris:
        basename = os.path.basename(table_uri)
        cropname = basename.split('_')[0].lower()
        if cropname != '':
            percentile_yield_dict[cropname] = pygeo.get_lookup_from_csv(
                table_uri, 'climate_bin')
            for c_bin in percentile_yield_dict[cropname].keys():
                del percentile_yield_dict[cropname][c_bin]['climate_bin']
                percentile_yield_dict[cropname][c_bin] = _init_empty_items(
                    percentile_yield_dict[cropname][c_bin])

            zero_bin_dict = {}
            for key in percentile_yield_dict[cropname][percentile_yield_dict[
                    cropname].keys()[0]].keys():
                zero_bin_dict[key] = ''
            percentile_yield_dict[cropname][0] = _init_empty_items(
                zero_bin_dict)

    # Add Assertion Statements?

    vars_dict['percentile_yield_dict'] = convert_dict_to_unicode(
        percentile_yield_dict)

    return vars_dict


def read_regression_model_yield_tables(vars_dict):
    '''
    Reads the regression model yield tables and returns a dictionary of values

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'modeled_yield_tables_dir': 'path/to/modeled_yield_tables_dir/',
            'modeled_yield_dict': {
                'crop': {
                    <climate_bin>: {
                        'yield_ceiling': '<float>',
                        'yield_ceiling_rf': '<float>',
                        'b_nut': '<float>',
                        'b_K2O': '<float>',
                        'c_N': '<float>',
                        'c_P2O5': '<float>',
                        'c_K2O': '<float>',
                    },
                },
                ...
            },
        }
    '''
    try:
        assert(os.path.exists(vars_dict['modeled_yield_tables_dir']))
    except:
        LOGGER.error("A filepath to the directory containing the regresison "
                     "yield tables must be provided to run the regression "
                     "yield model.")
        raise KeyError

    table_uris = _listdir(vars_dict['modeled_yield_tables_dir'])

    modeled_yield_dict = {}
    for table_uri in table_uris:
        basename = os.path.basename(table_uri)
        cropname = basename.split('_')[0].lower()
        if cropname != '':
            modeled_yield_dict[cropname] = pygeo.get_lookup_from_csv(
                table_uri, 'climate_bin')
            for c_bin in modeled_yield_dict[cropname].keys():
                del modeled_yield_dict[cropname][c_bin]['climate_bin']
                modeled_yield_dict[cropname][c_bin] = _init_empty_items(
                    modeled_yield_dict[cropname][c_bin])

            zero_bin_dict = {}
            for key in modeled_yield_dict[cropname][modeled_yield_dict[
                    cropname].keys()[0]].keys():
                zero_bin_dict[key] = ''
            modeled_yield_dict[cropname][0] = _init_empty_items(
                zero_bin_dict)

    vars_dict['modeled_yield_dict'] = convert_dict_to_unicode(
        modeled_yield_dict)

    return vars_dict


def fetch_fertilizer_maps(vars_dict):
    '''
    Fetches a dictionary of URIs to fertilizer maps with fertilizer names as
        keys.

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'fertilizer_maps_dict': {
                'nitrogen': 'path/to/nitrogen_fertilizer_map',
                'phosphorous': 'path/to/phosphorous_fertilizer_map',
                'potash': 'path/to/potash_fertilizer_map'
            },
        }
    '''
    fertilizer_list = ['nitrogen', 'phosphorous', 'potash']
    map_uris = _listdir(vars_dict['fertilizer_maps_dir'])

    fertilizer_maps_dict = {}
    for map_uri in map_uris:
        if not map_uri.endswith('.aux.xml'):
            basename = os.path.splitext(os.path.basename(map_uri))[0]
            fertilizer_name = basename.split('_')[0]
            if fertilizer_name.lower() in fertilizer_list:
                fertilizer_maps_dict[fertilizer_name.lower()] = map_uri

    # Assert that the dictionary contains maps for all three fertilizers
    try:
        assert(not set(fertilizer_list).difference(
            fertilizer_maps_dict.keys()))
    except:
        LOGGER.warning("Issue fetching fertilizer maps.  Please check that "
                       "the contents of the fertilizer maps folder are "
                       "properly formatted")

    vars_dict['fertilizer_maps_dict'] = convert_dict_to_unicode(
        fertilizer_maps_dict)

    return vars_dict


def read_nutrition_table(vars_dict):
    '''
    Reads in the Nutrition Table and returns a dictionary of values

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'nutrition_table_dict': {
                'crop': {
                    'fraction_refuse': <float>,
                    'protein': <float>,
                    'lipid': <float>,
                    'energy': <float>,
                    'ca': <float>,
                    'fe': <float>,
                    'mg': <float>,
                    'ph': <float>,
                    ...
                },
                ...
            },
        }
    '''
    input_dict = pygeo.get_lookup_from_csv(
        vars_dict['nutrition_table_uri'], 'crop')
    crops_in_aoi_list = vars_dict['crops_in_aoi_list']

    template_sub_dict = dict(input_dict[input_dict.keys()[0]])
    for i in template_sub_dict.keys():
        template_sub_dict[i] = 0
    template_sub_dict['fraction_refuse'] = 0

    nutrition_table_dict = {}
    for cropname in crops_in_aoi_list:
        try:
            sub_dict = input_dict[cropname]
            del sub_dict['crop']
            try:
                sub_dict['fraction_refuse']
            except:
                sub_dict['fraction_refuse'] = 0
            sub_dict = _init_empty_items(sub_dict)
            nutrition_table_dict[cropname.lower()] = sub_dict
        except:
            nutrition_table_dict[cropname.lower()] = template_sub_dict

    vars_dict['nutrition_table_dict'] = convert_dict_to_unicode(
        nutrition_table_dict)
    return vars_dict


def _init_empty_items(d):
    for i in d.keys():
        if d[i] == '':
            d[i] = float('nan')
    return d


def read_economics_table(vars_dict):
    '''
    Reads in the Economics Table and returns a dictionary of values

    Example Returns::

        vars_dict = {
            # ... previous vars ...

            'economics_table_dict': {
                'crop': {
                    'price': <float>,
                    'cost_nitrogen': <float>,
                    'cost_phosphorous': <float>,
                    'cost_potash': <float>,
                    'cost_labor': <float>,
                    'cost_mach': <float>,
                    'cost_seed': <float>,
                    'cost_irrigation': <float>
                }
            },
        }
    '''
    input_dict = pygeo.get_lookup_from_csv(
        vars_dict['economics_table_uri'], 'crop')

    economics_table_dict = {}
    for cropname in input_dict.keys():
        src = input_dict[cropname]
        del src['crop']
        src = _init_empty_items(src)
        economics_table_dict[cropname.lower()] = src

    vars_dict['economics_table_dict'] = convert_dict_to_unicode(input_dict)
    return vars_dict


# Helper Functions
def _listdir(path):
    '''
    A replacement for the standard os.listdir which, instead of returning
    only the filename, will include the entire path. This will use os as a
    base, then just lambda transform the whole list.

    Args:
        path (string): the location container from which we want to
            gather all files

    Returns:
        uris (list): A list of full URIs contained within 'path'
    '''
    file_names = os.listdir(path)
    uris = map(lambda x: os.path.join(path, x), file_names)

    return uris


def create_results_table(vars_dict, percentile=None, first=True):
    '''
    Creates a table of results for each yield function.  This includes
        production information as well as economic and nutrition information
        if the necessary inputs are provided.

    Example Args::

        vars_dict = {
            'crop_production_dict': {
                'corn': 12.3,
                'soy': 13.4,
                ...
            },
            'economics_table_dict': {
                'corn': {
                    'total_cost': <float>,
                    'total_revenue': <float>,
                    'total_returns': <float>,
                    ...
                }
            },
            'crop_total_nutrition_dict': {
                'corn': {...},
                ...
            },
        }
    '''
    crop_production_dict = vars_dict['crop_production_dict']

    # Build list of fieldnames
    fieldnames = ['crop', 'production']
    if percentile is not None:
        fieldnames += ['percentile']
    if vars_dict['do_economic_returns']:
        economics_table_dict = vars_dict['economics_table_dict']
        fieldnames += ['total_returns', 'total_revenue', 'total_cost']
    if vars_dict['do_nutrition']:
        crop_total_nutrition_dict = vars_dict['crop_total_nutrition_dict']
        nutrition_headers = crop_total_nutrition_dict[
            crop_total_nutrition_dict.iterkeys().next()].keys()
        fieldnames += nutrition_headers

    results_table_uri = os.path.join(
        vars_dict['output_yield_func_dir'], 'results_table.csv')

    if first:
        csvfile = open(results_table_uri, 'w')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    else:
        csvfile = open(results_table_uri, 'a')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for crop in crop_production_dict.keys():
        row = {}
        row['crop'] = crop
        row['production'] = crop_production_dict[crop]
        if percentile is not None:
            row['percentile'] = percentile
        if vars_dict['do_economic_returns']:
            row['total_returns'] = economics_table_dict[crop]['total_returns']
            row['total_revenue'] = economics_table_dict[crop]['total_revenue']
            row['total_cost'] = economics_table_dict[crop]['total_cost']
        if vars_dict['do_nutrition']:
            row = dict(row.items() + crop_total_nutrition_dict[crop].items())
        writer.writerow(row)

    csvfile.close()


def convert_dict_to_unicode(data):
    '''
    Converts strings and strings nested in dictionaries and lists
        to unicode.
    '''
    if isinstance(data, basestring):
        return data.decode('utf-8')
    elif isinstance(data, collections.Mapping):
        return dict(map(convert_dict_to_unicode, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert_dict_to_unicode, data))
    else:
        return data
