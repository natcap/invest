# -*- coding: utf-8 -*-
"""Coastal Blue Carbon Preprocessor."""
import os
import csv
from itertools import product
import logging
import ast
import copy

from osgeo import gdal
import pygeoprocessing.geoprocessing as geoprocess
from pygeoprocessing.geoprocessing import get_lookup_from_table
from pygeoprocessing import create_directories

from .. import utils as invest_utils

NODATA_INT = -9999  # typical integer nodata value used in rasters

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.coastal_blue_carbon.preprocessor')

_OUTPUT = {
    'transitions': 'transitions.csv',
    'carbon_pool_initial_template': 'carbon_pool_initial_template.csv',
    'carbon_pool_transient_template': 'carbon_pool_transient_template.csv'
}


def execute(args):
    """Coastal Blue Carbon Preprocessor.

    The preprocessor accepts a list of rasters and checks for cell-transitions
    across the rasters.  The preprocessor outputs a CSV file representing a
    matrix of land cover transitions, each cell prefilled with a string
    indicating whether carbon accumulates or is disturbed as a result of the
    transition, if a transition occurs.

    Args:
        workspace_dir (string): directory path to workspace
        results_suffix (string): append to outputs directory name if provided
        lulc_lookup_uri (string): filepath of lulc lookup table
        lulc_snapshot_list (list): a list of filepaths to lulc rasters

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'results_suffix': '',
            'lulc_lookup_uri': 'path/to/lookup.csv',
            'lulc_snapshot_list': ['path/to/raster1', 'path/to/raster2', ...]
        }
    """
    LOGGER.info('Starting Coastal Blue Carbon Preprocessor run...')

    # Inputs
    vars_dict = _get_inputs(args)

    # Run Preprocessor
    vars_dict['transition_matrix_dict'] = _preprocess_data(
        vars_dict['lulc_lookup_dict'], vars_dict['lulc_snapshot_list'])

    # Outputs
    base_file_path_list = [(_OUTPUT, vars_dict['output_dir'])]
    reg = invest_utils.build_file_registry(
        base_file_path_list,
        vars_dict['results_suffix'])

    _create_transition_table(
        reg['transitions'],
        vars_dict['lulc_to_code_dict'].keys(),
        vars_dict['transition_matrix_dict'],
        vars_dict['code_to_lulc_dict'])

    _create_carbon_pool_initial_table_template(
        reg['carbon_pool_initial_template'],
        vars_dict['code_to_lulc_dict'])

    _create_carbon_pool_transient_table_template(
        reg['carbon_pool_transient_template'],
        vars_dict['code_to_lulc_dict'])

    LOGGER.info('...Coastal Blue Carbon Preprocessor run complete.')


def _get_inputs(args):
    """Get Inputs.

    Args:
        args (dict): model arguments dictionary

    Returns:
        vars_dict (dict): processed data from args dictionary
    """
    LOGGER.info('Getting inputs...')
    vars_dict = dict(args.items())
    results_suffix = invest_utils.make_suffix_string(
        args, 'results_suffix')

    lulc_lookup_dict = get_lookup_from_table(args['lulc_lookup_uri'], 'code')

    for code in lulc_lookup_dict.keys():
        sub_dict = lulc_lookup_dict[code]
        val = sub_dict['is_coastal_blue_carbon_habitat'].strip().capitalize()
        if val in ['True', 'False']:
            sub_dict['is_coastal_blue_carbon_habitat'] = ast.literal_eval(val)
        else:
            raise ValueError('All land cover types must have an '
                             '\'is_coastal_blue_carbon_habitat\' '
                             'attribute set to either \'True\' or \'False\'')
        lulc_lookup_dict[code] = sub_dict

    code_to_lulc_dict = {key: lulc_lookup_dict[key][
        'lulc-class'] for key in lulc_lookup_dict.keys()}
    lulc_to_code_dict = {v: k for k, v in code_to_lulc_dict.items()}

    # Create workspace and output directories
    output_dir = os.path.join(args['workspace_dir'], 'outputs_preprocessor')
    create_directories([output_dir])

    _validate_inputs(args['lulc_snapshot_list'], lulc_lookup_dict)

    vars_dict = {
        'workspace_dir': args['workspace_dir'],
        'output_dir': output_dir,
        'results_suffix': results_suffix,
        'lulc_snapshot_list': args['lulc_snapshot_list'],
        'lulc_lookup_dict': lulc_lookup_dict,
        'code_to_lulc_dict': code_to_lulc_dict,
        'lulc_to_code_dict': lulc_to_code_dict
    }

    return vars_dict


def _validate_inputs(lulc_snapshot_list, lulc_lookup_dict):
    """Validate inputs.

    Args:
        lulc_snapshot_list (list): list of snapshot raster filepaths
        lulc_lookup_dict (dict): lookup table information
    """
    LOGGER.info('Validating inputs...')
    lulc_snapshot_list = lulc_snapshot_list
    lulc_lookup_dict = lulc_lookup_dict

    for snapshot_idx in xrange(0, len(lulc_snapshot_list)-1):
        nodata_1 = geoprocess.get_nodata_from_uri(
            lulc_snapshot_list[snapshot_idx])
        nodata_2 = geoprocess.get_nodata_from_uri(
            lulc_snapshot_list[snapshot_idx+1])
        if nodata_1 != nodata_2:
            raise ValueError('Provided rasters have different nodata values.')

    # assert all raster values in lookup table
    raster_val_set = set()
    for snapshot in lulc_snapshot_list:
        raster_val_set = raster_val_set.union(set(
            geoprocess.unique_raster_values_uri(snapshot)))

    code_set = set(lulc_lookup_dict.keys())
    code_set.add(geoprocess.get_nodata_from_uri(lulc_snapshot_list[0]))

    if raster_val_set.difference(code_set):
        msg = "These raster values are not in the lookup table: %s" % \
            raster_val_set.difference(code_set)
        raise ValueError(msg)


def _get_land_cover_transitions(raster_t1_uri, raster_t2_uri):
    """Get land cover transition.

    Args:
        raster_t1_uri (str): filepath to first raster
        raster_t2_uri (str): filepath to second raster

    Returns:
        transition_set (set): a set of all types of transitions
    """
    iter_r1 = geoprocess.iterblocks(raster_t1_uri)
    transition_nodata = geoprocess.get_nodata_from_uri(raster_t1_uri)
    transition_set = set()

    for d, a1 in iter_r1:
        a2 = read_from_raster(raster_t2_uri, d)
        transition_list = zip(a1.flatten(), a2.flatten())
        transition_set = transition_set.union(set(transition_list))

    # Remove transitions to or from cells with NODATA values
    # There may be times when the user's nodata may not match NODATA_INT
    expected_nodata_values = set([NODATA_INT, transition_nodata])
    s = copy.copy(transition_set)
    for i in s:
        for nodata_value in expected_nodata_values:
            if nodata_value in i:
                transition_set.remove(i)

    return transition_set


def read_from_raster(input_raster, offset_block):
    """Read block from raster.

    Args:
        input_raster (str): filepath to raster.
        offset_block (dict): where the block is indexed.

    Returns:
        a (np.array): the raster block.
    """
    ds = gdal.Open(input_raster)
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray(**offset_block)
    ds = None
    return a


def _mark_transition_type(lookup_dict, lulc_from, lulc_to):
    """Mark transition type, given lulc_from and lulc_to.

    Args:
        lookup_dict (dict): dictionary of lookup values
        lulc_from (int): lulc code of previous cell
        lulc_to (int): lulc code of next cell

    Returns:
        carbon (str): direction of carbon flow
    """
    from_is_habitat = \
        lookup_dict[lulc_from]['is_coastal_blue_carbon_habitat']
    to_is_habitat = \
        lookup_dict[lulc_to]['is_coastal_blue_carbon_habitat']

    if from_is_habitat and to_is_habitat:
        return 'accum'  # veg --> veg
    elif not from_is_habitat and to_is_habitat:
        return 'accum'  # non-veg --> veg
    elif from_is_habitat and not to_is_habitat:
        return 'disturb'  # veg --> non-veg
    else:
        return 'NCC'  # non-veg --> non-veg


def _preprocess_data(lulc_lookup_dict, lulc_snapshot_list):
    """Preprocess data.

    Args:
        lulc_lookup_dict (dict): dictionary of lookup values
        lulc_snapshot_list (list): list of raster paths

    Returns:
        transition_matrix_dict (dict): dictionary of transitions for transition
            matrix file.
    """
    LOGGER.info('Processing data...')

    # Transition Matrix
    transition_matrix_dict = dict(
        (i, '') for i in product(lulc_lookup_dict.keys(), repeat=2))

    # Determine Transitions and Directions
    for snapshot_idx in xrange(0, len(lulc_snapshot_list)-1):
        transition_set = _get_land_cover_transitions(
            lulc_snapshot_list[snapshot_idx],
            lulc_snapshot_list[snapshot_idx+1])

        for transition_tuple in transition_set:
            transition_matrix_dict[transition_tuple] = _mark_transition_type(
                lulc_lookup_dict, *transition_tuple)

    return transition_matrix_dict


def _create_transition_table(filepath, lulc_class_list, transition_matrix_dict,
                             code_to_lulc_dict):
    """Create transition table representing effect on emissions or sequestration.

    Args:
        filepath (str): output filepath
        lulc_class_list (list): list of lulc classes (strings)
        transition_matrix_dict (dict): dictionary of lulc transitions
        code_to_lulc_dict (dict): map lulc codes to lulc classes
    """
    LOGGER.info('Creating transition table as output...')

    code_list = code_to_lulc_dict.keys()
    code_list.sort()
    lulc_class_list_sorted = [code_to_lulc_dict[code] for code in code_list]

    transition_by_lulc_class_dict = dict(
        (lulc_class, {}) for lulc_class in lulc_class_list)

    for transition in transition_matrix_dict.keys():
        top_dict = transition_by_lulc_class_dict[
            code_to_lulc_dict[transition[0]]]
        top_dict[code_to_lulc_dict[transition[1]]] = transition_matrix_dict[
            transition]
        transition_by_lulc_class_dict[code_to_lulc_dict[transition[0]]] = \
            top_dict

    with open(filepath, 'w') as csv_file:
        fieldnames = ['lulc-class'] + lulc_class_list_sorted
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for code in code_list:
            lulc_class = code_to_lulc_dict[code]
            row = dict([('lulc-class', lulc_class)] +
                       transition_by_lulc_class_dict[lulc_class].items())
            writer.writerow(row)

    # Append legend
    with open(filepath, 'a') as csv_file:
        csv_file.write(",\n,legend")
        csv_file.write(
            "\n,empty cells indicate that no transitions occur of that type")
        csv_file.write("\n,disturb (disturbance): change to low- med- or "
                       "high-impact-disturb")
        csv_file.write("\n,accum (accumulation)")
        csv_file.write("\n,NCC (no-carbon-change)")


def _create_carbon_pool_initial_table_template(filepath, code_to_lulc_dict):
    """Create carbon pool initial values table.

    Args:
        filepath (str): filepath to carbon pool initial conditions
        code_to_lulc_dict (dict): map lulc codes to lulc classes
    """
    with open(filepath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['code', 'lulc-class', 'biomass', 'soil', 'litter'])
        for code in code_to_lulc_dict.keys():
            row = [code, code_to_lulc_dict[code]] + ['', '', '']
            writer.writerow(row)


def _create_carbon_pool_transient_table_template(filepath, code_to_lulc_dict):
    """Create carbon pool transient values table.

    Args:
        filepath (str): filepath to carbon pool initial conditions
        code_to_lulc_dict (dict): map lulc codes to lulc classes
    """
    with open(filepath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['code', 'lulc-class', 'biomass-half-life',
                         'biomass-low-impact-disturb',
                         'biomass-med-impact-disturb',
                         'biomass-high-impact-disturb',
                         'biomass-yearly-accumulation',
                         'soil-half-life', 'soil-low-impact-disturb',
                         'soil-med-impact-disturb', 'soil-high-impact-disturb',
                         'soil-yearly-accumulation'])
        for code in code_to_lulc_dict.keys():
            row = [code, code_to_lulc_dict[code]] + [''] * 10
            writer.writerow(row)
