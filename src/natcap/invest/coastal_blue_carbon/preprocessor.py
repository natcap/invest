# -*- coding: utf-8 -*-
"""Coastal Blue Carbon Preprocessor."""
import os
import csv
from itertools import product
import pprint as pp
import logging
import ast

from osgeo import gdal
import pygeoprocessing.geoprocessing as geoprocess
from pygeoprocessing.geoprocessing import get_lookup_from_csv
from pygeoprocessing import create_directories

NODATA_INT = -9999

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.coastal_blue_carbon.preprocessor')


def execute(args):
    """Execute preprocessor.

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
    filename = 'transitions%s.csv' % vars_dict['results_suffix']
    transition_table_filepath = os.path.join(vars_dict['output_dir'], filename)
    _create_transition_table(
        transition_table_filepath,
        vars_dict['lulc_class_list'],
        vars_dict['transition_matrix_dict'],
        vars_dict['code_to_lulc_dict'])

    filename = 'carbon_pool_initial_template%s.csv' % \
        vars_dict['results_suffix']
    initial_table_filepath = os.path.join(vars_dict['output_dir'], filename)
    _create_carbon_pool_initial_table_template(
        initial_table_filepath,
        vars_dict['lulc_class_list'],
        vars_dict['code_to_lulc_dict'])

    filename = 'carbon_pool_transient_template%s.csv' % \
        vars_dict['results_suffix']
    transient_table_filepath = os.path.join(vars_dict['output_dir'], filename)
    _create_carbon_pool_transient_table_template(
        transient_table_filepath,
        vars_dict['lulc_class_list'],
        vars_dict['code_to_lulc_dict'])

    LOGGER.info('...Coastal Blue Carbon Preprocessor run complete.')


def _get_inputs(args):
    """Get Inputs."""
    LOGGER.info('Getting inputs...')
    vars_dict = dict(args.items())
    try:
        vars_dict['results_suffix']
        vars_dict['results_suffix'] = '_' + vars_dict['results_suffix']
    except:
        vars_dict['results_suffix'] = ''

    lulc_lookup_dict = get_lookup_from_csv(
        vars_dict['lulc_lookup_uri'], 'code')

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

    vars_dict['lulc_lookup_dict'] = lulc_lookup_dict
    vars_dict['code_to_lulc_dict'] = code_to_lulc_dict
    vars_dict['lulc_to_code_dict'] = lulc_to_code_dict
    vars_dict['lulc_class_list'] = lulc_to_code_dict.keys()

    # Create workspace and output directories
    vars_dict['output_dir'] = os.path.join(
        vars_dict['workspace_dir'], 'outputs_preprocessor')
    create_directories([vars_dict['output_dir']])

    _validate_inputs(vars_dict)

    return vars_dict


def _validate_inputs(vars_dict):
    """Validate inputs."""
    LOGGER.info('Validating inputs...')
    lulc_snapshot_list = vars_dict['lulc_snapshot_list']
    lulc_lookup_dict = vars_dict['lulc_lookup_dict']

    for snapshot_idx in xrange(0, len(lulc_snapshot_list)-1):
        nodata_1 = geoprocess.get_nodata_from_uri(
            lulc_snapshot_list[snapshot_idx])
        nodata_2 = geoprocess.get_nodata_from_uri(
            lulc_snapshot_list[snapshot_idx+1])
        if nodata_1 != nodata_2:
            raise ValueError('Provided rasters have different nodata values.')

    # assert all raster values in lookup table
    raster_val_set = set()
    for snapshot_idx in xrange(0, len(lulc_snapshot_list)):
        raster_val_set = raster_val_set.union(set(
            geoprocess.unique_raster_values_uri(
                lulc_snapshot_list[snapshot_idx])))

    code_set = set(lulc_lookup_dict.keys())
    code_set.add(geoprocess.get_nodata_from_uri(
        lulc_snapshot_list[snapshot_idx]))

    if raster_val_set.difference(code_set):
        msg = "These raster values are not in the lookup table: %s" % \
            raster_val_set.difference(code_set)
        raise ValueError(msg)


def _get_land_cover_transitions(raster_t1_uri, raster_t2_uri):
    array_t1 = get_flattened_band(raster_t1_uri)
    array_t2 = get_flattened_band(raster_t2_uri)

    transition_list = zip(array_t1, array_t2)
    transition_set = set(transition_list)

    return transition_set


def get_flattened_band(uri):
    """Gets first band of raster."""
    ds = gdal.Open(uri)
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray().flatten()
    band = None
    ds = None
    return array


def _mark_transition_type(
        lookup_dict, lulc_from, lulc_to):
    """Mark transition type, given lulc_from and lulc_to."""
    from_is_habitat = bool(
        lookup_dict[lulc_from]['is_coastal_blue_carbon_habitat'])
    to_is_habitat = bool(
        lookup_dict[lulc_to]['is_coastal_blue_carbon_habitat'])

    if (lulc_from == NODATA_INT) or (lulc_to == NODATA_INT):
        pass
    elif from_is_habitat and to_is_habitat:
        return 'accum'  # veg --> veg
    elif not from_is_habitat and to_is_habitat:
        return 'accum'  # non-veg --> veg
    elif from_is_habitat and not to_is_habitat:
        return 'disturb'  # veg --> non-veg
    elif not from_is_habitat and not to_is_habitat:
        return 'NCC'  # non-veg --> non-veg
    else:
        raise Exception

    return transition_matrix_dict


def _preprocess_data(lulc_lookup_dict, lulc_snapshot_list):
    """Preprocess data."""
    LOGGER.info('Processing data...')

    # Transition Matrix
    transition_matrix_dict = dict(
        (i, '') for i in product(lulc_lookup_dict.keys(), repeat=2))

    # Determine Transitions and Directions
    lulc_snapshot_list = lulc_snapshot_list
    for snapshot_idx in xrange(0, len(lulc_snapshot_list)-1):
        transition_set = _get_land_cover_transitions(
            lulc_snapshot_list[snapshot_idx],
            lulc_snapshot_list[snapshot_idx+1])

        # make sure that no lulc transitions interact with nodata values
        for t in transition_set:
            if t[0] == NODATA_INT ^ t[1] != NODATA_INT:
                raise AssertionError(
                    'invalid transition from nodata value to lulc-code')

        for transition_tuple in transition_set:
            transition_matrix_dict[transition_tuple] = _mark_transition_type(
                lulc_lookup_dict, *transition_tuple)

    return transition_matrix_dict


def _create_transition_table(
        filepath, lulc_class_list, transition_matrix_dict, code_to_lulc_dict):
    """Create transition table representing the lulc transition effect on
    carbon emissions or sequestration."""

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
        csv_file.write(
            "\n,disturb (disturbance): change to low- med- or high-impact-disturb")
        csv_file.write("\n,accum (accumulation)")
        csv_file.write("\n,NCC (no-carbon-change)")


def _create_carbon_pool_initial_table_template(
        filepath, lulc_class_list, code_to_lulc_dict):
    """Create carbon pool initial values table."""
    with open(filepath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['code', 'lulc-class', 'biomass', 'soil', 'litter'])
        for code in code_to_lulc_dict.keys():
            row = [code, code_to_lulc_dict[code]] + ['', '', '']
            writer.writerow(row)


def _create_carbon_pool_transient_table_template(
        filepath, lulc_class_list, code_to_lulc_dict):
    """Create carbon pool transient values table."""
    with open(filepath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['code', 'lulc-class', 'pool', 'half-life',
                         'low-impact-disturb', 'med-impact-disturb',
                         'high-impact-disturb', 'yearly_accumulation'])
        for code in code_to_lulc_dict.keys():
            for pool in ['biomass', 'soil']:
                row = [code, code_to_lulc_dict[code]] + \
                    [pool, '', '', '', '', '']
                writer.writerow(row)
