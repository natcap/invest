"""Coastal Blue Carbon Preprocessor."""
import os
import csv
from itertools import product
import pprint as pp
import logging

import gdal
from pygeoprocessing.geoprocessing import get_lookup_from_csv

from natcap.invest.coastal_blue_carbon import NODATA_INT, NODATA_FLOAT, HA_PER_M2
from natcap.invest.coastal_blue_carbon.classes.raster import Raster

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.coastal_blue_carbon.preprocessor')


def execute(args):
    """Execute preprocessor.

    Parameters:
        workspace_dir (string)
        results_suffix (string)
        lulc_lookup_uri (string)
        lulc_snapshot_list (list)

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'results_suffix': '',
            'lulc_lookup_uri': 'path/to/lookup.csv',
            'lulc_snapshot_list': ['path/to/raster1', 'path/to/raster2', ...]
        }
    """
    LOGGER.info('Beginning execution of Coastal Blue Carbon model...')
    vars_dict = _get_inputs(args)
    vars_dict = _preprocess_data(vars_dict)
    _create_transition_table(vars_dict)
    _create_carbon_pool_initial_table_template(vars_dict)
    _create_carbon_pool_transient_table_template(vars_dict)
    LOGGER.info('Coastal Blue Carbon model execution complete.')


def _get_inputs(args):
    """Get Inputs."""
    LOGGER.info('Getting inputs...')
    vars_dict = dict(args.items())
    try:
        vars_dict['results_suffix']
    except:
        vars_dict['results_suffix'] = ''
    # ...
    vars_dict = _get_derivative_inputs(vars_dict)
    _validate_inputs(vars_dict)
    return vars_dict


def _get_derivative_inputs(vars_dict):
    """Create variables to help with preprocessing.

    Returns:

        lulc_lookup_dict
        code_to_lulc_dict
        lulc_to_code_dict
        lulc_class_list
        output_dir
    """
    # ...
    lulc_lookup_dict = get_lookup_from_csv(vars_dict['lulc_lookup_uri'], 'code')

    for code in lulc_lookup_dict.keys():
        sub_dict = lulc_lookup_dict[code]
        val = sub_dict['is_coastal_blue_carbon_habitat']
        sub_dict['is_coastal_blue_carbon_habitat'] = eval(val.capitalize())
        lulc_lookup_dict[code] = sub_dict

    code_to_lulc_dict = {key: lulc_lookup_dict[key][
        'lulc-class'] for key in lulc_lookup_dict.keys()}
    lulc_to_code_dict = {v: k for k, v in code_to_lulc_dict.items()}

    vars_dict['lulc_lookup_dict'] = lulc_lookup_dict
    vars_dict['code_to_lulc_dict'] = code_to_lulc_dict
    vars_dict['lulc_to_code_dict'] = lulc_to_code_dict
    vars_dict['lulc_class_list'] = lulc_to_code_dict.keys()

    # make workspace and output dir if necessary
    output_dir_name = 'outputs_preprocessor'
    if vars_dict['results_suffix'] != '':
        output_dir_name = output_dir_name + '_' + vars_dict['results_suffix']
    output_dir = os.path.join(vars_dict['workspace_dir'], output_dir_name)
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            LOGGER.error("Cannot create Workspace Directory")
            raise OSError
    vars_dict['output_dir'] = output_dir

    return vars_dict


def _validate_inputs(vars_dict):
    """Validate inputs."""
    LOGGER.info('Validating inputs...')
    lulc_snapshot_list = vars_dict['lulc_snapshot_list']
    lulc_lookup_dict = vars_dict['lulc_lookup_dict']

    # assert rasters aligned
    for snapshot_idx in range(0, len(lulc_snapshot_list)-1):
        raster1 = Raster.from_file(lulc_snapshot_list[snapshot_idx])
        raster2 = Raster.from_file(lulc_snapshot_list[snapshot_idx+1])

        try:
            assert(raster1.is_aligned(raster2))
        except:
            class MisalignedRasters(ValueError):
                def __init__(self, message):
                    self.message = message
            raise MisalignedRasters(
                "At least one raster is misaligned from the others")

    # assert all raster values in lookup table
    raster_val_set = set()
    for snapshot_idx in range(0, len(lulc_snapshot_list)):
        raster = Raster.from_file(lulc_snapshot_list[
            snapshot_idx]).set_datatype_and_nodata(gdal.GDT_Int32, NODATA_INT)
        raster_val_set = raster_val_set.union(set(raster.unique()))

    code_set = set(lulc_lookup_dict.keys())
    code_set.add(NODATA_INT)

    try:
        if raster_val_set.difference(code_set):
            raise AssertionError
    except:
        msg = "These raster values are not in the lookup table: %s" %\
            raster_val_set.difference(code_set)
        raise ValueError(msg)


def _preprocess_data(vars_dict):
    """Preprocess data."""

    def _get_land_cover_transitions(raster_t1_uri, raster_t2_uri):
        raster_t1 = Raster.from_file(
            raster_t1_uri).set_datatype_and_nodata(gdal.GDT_Int32, NODATA_INT)
        raster_t2 = Raster.from_file(
            raster_t2_uri).set_datatype_and_nodata(gdal.GDT_Int32, NODATA_INT)

        band_t1 = raster_t1.get_band(1).data.flatten()
        band_t2 = raster_t2.get_band(1).data.flatten()

        transition_list = zip(band_t1, band_t2)
        transition_set = set(transition_list)

        return transition_set


    def _mark_transition_type(lookup_dict, transition_matrix_dict, lulc_from, lulc_to):
        """Mark transition type, given lulc_from and lulc_to."""
        if (lulc_from == NODATA_INT) or (lulc_to == NODATA_INT):
            pass
        elif (bool(lookup_dict[lulc_from]['is_coastal_blue_carbon_habitat']) and
            bool(lookup_dict[lulc_to]['is_coastal_blue_carbon_habitat'])):
            # veg --> veg
            transition_matrix_dict[(lulc_from, lulc_to)] = 'accum'
        elif (not bool(lookup_dict[lulc_from]['is_coastal_blue_carbon_habitat']) and
            bool(lookup_dict[lulc_to]['is_coastal_blue_carbon_habitat'])):
            # non-veg --> veg
            transition_matrix_dict[(lulc_from, lulc_to)] = 'accum'
        elif (bool(lookup_dict[lulc_from]['is_coastal_blue_carbon_habitat']) and
            not bool(lookup_dict[lulc_to]['is_coastal_blue_carbon_habitat'])):
            # veg --> non-veg
            transition_matrix_dict[(lulc_from, lulc_to)] = 'disturb'
        elif (not bool(lookup_dict[lulc_from]['is_coastal_blue_carbon_habitat']) and
              not bool(lookup_dict[lulc_to]['is_coastal_blue_carbon_habitat'])):
            # non-veg --> non-veg
            transition_matrix_dict[(lulc_from, lulc_to)] = 'NCC'
        else:
            raise Exception

        return transition_matrix_dict

    LOGGER.info('Processing data...')
    # Transition Matrix
    lulc_lookup_dict = vars_dict['lulc_lookup_dict']
    p = product(lulc_lookup_dict.keys(), repeat=2)

    transition_matrix_dict = {}
    for i in p:
        transition_matrix_dict[i] = ''

    # Determine Transitions and Directions
    lulc_snapshot_list = vars_dict['lulc_snapshot_list']
    for snapshot_idx in range(0, len(lulc_snapshot_list)-1):
        transition_set = _get_land_cover_transitions(
            lulc_snapshot_list[snapshot_idx],
            lulc_snapshot_list[snapshot_idx+1])

        # make sure that no lulc transitions interact with nodata values
        _validate_transitions(transition_set)

        for transition_tuple in transition_set:
            transition_matrix_dict = _mark_transition_type(
                lulc_lookup_dict,
                transition_matrix_dict,
                *transition_tuple)

    vars_dict['transition_matrix_dict'] = transition_matrix_dict

    return vars_dict


def _validate_transitions(transition_set):
    """Asserts that transitions between rasters are nodata to nodata
    and lulc-code to lulc-code.

    Parameters:
        transition_set (set): a set of tuples

    Raises:
        AssertionError
    """
    for t in transition_set:
        if t[0] == NODATA_INT and t[1] != NODATA_INT:
            raise AssertionError(
                'invalid transition from nodata value to lulc-code')
        if t[0] != NODATA_INT and t[1] == NODATA_INT:
            raise AssertionError(
                'invalid transition from lulc-code to nodata value')


def _create_transition_table(vars_dict):
    """Create transition table representing the lulc transition effect on
    carbon emissions or sequestration."""

    LOGGER.info('Creating transition table as output...')
    lulc_class_list = vars_dict['lulc_class_list']
    transition_matrix_dict = vars_dict['transition_matrix_dict']
    code_to_lulc_dict = vars_dict['code_to_lulc_dict']
    code_list = code_to_lulc_dict.keys()
    code_list.sort()
    lulc_class_list_sorted = [code_to_lulc_dict[code] for code in code_list]

    transition_by_lulc_class_dict = dict(
        [(lulc_class, {}) for lulc_class in lulc_class_list])

    for transition in transition_matrix_dict.keys():
        top_dict = transition_by_lulc_class_dict[
            code_to_lulc_dict[transition[0]]]
        top_dict[code_to_lulc_dict[transition[1]]] = transition_matrix_dict[
            transition]
        transition_by_lulc_class_dict[code_to_lulc_dict[transition[0]]] = \
            top_dict

    fpath = os.path.join(vars_dict['output_dir'], 'transitions.csv')
    with open(fpath, 'wb') as csv_file:
        fieldnames = ['lulc-class'] + lulc_class_list_sorted
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for code in code_list:
            lulc_class = code_to_lulc_dict[code]
            row = dict([('lulc-class', lulc_class)] + transition_by_lulc_class_dict[lulc_class].items())
            writer.writerow(row)

    _append_legend(fpath)


def _append_legend(fpath):
    """Append legend to transition csv file."""
    with open(fpath, 'ab') as csv_file:
        csv_file.write(",\n,legend")
        csv_file.write("\n,empty cells indicate that no transitions occur of that type")
        csv_file.write("\n,disturb (disturbance): change to low- med- or high-impact-disturb")
        csv_file.write("\n,accum (accumulation)")
        csv_file.write("\n,NCC (no-carbon-change)")


def _create_carbon_pool_initial_table_template(vars_dict):
    """Create carbon pool initial values table."""
    lulc_class_list = vars_dict['lulc_class_list']

    fpath = os.path.join(
        vars_dict['output_dir'], 'carbon_pool_initial_template.csv')
    with open(fpath, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['code', 'lulc-class', 'biomass', 'soil', 'litter'])
        for code in vars_dict['code_to_lulc_dict'].keys():
            row = [code, vars_dict['code_to_lulc_dict'][code]] + \
                ['', '', '']
            writer.writerow(row)


def _create_carbon_pool_transient_table_template(vars_dict):
    """Create carbon pool transient values table."""
    lulc_class_list = vars_dict['lulc_class_list']

    fpath = os.path.join(
        vars_dict['output_dir'], 'carbon_pool_transient_template.csv')
    with open(fpath, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['code', 'lulc-class', 'pool', 'half-life',
                         'low-impact-disturb', 'med-impact-disturb',
                         'high-impact-disturb', 'yearly_accumulation'])
        for code in vars_dict['code_to_lulc_dict'].keys():
            for pool in ['biomass', 'soil']:
                row = [code, vars_dict['code_to_lulc_dict'][code]] + \
                    [pool, '', '', '', '', '']
                writer.writerow(row)
