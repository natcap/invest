# -*- coding: utf-8 -*-
"""Coastal Blue Carbon Preprocessor."""
import os
import itertools
from itertools import product
import logging
import copy
from functools import reduce

from osgeo import gdal
import numpy
import pygeoprocessing

from .. import utils
from .. import validation
from . import coastal_blue_carbon2


NODATA_INT = -9999  # typical integer nodata value used in rasters

LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": "Coastal Blue Carbon Preprocessor",
    "module": __name__,
    "userguide_html": "coastal_blue_carbon.html",
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "lulc_lookup_table_path": {
            "name": "LULC Lookup Table",
            "type": "csv",
            "about": (
                "A CSV table used to map lulc classes to their values "
                "in a raster, as well as to indicate whether or not "
                "the lulc class is a coastal blue carbon habitat."),
            "required": True,
            "validation_options": {
                "required_fields": ["lulc-class", "code",
                                    "is_coastal_blue_carbon_habitat"]
            },
        },
        "landcover_snapshot_csv": {
            "validation_options": {
                "required_fields": ["snapshot_year", "raster_path"],
            },
            "type": "csv",
            "required": True,
            "about": (
                "A CSV table where each row represents the year and path "
                "to a raster file on disk representing the landcover raster "
                "representing the state of the landscape in that year. "
                "Landcover codes match those in the LULC lookup table."
            ),
            "name": "Transitions Table",
        },
    }
}


ALIGNED_LULC_RASTER_TEMPLATE = 'aligned_lulc_{year}{suffix}.tif'
TRANSITION_TABLE = 'carbon_pool_transient_template{suffix}.tif'
BIOPHYSICAL_TABLE = 'carbon_biophysical_table_template{suffix}.tif'

_OUTPUT = {
    'aligned_lulc_template': 'aligned_lulc_%s.tif',
    'transitions': 'transitions.csv',
    'carbon_pool_initial_template': 'carbon_pool_initial_template.csv',
    'carbon_pool_transient_template': 'carbon_pool_transient_template.csv'
}


def execute(args):
    """Coastal Blue Carbon Preprocessor.

    The preprocessor accepts a list of rasters and checks for cell-transitions
    across the rasters.  The preprocessor outputs a CSV file representing a
    matrix of land cover transitions, each cell pre-filled with a string
    indicating whether carbon accumulates or is disturbed as a result of the
    transition, if a transition occurs.

    Args:
        args['workspace_dir'] (string): directory path to workspace
        args['results_suffix'] (string): append to outputs directory name if provided
        args['lulc_lookup_table_path'] (string): filepath of lulc lookup table
        args['landcover_csv_path'] (string): filepath to a CSV containing the
            year and filepath to snapshot rasters on disk.  The years may be in
            any order, but must be unique.

    Returns:
        ``None``
    """
    suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'], 'outputs_preprocessor')

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(
        taskgraph_cache_dir, n_workers, reporting_interval=5.0)

    snapshots_dict = (
        coastal_blue_carbon2._extract_snapshots_from_tabls(
            args['landcover_snapshot_csv']))

    # Align the raster stack for analyzing the various transitions.
    min_pixel_size = float('inf')
    source_snapshot_paths = []
    aligned_snapshot_paths = []
    for snapshot_year, raster_path in snapshots_dict.items():
        source_snapshot_paths.append(raster_path)
        aligned_snapshot_paths.append(ALIGNED_LULC_RASTER_TEMPLATE.format(
            year=snapshot_year, suffix=suffix))
        min_pixel_size = min(
            utils.mean_pixel_size_and_area(
                pygeoprocessing.get_raster_info(raster_path)['pixel_size'])[0],
            min_pixel_size)

    alignment_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(source_snapshot_paths,
              aligned_snapshot_paths,
              ['nearest']*len(source_snapshot_paths),
              (min_pixel_size, -min_pixel_size),
              'intersection'),
        hash_algorithm='md5',
        copy_duplicate_artifact=True,
        target_path_list=aligned_snapshot_paths,
        task_name='Align input landcover rasters')

    landcover_table = utils.build_lookup_from_csv(
        args['lulc_lookup_table_path'], 'code')

    target_transition_table = os.path.join(
        output_dir, TRANSITION_TABLE.format(suffix=suffix))
    transition_matrix_creation_task = task_graph.add_task(
        func=_create_transition_table,
        args=(landcover_table,
              sorted(snapshots_dict.values(), key=lambda x:[0]),
             target_transition_table),
        target_path_list=[target_transition_table],
        dependent_task_list=[alignment_task],
        task_name='Determine transitions and write transition table')

    # Creating this table should be cheap, so it's not in a task.
    # This is only likely to be expensive if the user provided a landcover
    # lookup with many many rows. If that happens, this tool will have other
    # problems (like exploding memory in the transition table creation).
    target_biophysical_table_path = os.path.join(
        output_dir, BIOPHYSICAL_TABLE.format(suffix=suffix))
    _create_biophysical_table(landcover_table, target_biophysical_table_path)

    task_graph.close()
    task_graph.join()


def execute_old(args):
    """Coastal Blue Carbon Preprocessor.

    The preprocessor accepts a list of rasters and checks for cell-transitions
    across the rasters.  The preprocessor outputs a CSV file representing a
    matrix of land cover transitions, each cell pre-filled with a string
    indicating whether carbon accumulates or is disturbed as a result of the
    transition, if a transition occurs.

    Args:
        workspace_dir (string): directory path to workspace
        results_suffix (string): append to outputs directory name if provided
        lulc_lookup_uri (string): filepath of lulc lookup table
        landcover_csv_path (string): filepath to a CSV containing the
            year and filepath to a raster on disk.

    Returns:
        ``None``
    """
    LOGGER.info('Starting Coastal Blue Carbon Preprocessor run...')

    # Inputs
    results_suffix = utils.make_suffix_string(
        args, 'results_suffix')
    vars_dict = _get_inputs(args)

    base_file_path_list = [(_OUTPUT, vars_dict['output_dir'])]
    reg = utils.build_file_registry(
        base_file_path_list,
        vars_dict['results_suffix'])

    aligned_lulcs = [reg['aligned_lulc_template'] % index
                     for index in range(len(vars_dict['lulc_snapshot_list']))]
    min_pixel_raster_info = min(
        (pygeoprocessing.get_raster_info(path) for path
         in vars_dict['lulc_snapshot_list']),
        key=lambda info: utils.mean_pixel_size_and_area(
            info['pixel_size'])[0])
    pygeoprocessing.align_and_resize_raster_stack(
        vars_dict['lulc_snapshot_list'],
        aligned_lulcs,
        ['near'] * len(aligned_lulcs),
        min_pixel_raster_info['pixel_size'],
        'intersection')

    # Run Preprocessor
    vars_dict['transition_matrix_dict'] = _preprocess_data(
        vars_dict['lulc_lookup_dict'], aligned_lulcs)

    # Outputs
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
    results_suffix = utils.make_suffix_string(
        args, 'results_suffix')

    lulc_lookup_dict = utils.build_lookup_from_csv(
        args['lulc_lookup_table_path'], 'code')

    for code in lulc_lookup_dict.keys():
        sub_dict = lulc_lookup_dict[code]
        if not isinstance(sub_dict['is_coastal_blue_carbon_habitat'], bool):
            raise ValueError(
                'All land cover types must have an '
                '\'is_coastal_blue_carbon_habitat\' '
                'attribute set to either \'True\' or \'False\'')
        lulc_lookup_dict[code] = sub_dict

    code_to_lulc_dict = {key: lulc_lookup_dict[key][
        'lulc-class'] for key in lulc_lookup_dict.keys()}
    lulc_to_code_dict = {v: k for k, v in code_to_lulc_dict.items()}

    # Create workspace and output directories
    output_dir = os.path.join(args['workspace_dir'], 'outputs_preprocessor')
    utils.make_directories([output_dir])

    snapshots_dict = (
        coastal_blue_carbon2._extract_transitions_from_table(
            args['landcover_snapshot_csv']))
    snapshots_list = sorted([
        raster for (year, raster) in sorted(
            snapshots_dict.items(), key=lambda x: x[0])])
    _validate_inputs(snapshots_list, lulc_lookup_dict)

    vars_dict = {
        'workspace_dir': args['workspace_dir'],
        'output_dir': output_dir,
        'results_suffix': results_suffix,
        'lulc_snapshot_list': snapshots_list,
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

    nodata_values = set([pygeoprocessing.get_raster_info(filepath)['nodata'][0]
                         for filepath in lulc_snapshot_list])
    if len(nodata_values) > 1:
        raise ValueError('Provided rasters have different nodata values')

    # assert all raster values in lookup table
    raster_val_set = set(reduce(
        lambda accum_value, x: numpy.unique(
            numpy.append(accum_value, next(x)[1].flat)),
        itertools.chain(pygeoprocessing.iterblocks((snapshot, 1))
                        for snapshot in lulc_snapshot_list),
        numpy.array([])))

    code_set = set(lulc_lookup_dict)
    code_set.add(
        pygeoprocessing.get_raster_info(lulc_snapshot_list[0])['nodata'][0])

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
    transition_nodata = pygeoprocessing.get_raster_info(
        raster_t1_uri)['nodata'][0]
    transition_set = set()

    for d, a1 in pygeoprocessing.iterblocks((raster_t1_uri, 1)):
        raster = gdal.OpenEx(raster_t2_uri, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        a2 = band.ReadAsArray(**d)
        band = None
        raster = None
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


def _create_transition_table(landcover_table, lulc_snapshot_list,
                             target_table_path):

    def _read_block(raster_path, offset_dict):
        try:
            raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
            band = raster.GetRasterBand(1)
            array = band.ReadAsArray(**offset_dict)
            nodata = band.GetNoDataValue()
        finally:
            band = None
            raster = None
        return array, nodata

    transition_pairs = set()
    for block_offsets in pygeoprocessing.iterblocks((lulc_snapshot_list, 1),
                                                    offset_only=True):
        # TODO: make this loop more efficient by not reading in each raster
        # twice
        for from_raster, to_raster in zip(lulc_snapshot_list[:-1],
                                          lulc_snapshot_list[1:]):
            from_array, from_nodata = _read_block(from_raster, block_offsets)
            to_array, to_nodata = _read_block(to_raster, block_offsets)

            # This comparison assumes that our landcover rasters are of an
            # integer type.  When int matrices, we can compare directly to
            # None.
            valid_pixels = ((from_array != from_nodata) &
                            (to_array != to_nodata))
            transition_pairs = transition_pairs.union(
                set(zip(from_array[valid_pixels].flatten(),
                        to_array[valid_pixels].flatten())))

    # Mapping of whether the from, to landcover types are coastal blue carbon
    # habitats to the string carbon transition type.
    # The keys are structured as a tuple of two booleans where:
    #  * tuple[0] = whether the FROM transition is CBC habitat
    #  * tuple[1] = whether the TO transition is CBC habitat
    transition_types = {
        (True, True): 'accum',  # veg --> veg
        (False, True): 'accum',  # non-veg --> veg
        (True, False): 'disturb',  # veg --> non-veg
        (False, False): 'NCC',  # non-veg --> non-veg
    }

    sparse_transition_table = {}
    for from_lucode, to_lucode in transition_pairs:
        from_is_cbc = landcover_table[
            from_lucode]['is_coastal_blue_carbon_habitat']
        to_is_cbc = landcover_table[
            to_lucode]['is_coastal_blue_carbon_habitat']

        sparse_transition_table[(from_lucode, to_lucode)] = (
            transition_types[(from_is_cbc, to_is_cbc)])

    code_list = sorted([code for code in landcover_table.keys()])
    lulc_class_list_sorted = [landcover_table[code]['lulc-class'] for code in code_list]
    with open(target_table_path, 'w') as csv_file:
        fieldnames = ['lulc-class'] + lulc_class_list_sorted
        csv_file.write(f"{','.join(fieldnames)}\n")
        for row_code in code_list:
            class_name = landcover_table[row_code]['lulc-class']
            row = [class_name]
            for col_code in code_list:
                try:
                    column_value = sparse_transition_table[
                        (row_code, col_code)]
                except KeyError:
                    # When there isn't a transition that we know about, just
                    # leave the table blank.
                    column_value = ''
                row.append(column_value)
            csv_file.write(','.join(row) + '\n')

        # Append legend
        csv_file.write(",\n,legend")
        csv_file.write(
            "\n,empty cells indicate that no transitions occur of that type")
        csv_file.write("\n,disturb (disturbance): change to low- med- or "
                       "high-impact-disturb")
        csv_file.write("\n,accum (accumulation)")
        csv_file.write("\n,NCC (no-carbon-change)")


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
    for snapshot_idx in range(0, len(lulc_snapshot_list)-1):
        transition_set = _get_land_cover_transitions(
            lulc_snapshot_list[snapshot_idx],
            lulc_snapshot_list[snapshot_idx+1])

        for transition_tuple in transition_set:
            transition_matrix_dict[transition_tuple] = _mark_transition_type(
                lulc_lookup_dict, *transition_tuple)

    return transition_matrix_dict


def _create_transition_table_old(filepath, lulc_class_list, transition_matrix_dict,
                             code_to_lulc_dict):
    """Create transition table representing effect on emissions or sequestration.

    Args:
        filepath (str): output filepath
        lulc_class_list (list): list of lulc classes (strings)
        transition_matrix_dict (dict): dictionary of lulc transitions
        code_to_lulc_dict (dict): map lulc codes to lulc classes
    """
    LOGGER.info('Creating transition table as output...')

    code_list = sorted(code_to_lulc_dict)
    lulc_class_list_sorted = [code_to_lulc_dict[code] for code in code_list]

    transition_by_lulc_class_dict = dict(
        (lulc_class, {}) for lulc_class in lulc_class_list)

    for transition in transition_matrix_dict:
        top_dict = transition_by_lulc_class_dict[
            code_to_lulc_dict[transition[0]]]
        top_dict[code_to_lulc_dict[transition[1]]] = transition_matrix_dict[
            transition]
        transition_by_lulc_class_dict[code_to_lulc_dict[transition[0]]] = \
            top_dict

    with open(filepath, 'w') as csv_file:
        fieldnames = ['lulc-class'] + lulc_class_list_sorted
        csv_file.write(','.join(fieldnames)+'\n')
        for code in code_list:
            lulc_class = code_to_lulc_dict[code]
            row = [lulc_class] + [
                transition_by_lulc_class_dict[lulc_class][x]
                for x in lulc_class_list_sorted]
            csv_file.write(','.join(row)+'\n')

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
        csv_file.write('code,lulc-class,biomass,soil,litter\n')
        for code in code_to_lulc_dict:
            csv_file.write('%s,%s,,,\n' % (code, code_to_lulc_dict[code]))


def _create_carbon_pool_transient_table_template(filepath, code_to_lulc_dict):
    """Create carbon pool transient values table.

    Args:
        filepath (str): filepath to carbon pool initial conditions
        code_to_lulc_dict (dict): map lulc codes to lulc classes
    """
    with open(filepath, 'w') as csv_file:
        csv_file.write(
            'code,lulc-class,biomass-half-life,biomass-low-impact-disturb,'
            'biomass-med-impact-disturb,biomass-high-impact-disturb,'
            'biomass-yearly-accumulation,soil-half-life,'
            'soil-low-impact-disturb,soil-med-impact-disturb,'
            'soil-high-impact-disturb,soil-yearly-accumulation\n')
        for code in code_to_lulc_dict:
            csv_file.write('%s,%s,,,,,,,,,,\n' % (
                code, code_to_lulc_dict[code]))


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Coastal Blue Carbon: Preprocessor.

    Parameters:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the string validation warning.
    """
    return validation.validate(args, ARGS_SPEC['args'])
