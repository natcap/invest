# -*- coding: utf-8 -*-
"""Coastal Blue Carbon Preprocessor."""
import os
import logging

from osgeo import gdal
import pygeoprocessing
import taskgraph

from .. import utils
from .. import validation
from . import coastal_blue_carbon2


LOGGER = logging.getLogger(__name__)
ARGS_SPEC = {
    "model_name": "Coastal Blue Carbon Preprocessor",
    "module": __name__,
    "userguide_html": "coastal_blue_carbon.html",
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
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
            "name": "LULC Snapshots Table",
        },
    }
}


ALIGNED_LULC_RASTER_TEMPLATE = 'aligned_lulc_{year}{suffix}.tif'
TRANSITION_TABLE = 'carbon_pool_transition_template{suffix}.csv'
BIOPHYSICAL_TABLE = 'carbon_biophysical_table_template{suffix}.csv'

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
        args['results_suffix'] (string): append to outputs directory name if
            provided
        args['lulc_lookup_table_path'] (string): filepath of lulc lookup table
        args['landcover_csv_path'] (string): filepath to a CSV containing the
            year and filepath to snapshot rasters on disk.  The years may be in
            any order, but must be unique.

    Returns:
        ``None``
    """
    suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'], 'outputs_preprocessor')
    taskgraph_cache_dir = os.path.join(args['workspace_dir'], 'task_cache')
    utils.make_directories([output_dir, taskgraph_cache_dir])

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
        coastal_blue_carbon2._extract_snapshots_from_table(
            args['landcover_snapshot_csv']))

    # Align the raster stack for analyzing the various transitions.
    min_pixel_size = float('inf')
    source_snapshot_paths = []
    aligned_snapshot_paths = []
    for snapshot_year, raster_path in snapshots_dict.items():
        source_snapshot_paths.append(raster_path)
        aligned_snapshot_paths.append(os.path.join(
            output_dir, ALIGNED_LULC_RASTER_TEMPLATE.format(
                year=snapshot_year, suffix=suffix)))
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
    _ = task_graph.add_task(
        func=_create_transition_table,
        args=(landcover_table,
              sorted(snapshots_dict.values(), key=lambda x: [0]),
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
    for block_offsets in pygeoprocessing.iterblocks((lulc_snapshot_list[0], 1),
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
        try:
            from_is_cbc = landcover_table[
                from_lucode]['is_coastal_blue_carbon_habitat']
        except KeyError:
            raise ValueError(
                'The landcover table is missing a row with the landuse '
                f'code {from_lucode}.')
        try:
            to_is_cbc = landcover_table[
                to_lucode]['is_coastal_blue_carbon_habitat']
        except KeyError:
            raise ValueError(
                'The landcover table is missing a row with the landuse '
                f'code {to_lucode}.')

        sparse_transition_table[(from_lucode, to_lucode)] = (
            transition_types[(from_is_cbc, to_is_cbc)])

    code_list = sorted([code for code in landcover_table.keys()])
    lulc_class_list_sorted = [
        landcover_table[code]['lulc-class'] for code in code_list]
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
        csv_file.write("\n,legend")
        csv_file.write(
            "\n,empty cells indicate that no transitions occur of that type")
        csv_file.write("\n,disturb (disturbance): change to low- med- or "
                       "high-impact-disturb")
        csv_file.write("\n,accum (accumulation)")
        csv_file.write("\n,NCC (no-carbon-change)")


def _create_biophysical_table(landcover_table, target_biophysical_table_path):
    """Write the biophysical table template to disk.

    The biophysical table templates contains all of the fields required by the
    main Coastal Blue Carbon model, and any field values that exist in the
    landcover table provided to this model will be carried over to the new
    table.

    Args:
        landcover_table (dict): A dict mapping int landcover codes to a dict
            with string keys that map to numeric or string column values.
        target_biophysical_table_path (string): The path to where the
            biophysical table template will be stored on disk.

    Returns:
        ``None``
    """
    target_column_names = [
        colname.lower() for colname in coastal_blue_carbon2.ARGS_SPEC['args'][
            'biophysical_table_path']['validation_options']['required_fields']]

    with open(target_biophysical_table_path, 'w') as bio_table:
        bio_table.write(f"{','.join(target_column_names)}\n")
        for lulc_code in sorted(landcover_table.keys()):
            # 2 columns are defined below, and we need 1 less comma to only
            # have commas between fields.
            row = []
            for colname in target_column_names:
                try:
                    # Use the user's defined value if it exists
                    row.append(str(landcover_table[lulc_code][colname]))
                except KeyError:
                    row.append('')
            bio_table.write(f"{','.join(row)}\n")


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
