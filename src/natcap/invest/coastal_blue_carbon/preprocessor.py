# -*- coding: utf-8 -*-
"""Coastal Blue Carbon Preprocessor."""
import time
import os
import logging

from osgeo import gdal
import pygeoprocessing
import taskgraph

from .. import utils
from .. import spec_utils
from ..spec_utils import u
from .. import validation
from ..model_metadata import MODEL_METADATA
from .. import gettext
from . import coastal_blue_carbon

LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": MODEL_METADATA["coastal_blue_carbon_preprocessor"].model_title,
    "pyname": MODEL_METADATA["coastal_blue_carbon_preprocessor"].pyname,
    "userguide": MODEL_METADATA["coastal_blue_carbon_preprocessor"].userguide,
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "lulc_lookup_table_path": {
            "name": gettext("LULC lookup table"),
            "type": "csv",
            "about": gettext(
                "A table mapping LULC codes from the snapshot rasters to the "
                "corresponding LULC class names, and whether or not the "
                "class is a coastal blue carbon habitat."),
            "columns": {
                "code": {
                    "type": "integer",
                    "about": gettext(
                        "LULC code. Every value in the "
                        "snapshot LULC maps must have a corresponding entry "
                        "in this column.")},
                "lulc-class": {
                    "type": "freestyle_string",
                    "about": gettext("Name of the LULC class.")},
                "is_coastal_blue_carbon_habitat": {
                    "type": "boolean",
                    "about": gettext(
                        "Enter TRUE if this LULC class is a coastal blue "
                        "carbon habitat, FALSE if not.")}
            }
        },
        "landcover_snapshot_csv": {
            "type": "csv",
            "columns": {
                "snapshot_year": {
                    "type": "number",
                    "units": u.year_AD,
                    "about": gettext("Year to snapshot.")},
                "raster_path": {
                    "type": "raster",
                    "bands": {1: {"type": "integer"}},
                    "about": gettext(
                        "Map of LULC in the snapshot year. "
                        "All values in this raster must have corresponding "
                        "entries in the LULC Lookup table.")
                }
            },
            "about": gettext(
                "A table mapping snapshot years to corresponding LULC maps "
                "for each year."),
            "name": gettext("LULC snapshots table"),
        },
    }
}


ALIGNED_LULC_RASTER_TEMPLATE = 'aligned_lulc_{year}{suffix}.tif'
TRANSITION_TABLE = 'carbon_pool_transition_template{suffix}.csv'
BIOPHYSICAL_TABLE = 'carbon_biophysical_table_template{suffix}.csv'


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
        coastal_blue_carbon._extract_snapshots_from_table(
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

    baseline_srs_wkt = pygeoprocessing.get_raster_info(
        snapshots_dict[min(snapshots_dict.keys())])['projection_wkt']
    alignment_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(source_snapshot_paths,
              aligned_snapshot_paths,
              (['near']*len(source_snapshot_paths)),
              (min_pixel_size, -min_pixel_size),
              'intersection'),
        kwargs={'target_projection_wkt': baseline_srs_wkt},
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

    target_biophysical_table_path = os.path.join(
        output_dir, BIOPHYSICAL_TABLE.format(suffix=suffix))
    _ = task_graph.add_task(
        func=_create_biophysical_table,
        args=(landcover_table, target_biophysical_table_path),
        target_path_list=[target_biophysical_table_path],
        task_name='Write biophysical table template')

    task_graph.close()
    task_graph.join()


def _create_transition_table(landcover_table, lulc_snapshot_list,
                             target_table_path):
    """Create the transition table from a series of landcover snapshots.

    Args:
        landcover_table (dict): A dict mapping integer landcover codes to dict
            values indicating the landcover class name in the ``lulc-class``
            field and ``True`` or ``False`` under the
            ``is_coastal_blue_carbon_habitat`` key.
        lulc_snapshot_list (list): A list of string paths to GDAL rasters on
            disk.  All rasters must have the same spatial reference, pixel size
            and dimensions and must also all be integer rasters, where all
            non-nodata pixel values must be represented in the
            ``landcover_table`` dict.
        target_table_path (string): A string path to where the target
            transition table should be written.

    Returns:
        ``None``.
    """
    n_rows, n_cols = pygeoprocessing.get_raster_info(
        lulc_snapshot_list[0])['raster_size']
    n_pixels_total = (n_rows * n_cols) * len(lulc_snapshot_list)
    n_pixels_processed = 0

    raster_tuple_list = []
    for raster_path in lulc_snapshot_list:
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        raster_tuple_list.append((raster, band, nodata))

    transition_pairs = set()
    last_log_time = time.time()
    for block_offsets in pygeoprocessing.iterblocks((lulc_snapshot_list[0], 1),
                                                    offset_only=True):
        _, from_band, from_nodata = raster_tuple_list[0]
        from_array = from_band.ReadAsArray(**block_offsets)
        from_band = None

        for (_, to_band, to_nodata) in raster_tuple_list[1:]:
            if time.time() - last_log_time >= 5.0:
                percent_complete = n_pixels_processed / n_pixels_total
                LOGGER.info(
                    "Determining landcover transitions, "
                    f"{percent_complete:.2f}% complete.")

            to_array = to_band.ReadAsArray(**block_offsets)

            # This comparison assumes that our landcover rasters are of an
            # integer type.  When int matrices, we can compare directly to
            # None.
            valid_pixels = (
                ~utils.array_equals_nodata(from_array, from_nodata) &
                ~utils.array_equals_nodata(to_array, to_nodata))
            transition_pairs = transition_pairs.union(
                set(zip(from_array[valid_pixels].flatten(),
                        to_array[valid_pixels].flatten())))

            # Swap the arrays around to use the current 'to_array', 'to_nodata'
            # as the 'from_array', 'from_nodata' in the next iteration.
            from_array, from_nodata = (to_array, to_nodata)
            n_pixels_processed += to_array.size
    raster_tuple_list = None
    to_band = None
    LOGGER.info("Determining landcover transitions, 100.00%% complete.")

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
            to_is_cbc = landcover_table[
                to_lucode]['is_coastal_blue_carbon_habitat']
        except KeyError:
            for variable in (from_lucode, to_lucode):
                if variable not in landcover_table:
                    raise ValueError(
                        'The landcover table is missing a row with the '
                        f'landuse code {variable}.')

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
        colname.lower() for colname in coastal_blue_carbon.ARGS_SPEC['args'][
            'biophysical_table_path']['columns']]

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

    Args:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the string validation warning.
    """
    return validation.validate(args, ARGS_SPEC['args'])
