"""Stormwater Retention"""
import logging
import numpy
import pygeoprocessing
import taskgraph

from . import validation
from . import utils

LOGGER = logging.getLogger(__name__)


ARGS_SPEC = {
    "model_name": "Stormwater Retention",
    "module": __name__,
    "userguide_html": "stormwater.html",
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "lulc_path": {
            "type": "raster",
            "bands": {1: {"type": "code"}},
            "required": True,
            "about": (
                "A GDAL-supported raster representing land use/land cover "
                "of the area"),
            "name": "land use/land cover"
        },
        "soil_groups_path": {
            "type": "raster",
            "bands": {
                1: {
                    "type": "option_string",
                    "options": ["1", "2", "3", "4"]
                }
            },
            "required": True,
            "about": (
                "Raster map of hydrologic soil groups, where 1, 2, 3, and 4 "
                "correspond to groups A, B, C, and D respectively"),
            "name": "soil groups"
        },
        "precipitation_path": {
            "type": "raster",
            "bands": {1: {"type": "number", "units": "millimeters"}},
            "required": True,
            "about": ("Precipitation raster"),
            "name": "precipitation"
        },
        "biophysical_table": {
            "type": "csv",
            "columns": {
                "lucode": {"type": "code"},
                "EMC_P": {"type": "number", "units": "mg/L"},
                "EMC_N": {"type": "number", "units": "mg/L"},
                "RC_A": {"type": "ratio"},
                "RC_B": {"type": "ratio"},
                "RC_C": {"type": "ratio"},
                "RC_D": {"type": "ratio"},
                "IR_A": {"type": "ratio"},
                "IR_B": {"type": "ratio"},
                "IR_C": {"type": "ratio"},
                "IR_D": {"type": "ratio"}
            },
            "required": True,
            "about": "biophysical table",
            "name": "biophysical table"
        },
        "road_centerlines_path": {
            "type": "vector",
            "fields": {},
            "required": False,
            "about": "Map of road centerlines",
            "name": "road centerlines"
        },
        "watersheds_path": {
            "type": "vector",
            "fields": {},
            "required": False,
            "about": "Aggregation areas",
            "name": "watersheds"
        },
        "replacement_cost": {
            "type": "number",
            "units": "currency",
            "required": False,
            "about": "Replacement cost of stormwater retention devices",
            "name": "replacement cost"
        }
    }
}


FILES = {
    'lulc_aligned_path': 'intermediate/lulc_aligned.tif',
    'soil_group_aligned_path': 'intermediate/soil_group_aligned.tif',
    'precipitation_aligned_path': 'intermediate/precipitation_aligned.tif',
    'retention_ratio_path': 'retention_ratio.tif',
    'retention_volume_path': 'retention_volume.tif',
    'infiltration_ratio_path': 'infiltration_ratio.tif',
    'infiltration_volume_path': 'infiltration_volume.tif'
}


def execute(args):

    align_inputs = [args['lulc_path'], args['soil_groups_path'], args['precipitation_path']]
    align_outputs = [
        FILES['lulc_aligned_path'],
        FILES['soil_group_aligned_path'], 
        FILES['precipitation_aligned_path']]

    task_graph = taskgraph.TaskGraph(args['workspace_dir'], args['n_workers'])



    # Align all three input rasters to the same projection
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            align_inputs, align_outputs, interpolate_list,
            pixel_size, 'intersection'),
        kwargs={
            'base_vector_path_list': (args['aoi_path'],),
            'raster_align_index': align_index},
        target_path_list=output_align_list,
        task_name='align rasters')


    # Build a lookup dictionary mapping each LULC code to its row
    biophysical_dict = utils.build_lookup_from_csv(
        args['biophysical_table'], 'lucode')

    # Make ratio lookup dictionaries mapping each LULC code to
    # a ratio for each soil group
    retention_ratio_dict = {
        lucode: {
            'A': row['RC_A'],
            'B': row['RC_B'],
            'C': row['RC_C'],
            'D': row['RC_D'],
        } for lucode, row in biophysical_dict
    }
    infiltration_ratio_dict = {
        lucode: {
            'A': row['IR_A'],
            'B': row['IR_B'],
            'C': row['IR_C'],
            'D': row['IR_D'],
        } for lucode, row in biophysical_dict
    }


    # Calculate stormwater retention ratio and volume from
    # LULC, soil groups, biophysical table, and precipitation

    retention_ratio_task = task_graph.add_task(
        func=calculate_stormwater_retention_ratio,
        args=(
            FILES['lulc_aligned_path'],
            FILES['soil_group_aligned_path'],
            retention_ratio_dict,
            FILES['retention_ratio_path']),
        target_path_list=[FILES['retention_ratio_path']],
        task_name='calculate stormwater retention ratio'
    )

    # (Optional) adjust stormwater retention ratio using roads
    adjust_retention_ratio_task = task_graph.add_task(
        func=adjust_stormwater_retention_ratio,
        args=(
            FILES['retention_ratio_path'],
            args['road_centerlines_path'],
            FILES['adjusted_retention_ratio_path']),
        target_path_list=[FILES['adjusted_retention_ratio_path']],
        task_name='adjust stormwater retention ratio'
    )

    retention_volume_task = task_graph.add_task(
        func=calculate_stormwater_retention_volume,
        args=(
            FILES['lulc_aligned_path'],
            FILES['soil_group_aligned_path'],
            args['biophysical_table'],
            FILES['retention_volume_path']),
        target_path_list=[FILES['retention_volume_path']],
        task_name='calculate stormwater retention volume'
    )


    # (Optional) Calculate stormwater infiltration ratio and volume from
    # LULC, soil groups, biophysical table, and precipitation

    infiltration_ratio_task = task_graph.add_task(
        func=calculate_stormwater_infiltration_ratio,
        args=(
            FILES['lulc_aligned_path'],
            FILES['soil_group_aligned_path'],
            infiltration_ratio_dict,
            FILES['infiltration_path']),
        target_path_list=[FILES['infiltration_path']],
        task_name='calculate stormwater infiltration'
    )

    infiltration_volume_task = task_graph.add_task(
        func=calculate_stormwater_retention_volume,
        args=(
            FILES['lulc_aligned_path'],
            FILES['soil_group_aligned_path'],
            args['biophysical_table'],
            FILES['retention_volume_path']),
        target_path_list=[FILES['retention_volume_path']],
        task_name='calculate stormwater retention volume'
    )

    # Calculate avoided pollutant load from retention volume and biophysical table

    avoided_pollutant_load_task = task_graph.add_task(
        func=calculate_avoided_pollutant_load,
        args=(
            FILES['retention_path'],
            args['biophysical_table'],
            FILES['avoided_pollutants']),
        target_path_list=[FILES['avoided_pollutants']],
        task_name='calculate avoided pollutant load'
    )


    # (Optional) Valuation

    valuation_task = task_graph.add_task(
        func=calculate_retention_value,
        args=(
            FILES['retention_path'],
            args['replacement_cost'],
            FILES['retention_value_path']),
        target_path_list=[FILES['retention_value_path']],
        task_name='calculate stormwater retention value'
    )


    # (Optional) Aggregate to watersheds
    aggregation_task = task_graph.add_task(
        func=aggregate_values,
        args=(
            ))


def calculate_stormwater_ratio(lulc_path, soil_group_path, 
        ratio_lookup, output_path):
    """Make stormwater retention or infiltration ratio map from LULC and
       soil group data.
    Args:
        lulc_path (str): path to a LULC raster whose LULC codes exist in the
            biophysical table
        soil_group_path (str): path to a soil group raster with pixel values
            1, 2, 3, and 4 corresponding to hydrologic soil groups A, B, C, and D
        ratio_lookup (dict): a lookup dictionary of ratios for each pair of 
            LULC code and soil group. Each LULC code is mapped to a dictionary
            with keys 'A', 'B', 'C', and 'D', which map to the ratio for that
            LULC code x soil group pair.
        output_path: path to write out the retention ratio raster to

    Returns:
        None
    """
    ratio_nodata = -1

    def ratio_op(lulc_array, soil_group_array):

        # initialize an array of the output nodata value
        ratio_array = numpy.full(ratio_nodata)
        nodata_mask = (lulc_array != lulc_nodata & 
                       soil_group_array != soil_group_nodata)

        for lucode in ratio_lookup:
            lucode_mask = (lulc_array == lucode)

            for soil_group in ['A', 'B', 'C', 'D']:
                soil_group_mask = (soil_group_array == soil_group)

                ratio_array[(lucode_mask & soil_group_mask)] = ratio_lookup[lucode][soil_group]

        return ratio_array


def calculate_stormwater_volume()




