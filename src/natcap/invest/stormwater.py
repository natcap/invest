"""Stormwater Retention"""
import logging
import math
import os

import numpy
from osgeo import gdal, ogr
import pygeoprocessing
import scipy.ndimage
import scipy.signal
import taskgraph

from . import validation
from . import utils

LOGGER = logging.getLogger(__name__)

# a constant nodata value to use for intermediates and outputs
FLOAT_NODATA = -1
UINT8_NODATA = 255
UINT16_NODATA = 65535

ARGS_SPEC = {
    "model_name": "Stormwater Retention",
    "module": __name__,
    "userguide_html": "stormwater.html",
    "args_with_spatial_overlap": {
        "spatial_keys": ["lulc_path", "soil_group_path", "precipitation_path",
                         "road_centerlines_path", "aggregate_areas_path"],
        "different_projections_ok": True
    },
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "lulc_path": {
            "type": "raster",
            "required": True,
            "about": (
                "A map of land use/land cover classes in the area of interest"),
            "name": "land use/land cover",
            "validation_options": {
                "projected": True
            }
        },
        "soil_group_path": {
            "type": "raster",
            "required": True,
            "about": (
                "Map of hydrologic soil groups, where pixel values 1, 2, 3, "
                "and 4 correspond to groups A, B, C, and D respectively"),
            "name": "soil groups"
        },
        "precipitation_path": {
            "type": "raster",
            "required": True,
            "about": ("Map of total annual precipitation"),
            "name": "precipitation"
        },
        "biophysical_table": {
            "type": "csv",
            "required": True,
            "about": "biophysical table",
            "name": "biophysical table"
        },
        "adjust_retention_ratios": {
            "type": "boolean",
            "required": True,
            "about": (
                "If true, adjust retention ratios. The adjustment algorithm "
                "accounts for drainage effects of nearby impervious surfaces "
                "which are directly connected to artifical urban drainage "
                "channels (typically roads, parking lots, etc.) Connected "
                "impervious surfaces are indicated by the is_impervious column"
                "in the biophysical table and/or the road centerlines vector."),
            "name": "adjust retention ratios"
        },
        "retention_radius": {
            "type": "number",
            "required": "adjust_retention_ratios",
            "about": (
                "Radius around each pixel to adjust retention ratios. For the "
                "adjustment algorithm, a pixel is 'near' a connected "
                "impervious surface if its centerpoint is within this radius "
                "of connected-impervious LULC and/or a road centerline."),
            "name": "retention radius"
        },
        "road_centerlines_path": {
            "type": "vector",
            "required": "adjust_retention_ratios",
            "about": "Map of road centerlines",
            "name": "road centerlines"
        },
        "aggregate_areas_path": {
            "type": "vector",
            "required": False,
            "about": (
                "Areas over which to aggregate results (typically watersheds "
                "or sewersheds). The aggregated data are: average retention "
                "ratio and total retention volume; average infiltration ratio "
                "and total infiltration volume if infiltration data was "
                "provided; total retention value if replacement cost was "
                "provided; and total avoided pollutant load for each "
                "pollutant provided."),
            "name": "watersheds"
        },
        "replacement_cost": {
            "type": "number",
            "required": False,
            "about": "Replacement cost of stormwater retention devices",
            "name": "replacement cost"
        }
    }
}

FILES = {
    'lulc_aligned_path': os.path.join(intermediate_dir, f'lulc_aligned{suffix}.tif'),
    'soil_group_aligned_path': os.path.join(intermediate_dir, f'soil_group_aligned{suffix}.tif'),
    'precipitation_aligned_path': os.path.join(intermediate_dir, f'precipitation_aligned{suffix}.tif'),
    'reprojected_centerlines_path': os.path.join(intermediate_dir, f'reprojected_centerlines{suffix}.gpkg'),
    'rasterized_centerlines_path': os.path.join(intermediate_dir, f'rasterized_centerlines{suffix}.tif'),
    'reprojected_aggregate_areas_path': os.path.join(output_dir, f'aggregate_data{suffix}.gpkg'),
    'retention_ratio_path': os.path.join(output_dir, f'retention_ratio{suffix}.tif'),
    'retention_volume_path': os.path.join(output_dir, f'retention_volume{suffix}.tif'),
    'infiltration_ratio_path': os.path.join(output_dir, f'infiltration_ratio{suffix}.tif'),
    'infiltration_volume_path': os.path.join(output_dir, f'infiltration_volume{suffix}.tif'),
    'retention_value_path': os.path.join(output_dir, f'retention_value{suffix}.tif'),
    'impervious_lulc_path': os.path.join(intermediate_dir, f'is_impervious_lulc{suffix}.tif'),
    'adjusted_retention_ratio_path': os.path.join(intermediate_dir, f'adjusted_retention_ratio{suffix}.tif'),
    'x_coords_path': os.path.join(intermediate_dir, f'x_coords{suffix}.tif'),
    'y_coords_path': os.path.join(intermediate_dir, f'y_coords{suffix}.tif'),
    'road_distance_path': os.path.join(intermediate_dir, f'road_distance{suffix}.tif'),
    'near_impervious_lulc_path': os.path.join(intermediate_dir, f'near_impervious_lulc{suffix}.tif'),
    'near_road_path': os.path.join(intermediate_dir, f'near_road{suffix}.tif'),
    'ratio_n_values_path': os.path.join(intermediate_dir, f'ratio_n_values{suffix}.tif'),
    'ratio_sum_path': os.path.join(intermediate_dir, f'ratio_sum{suffix}.tif'),
    'ratio_average_path': os.path.join(intermediate_dir, f'ratio_average{suffix}.tif')
}


def execute(args):
    """Execute the stormwater model.

    Args:
        args['workspace_dir'] (str): path to a directory to write intermediate
            and final outputs. May already exist or not.
        args['results_suffix'] (str, optional): string to append to all output
            file names from this model run
        args['n_workers'] (int): if present, indicates how many worker
            processes should be used in parallel processing. -1 indicates
            single process mode, 0 is single process but non-blocking mode,
            and >= 1 is number of processes.
        args['lulc_path'] (str): path to LULC raster
        args['soil_group_path'] (str): path to soil group raster, where pixel
            values 1, 2, 3, 4 correspond to groups A, B, C, D
        args['precipitation_path'] (str): path to raster of total annual
            precipitation in millimeters
        args['biophysical_table'] (str): path to biophysical table with columns
            'lucode', 'EMC_x' (event mean concentration mg/L) for each
            pollutant x, 'RC_y' (retention coefficient) and 'IR_y'
            (infiltration coefficient) for each soil group y, and
            'is_impervious' if args['adjust_retention_ratios'] is True
        args['adjust_retention_ratios'] (bool): If True, apply retention ratio
            adjustment algorithm.
        args['retention_radius'] (float): If args['adjust_retention_ratios']
            is True, use this radius in the adjustment algorithm.
        args['road_centerliens_path'] (str): Path to linestring vector of road
            centerlines. Only used if args['adjust_retention_ratios'] is True.
        args['aggregate_areas_path'] (str): Optional path to polygon vector of
            areas to aggregate results over.
        args['replacement_cost'] (float): Cost to replace stormwater retention
            devices in units currency per cubic meter

    Returns:
        None
    """

    # set up files and directories
    suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = args['workspace_dir']
    intermediate_dir = os.path.join(output_dir, 'intermediate')
    cache_dir = os.path.join(intermediate_dir, 'cache_dir')
    utils.make_directories(
        [args['workspace_dir'], intermediate_dir, cache_dir])

    align_inputs = [args['lulc_path'],
                    args['soil_group_path'], args['precipitation_path']]
    align_outputs = [
        FILES['lulc_aligned_path'],
        FILES['soil_group_aligned_path'],
        FILES['precipitation_aligned_path']]

    task_graph = taskgraph.TaskGraph(cache_dir, int(args.get('n_workers', -1)))
    source_lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_path'])
    pixel_size = source_lulc_raster_info['pixel_size']
    pixel_area = abs(pixel_size[0] * pixel_size[1])

    lulc_nodata = source_lulc_raster_info['nodata'][0]
    precipitation_nodata = pygeoprocessing.get_raster_info(
        args['precipitation_path'])['nodata'][0]
    soil_group_nodata = pygeoprocessing.get_raster_info(
        args['soil_group_path'])['nodata'][0]

    # Align all three input rasters to the same projection
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            align_inputs,
            align_outputs,
            ['near' for _ in align_inputs],
            pixel_size,
            'intersection'),
        kwargs={'raster_align_index': 0},
        target_path_list=align_outputs,
        task_name='align input rasters')

    # Build a lookup dictionary mapping each LULC code to its row
    biophysical_dict = utils.build_lookup_from_csv(
        args['biophysical_table'], 'lucode')
    # sort the LULC codes upfront because we use the sorted list in multiple
    # places. it's more efficient to do this once.
    sorted_lucodes = sorted(biophysical_dict)

    # convert the nested dictionary in to a 2D array where rows are LULC codes
    # in sorted order and columns correspond to soil groups in order
    # this facilitates efficiently looking up the ratio values with numpy

    # Biophysical table has runoff coefficents so subtract
    # from 1 to get retention coefficient.
    # add a placeholder in column 0 so that the soil groups 1, 2, 3, 4 line
    # up with their indices in the array. this is more efficient than
    # decrementing the whole soil group array by 1.
    retention_ratio_array = numpy.array([
        [numpy.nan] + [1 - biophysical_dict[lucode][f'rc_{soil_group}']
                       for soil_group in ['a', 'b', 'c', 'd']
                       ] for lucode in sorted_lucodes
    ], dtype=numpy.float32)

    # Calculate stormwater retention ratio and volume from
    # LULC, soil groups, biophysical data, and precipitation
    retention_ratio_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (FILES['lulc_aligned_path'], 1),
            (lulc_nodata, 'raw'),
            (FILES['soil_group_aligned_path'], 1),
            (soil_group_nodata, 'raw'),
            (retention_ratio_array, 'raw'),
            (sorted_lucodes, 'raw')],
            ratio_op,
            FILES['retention_ratio_path'],
            gdal.GDT_Float32,
            FLOAT_NODATA),
        target_path_list=[FILES['retention_ratio_path']],
        dependent_task_list=[align_task],
        task_name='calculate stormwater retention ratio'
    )

    # (Optional) adjust stormwater retention ratio using roads
    if args['adjust_retention_ratios']:
        # in raster coord system units
        radius = float(args['retention_radius'])
        # boolean mapping for each LULC code whether it's impervious
        is_impervious_map = {
            1 if biophysical_dict[lucode]['is_impervious'] else 0
            for lucode in biophysical_dict}

        reproject_roads_task = task_graph.add_task(
            func=pygeoprocessing.reproject_vector,
            args=(
                args['road_centerlines_path'],
                source_lulc_raster_info['projection_wkt'],
                FILES['reprojected_centerlines_path']),
            kwargs={'driver_name': 'GPKG'},
            target_path_list=[FILES['reprojected_centerlines_path']],
            task_name='reproject road centerlines vector to match rasters',
            dependent_task_list=[]
        )

        rasterize_centerlines_task = task_graph.add_task(
            func=pygeoprocessing.rasterize,
            args=(
                args['reprojected_centerlines_path'],
                FILES['rasterized_centerlines_path'],
                [1]),
            target_path_list=[FILES['rasterized_centerlines_path']],
            task_name='rasterize road centerlines vector',
            dependent_task_list=[reproject_roads_task])

        distance_transform_task = task_graph.add_task(
            func=pygeoprocessing.distance_transform_edt,
            args=(
                args['centerlines_raster_path'],
                args['road_distance_path']),
            target_path_list=[args['road_distance_path']],
            task_name='calculate distance from each pixel to centerline pixel')

        # Make a boolean raster showing which pixels are within the given
        # radius of a road centerline
        distance_threshold_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(FILES['road_distance_path'], 1), (radius, 'raw')],
                threshold_array,
                FILES['near_road_path'],
                gdal.GDT_Byte,
                UINT8_NODATA),
            target_path_list=[FILES['near_road_path']],
            task_name='find pixels within radius of road centerlines',
            dependent_task_list=[distance_transform_task])

        # Make a boolean raster indicating which pixels are directly
        # connected impervious LULC type
        impervious_lulc_task = task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (FILES['lulc_aligned_path'], 1),
                is_impervious_map,
                FILES['impervious_lulc_path'],
                gdal.GDT_Byte,
                UINT8_NODATA),
            target_path_list=[FILES['impervious_lulc_path']],
            task_name='calculate binary impervious lulc raster',
            dependent_task_list=[align_task]
        )

        average_ratios_task = task_graph.add_task(
            func=raster_average,
            args=(
                FILES['retention_ratio_path'],
                radius,
                FILES['ratio_n_values_path'],
                FILES['ratio_sum_path'],
                FILES['ratio_average_path']),
            target_path_list=[FILES['ratio_average_path']],
            task_name='average retention ratios within radius',
            dependent_task_list=[retention_ratio_task])

        # Using the averaged retention ratio raster and boolean
        # "within radius" rasters, adjust the retention ratios
        adjust_retention_ratio_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (FILES['retention_ratio_path'], 1),
                (FILES['ratio_average_path'], 1),
                (FILES['near_impervious_lulc_path'], 1),
                (FILES['near_road_path'], 1)],
                adjust_op,
                FILES['adjusted_retention_ratio_path'],
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[FILES['adjusted_retention_ratio_path']],
            task_name='adjust stormwater retention ratio',
            dependent_task_list=[retention_ratio_task, average_ratios_task,
                                 near_impervious_lulc_task, near_road_task])

        final_retention_ratio_path = FILES['adjusted_retention_ratio_path']
        final_retention_ratio_task = adjust_retention_ratio_task
    else:
        final_retention_ratio_path = FILES['retention_ratio_path']
        final_retention_ratio_task = retention_ratio_task

    # Calculate stormwater retention volume from ratios and precipitation
    retention_volume_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (final_retention_ratio_path, 1),
            (FILES['precipitation_aligned_path'], 1),
            (precipitation_nodata, 'raw'),
            (pixel_area, 'raw')],
            volume_op,
            FILES['retention_volume_path'],
            gdal.GDT_Float32,
            FLOAT_NODATA),
        target_path_list=[FILES['retention_volume_path']],
        dependent_task_list=[align_task, final_retention_ratio_task],
        task_name='calculate stormwater retention volume'
    )
    aggregation_task_dependencies = [retention_volume_task]
    data_to_aggregate = [
        # tuple of (raster path, output field name, op) for aggregation
        (FILES['retention_ratio_path'], 'RR_mean', 'mean'),
        (FILES['retention_volume_path'], 'RV_sum', 'sum')]

    # (Optional) Calculate stormwater infiltration ratio and volume from
    # LULC, soil groups, biophysical table, and precipitation
    if 'ir_a' in next(iter(biophysical_dict.values())):
        LOGGER.info('Infiltration data detected in biophysical table. '
                    'Will calculate infiltration ratio and volume rasters.')
        infiltration_ratio_array = numpy.array([
            [biophysical_dict[lucode][f'ir_{soil_group}']
                for soil_group in ['a', 'b', 'c', 'd']
             ] for lucode in sorted_lucodes
        ], dtype=numpy.float32)
        infiltration_ratio_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (FILES['lulc_aligned_path'], 1),
                (lulc_nodata, 'raw'),
                (FILES['soil_group_aligned_path'], 1),
                (soil_group_nodata, 'raw'),
                (infiltration_ratio_array, 'raw'),
                (sorted_lucodes, 'raw')],
                ratio_op,
                FILES['infiltration_ratio_path'],
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[FILES['infiltration_ratio_path']],
            dependent_task_list=[align_task],
            task_name='calculate stormwater infiltration ratio'
        )
        infiltration_volume_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (FILES['infiltration_ratio_path'], 1),
                (FILES['precipitation_aligned_path'], 1),
                (precipitation_nodata, 'raw'),
                (pixel_area, 'raw')],
                volume_op,
                FILES['infiltration_volume_path'],
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[FILES['infiltration_volume_path']],
            dependent_task_list=[align_task, infiltration_ratio_task],
            task_name='calculate stormwater retention volume'
        )
        aggregation_task_dependencies.append(infiltration_volume_task)
        data_to_aggregate.append(
            (FILES['infiltration_ratio_path'], 'IR_mean', 'mean'))
        data_to_aggregate.append(
            (FILES['infiltration_volume_path'], 'IV_sum', 'sum'))

    # get all EMC columns from an arbitrary row in the dictionary
    # strip the first four characters off 'EMC_pollutant' to get pollutant name
    pollutants = [key[4:] for key in next(iter(biophysical_dict.values()))
                  if key.startswith('emc_')]
    LOGGER.debug(f'Pollutants found in biophysical table: {pollutants}')

    # Calculate avoided pollutant load for each pollutant from retention volume
    # and biophysical table EMC value
    avoided_load_paths = []
    for pollutant in pollutants:
        # one output raster for each pollutant
        avoided_pollutant_load_path = os.path.join(
            output_dir, f'avoided_pollutant_load_{pollutant}{suffix}.tif')
        avoided_load_paths.append(avoided_pollutant_load_path)
        # make an array mapping each LULC code to the pollutant EMC value
        emc_array = numpy.array(
            [biophysical_dict[lucode][f'emc_{pollutant}']
                for lucode in sorted_lucodes], dtype=numpy.float32)

        avoided_load_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (FILES['lulc_aligned_path'], 1),
                (lulc_nodata, 'raw'),
                (FILES['retention_volume_path'], 1),
                (sorted_lucodes, 'raw'),
                (emc_array, 'raw')],
                avoided_pollutant_load_op,
                avoided_pollutant_load_path,
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[avoided_pollutant_load_path],
            dependent_task_list=[retention_volume_task],
            task_name=f'calculate avoided pollutant {pollutant} load'
        )
        aggregation_task_dependencies.append(avoided_load_task)
        data_to_aggregate.append(
            (avoided_pollutant_load_path, f'avoided_{pollutant}', 'sum'))

    # (Optional) Do valuation if a replacement cost is defined
    # you could theoretically have a cost of 0 which should be allowed
    if 'replacement_cost' in args and args['replacement_cost'] not in [
            None, '']:
        valuation_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (FILES['retention_volume_path'], 1),
                (float(args['replacement_cost']), 'raw')],
                retention_value_op,
                FILES['retention_value_path'],
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[FILES['retention_value_path']],
            dependent_task_list=[retention_volume_task],
            task_name='calculate stormwater retention value'
        )
        aggregation_task_dependencies.append(valuation_task)
        data_to_aggregate.append(
            (FILES['retention_value_path'], 'val_sum', 'sum'))
        valuation_path = FILES['retention_value_path']
    else:
        valuation_path = None

    # (Optional) Aggregate to watersheds if an aggregate vector is defined
    if 'aggregate_areas_path' in args and args['aggregate_areas_path']:
        reproject_aggregate_areas_task = task_graph.add_task(
            func=pygeoprocessing.reproject_vector,
            args=(
                args['aggregate_areas_path'],
                source_lulc_raster_info['projection_wkt'],
                FILES['reprojected_aggregate_areas_path']),
            kwargs={'driver_name': 'GPKG'},
            target_path_list=[FILES['reprojected_aggregate_areas_path']],
            task_name='reproject aggregate areas vector to match rasters',
            dependent_task_list=[]
        )
        aggregation_task_dependencies.append(reproject_aggregate_areas_task)
        aggregation_task = task_graph.add_task(
            func=aggregate_results,
            args=(
                FILES['reprojected_aggregate_areas_path'],
                data_to_aggregate),
            target_path_list=[FILES['reprojected_aggregate_areas_path']],
            dependent_task_list=aggregation_task_dependencies,
            task_name='aggregate data over polygons'
        )

    task_graph.close()
    task_graph.join()


def threshold_array(array, threshold):
    """Return a boolean array where 1 means less than or equal to the
    threshold value. Assume that nodata areas in the array are above
    the threshold.

    Args:
        array (numpy.ndarray): Array to threshold. It is assumed that
            the array's nodata value is the global FLOAT_NODATA.
        threshold (float): Threshold value to apply to the array.

    Returns:
        boolean numpy.ndarray
    """
    out = numpy.full(array.shape, 0, dtype=numpy.uint8)
    valid_mask = ~numpy.isclose(array, FLOAT_NODATA)
    out[valid_mask] = array[valid_mask] <= threshold
    return out


def ratio_op(lulc_array, lulc_nodata, soil_group_array, soil_group_nodata,
             ratio_lookup, sorted_lucodes):
    """Make an array of stormwater retention or infiltration ratios from
    arrays of LULC codes and hydrologic soil groups.

    Args:
        lulc_array (numpy.ndarray): 2D array of LULC codes
        lulc_nodata (int): nodata value for the LULC array
        soil_group_array (numpy.ndarray): 2D array with the same shape as
            ``lulc_array``. Values in {1, 2, 3, 4} corresponding to soil
            groups A, B, C, and D.
        soil_group_nodata (int): nodata value for the soil group array
        ratio_lookup (numpy.ndarray): 2D array where rows correspond to
            sorted LULC codes and columns 1, 2, 3, 4 correspond to soil groups
            A, B, C, D in order. Shape: (number of lulc codes, 5). Column 0
            is ignored, it's just there so that the existing soil group array
            values line up with their indexes.
        sorted_lucodes (list[int]): List of LULC codes sorted from smallest
            to largest. These correspond to the rows of ``ratio_lookup``.

    Returns:
        2D numpy array with the same shape as ``lulc_array`` and
        ``soil_group_array``. Each value is the corresponding ratio for that
        LULC code x soil group pair.
    """
    output_ratio_array = numpy.full(lulc_array.shape, FLOAT_NODATA,
                                    dtype=numpy.float32)
    valid_mask = ((lulc_array != lulc_nodata) &
                  (soil_group_array != soil_group_nodata))
    # the index of each lucode in the sorted lucodes array
    lulc_index = numpy.digitize(lulc_array[valid_mask], sorted_lucodes,
                                right=True)
    output_ratio_array[valid_mask] = ratio_lookup[lulc_index,
                                                  soil_group_array[valid_mask]]
    return output_ratio_array


def volume_op(ratio_array, precip_array, precip_nodata, pixel_area):
    """Calculate array of volumes (retention or infiltration) from arrays
    of precipitation values and stormwater ratios. This is meant to be used
    with raster_calculator.

    Args:
        ratio_array (numpy.ndarray): 2D array of stormwater ratios. Assuming
            that its nodata value is the global FLOAT_NODATA.
        precip_array (numpy.ndarray): 2D array of precipitation amounts
            in millimeters/year
        precip_nodata (float): nodata value for the precipitation array
        pixel_area (float): area of each pixel in m^2

    Returns:
        2D numpy.ndarray of precipitation volumes in m^3/year
    """
    volume_array = numpy.full(ratio_array.shape, FLOAT_NODATA,
                              dtype=numpy.float32)
    valid_mask = (
        ~numpy.isclose(ratio_array, FLOAT_NODATA) &
        ~numpy.isclose(precip_array, precip_nodata))

    # precipitation (mm/yr) * pixel area (m^2) *
    # 0.001 (m/mm) * ratio = volume (m^3/yr)
    volume_array[valid_mask] = (
        precip_array[valid_mask] *
        ratio_array[valid_mask] *
        pixel_area * 0.001)
    return volume_array


def avoided_pollutant_load_op(lulc_array, lulc_nodata, retention_volume_array,
                              sorted_lucodes, emc_array):
    """Calculate avoided pollutant loads from LULC codes retention volumes.
    This is intented to be used with pygeoprocessing.raster_calculator.

    Args:
        lulc_array (numpy.ndarray): 2D array of LULC codes
        lulc_nodata (int): nodata value for the LULC array
        retention_volume_array (numpy.ndarray): 2D array of stormwater
            retention volumes, with the same shape as ``lulc_array``. It is
            assumed that the retention volume nodata value is the global
            FLOAT_NODATA.
        sorted_lucodes (numpy.ndarray): 1D array of the LULC codes in order
            from smallest to largest
        emc_array (numpy.ndarray): 1D array of pollutant EMC values for each
            lucode. ``emc_array[i]`` is the EMC for the LULC class at
            ``sorted_lucodes[i]``.

    Returns:
        2D numpy.ndarray with the same shape as ``lulc_array``.
        Each value is the avoided pollutant load on that pixel in kg/yr,
        or FLOAT_NODATA if any of the inputs have nodata on that pixel.
    """
    load_array = numpy.full(
        lulc_array.shape, FLOAT_NODATA, dtype=numpy.float32)
    valid_mask = (
        (lulc_array != lulc_nodata) &
        ~numpy.isclose(retention_volume_array, FLOAT_NODATA))

    # bin each value in the LULC array such that
    # lulc_array[i,j] == sorted_lucodes[lulc_index[i,j]]. thus,
    # emc_array[lulc_index[i,j]] is the EMC for the lucode at lulc_array[i,j]
    lulc_index = numpy.digitize(lulc_array, sorted_lucodes, right=True)
    # EMC for pollutant (mg/L) * 1000 (L/m^3) * 0.000001 (kg/mg) *
    # retention (m^3/yr) = pollutant load (kg/yr)
    load_array[valid_mask] = (emc_array[lulc_index][valid_mask] *
                              0.001 * retention_volume_array[valid_mask])
    return load_array


def retention_value_op(retention_volume_array, replacement_cost):
    """Multiply array of retention volumes by the retention replacement
    cost to get an array of retention values. This is meant to be used with
    raster_calculator.

    Args:
        retention_volume_array (numpy.ndarray): 2D array of retention volumes.
            Assumes that the retention volume nodata value is the global
            FLOAT_NODATA.
        replacement_cost (float): Replacement cost per cubic meter of water

    Returns:
        numpy.ndarray of retention values with the same dimensions as the input
    """
    value_array = numpy.full(retention_volume_array.shape, FLOAT_NODATA,
                             dtype=numpy.float32)
    valid_mask = ~numpy.isclose(retention_volume_array, FLOAT_NODATA)

    # retention (m^3/yr) * replacement cost ($/m^3) = retention value ($/yr)
    value_array[valid_mask] = (
        retention_volume_array[valid_mask] * replacement_cost)
    return value_array


def adjust_op(ratio_array, avg_ratio_array, near_impervious_lulc_array,
              near_road_array):
    """Apply the retention ratio adjustment algorithm to an array of ratios.
    This is meant to be used with raster_calculator. Assumes that the nodata
    value for all four input arrays is the global FLOAT_NODATA.

    Args:
        ratio_array (numpy.ndarray): 2D array of stormwater retention ratios
        avg_ratio_array (numpy.ndarray): 2D array of averaged ratios
        near_impervious_lulc_array (numpy.ndarray): 2D boolean array where 1
            means this pixel is near a directly-connected LULC area
        near_road_array (numpy.ndarray): 2D boolean array where 1
            means this pixel is near a road centerline

    Returns:
        2D numpy array of adjusted retention ratios. Has the same shape as
        ``retention_ratio_array``.
    """
    adjusted_ratio_array = numpy.full(ratio_array.shape, FLOAT_NODATA,
                                      dtype=numpy.float32)
    adjustment_factor_array = numpy.full(ratio_array.shape, FLOAT_NODATA,
                                         dtype=numpy.float32)
    valid_mask = (
        ~numpy.isclose(ratio_array, FLOAT_NODATA) &
        ~numpy.isclose(avg_ratio_array, FLOAT_NODATA) &
        (near_impervious_lulc_array != FLOAT_NODATA) &
        (near_road_array != FLOAT_NODATA))

    # adjustment factor:
    # - 0 if any of the nearby pixels are impervious/connected;
    # - average of nearby pixels, otherwise
    is_not_impervious = ~(
        near_impervious_lulc_array[valid_mask] |
        near_road_array[valid_mask]).astype(bool)
    adjustment_factor_array[valid_mask] = (avg_ratio_array[valid_mask] *
                                           is_not_impervious)

    adjustment_factor_array[valid_mask] = (
        avg_ratio_array[valid_mask] * ~(
            near_impervious_lulc_array[valid_mask] |
            near_road_array[valid_mask]
        ).astype(bool))

    # equation 2-4: Radj_ij = R_ij + (1 - R_ij) * C_ij
    adjusted_ratio_array[valid_mask] = (ratio_array[valid_mask] +
                                        (1 - ratio_array[valid_mask]) * adjustment_factor_array[valid_mask])
    return adjusted_ratio_array


def aggregate_results(aggregate_areas_path, aggregations):
    """Aggregate outputs into regions of interest.

    Args:
        aggregate_areas_path (str): path to vector of polygon(s) to aggregate over.
            this should be a copy that it's okay to modify.
        aggregations (list[tuple(str,str,str)]): list of tuples describing the
            datasets to aggregate. Each tuple has 3 items. The first is the
            path to a raster to aggregate. The second is the field name for
            this aggregated data in the output vector. The third is either
            'mean' or 'sum' indicating the aggregation to perform.

    Returns:
        None
    """
    # create a copy of the aggregate areas vector to write to
    aggregate_vector = gdal.OpenEx(aggregate_areas_path, gdal.GA_Update)
    aggregate_layer = aggregate_vector.GetLayer()

    for raster_path, field_id, op in aggregations:
        # aggregate the raster by the vector region(s)
        aggregate_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregate_areas_path)

        # set up the field to hold the aggregate data
        aggregate_field = ogr.FieldDefn(field_id, ogr.OFTReal)
        aggregate_field.SetWidth(24)
        aggregate_field.SetPrecision(11)
        aggregate_layer.CreateField(aggregate_field)
        aggregate_layer.ResetReading()

        # save the aggregate data to the field for each feature
        for feature in aggregate_layer:
            feature_id = feature.GetFID()
            if op == 'mean':
                pixel_count = aggregate_stats[feature_id]['count']
                try:
                    value = (aggregate_stats[feature_id]['sum'] / pixel_count)
                except ZeroDivisionError:
                    LOGGER.warning(
                        f'Polygon {feature_id} does not overlap {raster_path}')
                    value = 0.0
            elif op == 'sum':
                value = aggregate_stats[feature_id]['sum']
            feature.SetField(field_id, float(value))
            aggregate_layer.SetFeature(feature)

    # save the aggregate vector layer and clean up references
    aggregate_layer.SyncToDisk()
    aggregate_layer = None
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    aggregate_vector = None


def is_near(input_path, search_kernel, output_path):
    """Take a boolean raster and create a new boolean raster where a pixel is
    assigned '1' iff it's within a search kernel of a '1' pixel in the original
    raster.

    Args:
        input_path (str): path to a boolean raster. It is assumed that this
            raster's nodata value is the global FLOAT_NODATA.
        search_kernel (numpy.ndarray): 2D numpy array to center on each pixel.
            Pixels that fall on a '1' in the search kernel are counted.
        output_path (str): path to write out the result raster. This is a
            boolean raster where 1 means this pixel's centerpoint is within the
            search kernel of the centerpoint of a '1' pixel in the input raster

    Returns:
        None
    """
    # open the input raster and create the output raster
    in_raster = gdal.OpenEx(input_path, gdal.OF_RASTER)
    in_band = in_raster.GetRasterBand(1)

    pygeoprocessing.new_raster_from_base(
        input_path, output_path, gdal.GDT_Byte, [UINT8_NODATA])
    out_raster = gdal.OpenEx(output_path, gdal.OF_RASTER)
    out_band = out_raster.GetRasterBand(1)

    # iterate over the raster by overlapping blocks
    overlap = int((search_kernel.shape[0] - 1) / 2)
    for block in overlap_iterblocks(input_path, overlap):
        in_array = in_band.ReadAsArray(
            block['xoff'] - block['left_overlap'],
            block['yoff'] - block['top_overlap'],
            block['xsize'] + block['left_overlap'] + block['right_overlap'],
            block['ysize'] + block['top_overlap'] + block['bottom_overlap'])
        in_array[in_array == FLOAT_NODATA] = 0
        padded_array = numpy.pad(in_array,
                                 pad_width=(
                                     (block['top_padding'],
                                      block['bottom_padding']),
                                     (block['left_padding'], block['right_padding'])),
                                 mode='constant',
                                 constant_values=0)
        nodata_mask = padded_array[overlap:-overlap,
                                   overlap:-overlap] == FLOAT_NODATA
        # sum up the values that fall within the search kernel of each pixel
        is_near = scipy.signal.convolve(
            padded_array,
            search_kernel,
            mode='valid') > 0
        is_near[nodata_mask] = UINT8_NODATA

        out_band.WriteArray(is_near, xoff=block['xoff'], yoff=block['yoff'])
    in_raster, in_band, out_raster, out_band = None, None, None, None


def overlap_iterblocks(raster_path, n_pixels):
    """Yield block dimensions and padding such that the raster blocks overlap
    by ``n_pixels`` as much as possible. Where a block can't be extended by
    ``n_pixels`` in any direction, it's extended as far as possible, and the
    rest is indicated by a ``padding`` value for that side. Thus, a block from
    (xoff, yoff) to (xoff + xsize, yoff + ysize) that's then padded with
    top_padding, left_padding, bottom_padding, and right_padding, will always
    have an extra n_pixels of data on each side.

    Args:
        raster_path (str): path to raster to iterate over
        n_pixels (int): Number of pixels by which to overlap the blocks

    Yields:
        dictionary with block dimensions and padding:

        - 'xoff' (int): x offset in pixels of the block's top-left corner
            relative to the raster

        - 'yoff' (int): y offset in pixels of the block's top-left corner
            relative to the raster

        - 'xsize' (int): width of the block in pixels

        - 'ysize' (int): height of the block in pixels

        and for side in 'top', 'left', 'bottom', 'right':

        - 'side_overlap' (int): number in the range [0, n_pixels] indicating how
            many rows you can extend the block on that side.

        - 'side_padding' (int): number in the range [0, n_pixels] indicating how
            many more rows of padding (filler data) to add to that side.

        for each side, side_overlap + side_padding = n_pixels. side_overlap is
        maximized until it hits the edge of the raster, then side_padding covers
        the rest.
    """
    raster_width, raster_height = pygeoprocessing.get_raster_info(
        raster_path)['raster_size']
    for block in pygeoprocessing.iterblocks(
            (raster_path, 1), offset_only=True):
        xoff, yoff = block['xoff'], block['yoff']
        xsize, ysize = block['win_xsize'], block['win_ysize']
        # end coordinates (exclusive)
        xend = block['xoff'] + block['win_xsize']
        yend = block['yoff'] + block['win_ysize']

        # the amount of the padding on each side that can be filled with
        # raster data (if n_pixels is greater than the distance to the edge,
        # it hangs over)
        left_overlap = min(n_pixels, xoff)
        top_overlap = min(n_pixels, yoff)
        right_overlap = min(n_pixels, raster_width - xend)
        bottom_overlap = min(n_pixels, raster_height - yend)

        # the amount of padding that couldn't be filled with raster data
        # (hanging over the edge). the calling function can decide how to
        # fill this space, typically with zeros.
        left_padding = n_pixels - left_overlap
        top_padding = n_pixels - top_overlap
        right_padding = n_pixels - right_overlap
        bottom_padding = n_pixels - bottom_overlap

        yield {
            'xoff': int(xoff),
            'yoff': int(yoff),
            'xsize': int(xsize),
            'ysize': int(ysize),
            'top_overlap': int(top_overlap),
            'left_overlap': int(left_overlap),
            'bottom_overlap': int(bottom_overlap),
            'right_overlap': int(right_overlap),
            'top_padding': int(top_padding),
            'left_padding': int(left_padding),
            'bottom_padding': int(bottom_padding),
            'right_padding': int(right_padding)
        }


def make_search_kernel(raster_path, radius, out_path):
    """Make a search kernel for a raster that marks pixels within a radius.
    Save the search kernel to a raster for use with pygeoprocessing.convolve_2d

    Args:
        raster_path (str): path to a raster to make kernel for
        radius (float): distance around each pixel's centerpoint to search
            in raster coordinate system units
        out_path (str): path to write out the search kernel as a raster

    Returns:
        2D boolean numpy.ndarray. '1' pixels are within ``radius`` of the
        center pixel, measured centerpoint-to-centerpoint. '0' pixels are
        outside the radius. The array dimensions are as small as possible
        while still including the entire radius.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pixel_radius = radius / raster_info['pixel_size'][0]
    pixel_margin = math.floor(pixel_radius)
    # the search kernel is just large enough to contain all pixels that
    # *could* be within the radius of the center pixel
    search_kernel_shape = tuple([pixel_margin * 2 + 1] * 2)
    # arrays of the column index and row index of each pixel
    col_indices, row_indices = numpy.indices(search_kernel_shape)
    # adjust them so that (0, 0) is the center pixel
    col_indices -= pixel_margin
    row_indices -= pixel_margin
    # hypotenuse_i = sqrt(col_indices_i**2 + row_indices_i**2) for each pixel i
    hypotenuse = numpy.hypot(col_indices, row_indices)
    # boolean kernel where 1=pixel centerpoint is within the radius of the
    # center pixel's centerpoint
    search_kernel = numpy.array(hypotenuse <= pixel_radius, dtype=numpy.uint8)
    LOGGER.debug(
        f'Search kernel for {raster_path} with radius {radius}:'
        f'\n{search_kernel}')

    return search_kernel


def raster_average():
    search_kernel = make_search_kernel(raster_path, radius, out_path)
    # create and open the output raster
    pygeoprocessing.new_raster_from_base(
        raster_path, kernel_path, gdal.GDT_Byte, [UINT8_NODATA])
    kernel_raster = gdal.OpenEx(kernel_path, gdal.OF_RASTER)
    kernel_band = kernel_raster.GetRasterBand(1)

    def ones_op(array):
        ones_array = numpy.zeros(array.shape, dtype=numpy.uint8)
        valid_mask = array != FLOAT_NODATA
        ones_array[valid_mask] = 1
        return ones_array

    pygeoprocessing.raster_calculator(
        [(ratio_path, 1)],
        ones_op,
        ones_path,
        gdal.GDT_Byte,
        UINT8_NODATA)

    # the numerator, the sum of values within the search kernel
    pygeoprocessing.convolve_2d(
        ratio_path,
        kernel_path,
        sum_path,
        ignore_nodata_and_edges=True,
        mask_nodata=True,
        target_dtype=gdal.GDT_Float32,
        target_nodata=FLOAT_NODATA)

    # the denominator, the number of valid values within the search kernel
    # nodata does not matter here because there are no nodata pixels in ones_path.
    # they have already been filled with 0s
    pygeoprocessing.convolve_2d(
        ones_path,
        kernel_path,
        n_values_path,
        ignore_nodata_and_edges=True,
        target_dtype=gdal.GDT_Float32,
        target_nodata=FLOAT_NODATA)

    def divide_op(numerator_array, denominator_array):
        out_array = numpy.full(numerator_array.shape, FLOAT_NODATA,
                               dtype=numpy.float32)
        valid_mask = (
            ~numpy.isclose(numerator_array, FLOAT_NODATA) &
            (denominator_array != FLOAT_NODATA))
        out_array[valid_mask] = (
            numerator_array[valid_mask] / denominator_array[valid_mask])
        return out_array

    # compute the average and save to a raster
    pygeoprocessing.raster_calculator(
        [(sum_path, 1), (n_values_path, 1)],
        divide_op,
        average_path,
        gdal.GDT_Float32,
        FLOAT_NODATA)


@validation.invest_validator
def validate(args):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.
    """
    return validation.validate(args, ARGS_SPEC['args'],
                               ARGS_SPEC['args_with_spatial_overlap'])
