"""Stormwater Retention"""
import logging
import math
import numpy
import os
from osgeo import gdal, ogr
import pygeoprocessing
import scipy.ndimage
import scipy.signal
import taskgraph

from . import validation
from . import utils

LOGGER = logging.getLogger(__name__)

# a constant nodata value to use for intermediates and outputs
NODATA = -1

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
            # "bands": {1: {"type": "code"}},
            "required": True,
            "about": (
                "A map of land use/land cover classes in the area of interest"),
            "name": "land use/land cover"
        },
        "soil_group_path": {
            "type": "raster",
            # "bands": {
            #     1: {
            #         "type": "option_string",
            #         "options": ["1", "2", "3", "4"]
            #     }
            # },
            "required": True,
            "about": (
                "Map of hydrologic soil groups, where pixel values 1, 2, 3, "
                "and 4 correspond to groups A, B, C, and D respectively"),
            "name": "soil groups"
        },
        "precipitation_path": {
            "type": "raster",
            # "bands": {1: {"type": "number", "units": "millimeters"}},
            "required": True,
            "about": ("Map of total annual precipitation"),
            "name": "precipitation"
        },
        "biophysical_table": {
            "type": "csv",
            # "columns": {
            #     "lucode": {"type": "code"},
            #     "is_connected": {"type": "boolean"},
            #     "EMC_P": {"type": "number", "units": "mg/L"},
            #     "EMC_N": {"type": "number", "units": "mg/L"},
            #     "RC_A": {"type": "ratio"},
            #     "RC_B": {"type": "ratio"},
            #     "RC_C": {"type": "ratio"},
            #     "RC_D": {"type": "ratio"},
            #     "IR_A": {"type": "ratio"},
            #     "IR_B": {"type": "ratio"},
            #     "IR_C": {"type": "ratio"},
            #     "IR_D": {"type": "ratio"}
            # },
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
                "impervious surfaces are indicated by the is_connected column"
                "in the biophysical table and/or the road centerlines vector."),
            "name": "adjust retention ratios"
        },
        "retention_radius": {
            "type": "number",
            # "units": "meters",
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
            # "fields": {},
            # "geometry": {'LINESTRING'},
            "required": "adjust_retention_ratios",
            "about": "Map of road centerlines",
            "name": "road centerlines"
        },
        "aggregate_areas_path": {
            "type": "vector",
            # "fields": {},
            # "geometry": {'POLYGON'},
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
            # "units": "currency/m^3",
            "required": False,
            "about": "Replacement cost of stormwater retention devices",
            "name": "replacement cost"
        }
    }
}


def execute(args):
    """Execute the stormwater model.
    
    Args:
        args['lulc_path'] (str): path to LULC raster
        args['soil_group_path'] (str): path to soil group raster, where pixel 
            values 1, 2, 3, 4 correspond to groups A, B, C, D
        args['precipitation_path'] (str): path to raster of total annual 
            precipitation in millimeters
        args['biophysical_table'] (str): path to biophysical table with columns
            'lucode', 'EMC_x' (event mean concentration mg/L) for each 
            pollutant x, 'RC_y' (retention coefficient) and 'IR_y' 
            (infiltration coefficient) for each soil group y, and 
            'is_connected' if args['adjust_retention_ratios'] is True
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
    cache_dir = os.path.join(output_dir, 'cache_dir')
    utils.make_directories([args['workspace_dir'], intermediate_dir, cache_dir])

    FILES = {
        'lulc_aligned_path': os.path.join(intermediate_dir, f'lulc_aligned{suffix}.tif'),
        'soil_group_aligned_path': os.path.join(intermediate_dir, f'soil_group_aligned{suffix}.tif'),
        'precipitation_aligned_path': os.path.join(intermediate_dir, f'precipitation_aligned{suffix}.tif'),
        'retention_ratio_path': os.path.join(output_dir, f'retention_ratio{suffix}.tif'),
        'retention_volume_path': os.path.join(output_dir, f'retention_volume{suffix}.tif'),
        'infiltration_ratio_path': os.path.join(output_dir, f'infiltration_ratio{suffix}.tif'),
        'infiltration_volume_path': os.path.join(output_dir, f'infiltration_volume{suffix}.tif'),
        'retention_value_path': os.path.join(output_dir, f'retention_value{suffix}.tif'),
        'aggregate_data_path': os.path.join(output_dir, f'aggregate{suffix}.gpkg'),
        'connected_lulc_path': os.path.join(intermediate_dir, f'is_connected_lulc{suffix}.tif'),
        'adjusted_retention_ratio_path': os.path.join(intermediate_dir, f'adjusted_retention_ratio{suffix}.tif'),
        'x_coords_path': os.path.join(intermediate_dir, f'x_coords{suffix}.tif'),
        'y_coords_path': os.path.join(intermediate_dir, f'y_coords{suffix}.tif'),
        'road_distance_path': os.path.join(intermediate_dir, f'road_distance{suffix}.tif'),
        'near_connected_lulc_path': os.path.join(intermediate_dir, f'near_connected_lulc{suffix}.tif'),
        'near_road_path': os.path.join(intermediate_dir, f'near_road{suffix}.tif'),
        'ratio_n_values_path': os.path.join(intermediate_dir, f'ratio_n_values{suffix}.tif'),
        'ratio_sum_path': os.path.join(intermediate_dir, f'ratio_sum{suffix}.tif'),
        'ratio_average_path': os.path.join(intermediate_dir, f'ratio_average{suffix}.tif'),
    }
    
    align_inputs = [args['lulc_path'], args['soil_group_path'], args['precipitation_path']]
    align_outputs = [
        FILES['lulc_aligned_path'],
        FILES['soil_group_aligned_path'], 
        FILES['precipitation_aligned_path']]

    radius = float(args['retention_radius'])  # in raster coord system units

    task_graph = taskgraph.TaskGraph(args['workspace_dir'], int(args.get('n_workers', -1)))

    # Align all three input rasters to the same projection
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(align_inputs, align_outputs, ['near' for _ in align_inputs],
            pixel_size, 'intersection'),
        kwargs={'raster_align_index': 0},
        target_path_list=align_outputs,
        task_name='align input rasters')

    # Build a lookup dictionary mapping each LULC code to its row
    biophysical_dict = utils.build_lookup_from_csv(
        args['biophysical_table'], 'lucode')

    # Make ratio lookup dictionaries mapping each LULC code to a ratio for 
    # each soil group. Biophysical table has runoff coefficents so subtract 
    # from 1 to get retention coefficient.
    retention_ratio_dict = {
        lucode: {
            'A': 1 - row['rc_a'],
            'B': 1 - row['rc_b'],
            'C': 1 - row['rc_c'],
            'D': 1 - row['rc_d'],
        } for lucode, row in biophysical_dict.items()
    }
    infiltration_ratio_dict = {
        lucode: {
            'A': row['ir_a'],
            'B': row['ir_b'],
            'C': row['ir_c'],
            'D': row['ir_d'],
        } for lucode, row in biophysical_dict.items()
    }

    # Calculate stormwater retention ratio and volume from
    # LULC, soil groups, biophysical data, and precipitation
    retention_ratio_task = task_graph.add_task(
        func=calculate_stormwater_ratio,
        args=(
            FILES['lulc_aligned_path'],
            FILES['soil_group_aligned_path'],
            retention_ratio_dict,
            FILES['retention_ratio_path']),
        target_path_list=[FILES['retention_ratio_path']],
        dependent_task_list=[align_task],
        task_name='calculate stormwater retention ratio'
    )

    # (Optional) adjust stormwater retention ratio using roads
    if args['adjust_retention_ratios']:
        # boolean mapping for each LULC code whether it's connected
        is_connected_lookup = {lucode: row['is_connected'] 
            for lucode, row in biophysical_dict.items()}
        # Make a boolean raster indicating which pixels are directly
        # connected impervious LULC type
        connected_lulc_task = task_graph.add_task(
            func=calculate_connected_lulc,
            args=(FILES['lulc_aligned_path'], is_connected_lookup, 
                FILES['connected_lulc_path']),
            target_path_list=[FILES['connected_lulc_path']],
            task_name='calculate binary connected lulc raster',
            dependent_task_list=[align_task]
        )
    
        # Make a boolean raster indicating which pixels are within the
        # given radius of a directly-connected impervious LULC type
        connected_lulc_search_kernel = make_search_kernel(
            FILES['connected_lulc_path'], radius)
        near_connected_lulc_task = task_graph.add_task(
            func=is_near,
            args=(FILES['connected_lulc_path'], connected_lulc_search_kernel,
                FILES['near_connected_lulc_path']),
            target_path_list=[FILES['near_connected_lulc_path']],
            task_name='find pixels within radius of connected LULC',
            dependent_task_list=[connected_lulc_task])

        # Make a raster of the distance from each pixel to the nearest 
        # road centerline
        coordinate_rasters_task = task_graph.add_task(
            func=make_coordinate_rasters,
            args=(FILES['retention_ratio_path'], 
                FILES['x_coords_path'], FILES['y_coords_path']),
            target_path_list=[FILES['x_coords_path'], FILES['y_coords_path']],
            task_name='make coordinate rasters',
            dependent_task_list=[retention_ratio_task]
        )
        distance_task = task_graph.add_task(
            func=distance_to_road_centerlines,
            args=(FILES['x_coords_path'], FILES['y_coords_path'],
                args['road_centerlines_path'], FILES['road_distance_path']),
            target_path_list=[FILES['road_distance_path']],
            task_name='calculate pixel distance to roads',
            dependent_task_list=[coordinate_rasters_task]
        )

        # Make a boolean raster showing which pixels are within the given
        # radius of a road centerline
        near_road_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(FILES['road_distance_path'], 1), (radius, 'raw')],
                threshold_array,
                FILES['near_road_path'],
                gdal.GDT_Float32, 
                NODATA),
            target_path_list=[FILES['near_road_path']],
            task_name='find pixels within radius of road centerlines',
            dependent_task_list=[distance_task])

        # boolean kernel where 1=pixel centerpoint is within the radius of the 
        # center pixel's centerpoint
        ratio_search_kernel = make_search_kernel(
            FILES['retention_ratio_path'], radius)
        # Average the retention ratio values around each pixel
        average_ratios_task = task_graph.add_task(
            func=raster_average,
            args=(
                FILES['retention_ratio_path'],
                ratio_search_kernel,
                FILES['ratio_n_values_path'],
                FILES['ratio_sum_path'],
                FILES['ratio_average_path']),
            target_path_list=[FILES['ratio_average_path']],
            task_name='average retention ratios within radius',
            dependent_task_list=[retention_ratio_task])

        # Using the averaged retention ratio raster and boolean 
        # "within radius" rasters, adjust the retention ratios
        adjust_retention_ratio_task = task_graph.add_task(
            func= pygeoprocessing.raster_calculator,
            args=([
                    (FILES['retention_ratio_path'], 1), 
                    (FILES['ratio_average_path'], 1),
                    (FILES['near_connected_lulc_path'], 1),
                    (FILES['near_road_path'], 1)
                ],
                adjust_op, 
                FILES['adjusted_retention_ratio_path'],
                gdal.GDT_Float32, 
                NODATA),
            target_path_list=[FILES['adjusted_retention_ratio_path']],
            task_name='adjust stormwater retention ratio',
            dependent_task_list=[retention_ratio_task, average_ratios_task, 
                near_connected_lulc_task, near_road_task])

        final_retention_ratio_path = FILES['adjusted_retention_ratio_path']
        final_retention_ratio_task = adjust_retention_ratio_task
    else:
        final_retention_ratio_path = FILES['retention_ratio_path']
        final_retention_ratio_task = retention_ratio_task

    # Calculate stormwater retention volume from ratios and precipitation
    retention_volume_task = task_graph.add_task(
        func=calculate_stormwater_volume,
        args=(
            final_retention_ratio_path,
            FILES['precipitation_aligned_path'],
            FILES['retention_volume_path']),
        target_path_list=[FILES['retention_volume_path']],
        dependent_task_list=[align_task, final_retention_ratio_task],
        task_name='calculate stormwater retention volume'
    )

    # (Optional) Calculate stormwater infiltration ratio and volume from
    # LULC, soil groups, biophysical table, and precipitation
    infiltration_ratio_task = task_graph.add_task(
        func=calculate_stormwater_ratio,
        args=(
            FILES['lulc_aligned_path'],
            FILES['soil_group_aligned_path'],
            infiltration_ratio_dict,
            FILES['infiltration_ratio_path']),
        target_path_list=[FILES['infiltration_ratio_path']],
        dependent_task_list=[align_task],
        task_name='calculate stormwater infiltration ratio'
    )
    infiltration_volume_task = task_graph.add_task(
        func=calculate_stormwater_volume,
        args=(
            FILES['infiltration_ratio_path'],
            FILES['precipitation_aligned_path'],
            FILES['infiltration_volume_path']),
        target_path_list=[FILES['infiltration_volume_path']],
        dependent_task_list=[align_task, infiltration_ratio_task],
        task_name='calculate stormwater retention volume'
    )

    # get all EMC columns from an arbitrary row in the dictionary
    # strip the first four characters off 'EMC_pollutant' to get pollutant name
    emc_columns = [key for key in next(iter(biophysical_dict.values()))
        if key.startswith('emc_')]
    pollutants = [key[4:] for key in  emc_columns]
    LOGGER.info(f'Pollutants found in biophysical table: {pollutants}')

    # Calculate avoided pollutant load for each pollutant from retention volume
    # and biophysical table EMC value
    avoided_load_paths = []
    aggregation_dependencies = [retention_volume_task, infiltration_volume_task]
    for pollutant in pollutants:
        # one output raster for each pollutant
        avoided_pollutant_load_path = os.path.join(
            output_dir, f'avoided_pollutant_load_{pollutant}{suffix}.tif')
        avoided_load_paths.append(avoided_pollutant_load_path)
        # make a dictionary mapping each LULC code to the pollutant EMC value
        lulc_emc_lookup = {
            lucode: row[f'emc_{pollutant}'] for lucode, row in biophysical_dict.items()
        }
        avoided_load_task = task_graph.add_task(
            func=calculate_avoided_pollutant_load,
            args=(
                FILES['lulc_aligned_path'],
                FILES['retention_volume_path'],
                lulc_emc_lookup,
                avoided_pollutant_load_path),
            target_path_list=[avoided_pollutant_load_path],
            dependent_task_list=[retention_volume_task],
            task_name=f'calculate avoided pollutant {pollutant} load'
        )
        aggregation_dependencies.append(avoided_load_task)

    # (Optional) Do valuation if a replacement cost is defined
    # you could theoretically have a cost of 0 which should be allowed
    if (args['replacement_cost'] not in [None, '']):
        valuation_task = task_graph.add_task(
            func=calculate_retention_value,
            args=(
                FILES['retention_volume_path'],
                args['replacement_cost'],
                FILES['retention_value_path']),
            target_path_list=[FILES['retention_value_path']],
            dependent_task_list=[retention_volume_task],
            task_name='calculate stormwater retention value'
        )
        aggregation_dependencies.append(valuation_task)
        valuation_path = FILES['retention_value_path']
    else:
        valuation_path = None

    # (Optional) Aggregate to watersheds if an aggregate vector is defined
    if (args['aggregate_areas_path']):
        aggregation_task = task_graph.add_task(
            func=aggregate_results,
            args=(
                args['aggregate_areas_path'],
                FILES['retention_ratio_path'],
                FILES['retention_volume_path'],
                FILES['infiltration_ratio_path'],
                FILES['infiltration_volume_path'],
                avoided_load_paths,
                valuation_path,
                FILES['aggregate_data_path']),
            target_path_list=[FILES['aggregate_data_path']],
            dependent_task_list=aggregation_dependencies,
            task_name='aggregate data over polygons'
        )

    task_graph.close()
    task_graph.join()

def threshold_array(array, value):
    return array <= value

def ratio_op(lulc_array, soil_group_array, ratio_lookup, sorted_lucodes):
    """Make an array of stormwater retention or infiltration ratios from 
    arrays of LULC codes and hydrologic soil groups.

    Args:
        lulc_array (numpy.ndarray): 2D array of LULC codes
        soil_group_array (numpy.ndarray): 2D array with the same shape as
            ``lulc_array``. Values in {1, 2, 3, 4} corresponding to soil 
            groups A, B, C, and D.
        ratio_lookup (numpy.ndarray): 2D array where rows correspond to 
            sorted LULC codes and columns correspond to soil groups
            A, B, C, D in order. Shape: (number of lulc codes, 4)
        sorted_lucodes (list[int]): List of LULC codes sorted from smallest 
            to largest. These correspond to the rows of ``ratio_lookup``.

    Returns:
        2D numpy array with the same shape as ``lulc_array`` and 
        ``soil_group_array``. Each value is the corresponding ratio for that
        LULC code x soil group pair.
    """
    sorted_soil_groups = [1, 2, 3, 4]
    # the index of each soil group in the sorted soil groups array
    soil_group_index = numpy.digitize(soil_group_array, sorted_soil_groups, 
        right=True)
    # the index of each lucode in the sorted lucodes array
    lulc_index = numpy.digitize(lulc_array, sorted_lucodes, right=True)
    
    output_ratio_array = ratio_lookup[lulc_index, soil_group_index]
    return output_ratio_array


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
    # convert the nested dictionary in to a 2D array where rows are LULC codes 
    # in sorted order and columns correspond to soil groups in order
    # this facilitates efficiently looking up the ratio values with numpy
    sorted_lucodes = sorted(list(ratio_lookup.keys()))
    lulc_soil_group_array = numpy.array([
        [ratio_lookup[lucode][soil_group] 
            for soil_group in ['A', 'B', 'C', 'D']
        ] for lucode in sorted_lucodes])

    # Apply ratio_op to each block of the LULC and soil group rasters
    # Write result to output_path as float32 with nodata=NODATA
    pygeoprocessing.raster_calculator(
        [(lulc_path, 1), (soil_group_path, 1), (lulc_soil_group_array, 'raw'), 
        (sorted_lucodes, 'raw')], ratio_op, output_path, gdal.GDT_Float32, 
        NODATA)


def calculate_stormwater_volume(ratio_path, precipitation_path, output_path):
    """Make stormwater retention or infiltration volume map from ratio and 
       precipitation.

    Args:
        ratio_path (str): path to a raster of stormwater ratios
        precipitation_path (str): path to a raster of precipitation amounts
        output_path (str): path to write out the volume results (raster)

    Returns:
        None
    """
    ratio_raster_info = pygeoprocessing.get_raster_info(ratio_path)
    ratio_nodata = ratio_raster_info['nodata'][0]
    pixel_area = abs(ratio_raster_info['pixel_size'][0] * 
        ratio_raster_info['pixel_size'][1])
    precipitation_nodata = pygeoprocessing.get_raster_info(
        precipitation_path)['nodata'][0]

    def volume_op(ratio_array, precipitation_array):
        """Calculate array of volumes (retention or infiltration) from arrays 
        of precipitation values and stormwater ratios"""

        volume_array = numpy.full(ratio_array.shape, NODATA, dtype=float)
        nodata_mask = (
            (ratio_array != ratio_nodata) & 
            (precipitation_array != precipitation_nodata))

        # precipitation (mm/yr) * pixel area (m^2) * 
        # 0.001 (m/mm) * ratio = volume (m^3/yr)
        volume_array[nodata_mask] = (
            precipitation_array[nodata_mask] *
            ratio_array[nodata_mask] *
            pixel_area * 0.001)
        return volume_array

    # Apply volume_op to each block in the ratio and precipitation rasters
    # Write result to output_path as float32 with nodata=NODATA
    pygeoprocessing.raster_calculator(
        [(ratio_path, 1), (precipitation_path, 1)],
        volume_op, output_path, gdal.GDT_Float32, NODATA)


def calculate_avoided_pollutant_load(lulc_path, retention_volume_path, 
        emc_lookup, output_path):
    """Make avoided pollutant load map from retention volumes and LULC event 
       mean concentration data.

    Args:
        lulc_path (str): path to a LULC raster whose LULC codes exist in the
            EMC lookup dictionary
        retention_volume_path: (str) path to a raster of stormwater retention
            volumes in m^3
        emc_lookup (dict): a lookup dictionary where keys are LULC codes 
            and values are event mean concentration (EMC) values in mg/L for 
            the pollutant in that LULC area.
        output_path (str): path to write out the results (raster)

    Returns:
        None
    """
    lulc_nodata = pygeoprocessing.get_raster_info(lulc_path)['nodata'][0]
    sorted_lucodes = sorted(list(emc_lookup.keys()))
    ordered_emc_array = numpy.array(
        [emc_lookup[lucode] for lucode in sorted_lucodes])

    def avoided_pollutant_load_op(lulc_array, retention_volume_array):
        """Calculate array of avoided pollutant load values from arrays of 
        LULC codes and stormwater retention volumes."""
        load_array = numpy.full(lulc_array.shape, NODATA, dtype=float)
        valid_mask = (
            (lulc_array != lulc_nodata) &
            (retention_volume_array != NODATA))

        lulc_index = numpy.digitize(lulc_array, sorted_lucodes, right=True)
        # EMC for pollutant (mg/L) * 1000 (L/m^3) * 0.000001 (kg/mg) * 
        # retention (m^3/yr) = pollutant load (kg/yr)
        load_array[valid_mask] = (ordered_emc_array[lulc_index][valid_mask] * 
            0.001 * retention_volume_array[valid_mask])
        return load_array

    # Apply avoided_pollutant_load_op to each block of the LULC and retention 
    # volume rasters. Write result to output_path as float32 with nodata=NODATA
    pygeoprocessing.raster_calculator(
        [(lulc_path, 1), (retention_volume_path, 1)],
        avoided_pollutant_load_op, output_path, gdal.GDT_Float32, NODATA)


def calculate_retention_value(retention_volume_path, replacement_cost, output_path):
    """Calculate retention value from retention volume and replacement cost.
    Args:
        retention_volume_path (str): path to retention volume raster (m^3/pixel)
        replacement_cost (float): value in currency units/m^3
        output_path (str): path to write out valuation results raster

    Returns:
        None
    """
    def retention_value_op(retention_volume_array):
        """Multiply array of retention volumes by the retention replacement 
        cost to get an array of retention values."""
        value_array = numpy.full(retention_volume_array.shape, NODATA, dtype=float)
        nodata_mask = (retention_volume_array != NODATA)

        # retention (m^3/yr) * replacement cost ($/m^3) = retention value ($/yr)
        value_array[nodata_mask] = (
            retention_volume_array[nodata_mask] * replacement_cost)
        return value_array

    # Apply retention_value_op to each block of the retention volume rasters
    # Write result to output_path as float32 with nodata=NODATA
    pygeoprocessing.raster_calculator(
        [(retention_volume_path, 1)],
        retention_value_op, output_path, gdal.GDT_Float32, NODATA)



def aggregate_results(aoi_path, r_ratio_path, r_volume_path, 
        i_ratio_path, i_volume_path, avoided_pollutant_loads, 
        retention_value, output_path):
    """Aggregate outputs into regions of interest.

    Args:
        aoi_path (str): path to vector of polygon(s) to aggregate over
        retention_ratio (str): path to stormwater retention ratio raster
        retention_volume (str): path to stormwater retention volume raster
        infiltration_ratio (str): path to stormwater infiltration ratio raster
        infiltration_volume (str): path to stormwater infiltration volume raster
        avoided_pollutant_loads (list[str]): list of paths to avoided pollutant
            load rasters
        retention_value (str): path to retention value raster
        output_path (str): path to write out aggregated vector data

    Returns:
        None
    """

    if os.path.exists(output_path):
        LOGGER.warning(
            '%s exists, deleting and writing new output',
            output_path)
        os.remove(output_path)

    original_aoi_vector = gdal.OpenEx(aoi_path, gdal.OF_VECTOR)

    # copy AOI vector to the output path and convert to GPKG if needed
    result = gdal.VectorTranslate(output_path, aoi_path)
    
    aggregate_vector = gdal.OpenEx(output_path, 1)
    aggregate_layer = aggregate_vector.GetLayer()

    aggregations = [
        (r_ratio_path, 'RR_mean', 'mean'),     # average retention ratio
        (r_volume_path, 'RV_sum', 'sum'),      # total retention volume
        (i_ratio_path, 'IR_mean', 'mean'),     # average infiltration ratio
        (i_volume_path, 'IV_sum', 'sum'),      # total infiltration volume
    ]
    if (retention_value):                      # total retention value
        aggregations.append((retention_value, 'val_sum', 'sum'))
    for avoided_load_path in avoided_pollutant_loads:
        pollutant = avoided_load_path.split('_')[-1]
        field = f'avoided_{pollutant}'
        aggregations.append((avoided_load_path, field, 'sum'))


    for raster_path, field_id, op in aggregations:
        # aggregate the raster by the vector region(s)
        aggregate_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), output_path)

        # set up the field to hold the aggregate data
        aggregate_field = ogr.FieldDefn(field_id, ogr.OFTReal)
        aggregate_field.SetWidth(24)
        aggregate_field.SetPrecision(11)
        aggregate_layer.CreateField(aggregate_field)
        aggregate_layer.ResetReading()

        # save the aggregate data to the field for each feature
        for polygon in aggregate_layer:
            feature_id = polygon.GetFID()
            if op == 'mean':
                pixel_count = aggregate_stats[feature_id]['count']
                if pixel_count != 0:
                    value = (aggregate_stats[feature_id]['sum'] / pixel_count)
                else:
                    LOGGER.warning(
                        "no coverage for polygon %s", ', '.join(
                            [str(polygon.GetField(_)) for _ in range(
                                polygon.GetFieldCount())]))
                    value = 0.0
            elif op == 'sum':
                value = aggregate_stats[feature_id]['sum']
            polygon.SetField(field_id, float(value))
            aggregate_layer.SetFeature(polygon)

    # save the aggregate vector layer and clean up references
    aggregate_layer.SyncToDisk()
    aggregate_layer = None
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    aggregate_vector = None


def calculate_connected_lulc(lulc_path, impervious_lookup, output_path):
    """Convert LULC raster to a binary raster where 1 is directly connected
    impervious LULC type and 0 is not.

    Args:
        lulc_path (str): path to a LULC raster
        impervious_lookup (dict): dictionary mapping each LULC code in the 
            LULC raster to a boolean value, where True means the LULC type 
            is a directly-connected impervious surface
        output_path (str): path to write out the binary raster

    Returns:
        None
    """
    lulc_nodata = pygeoprocessing.get_raster_info(lulc_path)['nodata'][0]
    # make a list of the LULC codes in order and a list of the corresponding
    # binary impervious values
    sorted_lucodes = sorted(list(impervious_lookup.keys()))
    impervious_lookup_array = numpy.array(
        [impervious_lookup[lucode] for lucode in sorted_lucodes])

    def connected_op(lulc_array):
        is_connected_array = numpy.full(lulc_array.shape, NODATA)
        valid_mask = (lulc_array != lulc_nodata)
        lulc_index = numpy.digitize(lulc_array, sorted_lucodes, right=True)
        is_connected_array[valid_mask] = (
            impervious_lookup_array[lulc_index][valid_mask])
        return is_connected_array

    pygeoprocessing.raster_calculator(
        [(lulc_path, 1)], connected_op, output_path, gdal.GDT_Float32, NODATA)


def is_near(input_path, search_kernel, output_path):
    """Take a boolean raster and create a new boolean raster where a pixel is
    assigned '1' iff it's within a search kernel of a '1' pixel in the original
    raster.

    Args:
        input_path (str): path to a boolean raster
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
    raster_width, raster_height = pygeoprocessing.get_raster_info(
        input_path)['raster_size']
    raster_driver = gdal.GetDriverByName('GTIFF')
    out_raster = raster_driver.Create(
        output_path, raster_width, raster_height, 1, gdal.GDT_Float32,
        options=pygeoprocessing.geoprocessing_core.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])
    out_band = out_raster.GetRasterBand(1)

    # iterate over the raster by overlapping blocks
    overlap = int((search_kernel.shape[0] - 1) / 2)
    for block in overlap_iterblocks(input_path, overlap):
        in_array = in_band.ReadAsArray(block['xoff'], block['yoff'], 
            block['xsize'], block['ysize'])
        padded_array = numpy.pad(in_array, 
            pad_width=((block['top_padding'], block['bottom_padding']), 
                (block['left_padding'], block['right_padding'])), 
            mode='constant', constant_values=0)
        convolved = scipy.signal.convolve(
            padded_array, 
            search_kernel, 
            mode='valid')
        valid_mask = padded_array[margin:-margin, margin:-margin] != NODATA
        is_near = numpy.full(convolved.shape, NODATA)
        is_near[valid_mask] = convolved[valid_mask] > 0

        print('writing to', block['xoff'] + (margin - block['left_padding']), block['yoff'] + (margin - block['top_padding']))
        out_band.WriteArray(is_near, 
            xoff=block['xoff'] + (margin - block['left_padding']), 
            yoff=block['yoff'] + (margin - block['top_padding']))

        

def line_distance_op(x_coords, y_coords, x1, y1, x2, y2):
    """Find the minimum distance from each array point to a line segment.

    Args:
        x_coords (numpy.ndarray): a 2D array where each element is the
            x-coordinate of a point in the same coordinate system as the
            line endpoints
        y_coords (numpy.ndarray): a 2D array where each element is the
            y-coordinate of a point in the same coordinate system as the
            line endpoints
        x1 (float): the x coord of the first endpoint of the line segment
        y1 (float): the y coord of the first endpoint of the line segment
        x2 (float): the x coord of the second endpoint of the line segment
            ((x2, y2) can't be identical to (x1, y1))
        y2 (float): the y coord of the second endpoint of the line segment
            ((x2, y2) can't be identical to (x1, y1))

    Returns:
        numpy.ndarray with the same shape as x_coords and y_coords. The
        value of an element at [a, b] is the minimum distance from the
        point (x_coords[a, b], y_coords[a, b]) to the line segment from 
        (x1, y1) to (x2, y2). 
    """
    # Using the algorithm from https://math.stackexchange.com/a/330329:
    # Parameterize the line segment by parameter t, which represents how far
    # along the line segment we are from endpoint 1 to endpoint 2.
    # x(t) = x1 + t(x2 - x1)
    # y(t) = y1 + t(y2 - y1)
    # (x(t), y(t)) is on the segment when t ∈ [0, 1]

    # the notation ⟨𝑝−𝑠1,𝑠2−𝑠1⟩ in the SE post means the dot product:
    # (𝑝-𝑠1)·(𝑠2−𝑠1) = (x-x1)*(x2-x1) + (y-y1)*(y2-y1)
    # the notation ‖𝑠2−𝑠1‖ means the pythagorean distance

    # solve for the optimal value of t, such that the distance from
    # (x_coord, y_coord) to (x(t), y(t)) is minimized
    t_optimal = (
        ((x_coords - x1) * (x2 - x1) + (y_coords - y1) * (y2 - y1)) / 
        ((x2 - x1)**2 + (y2 - y1)**2))
    # constrain t to the bounds of the line segment
    t_in_bounds = numpy.minimum(numpy.maximum(t_optimal, 0), 1)
    # solve for x(t) and y(t)
    nearest_x_coords = x1 + t_in_bounds * (x2 - x1)
    nearest_y_coords = y1 + t_in_bounds * (y2 - y1)
    # find the distance from each (x_coord, y_coord) to (x(t), y(t))
    distances = numpy.hypot(nearest_x_coords - x_coords, 
        nearest_y_coords - y_coords)
    return distances


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
        dictionary with block dimensions and padding.
        'xoff' (int): x offset in pixels of the block's top-left corner 
            relative to the raster
        'yoff' (int): y offset in pixels of the block's top-left corner 
            relative to the raster
        'xsize' (int): width of the block in pixels
        'ysize' (int): height of the block in pixels
        'top_padding' (int): number in the range [0, n_pixels] indicating how 
            many more rows of padding to add to the top of the block. E.g. a
            block on the top edge of the raster would have `top_padding = n_pixels`.
            A block that's `k` rows down from the top would have 
            `min(0, n_pixels - k)`.
    """
    raster_width, raster_height = pygeoprocessing.get_raster_info(
        raster_path)['raster_size']
    for block in pygeoprocessing.iterblocks((raster_path, 1), offset_only=True):
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

        if xoff > 0:
            xoff -= left_overlap
            xsize += left_overlap
        if yoff > 0:
            yoff -= top_overlap
            ysize += top_overlap

        if xend < raster_width:
            xsize += right_overlap
        if yend < raster_height:
            ysize += bottom_overlap

        print({
            'xoff': xoff,
            'yoff': yoff,
            'xsize': xsize,
            'ysize': ysize,
            'top_padding': top_padding,
            'left_padding': left_padding,
            'bottom_padding': bottom_padding,
            'right_padding': right_padding
        })

        yield {
            'xoff': int(xoff),
            'yoff': int(yoff),
            'xsize': int(xsize),
            'ysize': int(ysize),
            'top_padding': int(top_padding),
            'left_padding': int(left_padding),
            'bottom_padding': int(bottom_padding),
            'right_padding': int(right_padding)
        }


def make_search_kernel(raster_path, radius):
    """Make a search kernel for a raster that marks pixels within a radius

    Args:
        raster_path (str): path to a raster to make kernel for
        radius (float): distance around each pixel's centerpoint to search
            in raster coordinate system units

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
    search_kernel_shape = tuple([pixel_margin*2+1]*2)
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
    return search_kernel


def adjust_op(ratio_array, avg_ratio_array, near_connected_lulc_array, 
        near_road_array):
    """Apply the retention ratio adjustment algorithm to an array of ratios. 
    This is meant to be used with raster_calculator.

    Args:
        ratio_array (numpy.ndarray): 2D array of stormwater retention ratios
        avg_ratio_array (numpy.ndarray): 2D array of averaged ratios
        near_connected_lulc_array (numpy.ndarray): 2D boolean array where 1 
            means this pixel is near a directly-connected LULC area
        near_road_array (numpy.ndarray): 2D boolean array where 1 
            means this pixel is near a road centerline
        
    Returns:
        2D numpy array of adjusted retention ratios. Has the same shape as 
        ``retention_ratio_array``.
    """
    adjusted_ratio_array = numpy.full(ratio_array.shape, NODATA)
    adjustment_factor_array = numpy.full(ratio_array.shape, NODATA)
    valid_mask = (
        (ratio_array != NODATA) &
        (avg_ratio_array != NODATA) &
        (near_connected_lulc_array != NODATA) &
        (near_road_array != NODATA))

    is_connected = is_near_impervious_lulc | is_near_road
    # adjustment factor:
    # - 0 if any of the nearby pixels are impervious/connected;
    # - average of nearby pixels, otherwise
    adjustment_factor_array[valid_mask] = (averaged_ratio_array[valid_mask] * 
        ~(near_connected_lulc_array[valid_mask] | near_road_array[valid_mask]))

    # equation 2-4: Radj_ij = R_ij + (1 - R_ij) * C_ij
    adjusted_ratio_array[valid_mask] = (ratio_array[valid_mask] + 
        (1 - ratio_array[valid_mask]) * adjustment_factor_array[valid_mask])
    return adjusted_ratio_array


def adjust_retention_ratios(ratio_path, avg_ratio_path, 
        near_connected_lulc_path, near_road_path, output_path):

    pygeoprocessing.raster_calculator(
        [(ratio_path, 1), (avg_ratio_path, 1), (near_connected_lulc_path, 1), 
        (near_road_path, 1)], adjust_op, output_path, gdal.GDT_Float32, NODATA)




def raster_average(raster_path, search_kernel, n_values_path, sum_path, 
        average_path):
    """Average pixel values within a search kernel.

    For each pixel in a raster, center the search kernel on top of it. Then
    its "neighborhood" includes all the pixels that are below a '1' in the
    search kernel. Add up the neighborhood pixel values and divide by how 
    many there are.
    
    This accounts for edge pixels and nodata pixels. For instance, if the 
    kernel covers a 3x3 pixel area centered on each pixel, most pixels will 
    have 9 valid pixels in their neighborhood, most edge pixels will have 6, 
    and most corner pixels will have 4. Nodata pixels in the neighborhood 
    don't count towards the total.

    Args:
        raster_path (str): path to the raster to average
        search_kernel (numpy.ndarray): 2D numpy array to center on each pixel.
            Pixels that are on a '1' in the search kernel are counted.
        n_values_path (str): path to write out the number of valid pixels in 
            each pixel's neighborhood (this is the denominator in the average)
        sum_path (str): path to write out the sum of valid pixel values in 
            each pixel's neighborhood (this is the numerator in the average)
        average_path (str): path to write out the average of the valid pixel 
            values in each pixel's neighborhood (``n_values_path / sum_path``)

    Returns:
        None
    """
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_width, raster_height = raster_info['raster_size']

    # create and open the three output rasters
    raster_driver = gdal.GetDriverByName('GTIFF')
    n_values_raster = raster_driver.Create(
        n_values_path, raster_width, raster_height, 1, gdal.GDT_Float32,
        options=pygeoprocessing.geoprocessing_core.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])
    n_values_band = n_values_raster.GetRasterBand(1)
    sum_raster = raster_driver.Create(
        sum_path, raster_width, raster_height, 1, gdal.GDT_Float32,
        options=pygeoprocessing.geoprocessing_core.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])
    sum_band = sum_raster.GetRasterBand(1)
    average_raster = raster_driver.Create(
        average_path, raster_width, raster_height, 1, gdal.GDT_Float32,
        options=pygeoprocessing.geoprocessing_core.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])
    average_band = average_raster.GetRasterBand(1)

    # calculate the sum and n_values of the neighborhood of each pixel
    overlap = int((search_kernel.shape[0] - 1) / 2)
    for block in overlap_iterblocks(raster_path, overlap):

        ratio_array = band.ReadAsArray(block['xoff'], block['yoff'], 
            block['xsize'], block['ysize'])
        padded_ratio_array = numpy.pad(ratio_array, 
            pad_width=((block['top_padding'], block['bottom_padding']), 
                (block['left_padding'], block['right_padding'])), 
            mode='constant', constant_values=0)
        # add up the valid pixel values in the neighborhood of each pixel
        sum_array = scipy.signal.convolve(
            padded_ratio_array, 
            search_kernel, 
            mode='valid')

        # have to pad the array after 
        valid_pixels = (ratio_array != NODATA).astype(int)
        padded_valid_pixels = numpy.pad(valid_pixels, 
            pad_width=((block['top_padding'], block['bottom_padding']), 
                (block['left_padding'], block['right_padding'])), 
            mode='constant', constant_values=False)
        # array where each value is the number of valid values within the
        # search kernel. 
        # - for every kernel that doesn't extend past the edge of the original 
        #   array, this is search_kernel.size - number of nodata pixels in kernel
        # - for kernels that extend past the edge, this is the number of 
        #   elements that are within the original array minus the number of 
        #   those that are nodata
        n_values_array = scipy.signal.convolve(
            padded_valid_pixels, 
            search_kernel, 
            mode='valid')
        
        n_values_band.WriteArray(n_values_array, xoff=block['xoff'], yoff=block['yoff'])
        sum_band.WriteArray(sum_array, xoff=block['xoff'], yoff=block['yoff'])

    # Calculate the pixel-wise average from the n_values and sum rasters
    def avg_op(n_values_array, sum_array):
        average_array = numpy.full(n_values_array.shape, NODATA)
        valid_mask = (n_values_array != NODATA) & (sum_array != NODATA)
        average_array[valid_mask] = (
            sum_array[valid_mask] / n_values_array[valid_mask])
        return average_array

    pygeoprocessing.raster_calculator([(n_values_path, 1), (sum_path, 1)], 
        avg_op, average_path, gdal.GDT_Float32, NODATA)


def distance_to_road_centerlines(x_coords_path, y_coords_path, 
        centerlines_path, output_path):
    """Calculate the distance from each pixel centerpoint to the nearest 
    road centerline.

    Args:
        x_coords_path (str): path to a raster where each pixel value is the x 
            coordinate of that pixel in the raster coordinate system
        y_coords_path (str): path to a raster where each pixel value is the y
            coordinate of that pixel in the raster coordinate system
        centerlines_path (str): path to a linestring vector of road centerlines
        output_path (str): path to write out the distance raster. This is a
            raster of the same dimensions, pixel size, and coordinate system 
            as ``raster_path``, where each pixel value is the distance from 
            that pixel's centerpoint to the nearest road centerline. Distances 
            are in the same unit as the raster coordinate system.

    Returns:
        None
    """
    def linestring_geometry_op(x_coords, y_coords):
        segment_generator = iter_linestring_segments(centerlines_path)
        (x1, y1), (x2, y2) = next(segment_generator)
        min_distance = line_distance_op(x_coords, y_coords, x1, y1, x2, y2)

        for (x1, y1), (x2, y2) in segment_generator:
            if x2 == x1 and y2 == y1:
                continue  # ignore lines with length 0
            distance = line_distance_op(x_coords, y_coords, x1, y1, x2, y2)
            min_distance = numpy.minimum(min_distance, distance)
        return min_distance

    pygeoprocessing.raster_calculator(
        [(x_coords_path, 1), (y_coords_path, 1)], 
        linestring_geometry_op, output_path, gdal.GDT_Float32, NODATA)


def iter_linestring_segments(vector_path):
    """Yield (start, end) coordinate pairs for each segment of a linestring.

    Args:
        vector_path (str): path to a linestring vector to iterate over

    Yields:
        ((x1, y1), (x2, y2)) tuples representing the start and end point of a
        linestring segment. (x1, y1) of the nth yielded tuple equals (x2, y2)
        of the (n-1)th yielded tuple.
    """
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    for feature in layer:
        ref = feature.GetGeometryRef()
        assert ref.GetGeometryName() in ['LINESTRING', 'MULTILINESTRING']

        n_geometries = ref.GetGeometryCount()
        if ref.GetGeometryCount() > 0:  # a multi type
            geometries = [ref.GetGeometryRef(i) for i in range(n_geometries)]
        else:  # not a multi type
            geometries = [ref]

        for geometry in geometries:
            points = geometry.GetPoints()  # a list of (x, y) points
            # iterate over each pair of points (each segment) in the linestring
            for i in range(len(points) - 1):
                x1, y1, *_ = points[i]
                x2, y2, *_ = points[i + 1]
                yield (x1, y1), (x2, y2)


def make_coordinate_rasters(raster_path, x_output_path, y_output_path):
    """Make coordinate rasters where each pixel value is the x/y coordinate
    of that pixel's centerpoint in the raster coordinate system.

    Args:
        raster_path (str): raster to generate coordinates for
        x_output_path (str): raster path to write out x coordinates
        y_output_path (str): raster path to write out y coordinates

    Returns:
        None
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pixel_size_x, pixel_size_y = raster_info['pixel_size']
    n_cols, n_rows = raster_info['raster_size']
    x_origin = raster_info['geotransform'][0]
    y_origin = raster_info['geotransform'][3]

    # create the output rasters
    raster_driver = gdal.GetDriverByName('GTIFF')
    x_raster = raster_driver.Create(
        x_output_path, n_cols, n_rows, 1, gdal.GDT_Float32,
        options=pygeoprocessing.geoprocessing_core.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])
    y_raster = raster_driver.Create(
        y_output_path, n_cols, n_rows, 1, gdal.GDT_Float32,
        options=pygeoprocessing.geoprocessing_core.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])
    x_band, y_band = x_raster.GetRasterBand(1), y_raster.GetRasterBand(1)

    # can't use raster_calculator here because we need the block offset info
    # calculate coords for each block and write them to the output rasters
    for data, array in pygeoprocessing.iterblocks((raster_path, 1)):
        y_coords, x_coords = numpy.indices(array.shape)
        x_coords = (
            (x_coords * pixel_size_x) +  # convert to pixel size in meters
            (pixel_size_x / 2) +  # center the point on the pixel
            (data['xoff'] * pixel_size_x) +   # the offset of this block relative to the raster
            x_origin)  # the raster's offset relative to the coordinate system
        y_coords = (
            (y_coords * pixel_size_y) + 
            (pixel_size_y / 2) +
            (data['yoff'] * pixel_size_y) +
            y_origin)

        x_band.WriteArray(x_coords, xoff=data['xoff'], yoff=data['yoff'])
        y_band.WriteArray(y_coords, xoff=data['xoff'], yoff=data['yoff'])
    x_band, y_band, x_raster, y_raster = None, None, None, None

    
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
