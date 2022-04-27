"""InVEST Sediment Delivery Ratio (SDR) module.

The SDR method in this model is based on:
    Winchell, M. F., et al. "Extension and validation of a geographic
    information system-based method for calculating the Revised Universal
    Soil Loss Equation length-slope factor for erosion risk assessments in
    large watersheds." Journal of Soil and Water Conservation 63.3 (2008):
    105-111.
"""
import logging
import os

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph
from osgeo import gdal
from osgeo import ogr

from ..model_metadata import MODEL_METADATA
from .. import gettext
from .. import spec_utils
from .. import utils
from .. import validation
from ..spec_utils import u
from . import sdr_core


LOGGER = logging.getLogger(__name__)

INVALID_ID_MSG = gettext('{number} features have a non-integer ws_id field')

ARGS_SPEC = {
    "model_name": MODEL_METADATA["sdr"].model_title,
    "pyname": MODEL_METADATA["sdr"].pyname,
    "userguide": MODEL_METADATA["sdr"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["dem_path", "erosivity_path", "erodibility_path",
                         "lulc_path", "drainage_path", "watersheds_path", ],
        "different_projections_ok": False,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "dem_path": {
            **spec_utils.DEM,
            "projected": True
        },
        "erosivity_path": {
            "type": "raster",
            "bands": {1: {
                "type": "number",
                "units": u.megajoule*u.millimeter/(u.hectare*u.hour*u.year)}},
            "projected": True,
            "about": gettext(
                "Map of rainfall erosivity, reflecting the intensity and "
                "duration of rainfall in the area of interest."),
            "name": gettext("erosivity")
        },
        "erodibility_path": {
            "type": "raster",
            "bands": {1: {
                "type": "number",
                "units": u.metric_ton*u.hectare*u.hour/(u.hectare*u.megajoule*u.millimeter)}},
            "projected": True,
            "about": gettext(
                "Map of soil erodibility, the susceptibility of soil "
                "particles to detachment and transport by rainfall and "
                "runoff."),
            "name": gettext("soil erodibility")
        },
        "lulc_path": {
            **spec_utils.LULC,
            "projected": True,
            "about": gettext(
                f"{spec_utils.LULC['about']} All values in this raster must "
                "have corresponding entries in the Biophysical Table.")
        },
        "watersheds_path": {
            "type": "vector",
            "fields": {
                "ws_id": {
                    "type": "integer",
                    "about": gettext("Unique identifier for the watershed.")}
            },
            "geometries": spec_utils.POLYGONS,
            "projected": True,
            "about": gettext(
                "Map of the boundaries of the watershed(s) over which to "
                "aggregate results. Each watershed should contribute to a "
                "point of interest where water quality will be analyzed."),
            "name": gettext("Watersheds")
        },
        "biophysical_table_path": {
            "type": "csv",
            "columns": {
                "lucode": {
                    "type": "integer",
                    "about": gettext("LULC code from the LULC raster.")},
                "usle_c": {
                    "type": "ratio",
                    "about": gettext("Cover-management factor for the USLE")},
                "usle_p": {
                    "type": "ratio",
                    "about": gettext("Support practice factor for the USLE")}
            },
            "about": gettext(
                "A table mapping each LULC code to biophysical properties of "
                "that LULC class. All values in the LULC raster must have "
                "corresponding entries in this table."),
            "name": gettext("biophysical table")
        },
        "threshold_flow_accumulation": spec_utils.THRESHOLD_FLOW_ACCUMULATION,
        "k_param": {
            "type": "number",
            "units": u.none,
            "about": gettext("Borselli k parameter."),
            "name": gettext("Borselli k parameter")
        },
        "sdr_max": {
            "type": "ratio",
            "about": gettext("The maximum SDR value that a pixel can have."),
            "name": gettext("maximum SDR value")
        },
        "ic_0_param": {
            "type": "number",
            "units": u.none,
            "about": gettext("Borselli IC0 parameter."),
            "name": gettext("Borselli IC0 parameter")
        },
        "l_max": {
            "type": "number",
            "expression": "value > 0",
            "units": u.none,
            "about": gettext(
                "The maximum allowed value of the slope length parameter (L) "
                "in the LS factor."),
            "name": gettext("maximum l value"),
        },
        "drainage_path": {
            "type": "raster",
            "bands": {1: {"type": "number", "units": u.none}},
            "required": False,
            "about": gettext(
                "Map of locations of artificial drainages that drain to the "
                "watershed. Pixels with 1 are drainages and are treated like "
                "streams. Pixels with 0 are not drainages."),
            "name": gettext("drainages")
        }
    }
}

_OUTPUT_BASE_FILES = {
    'rkls_path': 'rkls.tif',
    'sed_export_path': 'sed_export.tif',
    'sed_retention_index_path': 'sed_retention_index.tif',
    'sed_retention_path': 'sed_retention.tif',
    'sed_deposition_path': 'sed_deposition.tif',
    'stream_and_drainage_path': 'stream_and_drainage.tif',
    'stream_path': 'stream.tif',
    'usle_path': 'usle.tif',
    'watershed_results_sdr_path': 'watershed_results_sdr.shp',
}

INTERMEDIATE_DIR_NAME = 'intermediate_outputs'

_INTERMEDIATE_BASE_FILES = {
    'cp_factor_path': 'cp.tif',
    'd_dn_bare_soil_path': 'd_dn_bare_soil.tif',
    'd_dn_path': 'd_dn.tif',
    'd_up_bare_soil_path': 'd_up_bare_soil.tif',
    'd_up_path': 'd_up.tif',
    'f_path': 'f.tif',
    'flow_accumulation_path': 'flow_accumulation.tif',
    'flow_direction_path': 'flow_direction.tif',
    'ic_bare_soil_path': 'ic_bare_soil.tif',
    'ic_path': 'ic.tif',
    'ls_path': 'ls.tif',
    'pit_filled_dem_path': 'pit_filled_dem.tif',
    's_accumulation_path': 's_accumulation.tif',
    's_bar_path': 's_bar.tif',
    's_inverse_path': 's_inverse.tif',
    'sdr_bare_soil_path': 'sdr_bare_soil.tif',
    'sdr_path': 'sdr_factor.tif',
    'slope_path': 'slope.tif',
    'thresholded_slope_path': 'slope_threshold.tif',
    'thresholded_w_path': 'w_threshold.tif',
    'w_accumulation_path': 'w_accumulation.tif',
    'w_bar_path': 'w_bar.tif',
    'w_path': 'w.tif',
    'ws_inverse_path': 'ws_inverse.tif',
    'e_prime_path': 'e_prime.tif',
    'weighted_avg_aspect_path': 'weighted_avg_aspect.tif',
    'drainage_mask': 'what_drains_to_stream.tif',
}

_TMP_BASE_FILES = {
    'aligned_dem_path': 'aligned_dem.tif',
    'aligned_drainage_path': 'aligned_drainage.tif',
    'aligned_erodibility_path': 'aligned_erodibility.tif',
    'aligned_erosivity_path': 'aligned_erosivity.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
}

# Target nodata is for general rasters that are positive, and _IC_NODATA are
# for rasters that are any range
_TARGET_NODATA = -1.0
_BYTE_NODATA = 255
_IC_NODATA = float(numpy.finfo('float32').min)


def execute(args):
    """Sediment Delivery Ratio.

    This function calculates the sediment export and retention of a landscape
    using the sediment delivery ratio model described in the InVEST user's
    guide.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['erosivity_path'] (string): path to rainfall erosivity index
            raster
        args['erodibility_path'] (string): a path to soil erodibility raster
        args['lulc_path'] (string): path to land use/land cover raster
        args['watersheds_path'] (string): path to vector of the watersheds
        args['biophysical_table_path'] (string): path to CSV file with
            biophysical information of each land use classes.  contain the
            fields 'usle_c' and 'usle_p'
        args['threshold_flow_accumulation'] (number): number of upslope pixels
            on the dem to threshold to a stream.
        args['k_param'] (number): k calibration parameter
        args['sdr_max'] (number): max value the SDR
        args['ic_0_param'] (number): ic_0 calibration parameter
        args['drainage_path'] (string): (optional) path to drainage raster that
            is used to add additional drainage areas to the internally
            calculated stream layer
        args['n_workers'] (int): if present, indicates how many worker
            processes should be used in parallel processing. -1 indicates
            single process mode, 0 is single process but non-blocking mode,
            and >= 1 is number of processes.

    Returns:
        None.

    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    biophysical_table = utils.build_lookup_from_csv(
        args['biophysical_table_path'], 'lucode')

    # Test to see if c or p values are outside of 0..1
    for table_key in ['usle_c', 'usle_p']:
        for (lulc_code, table) in biophysical_table.items():
            try:
                float(lulc_code)
            except ValueError:
                raise ValueError(
                    f'Value "{lulc_code}" from the "lucode" column of the '
                    f'biophysical table is not a number. Please check the '
                    f'formatting of {args["biophysical_table_path"]}')
            try:
                float_value = float(table[table_key])
                if float_value < 0 or float_value > 1:
                    raise ValueError(
                        f'{float_value} is not within range 0..1')
            except ValueError:
                raise ValueError(
                    f'A value in the biophysical table is not a number '
                    f'within range 0..1. The offending value is in '
                    f'column "{table_key}", lucode row "{lulc_code}", '
                    f'and has value "{table[table_key]}"')

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], INTERMEDIATE_DIR_NAME)
    output_dir = os.path.join(args['workspace_dir'])
    churn_dir = os.path.join(
        intermediate_output_dir, 'churn_dir_not_for_humans')
    utils.make_directories([output_dir, intermediate_output_dir, churn_dir])

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, churn_dir)], file_suffix)

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(
        churn_dir, n_workers, reporting_interval=5.0)

    base_list = []
    aligned_list = []
    for file_key in ['dem', 'lulc', 'erosivity', 'erodibility']:
        base_list.append(args[file_key + "_path"])
        aligned_list.append(f_reg["aligned_" + file_key + "_path"])
    # all continuous rasters can use bilinaer, but lulc should be mode
    interpolation_list = ['bilinear', 'mode', 'bilinear', 'bilinear']

    drainage_present = False
    if 'drainage_path' in args and args['drainage_path'] != '':
        drainage_present = True
        base_list.append(args['drainage_path'])
        aligned_list.append(f_reg['aligned_drainage_path'])
        interpolation_list.append('near')

    dem_raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
    min_pixel_size = numpy.min(numpy.abs(dem_raster_info['pixel_size']))
    target_pixel_size = (min_pixel_size, -min_pixel_size)

    target_sr_wkt = dem_raster_info['projection_wkt']
    vector_mask_options = {'mask_vector_path': args['watersheds_path']}
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            base_list, aligned_list, interpolation_list,
            target_pixel_size, 'intersection'),
        kwargs={
            'target_projection_wkt': target_sr_wkt,
            'base_vector_path_list': (args['watersheds_path'],),
            'raster_align_index': 0,
            'vector_mask_options': vector_mask_options,
        },
        target_path_list=aligned_list,
        task_name='align input rasters')

    pit_fill_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (f_reg['aligned_dem_path'], 1),
            f_reg['pit_filled_dem_path']),
        target_path_list=[f_reg['pit_filled_dem_path']],
        dependent_task_list=[align_task],
        task_name='fill pits')

    slope_task = task_graph.add_task(
        func=pygeoprocessing.calculate_slope,
        args=(
            (f_reg['pit_filled_dem_path'], 1),
            f_reg['slope_path']),
        dependent_task_list=[pit_fill_task],
        target_path_list=[f_reg['slope_path']],
        task_name='calculate slope')

    threshold_slope_task = task_graph.add_task(
        func=_threshold_slope,
        args=(f_reg['slope_path'], f_reg['thresholded_slope_path']),
        target_path_list=[f_reg['thresholded_slope_path']],
        dependent_task_list=[slope_task],
        task_name='threshold slope')

    flow_dir_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_mfd,
        args=(
            (f_reg['pit_filled_dem_path'], 1),
            f_reg['flow_direction_path']),
        target_path_list=[f_reg['flow_direction_path']],
        dependent_task_list=[pit_fill_task],
        task_name='flow direction calculation')

    weighted_avg_aspect_task = task_graph.add_task(
        func=sdr_core.calculate_average_aspect,
        args=(f_reg['flow_direction_path'],
              f_reg['weighted_avg_aspect_path']),
        target_path_list=[f_reg['weighted_avg_aspect_path']],
        dependent_task_list=[flow_dir_task],
        task_name='weighted average of multiple-flow aspects')

    flow_accumulation_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            f_reg['flow_accumulation_path']),
        target_path_list=[f_reg['flow_accumulation_path']],
        dependent_task_list=[flow_dir_task],
        task_name='flow accumulation calculation')

    ls_factor_task = task_graph.add_task(
        func=_calculate_ls_factor,
        args=(
            f_reg['flow_accumulation_path'],
            f_reg['slope_path'],
            f_reg['weighted_avg_aspect_path'],
            float(args['l_max']),
            f_reg['ls_path']),
        target_path_list=[f_reg['ls_path']],
        dependent_task_list=[
            flow_accumulation_task, slope_task,
            weighted_avg_aspect_task],
        task_name='ls factor calculation')

    stream_task = task_graph.add_task(
        func=pygeoprocessing.routing.extract_streams_mfd,
        args=(
            (f_reg['flow_accumulation_path'], 1),
            (f_reg['flow_direction_path'], 1),
            float(args['threshold_flow_accumulation']),
            f_reg['stream_path']),
        kwargs={'trace_threshold_proportion': 0.7},
        target_path_list=[f_reg['stream_path']],
        dependent_task_list=[flow_accumulation_task],
        task_name='extract streams')

    if drainage_present:
        drainage_task = task_graph.add_task(
            func=_add_drainage(
                f_reg['stream_path'],
                f_reg['aligned_drainage_path'],
                f_reg['stream_and_drainage_path']),
            target_path_list=[f_reg['stream_and_drainage_path']],
            dependent_task_list=[stream_task, align_task],
            task_name='add drainage')
        drainage_raster_path_task = (
            f_reg['stream_and_drainage_path'], drainage_task)
    else:
        drainage_raster_path_task = (
            f_reg['stream_path'], stream_task)

    threshold_w_task = task_graph.add_task(
        func=_calculate_w,
        args=(
            biophysical_table, f_reg['aligned_lulc_path'], f_reg['w_path'],
            f_reg['thresholded_w_path']),
        target_path_list=[f_reg['w_path'], f_reg['thresholded_w_path']],
        dependent_task_list=[align_task],
        task_name='calculate W')

    cp_task = task_graph.add_task(
        func=_calculate_cp,
        args=(
            biophysical_table, f_reg['aligned_lulc_path'],
            f_reg['cp_factor_path']),
        target_path_list=[f_reg['cp_factor_path']],
        dependent_task_list=[align_task],
        task_name='calculate CP')

    rkls_task = task_graph.add_task(
        func=_calculate_rkls,
        args=(
            f_reg['ls_path'],
            f_reg['aligned_erosivity_path'],
            f_reg['aligned_erodibility_path'],
            drainage_raster_path_task[0],
            f_reg['rkls_path']),
        target_path_list=[f_reg['rkls_path']],
        dependent_task_list=[
            align_task, drainage_raster_path_task[1], ls_factor_task],
        task_name='calculate RKLS')

    usle_task = task_graph.add_task(
        func=_calculate_usle,
        args=(
            f_reg['rkls_path'],
            f_reg['cp_factor_path'],
            drainage_raster_path_task[0],
            f_reg['usle_path']),
        target_path_list=[f_reg['usle_path']],
        dependent_task_list=[
            rkls_task, cp_task, drainage_raster_path_task[1]],
        task_name='calculate USLE')

    bar_task_map = {}
    for factor_path, factor_task, accumulation_path, out_bar_path, bar_id in [
            (f_reg['thresholded_w_path'], threshold_w_task,
             f_reg['w_accumulation_path'],
             f_reg['w_bar_path'],
             'w_bar'),
            (f_reg['thresholded_slope_path'], threshold_slope_task,
             f_reg['s_accumulation_path'],
             f_reg['s_bar_path'],
             's_bar')]:
        bar_task = task_graph.add_task(
            func=_calculate_bar_factor,
            args=(
                f_reg['flow_direction_path'], factor_path,
                f_reg['flow_accumulation_path'],
                accumulation_path, out_bar_path),
            target_path_list=[accumulation_path, out_bar_path],
            dependent_task_list=[
                align_task, factor_task, flow_accumulation_task,
                flow_dir_task],
            task_name='calculate %s' % bar_id)
        bar_task_map[bar_id] = bar_task

    d_up_task = task_graph.add_task(
        func=_calculate_d_up,
        args=(
            f_reg['w_bar_path'], f_reg['s_bar_path'],
            f_reg['flow_accumulation_path'], f_reg['d_up_path']),
        target_path_list=[f_reg['d_up_path']],
        dependent_task_list=[
            bar_task_map['s_bar'], bar_task_map['w_bar'],
            flow_accumulation_task],
        task_name='calculate Dup')

    inverse_ws_factor_task = task_graph.add_task(
        func=_calculate_inverse_ws_factor,
        args=(
            f_reg['thresholded_slope_path'], f_reg['thresholded_w_path'],
            f_reg['ws_inverse_path']),
        target_path_list=[f_reg['ws_inverse_path']],
        dependent_task_list=[threshold_slope_task, threshold_w_task],
        task_name='calculate inverse ws factor')

    d_dn_task = task_graph.add_task(
        func=pygeoprocessing.routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            (drainage_raster_path_task[0], 1),
            f_reg['d_dn_path']),
        kwargs={'weight_raster_path_band': (f_reg['ws_inverse_path'], 1)},
        target_path_list=[f_reg['d_dn_path']],
        dependent_task_list=[
            flow_dir_task, drainage_raster_path_task[1],
            inverse_ws_factor_task],
        task_name='calculating d_dn')

    ic_task = task_graph.add_task(
        func=_calculate_ic,
        args=(
            f_reg['d_up_path'], f_reg['d_dn_path'], f_reg['ic_path']),
        target_path_list=[f_reg['ic_path']],
        dependent_task_list=[d_up_task, d_dn_task],
        task_name='calculate ic')

    sdr_task = task_graph.add_task(
        func=_calculate_sdr,
        args=(
            float(args['k_param']), float(args['ic_0_param']),
            float(args['sdr_max']), f_reg['ic_path'],
            drainage_raster_path_task[0], f_reg['sdr_path']),
        target_path_list=[f_reg['sdr_path']],
        dependent_task_list=[ic_task],
        task_name='calculate sdr')

    sed_export_task = task_graph.add_task(
        func=_calculate_sed_export,
        args=(
            f_reg['usle_path'], f_reg['sdr_path'], f_reg['sed_export_path']),
        target_path_list=[f_reg['sed_export_path']],
        dependent_task_list=[usle_task, sdr_task],
        task_name='calculate sed export')

    e_prime_task = task_graph.add_task(
        func=_calculate_e_prime,
        args=(
            f_reg['usle_path'], f_reg['sdr_path'], f_reg['e_prime_path']),
        target_path_list=[f_reg['e_prime_path']],
        dependent_task_list=[usle_task, sdr_task],
        task_name='calculate export prime')

    _ = task_graph.add_task(
        func=sdr_core.calculate_sediment_deposition,
        args=(
            f_reg['flow_direction_path'], f_reg['e_prime_path'],
            f_reg['f_path'], f_reg['sdr_path'],
            f_reg['sed_deposition_path']),
        dependent_task_list=[e_prime_task, sdr_task, flow_dir_task],
        target_path_list=[f_reg['sed_deposition_path'], f_reg['f_path']],
        task_name='sediment deposition')

    _ = task_graph.add_task(
        func=_calculate_sed_retention_index,
        args=(
            f_reg['rkls_path'], f_reg['usle_path'], f_reg['sdr_path'],
            float(args['sdr_max']), f_reg['sed_retention_index_path']),
        target_path_list=[f_reg['sed_retention_index_path']],
        dependent_task_list=[rkls_task, usle_task, sdr_task],
        task_name='calculate sediment retention index')

    # This next section is for calculating the bare soil part.
    s_inverse_task = task_graph.add_task(
        func=_calculate_inverse_s_factor,
        args=(f_reg['thresholded_slope_path'], f_reg['s_inverse_path']),
        target_path_list=[f_reg['s_inverse_path']],
        dependent_task_list=[threshold_slope_task],
        task_name='calculate S factor')

    d_dn_bare_task = task_graph.add_task(
        func=pygeoprocessing.routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            (drainage_raster_path_task[0], 1),
            f_reg['d_dn_bare_soil_path']),
        kwargs={'weight_raster_path_band': (f_reg['s_inverse_path'], 1)},
        target_path_list=[f_reg['d_dn_bare_soil_path']],
        dependent_task_list=[
            flow_dir_task, drainage_raster_path_task[1], s_inverse_task],
        task_name='calculating d_dn soil')

    d_up_bare_task = task_graph.add_task(
        func=_calculate_d_up_bare,
        args=(
            f_reg['s_bar_path'], f_reg['flow_accumulation_path'],
            f_reg['d_up_bare_soil_path']),
        target_path_list=[f_reg['d_up_bare_soil_path']],
        dependent_task_list=[bar_task_map['s_bar'], flow_accumulation_task],
        task_name='calculating d_up bare soil')

    ic_bare_task = task_graph.add_task(
        func=_calculate_ic,
        args=(
            f_reg['d_up_bare_soil_path'], f_reg['d_dn_bare_soil_path'],
            f_reg['ic_bare_soil_path']),
        target_path_list=[f_reg['ic_bare_soil_path']],
        dependent_task_list=[d_up_bare_task, d_dn_bare_task],
        task_name='calculate bare soil ic')

    sdr_bare_task = task_graph.add_task(
        func=_calculate_sdr,
        args=(
            float(args['k_param']), float(args['ic_0_param']),
            float(args['sdr_max']), f_reg['ic_bare_soil_path'],
            drainage_raster_path_task[0], f_reg['sdr_bare_soil_path']),
        target_path_list=[f_reg['sdr_bare_soil_path']],
        dependent_task_list=[ic_bare_task, drainage_raster_path_task[1]],
        task_name='calculate bare SDR')

    sed_retention_task = task_graph.add_task(
        func=_calculate_sed_retention,
        args=(
            f_reg['rkls_path'], f_reg['usle_path'],
            drainage_raster_path_task[0], f_reg['sdr_path'],
            f_reg['sdr_bare_soil_path'], f_reg['sed_retention_path']),
        target_path_list=[f_reg['sed_retention_path']],
        dependent_task_list=[
            rkls_task, usle_task, drainage_raster_path_task[1], sdr_task,
            sdr_bare_task],
        task_name='calculate sediment retention')

    _ = task_graph.add_task(
        func=_calculate_what_drains_to_stream,
        args=(f_reg['flow_direction_path'], f_reg['d_dn_path'],
              f_reg['drainage_mask']),
        target_path_list=[f_reg['drainage_mask']],
        dependent_task_list=[flow_dir_task, d_dn_task],
        task_name='write mask of what drains to stream')

    _ = task_graph.add_task(
        func=_generate_report,
        args=(
            args['watersheds_path'], f_reg['usle_path'],
            f_reg['sed_export_path'], f_reg['sed_retention_path'],
            f_reg['sed_deposition_path'], f_reg['watershed_results_sdr_path']),
        target_path_list=[f_reg['watershed_results_sdr_path']],
        dependent_task_list=[
            usle_task, sed_export_task, sed_retention_task],
        task_name='generate report')

    task_graph.close()
    task_graph.join()


def _calculate_what_drains_to_stream(
        flow_dir_mfd_path, dist_to_channel_mfd_path, target_mask_path):
    """Create a mask indicating regions that do or do not drain to a stream.

    This is useful because ``pygeoprocessing.distance_to_stream_mfd`` may leave
    some unexpected regions as nodata if they do not drain to a stream.  This
    may be confusing behavior, so this mask is intended to locate what drains
    to a stream and what does not. A pixel doesn't drain to a stream if it has
    a defined flow direction but undefined distance to stream.

    Args:
        flow_dir_mfd_path (string): The path to an MFD flow direction raster.
            This raster must have a nodata value defined.
        dist_to_channel_mfd_path (string): The path to an MFD
            distance-to-channel raster.  This raster must have a nodata value
            defined.
        target_mask_path (string): The path to where the mask raster should be
            written.

    Returns:
        ``None``
    """
    flow_dir_mfd_nodata = pygeoprocessing.get_raster_info(
        flow_dir_mfd_path)['nodata'][0]
    dist_to_channel_nodata = pygeoprocessing.get_raster_info(
        dist_to_channel_mfd_path)['nodata'][0]

    def _what_drains_to_stream(flow_dir_mfd, dist_to_channel):
        """Determine which pixels do and do not drain to a stream.

        Args:
            flow_dir_mfd (numpy.array): A numpy array of MFD flow direction
                values.
            dist_to_channel (numpy.array): A numpy array of calculated
                distances to the nearest channel.

        Returns:
            A ``numpy.array`` of dtype ``numpy.uint8`` with pixels where:

                * ``255`` where ``flow_dir_mfd`` is nodata (and thus
                  ``dist_to_channel`` is also nodata).
                * ``0`` where ``flow_dir_mfd`` has data and ``dist_to_channel``
                  does not
                * ``1`` where ``flow_dir_mfd`` has data, and
                  ``dist_to_channel`` also has data.
        """
        drains_to_stream = numpy.full(
            flow_dir_mfd.shape, _BYTE_NODATA, dtype=numpy.uint8)
        valid_flow_dir = ~utils.array_equals_nodata(
            flow_dir_mfd, flow_dir_mfd_nodata)
        valid_dist_to_channel = (
            ~utils.array_equals_nodata(
                dist_to_channel, dist_to_channel_nodata) &
            valid_flow_dir)

        # Nodata where both flow_dir and dist_to_channel are nodata
        # 1 where flow_dir and dist_to_channel have values (drains to stream)
        # 0 where flow_dir has data and dist_to_channel doesn't (doesn't drain)
        drains_to_stream[valid_flow_dir & valid_dist_to_channel] = 1
        drains_to_stream[valid_flow_dir & ~valid_dist_to_channel] = 0
        return drains_to_stream

    pygeoprocessing.raster_calculator(
        [(flow_dir_mfd_path, 1), (dist_to_channel_mfd_path, 1)],
        _what_drains_to_stream, target_mask_path, gdal.GDT_Byte, _BYTE_NODATA)


def _calculate_ls_factor(
        flow_accumulation_path, slope_path, avg_aspect_path, l_max,
        target_ls_prime_factor_path):
    """Calculate LS factor.

    Calculates a modified LS factor as Equation 3 from "Extension and
    validation of a geographic information system-based method for calculating
    the Revised Universal Soil Loss Equation length-slope factor for erosion
    risk assessments in large watersheds" where the ``x`` term is the average
    aspect ratio weighted by proportional flow to account for multiple flow
    direction.

    Args:
        flow_accumulation_path (string): path to raster, pixel values are the
            contributing upslope area at that cell. Pixel size is square.
        slope_path (string): path to slope raster as a percent
        avg_aspect_path (string): The path to to raster of the weighted average
            of aspects based on proportional flow.
        l_max (float): if the calculated value of L exceeds this value
            it is clamped to this value.
        target_ls_prime_factor_path (string): path to output ls_prime_factor
            raster

    Returns:
        None

    """
    slope_nodata = pygeoprocessing.get_raster_info(slope_path)['nodata'][0]
    avg_aspect_nodata = pygeoprocessing.get_raster_info(
        avg_aspect_path)['nodata'][0]

    flow_accumulation_info = pygeoprocessing.get_raster_info(
        flow_accumulation_path)
    flow_accumulation_nodata = flow_accumulation_info['nodata'][0]
    cell_size = abs(flow_accumulation_info['pixel_size'][0])
    cell_area = cell_size ** 2

    def ls_factor_function(
            percent_slope, flow_accumulation, avg_aspect, l_max):
        """Calculate the LS' factor.

        Args:
            percent_slope (numpy.ndarray): slope in percent
            flow_accumulation (numpy.ndarray): upslope pixels
            avg_aspect (numpy.ndarray): the weighted average aspect from MFD
            l_max (float): max L factor, clamp to this value if L exceeds it

        Returns:
            ls_factor

        """
        # avg aspect intermediate output should always have a defined
        # nodata value from pygeoprocessing
        valid_mask = (
            (~utils.array_equals_nodata(avg_aspect, avg_aspect_nodata)) &
            ~utils.array_equals_nodata(percent_slope, slope_nodata) &
            ~utils.array_equals_nodata(
                flow_accumulation, flow_accumulation_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA

        contributing_area = (flow_accumulation[valid_mask]-1) * cell_area
        slope_in_radians = numpy.arctan(percent_slope[valid_mask] / 100.0)

        # From Equation 4 in "Extension and validation of a geographic
        # information system ..."
        slope_factor = numpy.where(
            percent_slope[valid_mask] < 9.0,
            10.8 * numpy.sin(slope_in_radians) + 0.03,
            16.8 * numpy.sin(slope_in_radians) - 0.5)

        beta = (
            (numpy.sin(slope_in_radians) / 0.0896) /
            (3 * numpy.sin(slope_in_radians)**0.8 + 0.56))

        # Set m value via lookup table: Table 1 in
        # InVEST Sediment Model_modifications_10-01-2012_RS.docx
        # note slope_table in percent
        slope_table = numpy.array([1., 3.5, 5., 9.])
        m_table = numpy.array([0.2, 0.3, 0.4, 0.5])
        # mask where slopes are larger than lookup table
        big_slope_mask = percent_slope[valid_mask] > slope_table[-1]
        m_indexes = numpy.digitize(
            percent_slope[valid_mask][~big_slope_mask], slope_table,
            right=True)
        m_exp = numpy.empty(big_slope_mask.shape, dtype=numpy.float32)
        m_exp[big_slope_mask] = (
            beta[big_slope_mask] / (1 + beta[big_slope_mask]))
        m_exp[~big_slope_mask] = m_table[m_indexes]

        l_factor = (
            ((contributing_area + cell_area)**(m_exp+1) -
             contributing_area ** (m_exp+1)) /
            ((cell_size ** (m_exp + 2)) * (avg_aspect[valid_mask]**m_exp) *
             (22.13**m_exp)))

        # threshold L factor to l_max
        l_factor[l_factor > l_max] = l_max

        result[valid_mask] = l_factor * slope_factor
        return result

    # call vectorize datasets to calculate the ls_factor
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            slope_path, flow_accumulation_path, avg_aspect_path]] + [
            (l_max, 'raw')],
        ls_factor_function, target_ls_prime_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA)


def _calculate_rkls(
        ls_factor_path, erosivity_path, erodibility_path, stream_path,
        rkls_path):
    """Calculate potential soil loss (tons / (pixel * year)) using RKLS.

    (revised universal soil loss equation with no C or P).

    Args:
        ls_factor_path (string): path to LS raster that has square pixels in
            meter units.
        erosivity_path (string): path to erosivity raster
            (MJ * mm / (ha * hr * yr))
        erodibility_path (string): path to erodibility raster
            (t * ha * hr / (MJ * ha * mm))
        stream_path (string): path to drainage raster
            (1 is drainage, 0 is not)
        rkls_path (string): path to RKLS raster

    Returns:
        None

    """
    erosivity_nodata = pygeoprocessing.get_raster_info(
        erosivity_path)['nodata'][0]
    erodibility_nodata = pygeoprocessing.get_raster_info(
        erodibility_path)['nodata'][0]
    stream_nodata = pygeoprocessing.get_raster_info(
        stream_path)['nodata'][0]

    cell_size = abs(
        pygeoprocessing.get_raster_info(ls_factor_path)['pixel_size'][0])
    cell_area_ha = cell_size**2 / 10000.0  # hectares per pixel

    def rkls_function(ls_factor, erosivity, erodibility, stream):
        """Calculate the RKLS equation.

        Args:
            ls_factor (numpy.ndarray): length/slope factor. unitless.
            erosivity (numpy.ndarray): related to peak rainfall events. units:
                MJ * mm / (ha * hr * yr)
            erodibility (numpy.ndarray): related to the potential for soil to
                erode. units: t * ha * hr / (MJ * ha * mm)
            stream (numpy.ndarray): stream mask (1 stream, 0 no stream)

        Returns:
            numpy.ndarray of RKLS values in tons / (pixel * year))
        """
        rkls = numpy.empty(ls_factor.shape, dtype=numpy.float32)
        nodata_mask = (
            ~utils.array_equals_nodata(ls_factor, _TARGET_NODATA) &
            ~utils.array_equals_nodata(stream, stream_nodata))
        if erosivity_nodata is not None:
            nodata_mask &= ~utils.array_equals_nodata(
                erosivity, erosivity_nodata)
        if erodibility_nodata is not None:
            nodata_mask &= ~utils.array_equals_nodata(
                erodibility, erodibility_nodata)

        valid_mask = nodata_mask & (stream == 0)
        rkls[:] = _TARGET_NODATA

        rkls[valid_mask] = (           # rkls units are tons / (pixel * year)
            ls_factor[valid_mask] *    # unitless
            erosivity[valid_mask] *    # MJ * mm / (ha * hr * yr)
            erodibility[valid_mask] *  # t * ha * hr / (MJ * ha * mm)
            cell_area_ha)              # ha / pixel

        # rkls is 1 on the stream
        rkls[nodata_mask & (stream == 1)] = 1
        return rkls

    # aligning with index 3 that's the stream and the most likely to be
    # aligned with LULCs
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            ls_factor_path, erosivity_path, erodibility_path, stream_path]],
        rkls_function, rkls_path, gdal.GDT_Float32, _TARGET_NODATA)


def _threshold_slope(slope_path, out_thresholded_slope_path):
    """Threshold the slope between 0.005 and 1.0.

    Args:
        slope_path (string): path to a raster of slope in percent
        out_thresholded_slope_path (string): path to output raster of
            thresholded slope between 0.005 and 1.0

    Returns:
        None

    """
    slope_nodata = pygeoprocessing.get_raster_info(slope_path)['nodata'][0]

    def threshold_slope(slope):
        """Convert slope to m/m and clamp at 0.005 and 1.0.

        As desribed in Cavalli et al., 2013.
        """
        valid_slope = ~utils.array_equals_nodata(slope, slope_nodata)
        slope_m = slope[valid_slope] / 100.0
        slope_m[slope_m < 0.005] = 0.005
        slope_m[slope_m > 1.0] = 1.0
        result = numpy.empty(valid_slope.shape, dtype=numpy.float32)
        result[:] = slope_nodata
        result[valid_slope] = slope_m
        return result

    pygeoprocessing.raster_calculator(
        [(slope_path, 1)], threshold_slope, out_thresholded_slope_path,
        gdal.GDT_Float32, slope_nodata)


def _add_drainage(stream_path, drainage_path, out_stream_and_drainage_path):
    """Combine stream and drainage masks into one raster mask.

    Args:
        stream_path (string): path to stream raster mask where 1 indicates
            a stream, and 0 is a valid landscape pixel but not a stream.
        drainage_raster_path (string): path to 1/0 mask of drainage areas.
            1 indicates any water reaching that pixel drains to a stream.
        out_stream_and_drainage_path (string): output raster of a logical
            OR of stream and drainage inputs

    Returns:
        None

    """
    def add_drainage_op(stream, drainage):
        """Add drainage mask to stream layer."""
        return numpy.where(drainage == 1, 1, stream)

    stream_nodata = pygeoprocessing.get_raster_info(stream_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [stream_path, drainage_path]], add_drainage_op,
        out_stream_and_drainage_path, gdal.GDT_Byte, stream_nodata)


def _calculate_w(
        biophysical_table, lulc_path, w_factor_path,
        out_thresholded_w_factor_path):
    """W factor: map C values from LULC and lower threshold to 0.001.

    W is a factor in calculating d_up accumulation for SDR.

    Args:
        biophysical_table (dict): map of LULC codes to dictionaries that
            contain at least a 'usle_c' field
        lulc_path (string): path to LULC raster
        w_factor_path (string): path to outputed raw W factor
        out_thresholded_w_factor_path (string): W factor from `w_factor_path`
            thresholded to be no less than 0.001.

    Returns:
        None

    """
    lulc_to_c = dict(
        [(lulc_code, float(table['usle_c'])) for
         (lulc_code, table) in biophysical_table.items()])
    if pygeoprocessing.get_raster_info(lulc_path)['nodata'][0] is None:
        # will get a case where the raster might be masked but nothing to
        # replace so 0 is used by default. Ensure this exists in lookup.
        if 0 not in lulc_to_c:
            lulc_to_c = lulc_to_c.copy()
            lulc_to_c[0] = 0.0

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}

    utils.reclassify_raster(
        (lulc_path, 1), lulc_to_c, w_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA, reclass_error_details)

    def threshold_w(w_val):
        """Threshold w to 0.001."""
        w_val_copy = w_val.copy()
        nodata_mask = utils.array_equals_nodata(w_val, _TARGET_NODATA)
        w_val_copy[w_val < 0.001] = 0.001
        w_val_copy[nodata_mask] = _TARGET_NODATA
        return w_val_copy

    pygeoprocessing.raster_calculator(
        [(w_factor_path, 1)], threshold_w, out_thresholded_w_factor_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_cp(biophysical_table, lulc_path, cp_factor_path):
    """Map LULC to C*P value.

    Args:
        biophysical_table (dict): map of lulc codes to dictionaries that
            contain at least the entry 'usle_c" and 'usle_p' corresponding to
            those USLE components.
        lulc_path (string): path to LULC raster
        cp_factor_path (string): path to output raster of LULC mapped to C*P
            values

    Returns:
        None

    """
    lulc_to_cp = dict(
        [(lulc_code, float(table['usle_c']) * float(table['usle_p'])) for
         (lulc_code, table) in biophysical_table.items()])
    if pygeoprocessing.get_raster_info(lulc_path)['nodata'][0] is None:
        # will get a case where the raster might be masked but nothing to
        # replace so 0 is used by default. Ensure this exists in lookup.
        if 0 not in lulc_to_cp:
            lulc_to_cp[0] = 0.0

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}

    utils.reclassify_raster(
        (lulc_path, 1), lulc_to_cp, cp_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA, reclass_error_details)


def _calculate_usle(
        rkls_path, cp_factor_path, drainage_raster_path, out_usle_path):
    """Calculate USLE, multiply RKLS by CP and set to 1 on drains."""
    def usle_op(rkls, cp_factor, drainage):
        """Calculate USLE."""
        result = numpy.empty(rkls.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        valid_mask = (
            ~utils.array_equals_nodata(rkls, _TARGET_NODATA) &
            ~utils.array_equals_nodata(cp_factor, _TARGET_NODATA))
        result[valid_mask] = rkls[valid_mask] * cp_factor[valid_mask] * (
            1 - drainage[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            rkls_path, cp_factor_path, drainage_raster_path]], usle_op,
        out_usle_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_bar_factor(
        flow_direction_path, factor_path, flow_accumulation_path,
        accumulation_path, out_bar_path):
    """Route user defined source across DEM.

    Used for calculating S and W bar in the SDR operation.

    Args:
        dem_path (string): path to DEM raster
        factor_path (string): path to arbitrary factor raster
        flow_accumulation_path (string): path to flow accumulation raster
        flow_direction_path (string): path to flow direction path (in radians)
        accumulation_path (string): path to a raster that can be used to
            save the accumulation of the factor.  Temporary file.
        out_bar_path (string): path to output raster that is the result of
            the factor accumulation raster divided by the flow accumulation
            raster.

    Returns:
        None.

    """
    flow_accumulation_nodata = pygeoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    LOGGER.debug("doing flow accumulation mfd on %s", factor_path)
    # manually setting compression to DEFLATE because we got some LZW
    # errors when testing with large data.
    pygeoprocessing.routing.flow_accumulation_mfd(
        (flow_direction_path, 1), accumulation_path,
        weight_raster_path_band=(factor_path, 1),
        raster_driver_creation_tuple=('GTIFF', [
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
            'PREDICTOR=3']))

    def bar_op(base_accumulation, flow_accumulation):
        """Aggregate accumulation from base divided by the flow accum."""
        result = numpy.empty(base_accumulation.shape, dtype=numpy.float32)
        # flow accumulation intermediate output should always have a defined
        # nodata value from pygeoprocessing
        valid_mask = ~(
            utils.array_equals_nodata(base_accumulation, _TARGET_NODATA) |
            utils.array_equals_nodata(
                flow_accumulation, flow_accumulation_nodata))
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            base_accumulation[valid_mask] / flow_accumulation[valid_mask])
        return result
    pygeoprocessing.raster_calculator(
        [(accumulation_path, 1), (flow_accumulation_path, 1)], bar_op,
        out_bar_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_d_up(
        w_bar_path, s_bar_path, flow_accumulation_path, out_d_up_path):
    """Calculate w_bar * s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = abs(
        pygeoprocessing.get_raster_info(w_bar_path)['pixel_size'][0])**2
    flow_accumulation_nodata = pygeoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    def d_up_op(w_bar, s_bar, flow_accumulation):
        """Calculate the d_up index.

        w_bar * s_bar * sqrt(upslope area)

        """
        valid_mask = (
            ~utils.array_equals_nodata(w_bar, _TARGET_NODATA) &
            ~utils.array_equals_nodata(s_bar, _TARGET_NODATA) &
            ~utils.array_equals_nodata(
                flow_accumulation, flow_accumulation_nodata))
        d_up_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        d_up_array[:] = _TARGET_NODATA
        d_up_array[valid_mask] = (
            w_bar[valid_mask] * s_bar[valid_mask] * numpy.sqrt(
                flow_accumulation[valid_mask] * cell_area))
        return d_up_array

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            w_bar_path, s_bar_path, flow_accumulation_path]], d_up_op,
        out_d_up_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_d_up_bare(
        s_bar_path, flow_accumulation_path, out_d_up_bare_path):
    """Calculate s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = abs(
        pygeoprocessing.get_raster_info(s_bar_path)['pixel_size'][0])**2
    flow_accumulation_nodata = pygeoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    def d_up_op(s_bar, flow_accumulation):
        """Calculate the bare d_up index.

        s_bar * sqrt(upslope area)

        """
        valid_mask = (
            ~utils.array_equals_nodata(
                flow_accumulation, flow_accumulation_nodata) &
            ~utils.array_equals_nodata(s_bar, _TARGET_NODATA))
        d_up_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        d_up_array[:] = _TARGET_NODATA
        d_up_array[valid_mask] = (
            numpy.sqrt(flow_accumulation[valid_mask] * cell_area) *
            s_bar[valid_mask])
        return d_up_array

    pygeoprocessing.raster_calculator(
        [(s_bar_path, 1), (flow_accumulation_path, 1)], d_up_op,
        out_d_up_bare_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_inverse_ws_factor(
        thresholded_slope_path, thresholded_w_factor_path,
        out_ws_factor_inverse_path):
    """Calculate 1/(w*s)."""
    slope_nodata = pygeoprocessing.get_raster_info(
        thresholded_slope_path)['nodata'][0]

    def ws_op(w_factor, s_factor):
        """Calculate the inverse ws factor."""
        valid_mask = (
            ~utils.array_equals_nodata(w_factor, _TARGET_NODATA) &
            ~utils.array_equals_nodata(s_factor, slope_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            1.0 / (w_factor[valid_mask] * s_factor[valid_mask]))
        return result

    pygeoprocessing.raster_calculator(
        [(thresholded_w_factor_path, 1), (thresholded_slope_path, 1)], ws_op,
        out_ws_factor_inverse_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_inverse_s_factor(
        thresholded_slope_path, out_s_factor_inverse_path):
    """Calculate 1/s."""
    slope_nodata = pygeoprocessing.get_raster_info(
        thresholded_slope_path)['nodata'][0]

    def s_op(s_factor):
        """Calculate the inverse s factor."""
        valid_mask = ~utils.array_equals_nodata(s_factor, slope_nodata)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = 1.0 / s_factor[valid_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(thresholded_slope_path, 1)], s_op,
        out_s_factor_inverse_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_ic(d_up_path, d_dn_path, out_ic_factor_path):
    """Calculate log10(d_up/d_dn)."""
    # ic can be positive or negative, so float.min is a reasonable nodata value
    d_dn_nodata = pygeoprocessing.get_raster_info(d_dn_path)['nodata'][0]

    def ic_op(d_up, d_dn):
        """Calculate IC factor."""
        valid_mask = (
            ~utils.array_equals_nodata(d_up, _TARGET_NODATA) &
            ~utils.array_equals_nodata(d_dn, d_dn_nodata) & (d_dn != 0) &
            (d_up != 0))
        ic_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        ic_array[:] = _IC_NODATA
        ic_array[valid_mask] = numpy.log10(
            d_up[valid_mask] / d_dn[valid_mask])
        return ic_array

    pygeoprocessing.raster_calculator(
        [(d_up_path, 1), (d_dn_path, 1)], ic_op, out_ic_factor_path,
        gdal.GDT_Float32, _IC_NODATA)


def _calculate_sdr(
        k_factor, ic_0, sdr_max, ic_path, stream_path, out_sdr_path):
    """Derive SDR from k, ic0, ic; 0 on the stream and clamped to sdr_max."""
    def sdr_op(ic_factor, stream):
        """Calculate SDR factor."""
        valid_mask = (
            ~utils.array_equals_nodata(ic_factor, _IC_NODATA) & (stream != 1))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            sdr_max / (1+numpy.exp((ic_0-ic_factor[valid_mask])/k_factor)))
        result[stream == 1] = 0.0
        return result

    pygeoprocessing.raster_calculator(
        [(ic_path, 1), (stream_path, 1)], sdr_op, out_sdr_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_sed_export(usle_path, sdr_path, target_sed_export_path):
    """Calculate USLE * SDR."""
    def sed_export_op(usle, sdr):
        """Sediment export."""
        valid_mask = (
            ~utils.array_equals_nodata(usle, _TARGET_NODATA) &
            ~utils.array_equals_nodata(sdr, _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = usle[valid_mask] * sdr[valid_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(usle_path, 1), (sdr_path, 1)], sed_export_op,
        target_sed_export_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_e_prime(usle_path, sdr_path, target_e_prime):
    """Calculate USLE * (1-SDR)."""
    def e_prime_op(usle, sdr):
        """Wash that does not reach stream."""
        valid_mask = (
            ~utils.array_equals_nodata(usle, _TARGET_NODATA) &
            ~utils.array_equals_nodata(sdr, _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = usle[valid_mask] * (1-sdr[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(usle_path, 1), (sdr_path, 1)], e_prime_op, target_e_prime,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_sed_retention_index(
        rkls_path, usle_path, sdr_path, sdr_max,
        out_sed_retention_index_path):
    """Calculate (rkls-usle) * sdr  / sdr_max."""
    def sediment_index_op(rkls, usle, sdr_factor):
        """Calculate sediment retention index."""
        valid_mask = (
            ~utils.array_equals_nodata(rkls, _TARGET_NODATA) &
            ~utils.array_equals_nodata(usle, _TARGET_NODATA) &
            ~utils.array_equals_nodata(sdr_factor, _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            (rkls[valid_mask] - usle[valid_mask]) *
            sdr_factor[valid_mask] / sdr_max)
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [rkls_path, usle_path, sdr_path]],
        sediment_index_op, out_sed_retention_index_path, gdal.GDT_Float32,
        _TARGET_NODATA)


def _calculate_sed_retention(
        rkls_path, usle_path, stream_path, sdr_path, sdr_bare_soil_path,
        out_sed_ret_bare_soil_path):
    """Difference in exported sediments on basic and bare watershed.

    Calculates the difference of sediment export on the real landscape and
    a bare soil landscape given that SDR has been calculated for bare soil.
    Essentially:

        RKLS * SDR_bare - USLE * SDR

    Args:
        rkls_path (string): path to RKLS raster
        usle_path (string): path to USLE raster
        stream_path (string): path to stream/drainage mask
        sdr_path (string): path to SDR raster
        sdr_bare_soil_path (string): path to SDR raster calculated for a bare
            watershed
        out_sed_ret_bare_soil_path (string): path to output raster indicating
            where sediment is retained

    Returns:
        None

    """
    stream_nodata = pygeoprocessing.get_raster_info(stream_path)['nodata'][0]

    def sediment_retention_bare_soil_op(
            rkls, usle, stream_factor, sdr_factor, sdr_factor_bare_soil):
        """Subtract bare soil export from real landcover."""
        valid_mask = (
            ~utils.array_equals_nodata(rkls, _TARGET_NODATA) &
            ~utils.array_equals_nodata(usle, _TARGET_NODATA) &
            ~utils.array_equals_nodata(stream_factor, stream_nodata) &
            ~utils.array_equals_nodata(sdr_factor, _TARGET_NODATA) &
            ~utils.array_equals_nodata(sdr_factor_bare_soil, _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            rkls[valid_mask] * sdr_factor_bare_soil[valid_mask] -
            usle[valid_mask] * sdr_factor[valid_mask]) * (
                1 - stream_factor[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            rkls_path, usle_path, stream_path, sdr_path, sdr_bare_soil_path]],
        sediment_retention_bare_soil_op, out_sed_ret_bare_soil_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _generate_report(
        watersheds_path, usle_path, sed_export_path, sed_retention_path,
        sed_deposition_path, watershed_results_sdr_path):
    """Create shapefile with USLE, sed export, retention, and deposition."""
    original_datasource = gdal.OpenEx(watersheds_path, gdal.OF_VECTOR)
    if os.path.exists(watershed_results_sdr_path):
        LOGGER.warning(f'overwriting results at {watershed_results_sdr_path}')
        os.remove(watershed_results_sdr_path)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    target_vector = driver.CreateCopy(
        watershed_results_sdr_path, original_datasource)

    target_layer = target_vector.GetLayer()
    target_layer.SyncToDisk()

    field_summaries = {
        'usle_tot': pygeoprocessing.zonal_statistics(
            (usle_path, 1), watershed_results_sdr_path),
        'sed_export': pygeoprocessing.zonal_statistics(
            (sed_export_path, 1), watershed_results_sdr_path),
        'sed_retent': pygeoprocessing.zonal_statistics(
            (sed_retention_path, 1), watershed_results_sdr_path),
        'sed_dep': pygeoprocessing.zonal_statistics(
            (sed_deposition_path, 1), watershed_results_sdr_path),
    }

    for field_name in field_summaries:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_def.SetWidth(24)
        field_def.SetPrecision(11)
        target_layer.CreateField(field_def)

    target_layer.ResetReading()
    for feature in target_layer:
        feature_id = feature.GetFID()
        for field_name in field_summaries:
            feature.SetField(
                field_name,
                float(field_summaries[field_name][feature_id]['sum']))
        target_layer.SetFeature(feature)
    target_vector = None
    target_layer = None


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    validation_warnings = validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])

    invalid_keys = validation.get_invalid_keys(validation_warnings)
    sufficient_keys = validation.get_sufficient_keys(args)

    if ('watersheds_path' not in invalid_keys and
            'watersheds_path' in sufficient_keys):
        # The watersheds vector must have an integer column called WS_ID.
        vector = gdal.OpenEx(args['watersheds_path'], gdal.OF_VECTOR)
        layer = vector.GetLayer()
        n_invalid_features = 0
        for feature in layer:
            try:
                int(feature.GetFieldAsString('ws_id'))
            except ValueError:
                n_invalid_features += 1

        if n_invalid_features:
            validation_warnings.append((
                ['watersheds_path'],
                INVALID_ID_MSG.format(number=n_invalid_features)))
            invalid_keys.add('watersheds_path')

    return validation_warnings
