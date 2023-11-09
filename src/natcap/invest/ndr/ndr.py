"""InVEST Nutrient Delivery Ratio (NDR) module."""
import copy
import itertools
import logging
import os
import pickle

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph
from osgeo import gdal
from osgeo import ogr

from .. import gettext
from .. import spec_utils
from .. import utils
from .. import validation
from ..model_metadata import MODEL_METADATA
from ..sdr import sdr
from ..unit_registry import u
from . import ndr_core

LOGGER = logging.getLogger(__name__)

MISSING_NUTRIENT_MSG = gettext('Either calc_n or calc_p must be True')

MODEL_SPEC = {
    "model_name": MODEL_METADATA["ndr"].model_title,
    "pyname": MODEL_METADATA["ndr"].pyname,
    "userguide": MODEL_METADATA["ndr"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["dem_path", "lulc_path", "runoff_proxy_path",
                         "watersheds_path"],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "dem_path": {
            **spec_utils.DEM,
            "projected": True
        },
        "lulc_path": {
            **spec_utils.LULC,
            "projected": True,
            "about": spec_utils.LULC['about'] + " " + gettext(
                "All values in this raster must "
                "have corresponding entries in the Biophysical table.")
        },
        "runoff_proxy_path": {
            "type": "raster",
            "bands": {1: {
                "type": "number",
                "units": u.none
            }},
            "about": gettext(
                "Map of runoff potential, the capacity to transport "
                "nutrients downslope. This can be a quickflow index "
                "or annual precipitation. Any units are allowed since "
                "the values will be normalized by their average."),
            "name": gettext("nutrient runoff proxy")
        },
        "watersheds_path": {
            "type": "vector",
            "projected": True,
            "geometries": spec_utils.POLYGONS,
            "fields": {},
            "about": gettext(
                "Map of the boundaries of the watershed(s) over which to "
                "aggregate the model results."),
            "name": gettext("watersheds")
        },
        "biophysical_table_path": {
            "type": "csv",
            "index_col": "lucode",
            "columns": {
                "lucode": spec_utils.LULC_TABLE_COLUMN,
                "load_[NUTRIENT]": {  # nitrogen or phosphorus nutrient loads
                    "type": "number",
                    "units": u.kilogram/u.hectare/u.year,
                    "about": gettext(
                        "The nutrient loading for this land use class.")},
                "eff_[NUTRIENT]": {  # nutrient retention capacities
                    "type": "ratio",
                    "about": gettext(
                        "Maximum nutrient retention efficiency. This is the "
                        "maximum proportion of the nutrient that is retained "
                        "on this LULC class.")},
                "crit_len_[NUTRIENT]": {  # nutrient critical lengths
                    "type": "number",
                    "units": u.meter,
                    "about": gettext(
                        "The distance after which it is assumed that this "
                        "LULC type retains the nutrient at its maximum "
                        "capacity. If nutrients travel a shorter distance "
                        "that this, the retention "
                        "efficiency will be less than the maximum value "
                        "eff_x, following an exponential decay.")},
                "proportion_subsurface_n": {
                    "type": "ratio",
                    "required": "calc_n",
                    "about": gettext(
                        "The proportion of the total amount of nitrogen that "
                        "are dissolved into the subsurface. By default, this "
                        "value should be set to 0, indicating that all "
                        "nutrients are delivered via surface flow. There is "
                        "no equivalent of this for phosphorus.")}
            },
            "about": gettext(
                "A table mapping each LULC class to its biophysical "
                "properties related to nutrient load and retention. Replace "
                "'[NUTRIENT]' in the column names with 'n' or 'p' for "
                "nitrogen or phosphorus respectively. Nitrogen data must be "
                "provided if Calculate Nitrogen is selected. Phosphorus data "
                "must be provided if Calculate Phosphorus is selected. All "
                "LULC codes in the LULC raster must have corresponding "
                "entries in this table."),
            "name": gettext("biophysical table")
        },
        "calc_p": {
            "type": "boolean",
            "about": gettext("Calculate phosphorus retention and export."),
            "name": gettext("calculate phosphorus")
        },
        "calc_n": {
            "type": "boolean",
            "about": gettext("Calculate nitrogen retention and export."),
            "name": gettext("calculate nitrogen")
        },
        "threshold_flow_accumulation": {
            **spec_utils.THRESHOLD_FLOW_ACCUMULATION
        },
        "k_param": {
            "type": "number",
            "units": u.none,
            "about": gettext(
                "Calibration parameter that determines the shape of the "
                "relationship between hydrologic connectivity (the degree of "
                "connection from patches of land to the stream) and the "
                "nutrient delivery ratio (percentage of nutrient that "
                "actually reaches the stream)."),
            "name": gettext("Borselli k parameter"),
        },
        "subsurface_critical_length_n": {
            "type": "number",
            "units": u.meter,
            "required": "calc_n",
            "name": gettext("subsurface critical length (nitrogen)"),
            "about": gettext(
                "The distance traveled (subsurface and downslope) after which "
                "it is assumed that soil retains nitrogen at its maximum "
                "capacity. Required if Calculate Nitrogen is selected."),
        },
        "subsurface_eff_n": {
            "type": "ratio",
            "required": "calc_n",
            "name": gettext("subsurface maximum retention efficiency (nitrogen)"),
            "about": gettext(
                "The maximum nitrogen retention efficiency that can be "
                "reached through subsurface flow. This characterizes the "
                "retention due to biochemical degradation in soils. Required "
                "if Calculate Nitrogen is selected.")
        }
    },
    "outputs": {
        "watershed_results_ndr.gpkg": {
            "about": "Vector with aggregated nutrient model results per watershed.",
            "geometries": spec_utils.POLYGONS,
            "fields": {
                "p_surface_load": {
                    "type": "number",
                    "units": u.kilogram/u.year,
                    "about": "Total phosphorus loads (sources) in the watershed, i.e. the sum of the nutrient contribution from all surface LULC without filtering by the landscape."
                },
                "n_surface_load": {
                    "type": "number",
                    "units": u.kilogram/u.year,
                    "about": "Total nitrogen loads (sources) in the watershed, i.e. the sum of the nutrient contribution from all surface LULC without filtering by the landscape."
                },
                "n_subsurface_load": {
                    "type": "number",
                    "units": u.kilogram/u.year,
                    "about": "Total subsurface nitrogen loads in the watershed."
                },
                "p_surface_export": {
                    "type": "number",
                    "units": u.kilogram/u.year,
                    "about": "Total phosphorus export from the watershed by surface flow."
                },
                "n_surface_export": {
                    "type": "number",
                    "units": u.kilogram/u.year,
                    "about": "Total nitrogen export from the watershed by surface flow."
                },
                "n_subsurface_export": {
                    "type": "number",
                    "units": u.kilogram/u.year,
                    "about": "Total nitrogen export from the watershed by subsurface flow."
                },
                "n_total_export": {
                    "type": "number",
                    "units": u.kilogram/u.year,
                    "about": "Total nitrogen export from the watershed by surface and subsurface flow."
                }
            }
        },
        "p_surface_export.tif": {
            "about": "A pixel level map showing how much phosphorus from each pixel eventually reaches the stream by surface flow.",
            "bands": {1: {
                "type": "number",
                "units": u.kilogram/u.pixel
            }}
        },
        "n_surface_export.tif": {
            "about": "A pixel level map showing how much nitrogen from each pixel eventually reaches the stream by surface flow.",
            "bands": {1: {
                "type": "number",
                "units": u.kilogram/u.pixel
            }}
        },
        "n_subsurface_export.tif": {
            "about": "A pixel level map showing how much nitrogen from each pixel eventually reaches the stream by subsurface flow.",
            "bands": {1: {
                "type": "number",
                "units": u.kilogram/u.pixel
            }}
        },
        "n_total_export.tif": {
            "about": "A pixel level map showing how much nitrogen from each pixel eventually reaches the stream by either flow.",
            "bands": {1: {
                "type": "number",
                "units": u.kilogram/u.pixel
            }}
        },
        "intermediate_outputs": {
            "type": "directory",
            "contents": {
                "crit_len_n.tif": {
                    "about": (
                        "Nitrogen retention length, found in the biophysical table"),
                    "bands": {1: {"type": "number", "units": u.meter}}
                },
                "crit_len_p.tif": {
                    "about": (
                        "Phosphorus retention length, found in the biophysical table"),
                    "bands": {1: {"type": "number", "units": u.meter}}
                },
                "d_dn.tif": {
                    "about": "Downslope factor of the index of connectivity",
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "d_up.tif": {
                    "about": "Upslope factor of the index of connectivity",
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "dist_to_channel.tif": {
                    "about": "Average downslope distance from a pixel to the stream",
                    "bands": {1: {"type": "number", "units": u.pixel}}
                },
                "eff_n.tif": {
                    "about": "Raw per-landscape cover retention efficiency for nitrogen.",
                    "bands": {1: {"type": "ratio"}}
                },
                "eff_p.tif": {
                    "about": "Raw per-landscape cover retention efficiency for phosphorus",
                    "bands": {1: {"type": "ratio"}}
                },
                "effective_retention_n.tif": {
                    "about": "Effective nitrogen retention provided by the downslope flow path for each pixel",
                    "bands": {1: {"type": "ratio"}}
                },
                "effective_retention_p.tif": {
                    "about": "Effective phosphorus retention provided by the downslope flow path for each pixel",
                    "bands": {1: {"type": "ratio"}}
                },
                "flow_accumulation.tif": spec_utils.FLOW_ACCUMULATION,
                "flow_direction.tif": spec_utils.FLOW_DIRECTION,
                "ic_factor.tif": {
                    "about": "Index of connectivity",
                    "bands": {1: {"type": "ratio"}}
                },
                "load_n.tif": {
                    "about": "Nitrogen load (for surface transport) per pixel",
                    "bands": {1: {
                        "type": "number",
                        "units": u.kilogram/u.year
                    }}
                },
                "load_p.tif": {
                    "about": "Phosphorus load (for surface transport) per pixel",
                    "bands": {1: {
                        "type": "number",
                        "units": u.kilogram/u.year
                    }}
                },
                "modified_load_n.tif": {
                    "about": "Raw nitrogen load scaled by the runoff proxy index.",
                    "bands": {1: {
                        "type": "number",
                        "units": u.kilogram/u.year
                    }}
                },
                "modified_load_p.tif": {
                    "about": "Raw phosphorus load scaled by the runoff proxy index.",
                    "bands": {1: {
                        "type": "number",
                        "units": u.kilogram/u.year
                    }}
                },
                "ndr_n.tif": {
                    "about": "NDR values for nitrogen",
                    "bands": {1: {"type": "ratio"}}
                },
                "ndr_p.tif": {
                    "about": "NDR values for phosphorus",
                    "bands": {1: {"type": "ratio"}}
                },
                "runoff_proxy_index.tif": {
                    "about": "Normalized values for the Runoff Proxy input to the model",
                    "bands": {1: {"type": "ratio"}}
                },
                "s_accumulation.tif": {
                    "about": "Flow accumulation weighted by slope",
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "s_bar.tif": {
                    "about": "Average slope gradient of the upslope contributing area",
                    "bands": {1: {"type": "ratio"}}
                },
                "s_factor_inverse.tif": {
                    "about": "Inverse of slope",
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "stream.tif": spec_utils.STREAM,
                "sub_load_n.tif": {
                    "about": "Nitrogen loads for subsurface transport",
                    "bands": {1: {
                        "type": "number",
                        "units": u.kilogram/u.year
                    }}
                },
                "sub_ndr_n.tif": {
                    "about": "Subsurface nitrogen NDR values",
                    "bands": {1: {"type": "ratio"}}
                },
                "surface_load_n.tif": {
                    "about": "Above ground nitrogen loads",
                    "bands": {1: {
                        "type": "number",
                        "units": u.kilogram/u.year
                    }}
                },
                "surface_load_p.tif": {
                    "about": "Above ground phosphorus loads",
                    "bands": {1: {
                        "type": "number",
                        "units": u.kilogram/u.year
                    }}
                },
                "thresholded_slope.tif": {
                    "about": (
                        "Percent slope thresholded for correct calculation of IC."),
                    "bands": {1: {"type": "percent"}}
                },
                "what_drains_to_stream.tif": {
                    "about": (
                        "Map of which pixels drain to a stream. A value of 1 "
                        "means that at least some of the runoff from that "
                        "pixel drains to a stream in stream.tif. A value of 0 "
                        "means that it does not drain at all to any stream in "
                        "stream.tif."),
                    "bands": {1: {
                        "type": "integer"
                    }}
                },
                "aligned_dem.tif": {
                    "about": "Copy of the DEM clipped to the extent of the other inputs",
                    "bands": {1: {"type": "number", "units": u.meter}}
                },
                "aligned_lulc.tif": {
                    "about": (
                        "Copy of the LULC clipped to the extent of the other inputs "
                        "and reprojected to the DEM projection"),
                    "bands": {1: {"type": "integer"}}
                },
                "aligned_runoff_proxy.tif": {
                    "about": (
                        "Copy of the runoff proxy clipped to the extent of the other inputs "
                        "and reprojected to the DEM projection"),
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "masked_dem.tif": {
                    "about": "DEM input masked to exclude pixels outside the watershed",
                    "bands": {1: {"type": "number", "units": u.meter}}
                },
                "masked_lulc.tif": {
                    "about": "LULC input masked to exclude pixels outside the watershed",
                    "bands": {1: {"type": "integer"}}
                },
                "masked_runoff_proxy.tif": {
                    "about": "Runoff proxy input masked to exclude pixels outside the watershed",
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "filled_dem.tif": spec_utils.FILLED_DEM,
                "slope.tif": spec_utils.SLOPE,
                "subsurface_export_n.pickle": {
                    "about": "Pickled zonal statistics of nitrogen subsurface export"
                },
                "subsurface_load_n.pickle": {
                    "about": "Pickled zonal statistics of nitrogen subsurface load"
                },
                "surface_export_n.pickle": {
                    "about": "Pickled zonal statistics of nitrogen surface export"
                },
                "surface_export_p.pickle": {
                    "about": "Pickled zonal statistics of phosphorus surface export"
                },
                "surface_load_n.pickle": {
                    "about": "Pickled zonal statistics of nitrogen surface load"
                },
                "surface_load_p.pickle": {
                    "about": "Pickled zonal statistics of phosphorus surface load"
                },
                "total_export_n.pickle": {
                    "about": "Pickled zonal statistics of total nitrogen export"
                }
            }
        },
        "taskgraph_cache": spec_utils.TASKGRAPH_DIR
    }
}

_OUTPUT_BASE_FILES = {
    'n_surface_export_path': 'n_surface_export.tif',
    'n_subsurface_export_path': 'n_subsurface_export.tif',
    'n_total_export_path': 'n_total_export.tif',
    'p_surface_export_path': 'p_surface_export.tif',
    'watershed_results_ndr_path': 'watershed_results_ndr.gpkg',
}

INTERMEDIATE_DIR_NAME = 'intermediate_outputs'

_INTERMEDIATE_BASE_FILES = {
    'mask_path': 'watersheds_mask.tif',
    'ic_factor_path': 'ic_factor.tif',
    'load_n_path': 'load_n.tif',
    'load_p_path': 'load_p.tif',
    'modified_load_n_path': 'modified_load_n.tif',
    'modified_load_p_path': 'modified_load_p.tif',
    'ndr_n_path': 'ndr_n.tif',
    'ndr_p_path': 'ndr_p.tif',
    'runoff_proxy_index_path': 'runoff_proxy_index.tif',
    's_accumulation_path': 's_accumulation.tif',
    's_bar_path': 's_bar.tif',
    's_factor_inverse_path': 's_factor_inverse.tif',
    'stream_path': 'stream.tif',
    'sub_load_n_path': 'sub_load_n.tif',
    'surface_load_n_path': 'surface_load_n.tif',
    'surface_load_p_path': 'surface_load_p.tif',
    'sub_ndr_n_path': 'sub_ndr_n.tif',
    'crit_len_n_path': 'crit_len_n.tif',
    'crit_len_p_path': 'crit_len_p.tif',
    'd_dn_path': 'd_dn.tif',
    'd_up_path': 'd_up.tif',
    'eff_n_path': 'eff_n.tif',
    'eff_p_path': 'eff_p.tif',
    'effective_retention_n_path': 'effective_retention_n.tif',
    'effective_retention_p_path': 'effective_retention_p.tif',
    'flow_accumulation_path': 'flow_accumulation.tif',
    'flow_direction_path': 'flow_direction.tif',
    'thresholded_slope_path': 'thresholded_slope.tif',
    'dist_to_channel_path': 'dist_to_channel.tif',
    'drainage_mask': 'what_drains_to_stream.tif',
    'filled_dem_path': 'filled_dem.tif',
    'aligned_dem_path': 'aligned_dem.tif',
    'masked_dem_path': 'masked_dem.tif',
    'slope_path': 'slope.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
    'masked_lulc_path': 'masked_lulc.tif',
    'aligned_runoff_proxy_path': 'aligned_runoff_proxy.tif',
    'masked_runoff_proxy_path': 'masked_runoff_proxy.tif',
    'surface_load_n_pickle_path': 'surface_load_n.pickle',
    'surface_load_p_pickle_path': 'surface_load_p.pickle',
    'subsurface_load_n_pickle_path': 'subsurface_load_n.pickle',
    'surface_export_n_pickle_path': 'surface_export_n.pickle',
    'surface_export_p_pickle_path': 'surface_export_p.pickle',
    'subsurface_export_n_pickle_path': 'subsurface_export_n.pickle',
    'total_export_n_pickle_path': 'total_export_n.pickle'
}

_TARGET_NODATA = -1


def execute(args):
    """Nutrient Delivery Ratio.

    Args:
        args['workspace_dir'] (string):  path to current workspace
        args['dem_path'] (string): path to digital elevation map raster
        args['lulc_path'] (string): a path to landcover map raster
        args['runoff_proxy_path'] (string): a path to a runoff proxy raster
        args['watersheds_path'] (string): path to the watershed shapefile
        args['biophysical_table_path'] (string): path to csv table on disk
            containing nutrient retention values.

            For each nutrient type [t] in args['calc_[t]'] that is true, must
            contain the following headers:

            'load_[t]', 'eff_[t]', 'crit_len_[t]'

            If args['calc_n'] is True, must also contain the header
            'proportion_subsurface_n' field.

        args['calc_p'] (boolean): if True, phosphorus is modeled,
            additionally if True then biophysical table must have p fields in
            them
        args['calc_n'] (boolean): if True nitrogen will be modeled,
            additionally biophysical table must have n fields in them.
        args['results_suffix'] (string): (optional) a text field to append to
            all output files
        args['threshold_flow_accumulation']: a number representing the flow
            accumulation in terms of upslope pixels.
        args['k_param'] (number): The Borselli k parameter. This is a
            calibration parameter that determines the shape of the
            relationship between hydrologic connectivity.
        args['subsurface_critical_length_n'] (number): The distance (traveled
            subsurface and downslope) after which it is assumed that soil
            retains nutrient at its maximum capacity, given in meters. If
            dissolved nutrients travel a distance smaller than Subsurface
            Critical Length, the retention efficiency will be lower than the
            Subsurface Maximum Retention Efficiency value defined. Setting this
            value to a distance smaller than the pixel size will result in the
            maximum retention efficiency being reached within one pixel only.
            Required if ``calc_n``.
        args['subsurface_eff_n'] (number): The maximum nutrient retention
            efficiency that can be reached through subsurface flow, a floating
            point value between 0 and 1. This field characterizes the retention
            due to biochemical degradation in soils.  Required if ``calc_n``.
        args['n_workers'] (int): if present, indicates how many worker
            processes should be used in parallel processing. -1 indicates
            single process mode, 0 is single process but non-blocking mode,
            and >= 1 is number of processes.

    Returns:
        None

    """
    # Load all the tables for preprocessing
    output_dir = os.path.join(args['workspace_dir'])
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], INTERMEDIATE_DIR_NAME)
    utils.make_directories([output_dir, intermediate_output_dir])

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(
        os.path.join(args['workspace_dir'], 'taskgraph_cache'),
        n_workers, reporting_interval=5.0)

    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir)], file_suffix)

    # Build up a list of nutrients to process based on what's checked on
    nutrients_to_process = []
    for nutrient_id in ['n', 'p']:
        if args['calc_' + nutrient_id]:
            nutrients_to_process.append(nutrient_id)

    biophysical_df = validation.get_validated_dataframe(
        args['biophysical_table_path'],
        **MODEL_SPEC['args']['biophysical_table_path'])

    # these are used for aggregation in the last step
    field_pickle_map = {}

    create_vector_task = task_graph.add_task(
        func=create_vector_copy,
        args=(args['watersheds_path'], f_reg['watershed_results_ndr_path']),
        target_path_list=[f_reg['watershed_results_ndr_path']],
        task_name='create target vector')

    dem_info = pygeoprocessing.get_raster_info(args['dem_path'])

    base_raster_list = [
        args['dem_path'], args['lulc_path'], args['runoff_proxy_path']]
    aligned_raster_list = [
        f_reg['aligned_dem_path'], f_reg['aligned_lulc_path'],
        f_reg['aligned_runoff_proxy_path']]
    align_raster_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            base_raster_list, aligned_raster_list,
            ['near']*len(base_raster_list), dem_info['pixel_size'],
            'intersection'),
        kwargs={'base_vector_path_list': [args['watersheds_path']]},
        target_path_list=aligned_raster_list,
        task_name='align rasters')

    # Since we mask multiple rasters using the same vector, we can just do the
    # rasterization once.  Calling pygeoprocessing.mask_raster() multiple times
    # unfortunately causes the rasterization to happen once per call.
    mask_task = task_graph.add_task(
        func=_create_mask_raster,
        kwargs={
            'source_raster_path': f_reg['aligned_dem_path'],
            'source_vector_path': args['watersheds_path'],
            'target_raster_path': f_reg['mask_path']
        },
        target_path_list=[f_reg['mask_path']],
        dependent_task_list=[align_raster_task],
        task_name='create watersheds mask'
    )
    mask_runoff_proxy_task = task_graph.add_task(
        func=_mask_raster,
        kwargs={
            'source_raster_path': f_reg['aligned_runoff_proxy_path'],
            'mask_raster_path': f_reg['mask_path'],
            'target_masked_raster_path': f_reg['masked_runoff_proxy_path'],
            'target_dtype': gdal.GDT_Float32,
            'default_nodata': _TARGET_NODATA,
        },
        dependent_task_list=[mask_task, align_raster_task],
        target_path_list=[f_reg['masked_runoff_proxy_path']],
        task_name='mask runoff proxy raster',
    )
    mask_dem_task = task_graph.add_task(
        func=_mask_raster,
        kwargs={
            'source_raster_path': f_reg['aligned_dem_path'],
            'mask_raster_path': f_reg['mask_path'],
            'target_masked_raster_path': f_reg['masked_dem_path'],
            'target_dtype': gdal.GDT_Float32,
            'default_nodata': float(numpy.finfo(numpy.float32).min),
        },
        dependent_task_list=[mask_task, align_raster_task],
        target_path_list=[f_reg['masked_dem_path']],
        task_name='mask dem raster',
    )
    mask_lulc_task = task_graph.add_task(
        func=_mask_raster,
        kwargs={
            'source_raster_path': f_reg['aligned_lulc_path'],
            'mask_raster_path': f_reg['mask_path'],
            'target_masked_raster_path': f_reg['masked_lulc_path'],
            'target_dtype': gdal.GDT_Int32,
            'default_nodata': numpy.iinfo(numpy.int32).min,
        },
        dependent_task_list=[mask_task, align_raster_task],
        target_path_list=[f_reg['masked_lulc_path']],
        task_name='mask lulc raster',
    )

    fill_pits_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (f_reg['masked_dem_path'], 1), f_reg['filled_dem_path']),
        kwargs={'working_dir': intermediate_output_dir},
        dependent_task_list=[align_raster_task, mask_dem_task],
        target_path_list=[f_reg['filled_dem_path']],
        task_name='fill pits')

    flow_dir_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_mfd,
        args=(
            (f_reg['filled_dem_path'], 1), f_reg['flow_direction_path']),
        kwargs={'working_dir': intermediate_output_dir},
        dependent_task_list=[fill_pits_task],
        target_path_list=[f_reg['flow_direction_path']],
        task_name='flow dir')

    flow_accum_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            f_reg['flow_accumulation_path']),
        target_path_list=[f_reg['flow_accumulation_path']],
        dependent_task_list=[flow_dir_task],
        task_name='flow accum')

    stream_extraction_task = task_graph.add_task(
        func=pygeoprocessing.routing.extract_streams_mfd,
        args=(
            (f_reg['flow_accumulation_path'], 1),
            (f_reg['flow_direction_path'], 1),
            float(args['threshold_flow_accumulation']),
            f_reg['stream_path']),
        target_path_list=[f_reg['stream_path']],
        dependent_task_list=[flow_accum_task],
        task_name='stream extraction')

    calculate_slope_task = task_graph.add_task(
        func=pygeoprocessing.calculate_slope,
        args=((f_reg['filled_dem_path'], 1), f_reg['slope_path']),
        target_path_list=[f_reg['slope_path']],
        dependent_task_list=[fill_pits_task],
        task_name='calculate slope')

    threshold_slope_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=_slope_proportion_and_threshold_op,
            rasters=[f_reg['slope_path']],
            target_path=f_reg['thresholded_slope_path']),
        target_path_list=[f_reg['thresholded_slope_path']],
        dependent_task_list=[calculate_slope_task],
        task_name='threshold slope')

    runoff_proxy_index_task = task_graph.add_task(
        func=_normalize_raster,
        args=((f_reg['masked_runoff_proxy_path'], 1),
              f_reg['runoff_proxy_index_path']),
        target_path_list=[f_reg['runoff_proxy_index_path']],
        dependent_task_list=[align_raster_task, mask_runoff_proxy_task],
        task_name='runoff proxy mean')

    s_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=((f_reg['flow_direction_path'], 1), f_reg['s_accumulation_path']),
        kwargs={
            'weight_raster_path_band': (f_reg['thresholded_slope_path'], 1)},
        target_path_list=[f_reg['s_accumulation_path']],
        dependent_task_list=[flow_dir_task, threshold_slope_task],
        task_name='route s')

    s_bar_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=numpy.divide,  # s_bar = s_accum / flow_accum
            rasters=[f_reg['s_accumulation_path'], f_reg['flow_accumulation_path']],
            target_path=f_reg['s_bar_path'],
            target_dtype=numpy.float32,
            target_nodata=_TARGET_NODATA),
        target_path_list=[f_reg['s_bar_path']],
        dependent_task_list=[s_task, flow_accum_task],
        task_name='calculate s bar')

    d_up_task = task_graph.add_task(
        func=d_up_calculation,
        args=(f_reg['s_bar_path'], f_reg['flow_accumulation_path'],
              f_reg['d_up_path']),
        target_path_list=[f_reg['d_up_path']],
        dependent_task_list=[s_bar_task, flow_accum_task],
        task_name='d up')

    s_inv_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=_inverse_op,
            rasters=[f_reg['thresholded_slope_path']],
            target_path=f_reg['s_factor_inverse_path'],
            target_nodata=_TARGET_NODATA),
        target_path_list=[f_reg['s_factor_inverse_path']],
        dependent_task_list=[threshold_slope_task],
        task_name='s inv')

    d_dn_task = task_graph.add_task(
        func=pygeoprocessing.routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            (f_reg['stream_path'], 1),
            f_reg['d_dn_path']),
        kwargs={'weight_raster_path_band': (
            f_reg['s_factor_inverse_path'], 1)},
        dependent_task_list=[stream_extraction_task, s_inv_task],
        target_path_list=[f_reg['d_dn_path']],
        task_name='d dn')

    dist_to_channel_task = task_graph.add_task(
        func=pygeoprocessing.routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            (f_reg['stream_path'], 1),
            f_reg['dist_to_channel_path']),
        dependent_task_list=[stream_extraction_task],
        target_path_list=[f_reg['dist_to_channel_path']],
        task_name='dist to channel')

    _ = task_graph.add_task(
        func=sdr._calculate_what_drains_to_stream,
        args=(f_reg['flow_direction_path'],
              f_reg['dist_to_channel_path'],
              f_reg['drainage_mask']),
        target_path_list=[f_reg['drainage_mask']],
        dependent_task_list=[flow_dir_task, dist_to_channel_task],
        task_name='write mask of what drains to stream')

    ic_task = task_graph.add_task(
        func=calculate_ic,
        args=(
            f_reg['d_up_path'], f_reg['d_dn_path'], f_reg['ic_factor_path']),
        target_path_list=[f_reg['ic_factor_path']],
        dependent_task_list=[d_dn_task, d_up_task],
        task_name='calc ic')

    for nutrient in nutrients_to_process:
        load_path = f_reg[f'load_{nutrient}_path']
        modified_load_path = f_reg[f'modified_load_{nutrient}_path']
        # Perrine says that 'n' is the only case where we could consider a
        # prop subsurface component.  So there's a special case for that.
        if nutrient == 'n':
            subsurface_proportion_map = (
                biophysical_df['proportion_subsurface_n'].to_dict())
        else:
            subsurface_proportion_map = None
        load_task = task_graph.add_task(
            func=_calculate_load,
            args=(
                f_reg['masked_lulc_path'],
                biophysical_df[f'load_{nutrient}'],
                load_path),
            dependent_task_list=[align_raster_task, mask_lulc_task],
            target_path_list=[load_path],
            task_name=f'{nutrient} load')

        modified_load_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=_mult_op,
                rasters=[load_path, f_reg['runoff_proxy_index_path']],
                target_path=modified_load_path,
                target_nodata=_TARGET_NODATA),
            target_path_list=[modified_load_path],
            dependent_task_list=[load_task, runoff_proxy_index_task],
            task_name=f'modified load {nutrient}')

        surface_load_path = f_reg[f'surface_load_{nutrient}_path']
        surface_load_task = task_graph.add_task(
            func=_map_surface_load,
            args=(modified_load_path, f_reg['masked_lulc_path'],
                  subsurface_proportion_map, surface_load_path),
            target_path_list=[surface_load_path],
            dependent_task_list=[modified_load_task, align_raster_task],
            task_name=f'map surface load {nutrient}')

        eff_path = f_reg[f'eff_{nutrient}_path']
        eff_task = task_graph.add_task(
            func=_map_lulc_to_val_mask_stream,
            args=(
                f_reg['masked_lulc_path'], f_reg['stream_path'],
                biophysical_df[f'eff_{nutrient}'].to_dict(), eff_path),
            target_path_list=[eff_path],
            dependent_task_list=[align_raster_task, stream_extraction_task],
            task_name=f'ret eff {nutrient}')

        crit_len_path = f_reg[f'crit_len_{nutrient}_path']
        crit_len_task = task_graph.add_task(
            func=_map_lulc_to_val_mask_stream,
            args=(
                f_reg['masked_lulc_path'], f_reg['stream_path'],
                biophysical_df[f'crit_len_{nutrient}'].to_dict(),
                crit_len_path),
            target_path_list=[crit_len_path],
            dependent_task_list=[align_raster_task, stream_extraction_task],
            task_name=f'ret eff {nutrient}')

        effective_retention_path = (
            f_reg[f'effective_retention_{nutrient}_path'])
        ndr_eff_task = task_graph.add_task(
            func=ndr_core.ndr_eff_calculation,
            args=(
                f_reg['flow_direction_path'],
                f_reg['stream_path'], eff_path,
                crit_len_path, effective_retention_path),
            target_path_list=[effective_retention_path],
            dependent_task_list=[
                stream_extraction_task, eff_task, crit_len_task],
            task_name=f'eff ret {nutrient}')

        ndr_path = f_reg[f'ndr_{nutrient}_path']
        ndr_task = task_graph.add_task(
            func=_calculate_ndr,
            args=(
                effective_retention_path, f_reg['ic_factor_path'],
                float(args['k_param']), ndr_path),
            target_path_list=[ndr_path],
            dependent_task_list=[ndr_eff_task, ic_task],
            task_name=f'calc ndr {nutrient}')

        surface_export_path = f_reg[f'{nutrient}_surface_export_path']
        surface_export_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=numpy.multiply,  # export = load * ndr
                rasters=[surface_load_path, ndr_path],
                target_path=surface_export_path,
                target_nodata=_TARGET_NODATA),
            target_path_list=[surface_export_path],
            dependent_task_list=[
                load_task, ndr_task, surface_load_task],
            task_name=f'surface export {nutrient}')

        field_pickle_map[f'{nutrient}_surface_load'] = (
            f_reg[f'surface_load_{nutrient}_pickle_path'])
        field_pickle_map[f'{nutrient}_surface_export'] = (
            f_reg[f'surface_export_{nutrient}_pickle_path'])

        # only calculate subsurface things for nitrogen
        if nutrient == 'n':
            proportion_subsurface_map = (
                biophysical_df['proportion_subsurface_n'].to_dict())
            subsurface_load_task = task_graph.add_task(
                func=_map_subsurface_load,
                args=(modified_load_path, f_reg['masked_lulc_path'],
                      proportion_subsurface_map, f_reg['sub_load_n_path']),
                target_path_list=[f_reg['sub_load_n_path']],
                dependent_task_list=[modified_load_task, align_raster_task],
                task_name='map subsurface load n')

            subsurface_ndr_task = task_graph.add_task(
                func=_calculate_sub_ndr,
                args=(
                    float(args['subsurface_eff_n']),
                    float(args['subsurface_critical_length_n']),
                    f_reg['dist_to_channel_path'], f_reg['sub_ndr_n_path']),
                target_path_list=[f_reg['sub_ndr_n_path']],
                dependent_task_list=[dist_to_channel_task],
                task_name='sub ndr n')

            subsurface_export_task = task_graph.add_task(
                func=pygeoprocessing.raster_map,
                kwargs=dict(
                    op=numpy.multiply,  # export = load * ndr
                    rasters=[f_reg['sub_load_n_path'], f_reg['sub_ndr_n_path']],
                    target_path=f_reg['n_subsurface_export_path'],
                    target_nodata=_TARGET_NODATA),
                target_path_list=[f_reg['n_subsurface_export_path']],
                dependent_task_list=[
                    subsurface_load_task, subsurface_ndr_task],
                task_name='subsurface export n')

            # only need to calculate total for nitrogen because
            # phosphorus only has surface export
            total_export_task = task_graph.add_task(
                func=pygeoprocessing.raster_map,
                kwargs=dict(
                    op=_sum_op,
                    rasters=[surface_export_path, f_reg['n_subsurface_export_path']],
                    target_path=f_reg['n_total_export_path'],
                    target_nodata=_TARGET_NODATA),
                target_path_list=[f_reg['n_total_export_path']],
                dependent_task_list=[
                    surface_export_task, subsurface_export_task],
                task_name='total export n')

            _ = task_graph.add_task(
                func=_aggregate_and_pickle_total,
                args=(
                    (f_reg['n_subsurface_export_path'], 1),
                    f_reg['watershed_results_ndr_path'],
                    f_reg['subsurface_export_n_pickle_path']),
                target_path_list=[f_reg['subsurface_export_n_pickle_path']],
                dependent_task_list=[
                    subsurface_export_task, create_vector_task],
                task_name='aggregate n subsurface export')

            _ = task_graph.add_task(
                func=_aggregate_and_pickle_total,
                args=(
                    (f_reg['n_total_export_path'], 1),
                    f_reg['watershed_results_ndr_path'],
                    f_reg['total_export_n_pickle_path']),
                target_path_list=[
                    f_reg[f'total_export_{nutrient}_pickle_path']],
                dependent_task_list=[total_export_task, create_vector_task],
                task_name='aggregate n total export')

            _ = task_graph.add_task(
                func=_aggregate_and_pickle_total,
                args=(
                    (f_reg['sub_load_n_path'], 1),
                    f_reg['watershed_results_ndr_path'],
                    f_reg[f'subsurface_load_{nutrient}_pickle_path']),
                target_path_list=[
                    f_reg[f'subsurface_load_{nutrient}_pickle_path']],
                dependent_task_list=[subsurface_load_task, create_vector_task],
                task_name=f'aggregate {nutrient} subsurface load')

            field_pickle_map['n_subsurface_export'] = f_reg[
                'subsurface_export_n_pickle_path']
            field_pickle_map['n_total_export'] = f_reg[
                'total_export_n_pickle_path']
            field_pickle_map['n_subsurface_load'] = f_reg[
                'subsurface_load_n_pickle_path']

        _ = task_graph.add_task(
            func=_aggregate_and_pickle_total,
            args=(
                (surface_export_path, 1), f_reg['watershed_results_ndr_path'],
                f_reg[f'surface_export_{nutrient}_pickle_path']),
            target_path_list=[f_reg[f'surface_export_{nutrient}_pickle_path']],
            dependent_task_list=[surface_export_task, create_vector_task],
            task_name=f'aggregate {nutrient} export')

        _ = task_graph.add_task(
            func=_aggregate_and_pickle_total,
            args=(
                (surface_load_path, 1), f_reg['watershed_results_ndr_path'],
                f_reg[f'surface_load_{nutrient}_pickle_path']),
            target_path_list=[f_reg[f'surface_load_{nutrient}_pickle_path']],
            dependent_task_list=[surface_load_task, create_vector_task],
            task_name=f'aggregate {nutrient} surface load')

    task_graph.close()
    task_graph.join()

    LOGGER.info('Writing summaries to output shapefile')
    _add_fields_to_shapefile(
        field_pickle_map, f_reg['watershed_results_ndr_path'])

    LOGGER.info(r'NDR complete!')
    LOGGER.info(r'  _   _    ____    ____     ')
    LOGGER.info(r' | \ |"|  |  _"\U |  _"\ u  ')
    LOGGER.info(r'<|  \| |>/| | | |\| |_) |/  ')
    LOGGER.info(r'U| |\  |uU| |_| |\|  _ <    ')
    LOGGER.info(r' |_| \_|  |____/ u|_| \_\   ')
    LOGGER.info(r' ||   \\,-.|||_   //   \\_  ')
    LOGGER.info(r' (_")  (_/(__)_) (__)  (__) ')


# raster_map equation: Multiply a series of arrays element-wise
def _mult_op(*array_list): return numpy.prod(numpy.stack(array_list), axis=0)

# raster_map equation: Sum a list of arrays element-wise
def _sum_op(*array_list): return numpy.sum(array_list, axis=0)

# raster_map equation: calculate inverse of S factor
def _inverse_op(base_val): return numpy.where(base_val == 0, 0, 1 / base_val)

# raster_map equation: rescale and threshold slope between 0.005 and 1
def _slope_proportion_and_threshold_op(slope):
    slope_fraction = slope / 100
    slope_fraction[slope_fraction < 0.005] = 0.005
    slope_fraction[slope_fraction > 1] = 1
    return slope_fraction


def _create_mask_raster(source_raster_path, source_vector_path,
                        target_raster_path):
    """Create a mask raster from a vector.

    Masking like this is more tolerant of geometry errors than using gdalwarp's
    cutline functionality, which fails on even simple geometry errors.

    Args:
        source_raster_path (str): The path to a source raster from which the
            raster size, geotransform and spatial reference will be copied.
        source_vector_path (str): The path to a vector on disk to be
            rasterized onto a new raster matching the attributes of the raster
            at ``source_raster_path``.
        target_raster_path (str): The path to where the output raster should be
            written.

    Returns:
        ``None``
    """
    pygeoprocessing.new_raster_from_base(
        source_raster_path, target_raster_path, gdal.GDT_Byte, [255], [0])
    pygeoprocessing.rasterize(source_vector_path, target_raster_path, [1],
                              option_list=['ALL_TOUCHED=FALSE'])


def _mask_raster(source_raster_path, mask_raster_path,
                 target_masked_raster_path, default_nodata, target_dtype):
    """Using a raster of 1s and 0s, determine which pixels remain in output.

    Args:
        source_raster_path (str): The path to a source raster that contains
            pixel values, some of which will propagate through to the target
            raster.
        mask_raster_path (str): The path to a raster of 1s and 0s indicating
            whether a pixel should (1) or should not (0) be copied to the
            target raster.
        target_masked_raster_path (str): The path to where the target raster
            should be written.
        default_nodata (int, float, None): The nodata value that should be used
            if ``source_raster_path`` does not have a defined nodata value.
        target_dtype (int): The ``gdal.GDT_*`` datatype of the target raster.

    Returns:
        ``None``
    """
    source_raster_info = pygeoprocessing.get_raster_info(source_raster_path)
    source_nodata = source_raster_info['nodata'][0]
    nodata = source_nodata
    if nodata is None:
        nodata = default_nodata

    def _mask_op(mask, raster):
        result = numpy.full(mask.shape, nodata,
                            dtype=source_raster_info['numpy_type'])
        valid_pixels = (
            ~pygeoprocessing.array_equals_nodata(raster, nodata) &
            (mask == 1))
        result[valid_pixels] = raster[valid_pixels]
        return result

    pygeoprocessing.raster_calculator(
        [(mask_raster_path, 1), (source_raster_path, 1)], _mask_op,
        target_masked_raster_path, target_dtype, nodata)


def _add_fields_to_shapefile(field_pickle_map, target_vector_path):
    """Add fields and values to an OGR layer open for writing.

    Args:
        field_pickle_map (dict): maps field name to a pickle file that is a
            result of pygeoprocessing.zonal_stats with FIDs that match
            `target_vector_path`. Fields will be written in the order they
            appear in this dictionary.
        target_vector_path (string): path to target vector file.
    Returns:
        None.
    """
    target_vector = gdal.OpenEx(
        target_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_layer = target_vector.GetLayer()
    field_summaries = {}
    for field_name, pickle_file_name in field_pickle_map.items():
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_def.SetWidth(24)
        field_def.SetPrecision(11)
        target_layer.CreateField(field_def)
        with open(pickle_file_name, 'rb') as pickle_file:
            field_summaries[field_name] = pickle.load(pickle_file)

    for feature in target_layer:
        fid = feature.GetFID()
        for field_name in field_pickle_map:
            feature.SetField(
                field_name, float(field_summaries[field_name][fid]['sum']))
        # Save back to datasource
        target_layer.SetFeature(feature)
    target_layer = None
    target_vector = None


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
    spec_copy = copy.deepcopy(MODEL_SPEC['args'])
    # Check required fields given the state of ``calc_n`` and ``calc_p``
    nutrients_selected = []
    for nutrient_letter in ('n', 'p'):
        if f'calc_{nutrient_letter}' in args and args[f'calc_{nutrient_letter}']:
            nutrients_selected.append(nutrient_letter)

    for param in ['load', 'eff', 'crit_len']:
        for nutrient in nutrients_selected:
            spec_copy['biophysical_table_path']['columns'][f'{param}_{nutrient}'] = (
                spec_copy['biophysical_table_path']['columns'][f'{param}_[NUTRIENT]'])
            spec_copy['biophysical_table_path']['columns'][f'{param}_{nutrient}']['required'] = True
        spec_copy['biophysical_table_path']['columns'].pop(f'{param}_[NUTRIENT]')

    if 'n' in nutrients_selected:
        spec_copy['biophysical_table_path']['columns']['proportion_subsurface_n'][
            'required'] = True

    validation_warnings = validation.validate(
        args, spec_copy, MODEL_SPEC['args_with_spatial_overlap'])

    if not nutrients_selected:
        validation_warnings.append(
            (['calc_n', 'calc_p'], MISSING_NUTRIENT_MSG))

    return validation_warnings


def _normalize_raster(base_raster_path_band, target_normalized_raster_path):
    """Calculate normalize raster by dividing by the mean value.

    Args:
        base_raster_path_band (tuple): raster path/band tuple to calculate
            mean.
        target_normalized_raster_path (string): path to target normalized
            raster from base_raster_path_band.

    Returns:
        None.

    """
    value_sum, value_count = pygeoprocessing.raster_reduce(
        function=lambda sum_count, block:  # calculate both in one pass
            (sum_count[0] + numpy.sum(block), sum_count[1] + block.size),
        raster_path_band=base_raster_path_band,
        initializer=(0, 0))

    value_mean = value_sum
    if value_count > 0:
        value_mean /= value_count

    pygeoprocessing.raster_map(
        op=lambda array: array if value_mean == 0 else array / value_mean,
        rasters=[base_raster_path_band[0]],
        target_path=target_normalized_raster_path,
        target_dtype=numpy.float32)


def _calculate_load(lulc_raster_path, lucode_to_load, target_load_raster):
    """Calculate load raster by mapping landcover and multiplying by area.

    Args:
        lulc_raster_path (string): path to integer landcover raster.
        lucode_to_load (dict): a mapping of landcover IDs to per-area
            nutrient load.
        target_load_raster (string): path to target raster that will have
            total load per pixel.

    Returns:
        None.

    """
    cell_area_ha = abs(numpy.prod(pygeoprocessing.get_raster_info(
        lulc_raster_path)['pixel_size'])) * 0.0001

    def _map_load_op(lucode_array):
        """Convert unit load to total load & handle nodata."""
        result = numpy.empty(lucode_array.shape)
        for lucode in numpy.unique(lucode_array):
            try:
                result[lucode_array == lucode] = (
                    lucode_to_load[lucode] * cell_area_ha)
            except KeyError:
                raise KeyError(
                    'lucode: %d is present in the landuse raster but '
                    'missing from the biophysical table' % lucode)
        return result

    pygeoprocessing.raster_map(
        op=_map_load_op,
        rasters=[lulc_raster_path],
        target_path=target_load_raster,
        target_dtype=numpy.float32,
        target_nodata=_TARGET_NODATA)


def _map_surface_load(
        modified_load_path, lulc_raster_path, lucode_to_subsurface_proportion,
        target_surface_load_path):
    """Calculate surface load from landcover raster.

    Args:
        modified_load_path (string): path to modified load raster with units
            of kg/pixel.
        lulc_raster_path (string): path to landcover raster.
        lucode_to_subsurface_proportion (dict): maps landcover codes to
            subsurface proportion values. Or if None, no subsurface transfer
            is mapped.
        target_surface_load_path (string): path to target raster.

    Returns:
        None.

    """
    if lucode_to_subsurface_proportion is not None:
        keys = sorted(lucode_to_subsurface_proportion.keys())
        subsurface_values = numpy.array(
            [lucode_to_subsurface_proportion[x] for x in keys])

    def _map_surface_load_op(lucode_array, modified_load_array):
        """Convert unit load to total load & handle nodata."""
        # If we don't have subsurface, just return 0.0.
        if lucode_to_subsurface_proportion is None:
            return modified_load_array
        index = numpy.digitize(lucode_array.ravel(), keys, right=True)
        return modified_load_array * (1 - subsurface_values[index])

    pygeoprocessing.raster_map(
        op=_map_surface_load_op,
        rasters=[lulc_raster_path, modified_load_path],
        target_path=target_surface_load_path,
        target_dtype=numpy.float32,
        target_nodata=_TARGET_NODATA)


def _map_subsurface_load(
        modified_load_path, lulc_raster_path, proportion_subsurface_map,
        target_sub_load_path):
    """Calculate subsurface load from landcover raster.

    Args:
        modified_load_path (string): path to modified load raster.
        lulc_raster_path (string): path to landcover raster.
        proportion_subsurface_map (dict): maps each landcover code to its
            subsurface permeance value.
        target_sub_load_path (string): path to target raster.

    Returns:
        None.

    """
    keys = sorted(numpy.array(list(proportion_subsurface_map)))
    subsurface_permeance_values = numpy.array(
        [proportion_subsurface_map[x] for x in keys])
    pygeoprocessing.raster_map(
        op=lambda lulc, modified_load: (
            modified_load * subsurface_permeance_values[
                numpy.digitize(lulc.ravel(), keys, right=True)]),
        rasters=[lulc_raster_path, modified_load_path],
        target_path=target_sub_load_path,
        target_dtype=numpy.float32,
        target_nodata=_TARGET_NODATA)


def _map_lulc_to_val_mask_stream(
        lulc_raster_path, stream_path, lucodes_to_vals, target_eff_path):
    """Make retention efficiency raster from landcover.

    Args:
        lulc_raster_path (string): path to landcover raster.
        stream_path (string) path to stream layer 0, no stream 1 stream.
        lucodes_to_val (dict) mapping of landcover codes to values
        target_eff_path (string): target raster that contains the mapping of
            landcover codes to retention efficiency values except where there
            is a stream in which case the retention efficiency is 0.

    Returns:
        None.

    """
    lucodes = sorted(lucodes_to_vals.keys())
    values = numpy.array([lucodes_to_vals[x] for x in lucodes])
    pygeoprocessing.raster_map(
        op=lambda lulc, stream: (
            values[numpy.digitize(lulc.ravel(), lucodes, right=True)] *
            (1 - stream)),
        rasters=[lulc_raster_path, stream_path],
        target_path=target_eff_path,
        target_dtype=numpy.float32,
        target_nodata=_TARGET_NODATA)


def d_up_calculation(s_bar_path, flow_accum_path, target_d_up_path):
    """Calculate d_up = s_bar * sqrt(upslope area)."""
    cell_area_m2 = abs(numpy.prod(pygeoprocessing.get_raster_info(
        s_bar_path)['pixel_size']))
    pygeoprocessing.raster_map(
        op=lambda s_bar, flow_accum: (
            s_bar * numpy.sqrt(flow_accum * cell_area_m2)),
        rasters=[s_bar_path, flow_accum_path],
        target_path=target_d_up_path,
        target_dtype=numpy.float32,
        target_nodata=_TARGET_NODATA)


def calculate_ic(d_up_path, d_dn_path, target_ic_path):
    """Calculate IC as log_10(d_up/d_dn)."""
    ic_nodata = float(numpy.finfo(numpy.float32).min)
    d_up_nodata = pygeoprocessing.get_raster_info(d_up_path)['nodata'][0]
    d_dn_nodata = pygeoprocessing.get_raster_info(d_dn_path)['nodata'][0]

    def _ic_op(d_up, d_dn):
        """Calculate IC0."""
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(d_up, d_up_nodata) &
            ~pygeoprocessing.array_equals_nodata(d_dn, d_dn_nodata) & (d_up != 0) &
            (d_dn != 0))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = ic_nodata
        result[valid_mask] = numpy.log10(d_up[valid_mask] / d_dn[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(d_up_path, 1), (d_dn_path, 1)], _ic_op,
        target_ic_path, gdal.GDT_Float32, ic_nodata)


def _calculate_ndr(
        effective_retention_path, ic_factor_path, k_param, target_ndr_path):
    """Calculate NDR as a function of Equation 4 in the user's guide."""
    ic_factor_raster = gdal.OpenEx(ic_factor_path, gdal.OF_RASTER)
    ic_factor_band = ic_factor_raster.GetRasterBand(1)
    ic_min, ic_max, _, _ = ic_factor_band.GetStatistics(0, 1)
    ic_factor_band = None
    ic_factor_raster = None
    ic_0_param = (ic_min + ic_max) / 2

    pygeoprocessing.raster_map(
        op=lambda eff, ic: ((1 - eff) /
                            (1 + numpy.exp((ic_0_param - ic) / k_param))),
        rasters=[effective_retention_path, ic_factor_path],
        target_path=target_ndr_path,
        target_nodata=_TARGET_NODATA)


def _calculate_sub_ndr(
        eff_sub, crit_len_sub, dist_to_channel_path, target_sub_ndr_path):
    """Calculate subsurface: subndr = eff_sub(1-e^(-5*l/crit_len)."""
    pygeoprocessing.raster_map(
        op=lambda dist_to_channel: (
            1 - eff_sub *
            (1 - numpy.exp(-5 * dist_to_channel / crit_len_sub))),
        rasters=[dist_to_channel_path],
        target_path=target_sub_ndr_path,
        target_nodata=_TARGET_NODATA)


def _aggregate_and_pickle_total(
        base_raster_path_band, aggregate_vector_path, target_pickle_path):
    """Aggregate base raster path to vector path FIDs and pickle result.

    Args:
        base_raster_path_band (tuple): raster/path band to aggregate over.
        aggregate_vector_path (string): path to vector to use geometry to
            aggregate over.
        target_pickle_path (string): path to a file that will contain the
            result of a pygeoprocessing.zonal_statistics call over
            base_raster_path_band from aggregate_vector_path.

    Returns:
        None.

    """
    result = pygeoprocessing.zonal_statistics(
        base_raster_path_band, aggregate_vector_path,
        working_dir=os.path.dirname(target_pickle_path))

    with open(target_pickle_path, 'wb') as target_pickle_file:
        pickle.dump(result, target_pickle_file)


def create_vector_copy(base_vector_path, target_vector_path):
    """Create a copy of base vector."""
    if os.path.exists(target_vector_path):
        os.remove(target_vector_path)

    base_wkt = pygeoprocessing.get_vector_info(
        base_vector_path)['projection_wkt']
    # use reproject_vector to create a copy in geopackage format
    # keeping the original projection
    pygeoprocessing.reproject_vector(
        base_vector_path, base_wkt, target_vector_path, driver_name='GPKG')
