"""InVEST Nutrient Delivery Ratio (NDR) module."""
import copy
import logging
import os
import pickle

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph
from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr

from .. import gettext
from .. import spec
from .. import utils
from .. import validation
from ..sdr import sdr
from ..file_registry import FileRegistry
from ..unit_registry import u
from . import ndr_core

LOGGER = logging.getLogger(__name__)

MISSING_NUTRIENT_MSG = gettext('Either calc_n or calc_p must be True')

MODEL_SPEC = spec.ModelSpec(
    model_id="ndr",
    model_title=gettext("Nutrient Delivery Ratio"),
    userguide="ndr.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=(),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["dem_path", "lulc_path", "runoff_proxy_path",
         "watersheds_path", "biophysical_table_path"],
        ["calc_p"],
        ["calc_n", "subsurface_critical_length_n", "subsurface_eff_n"],
        ["flow_dir_algorithm", "threshold_flow_accumulation",
         "k_param", "runoff_proxy_av"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.PROJECTED_DEM,
        spec.SingleBandRasterInput(
            id="lulc_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes. Each land use/land cover type must be"
                " assigned a unique integer code. All values in this raster must have"
                " corresponding entries in the Biophysical table."
            ),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="runoff_proxy_path",
            name=gettext("nutrient runoff proxy"),
            about=gettext(
                "Map of runoff potential, the capacity to transport nutrients downslope."
                " This can be a quickflow index or annual precipitation. Any units are"
                " allowed since the values will be normalized by their average."
            ),
            data_type=float,
            units=u.none,
            projected=None
        ),
        spec.VectorInput(
            id="watersheds_path",
            name=gettext("watersheds"),
            about=gettext(
                "Map of the boundaries of the watershed(s) over which to aggregate the"
                " model results."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[],
            projected=True
        ),
        spec.CSVInput(
            id="biophysical_table_path",
            name=gettext("biophysical table"),
            about=gettext(
                "A table mapping each LULC class to its biophysical properties related to"
                " nutrient load and retention. Nitrogen data must be provided if"
                " Calculate Nitrogen is selected. Phosphorus data must be provided if"
                " Calculate Phosphorus is selected. All LULC codes in the LULC raster"
                " must have corresponding entries in this table."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.OptionStringInput(
                    id="load_type_p",
                    about=(
                        "Whether the nutrient load in column load_p should be treated as"
                        " nutrient application rate or measured contaminant runoff."
                    ),
                    required="calc_p",
                    options=[
                        spec.Option(
                            key="application-rate",
                            about=(
                                "Treat the load value as nutrient application rates"
                                " (e.g. fertilizer, livestock waste, ...).The model will"
                                " adjust the load using the application rate and"
                                " retention efficiency: load_p * (1 - eff_p).")),
                        spec.Option(
                            key="measured-runoff",
                            about="Treat the load value as measured contaminant runoff.")
                    ]
                ),
                spec.OptionStringInput(
                    id="load_type_n",
                    about=(
                        "Whether the nutrient load in column load_n should be treated as"
                        " nutrient application rate or measured contaminant runoff."
                    ),
                    required="calc_n",
                    options=[
                        spec.Option(
                            key="application-rate",
                            about=(
                                "Treat the load values as nutrient application rates"
                                " (e.g. fertilizer, livestock waste, ...).The model will"
                                " adjust the load using the application rate and"
                                " retention efficiency: load_n * (1 - eff_n).")),
                        spec.Option(
                            key="measured-runoff",
                            about="Treat the load values as measured contaminant runoff.")
                    ]
                ),
                spec.NumberInput(
                    id="load_n",
                    about=gettext("The nitrogen loading for this land use class."),
                    required="calc_n",
                    units=u.kilogram / u.hectare / u.year
                ),
                spec.NumberInput(
                    id="load_p",
                    about=gettext("The phosphorus loading for this land use class."),
                    required="calc_p",
                    units=u.kilogram / u.hectare / u.year
                ),
                spec.RatioInput(
                    id="eff_n",
                    about=gettext(
                        "Maximum nitrogen retention efficiency. This is the maximum"
                        " proportion of the nitrogen that is retained on this LULC class."
                    ),
                    required="calc_n",
                    units=None
                ),
                spec.RatioInput(
                    id="eff_p",
                    about=gettext(
                        "Maximum phosphorus retention efficiency. This is the maximum"
                        " proportion of the phosphorus that is retained on this LULC"
                        " class."
                    ),
                    required="calc_p",
                    units=None
                ),
                spec.NumberInput(
                    id="crit_len_n",
                    about=gettext(
                        "The distance after which it is assumed that this LULC type"
                        " retains nitrogen at its maximum capacity."
                    ),
                    required="calc_n",
                    units=u.meter
                ),
                spec.NumberInput(
                    id="crit_len_p",
                    about=gettext(
                        "The distance after which it is assumed that this LULC type"
                        " retains phosphorus at its maximum capacity."
                    ),
                    required="calc_p",
                    units=u.meter
                ),
                spec.RatioInput(
                    id="proportion_subsurface_n",
                    about=gettext(
                        "The proportion of the total amount of nitrogen that is dissolved"
                        " into the subsurface. By default, this value should be set to 0,"
                        " indicating that all nutrients are delivered via surface flow."
                        " There is no equivalent of this for phosphorus."
                    ),
                    required="calc_n",
                    units=None
                )
            ],
            index_col="lucode"
        ),
        spec.BooleanInput(
            id="calc_p",
            name=gettext("calculate phosphorus"),
            about=gettext("Calculate phosphorus retention and export.")
        ),
        spec.BooleanInput(
            id="calc_n",
            name=gettext("calculate nitrogen"),
            about=gettext("Calculate nitrogen retention and export.")
        ),
        spec.THRESHOLD_FLOW_ACCUMULATION,
        spec.NumberInput(
            id="k_param",
            name=gettext("Borselli k parameter"),
            about=gettext(
                "Calibration parameter that determines the shape of the relationship"
                " between hydrologic connectivity (the degree of connection from patches"
                " of land to the stream) and the nutrient delivery ratio (percentage of"
                " nutrient that actually reaches the stream)."
            ),
            units=u.none
        ),
        spec.NumberInput(
            id="runoff_proxy_av",
            name=gettext("average runoff proxy"),
            about=gettext(
                "This parameter allows the user to specify a predefined average value for"
                " the runoff proxy. This value is used to normalize the Runoff Proxy"
                " raster when calculating the Runoff Proxy Index (RPI). If a user does"
                " not specify the runoff proxy average, this value will be automatically"
                " calculated from the Runoff Proxy raster. The units will be the same as"
                " those in the Runoff Proxy raster."
            ),
            required=False,
            units=u.none,
            expression="value > 0"
        ),
        spec.NumberInput(
            id="subsurface_critical_length_n",
            name=gettext("subsurface critical length (nitrogen)"),
            about=gettext(
                "The distance traveled (subsurface and downslope) after which it is"
                " assumed that soil retains nitrogen at its maximum capacity. Required if"
                " Calculate Nitrogen is selected."
            ),
            required="calc_n",
            allowed="calc_n",
            units=u.meter
        ),
        spec.RatioInput(
            id="subsurface_eff_n",
            name=gettext("subsurface maximum retention efficiency (nitrogen)"),
            about=gettext(
                "The maximum nitrogen retention efficiency that can be reached through"
                " subsurface flow. This characterizes the retention due to biochemical"
                " degradation in soils. Required if Calculate Nitrogen is selected."
            ),
            required="calc_n",
            allowed="calc_n",
            units=None
        ),
        spec.FLOW_DIR_ALGORITHM
    ],
    outputs=[
        spec.VectorOutput(
            id="watershed_results_ndr",
            path="watershed_results_ndr.gpkg",
            about=gettext("Vector with aggregated nutrient model results per watershed."),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.NumberOutput(
                    id="p_surface_load",
                    about=gettext(
                        "Total phosphorus loads (sources) in the watershed, i.e. the sum"
                        " of the nutrient contribution from all surface LULC without"
                        " filtering by the landscape."
                    ),
                    units=u.kilogram / u.year
                ),
                spec.NumberOutput(
                    id="n_surface_load",
                    about=gettext(
                        "Total nitrogen loads (sources) in the watershed, i.e. the sum of"
                        " the nutrient contribution from all surface LULC without"
                        " filtering by the landscape."
                    ),
                    units=u.kilogram / u.year
                ),
                spec.NumberOutput(
                    id="n_subsurface_load",
                    about=gettext("Total subsurface nitrogen loads in the watershed."),
                    units=u.kilogram / u.year
                ),
                spec.NumberOutput(
                    id="p_surface_export",
                    about=gettext(
                        "Total phosphorus export from the watershed by surface flow."
                    ),
                    units=u.kilogram / u.year
                ),
                spec.NumberOutput(
                    id="n_surface_export",
                    about=gettext(
                        "Total nitrogen export from the watershed by surface flow."
                    ),
                    units=u.kilogram / u.year
                ),
                spec.NumberOutput(
                    id="n_subsurface_export",
                    about=gettext(
                        "Total nitrogen export from the watershed by subsurface flow."
                    ),
                    units=u.kilogram / u.year
                ),
                spec.NumberOutput(
                    id="n_total_export",
                    about=gettext(
                        "Total nitrogen export from the watershed by surface and"
                        " subsurface flow."
                    ),
                    units=u.kilogram / u.year
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="p_surface_export",
            path="p_surface_export.tif",
            about=gettext(
                "A pixel level map showing how much phosphorus from each pixel eventually"
                " reaches the stream by surface flow."
            ),
            data_type=float,
            units=u.kilogram / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="n_surface_export",
            path="n_surface_export.tif",
            about=gettext(
                "A pixel level map showing how much nitrogen from each pixel eventually"
                " reaches the stream by surface flow."
            ),
            data_type=float,
            units=u.kilogram / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="n_subsurface_export",
            path="n_subsurface_export.tif",
            about=gettext(
                "A pixel level map showing how much nitrogen from each pixel eventually"
                " reaches the stream by subsurface flow."
            ),
            data_type=float,
            units=u.kilogram / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="n_total_export",
            path="n_total_export.tif",
            about=gettext(
                "A pixel level map showing how much nitrogen from each pixel eventually"
                " reaches the stream by either flow."
            ),
            data_type=float,
            units=u.kilogram / u.hectare
        ),
        spec.STREAM,
        spec.SingleBandRasterOutput(
            id="mask",
            path="intermediate_outputs/watersheds_mask.tif",
            about=gettext("Watersheds mask raster"),
            data_type=int,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="crit_len_n",
            path="intermediate_outputs/crit_len_n.tif",
            about=gettext(
                "Nitrogen retention length, found in the biophysical table"
            ),
            data_type=float,
            units=u.meter
        ),
        spec.SingleBandRasterOutput(
            id="crit_len_p",
            path="intermediate_outputs/crit_len_p.tif",
            about=gettext(
                "Phosphorus retention length, found in the biophysical table"
            ),
            data_type=float,
            units=u.meter
        ),
        spec.SingleBandRasterOutput(
            id="d_dn",
            path="intermediate_outputs/d_dn.tif",
            about=gettext("Downslope factor of the index of connectivity"),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="d_up",
            path="intermediate_outputs/d_up.tif",
            about=gettext("Upslope factor of the index of connectivity"),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="dist_to_channel",
            path="intermediate_outputs/dist_to_channel.tif",
            about=gettext(
                "Average downslope distance from a pixel to the stream"
            ),
            data_type=float,
            units=u.pixel
        ),
        spec.SingleBandRasterOutput(
            id="eff_n",
            path="intermediate_outputs/eff_n.tif",
            about=gettext(
                "Raw per-landscape cover retention efficiency for nitrogen."
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="eff_p",
            path="intermediate_outputs/eff_p.tif",
            about=gettext(
                "Raw per-landscape cover retention efficiency for phosphorus"
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="effective_retention_n",
            path="intermediate_outputs/effective_retention_n.tif",
            about=gettext(
                "Effective nitrogen retention provided by the downslope flow path"
                " for each pixel"
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="effective_retention_p",
            path="intermediate_outputs/effective_retention_p.tif",
            about=gettext(
                "Effective phosphorus retention provided by the downslope flow"
                " path for each pixel"
            ),
            data_type=float,
            units=None
        ),
        spec.FLOW_ACCUMULATION.model_copy(update=dict(
            path="intermediate_outputs/flow_accumulation.tif")),
        spec.FLOW_DIRECTION.model_copy(update=dict(
            path="intermediate_outputs/flow_direction.tif")),
        spec.SingleBandRasterOutput(
            id="ic_factor",
            path="intermediate_outputs/ic_factor.tif",
            about=gettext("Index of connectivity"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="load_n",
            path="intermediate_outputs/load_n.tif",
            about=gettext("Nitrogen load (for surface transport) per pixel"),
            data_type=float,
            units=u.kilogram / u.year
        ),
        spec.SingleBandRasterOutput(
            id="load_p",
            path="intermediate_outputs/load_p.tif",
            about=gettext("Phosphorus load (for surface transport) per pixel"),
            data_type=float,
            units=u.kilogram / u.year
        ),
        spec.SingleBandRasterOutput(
            id="modified_load_n",
            path="intermediate_outputs/modified_load_n.tif",
            about=gettext("Raw nitrogen load scaled by the runoff proxy index."),
            data_type=float,
            units=u.kilogram / u.year
        ),
        spec.SingleBandRasterOutput(
            id="modified_load_p",
            path="intermediate_outputs/modified_load_p.tif",
            about=gettext(
                "Raw phosphorus load scaled by the runoff proxy index."
            ),
            data_type=float,
            units=u.kilogram / u.year
        ),
        spec.SingleBandRasterOutput(
            id="ndr_n",
            path="intermediate_outputs/ndr_n.tif",
            about=gettext("NDR values for nitrogen"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="ndr_p",
            path="intermediate_outputs/ndr_p.tif",
            about=gettext("NDR values for phosphorus"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="runoff_proxy_index",
            path="intermediate_outputs/runoff_proxy_index.tif",
            about=gettext(
                "Normalized values for the Runoff Proxy input to the model"
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="s_accumulation",
            path="intermediate_outputs/s_accumulation.tif",
            about=gettext("Flow accumulation weighted by slope"),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="s_bar",
            path="intermediate_outputs/s_bar.tif",
            about=gettext(
                "Average slope gradient of the upslope contributing area"
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="s_factor_inverse",
            path="intermediate_outputs/s_factor_inverse.tif",
            about=gettext("Inverse of slope"),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="sub_load_n",
            path="intermediate_outputs/sub_load_n.tif",
            about=gettext("Nitrogen loads for subsurface transport"),
            data_type=float,
            units=u.kilogram / u.year
        ),
        spec.SingleBandRasterOutput(
            id="sub_ndr_n",
            path="intermediate_outputs/sub_ndr_n.tif",
            about=gettext("Subsurface nitrogen NDR values"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="surface_load_n",
            path="intermediate_outputs/surface_load_n.tif",
            about=gettext("Above ground nitrogen loads"),
            data_type=float,
            units=u.kilogram / u.hectare / u.year
        ),
        spec.SingleBandRasterOutput(
            id="surface_load_p",
            path="intermediate_outputs/surface_load_p.tif",
            about=gettext("Above ground phosphorus loads"),
            data_type=float,
            units=u.kilogram / u.hectare / u.year
        ),
        spec.SingleBandRasterOutput(
            id="thresholded_slope",
            path="intermediate_outputs/thresholded_slope.tif",
            about=gettext(
                "Percent slope thresholded for correct calculation of IC."
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="what_drains_to_stream",
            path="intermediate_outputs/what_drains_to_stream.tif",
            about=gettext(
                "Map of which pixels drain to a stream. A value of 1 means that"
                " at least some of the runoff from that pixel drains to a stream"
                " in stream.tif. A value of 0 means that it does not drain at all"
                " to any stream in stream.tif."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="aligned_dem",
            path="intermediate_outputs/aligned_dem.tif",
            about=gettext(
                "Copy of the DEM clipped to the extent of the other inputs"
            ),
            data_type=float,
            units=u.meter
        ),
        spec.SingleBandRasterOutput(
            id="aligned_lulc",
            path="intermediate_outputs/aligned_lulc.tif",
            about=gettext(
                "Copy of the LULC clipped to the extent of the other inputs and"
                " reprojected to the DEM projection"
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="aligned_runoff_proxy",
            path="intermediate_outputs/aligned_runoff_proxy.tif",
            about=gettext(
                "Copy of the runoff proxy clipped to the extent of the other"
                " inputs and reprojected to the DEM projection"
            ),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="masked_dem",
            path="intermediate_outputs/masked_dem.tif",
            about=gettext(
                "DEM input masked to exclude pixels outside the watershed"
            ),
            data_type=float,
            units=u.meter
        ),
        spec.SingleBandRasterOutput(
            id="masked_lulc",
            path="intermediate_outputs/masked_lulc.tif",
            about=gettext(
                "LULC input masked to exclude pixels outside the watershed"
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="masked_runoff_proxy",
            path="intermediate_outputs/masked_runoff_proxy.tif",
            about=gettext(
                "Runoff proxy input masked to exclude pixels outside the"
                " watershed"
            ),
            data_type=float,
            units=u.none
        ),
        spec.FILLED_DEM.model_copy(update=dict(
            path="intermediate_outputs/filled_dem.tif")),
        spec.SLOPE.model_copy(update=dict(
            path="intermediate_outputs/slope.tif")),
        spec.FileOutput(
            id="subsurface_export_n_pickle",
            path="intermediate_outputs/subsurface_export_n.pickle",
            about=gettext(
                "Pickled zonal statistics of nitrogen subsurface export"
            )
        ),
        spec.FileOutput(
            id="subsurface_load_n_pickle",
            path="intermediate_outputs/subsurface_load_n.pickle",
            about=gettext("Pickled zonal statistics of nitrogen subsurface load")
        ),
        spec.FileOutput(
            id="surface_export_n_pickle",
            path="intermediate_outputs/surface_export_n.pickle",
            about=gettext("Pickled zonal statistics of nitrogen surface export")
        ),
        spec.FileOutput(
            id="surface_export_p_pickle",
            path="intermediate_outputs/surface_export_p.pickle",
            about=gettext(
                "Pickled zonal statistics of phosphorus surface export"
            )
        ),
        spec.FileOutput(
            id="surface_load_n_pickle",
            path="intermediate_outputs/surface_load_n.pickle",
            about=gettext("Pickled zonal statistics of nitrogen surface load")
        ),
        spec.FileOutput(
            id="surface_load_p_pickle",
            path="intermediate_outputs/surface_load_p.pickle",
            about=gettext("Pickled zonal statistics of phosphorus surface load")
        ),
        spec.FileOutput(
            id="total_export_n_pickle",
            path="intermediate_outputs/total_export_n.pickle",
            about=gettext("Pickled zonal statistics of total nitrogen export")
        ),
        spec.TASKGRAPH_CACHE
    ]
)

INTERMEDIATE_DIR_NAME = 'intermediate_outputs'
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
        args['runoff_proxy_av'] (number): (optional) The average runoff proxy.
            Used to calculate the runoff proxy index. If not specified,
            it will be automatically calculated.
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
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, f_reg, task_graph = MODEL_SPEC.setup(args)

    # Build up a list of nutrients to process based on what's checked on
    nutrients_to_process = []
    for nutrient_id in ['n', 'p']:
        if args['calc_' + nutrient_id]:
            nutrients_to_process.append(nutrient_id)

    biophysical_df = MODEL_SPEC.get_input(
        'biophysical_table_path').get_validated_dataframe(
        args['biophysical_table_path'])

    # Ensure that if user doesn't explicitly assign a value,
    # runoff_proxy_av = None
    runoff_proxy_av = args.get("runoff_proxy_av")
    runoff_proxy_av = float(runoff_proxy_av) if runoff_proxy_av else None

    # these are used for aggregation in the last step
    field_pickle_map = {}

    create_vector_task = task_graph.add_task(
        func=create_vector_copy,
        args=(args['watersheds_path'], f_reg['watershed_results_ndr']),
        target_path_list=[f_reg['watershed_results_ndr']],
        task_name='create target vector')

    dem_info = pygeoprocessing.get_raster_info(args['dem_path'])

    base_raster_list = [
        args['dem_path'], args['lulc_path'], args['runoff_proxy_path']]
    aligned_raster_list = [
        f_reg['aligned_dem'], f_reg['aligned_lulc'],
        f_reg['aligned_runoff_proxy']]
    align_raster_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            base_raster_list, aligned_raster_list,
            ['near']*len(base_raster_list), dem_info['pixel_size'],
            'intersection'),
        kwargs={
            'base_vector_path_list': [args['watersheds_path']],
            'raster_align_index': 0  # align to the grid of the DEM
        },
        target_path_list=aligned_raster_list,
        task_name='align rasters')

    # Since we mask multiple rasters using the same vector, we can just do the
    # rasterization once.  Calling pygeoprocessing.mask_raster() multiple times
    # unfortunately causes the rasterization to happen once per call.
    mask_task = task_graph.add_task(
        func=_create_mask_raster,
        kwargs={
            'source_raster_path': f_reg['aligned_dem'],
            'source_vector_path': args['watersheds_path'],
            'target_raster_path': f_reg['mask']
        },
        target_path_list=[f_reg['mask']],
        dependent_task_list=[align_raster_task],
        task_name='create watersheds mask'
    )
    mask_runoff_proxy_task = task_graph.add_task(
        func=_mask_raster,
        kwargs={
            'source_raster_path': f_reg['aligned_runoff_proxy'],
            'mask_raster_path': f_reg['mask'],
            'target_masked_raster_path': f_reg['masked_runoff_proxy'],
            'target_dtype': gdal.GDT_Float32,
            'target_nodata': _TARGET_NODATA,
        },
        dependent_task_list=[mask_task, align_raster_task],
        target_path_list=[f_reg['masked_runoff_proxy']],
        task_name='mask runoff proxy raster'
    )
    mask_dem_task = task_graph.add_task(
        func=_mask_raster,
        kwargs={
            'source_raster_path': f_reg['aligned_dem'],
            'mask_raster_path': f_reg['mask'],
            'target_masked_raster_path': f_reg['masked_dem'],
            'target_dtype': gdal.GDT_Float32,
            'target_nodata': float(numpy.finfo(numpy.float32).min),
        },
        dependent_task_list=[mask_task, align_raster_task],
        target_path_list=[f_reg['masked_dem']],
        task_name='mask dem raster'
    )
    mask_lulc_task = task_graph.add_task(
        func=_mask_raster,
        kwargs={
            'source_raster_path': f_reg['aligned_lulc'],
            'mask_raster_path': f_reg['mask'],
            'target_masked_raster_path': f_reg['masked_lulc'],
            'target_dtype': gdal.GDT_Int32,
            'target_nodata': numpy.iinfo(numpy.int32).min,
        },
        dependent_task_list=[mask_task, align_raster_task],
        target_path_list=[f_reg['masked_lulc']],
        task_name='mask lulc raster'
    )

    fill_pits_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (f_reg['masked_dem'], 1), f_reg['filled_dem']),
        kwargs={'working_dir': args['workspace_dir']},
        dependent_task_list=[align_raster_task, mask_dem_task],
        target_path_list=[f_reg['filled_dem']],
        task_name='fill pits')

    calculate_slope_task = task_graph.add_task(
        func=pygeoprocessing.calculate_slope,
        args=((f_reg['filled_dem'], 1), f_reg['slope']),
        target_path_list=[f_reg['slope']],
        dependent_task_list=[fill_pits_task],
        task_name='calculate slope')

    threshold_slope_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=_slope_proportion_and_threshold_op,
            rasters=[f_reg['slope']],
            target_path=f_reg['thresholded_slope']),
        target_path_list=[f_reg['thresholded_slope']],
        dependent_task_list=[calculate_slope_task],
        task_name='threshold slope')

    if args['flow_dir_algorithm'] == 'mfd':
        flow_dir_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_dir_mfd,
            args=(
                (f_reg['filled_dem'], 1), f_reg['flow_direction']),
            kwargs={'working_dir': args['workspace_dir']},
            dependent_task_list=[fill_pits_task],
            target_path_list=[f_reg['flow_direction']],
            task_name='flow dir')

        flow_accum_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_accumulation_mfd,
            args=(
                (f_reg['flow_direction'], 1),
                f_reg['flow_accumulation']),
            target_path_list=[f_reg['flow_accumulation']],
            dependent_task_list=[flow_dir_task],
            task_name='flow accum')

        stream_extraction_task = task_graph.add_task(
            func=pygeoprocessing.routing.extract_streams_mfd,
            args=(
                (f_reg['flow_accumulation'], 1),
                (f_reg['flow_direction'], 1),
                float(args['threshold_flow_accumulation']),
                f_reg['stream']),
            target_path_list=[f_reg['stream']],
            dependent_task_list=[flow_accum_task],
            task_name='stream extraction')
        s_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_accumulation_mfd,
            args=((f_reg['flow_direction'], 1), f_reg['s_accumulation']),
            kwargs={
                'weight_raster_path_band': (f_reg['thresholded_slope'], 1)},
            target_path_list=[f_reg['s_accumulation']],
            dependent_task_list=[flow_dir_task, threshold_slope_task],
            task_name='route s')
    else:  # D8
        flow_dir_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_dir_d8,
            args=(
                (f_reg['filled_dem'], 1), f_reg['flow_direction']),
            kwargs={'working_dir': args['workspace_dir']},
            dependent_task_list=[fill_pits_task],
            target_path_list=[f_reg['flow_direction']],
            task_name='flow dir')

        flow_accum_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_accumulation_d8,
            args=(
                (f_reg['flow_direction'], 1),
                f_reg['flow_accumulation']),
            target_path_list=[f_reg['flow_accumulation']],
            dependent_task_list=[flow_dir_task],
            task_name='flow accum')

        stream_extraction_task = task_graph.add_task(
            func=pygeoprocessing.routing.extract_streams_d8,
            kwargs=dict(
                flow_accum_raster_path_band=(f_reg['flow_accumulation'], 1),
                flow_threshold=float(args['threshold_flow_accumulation']),
                target_stream_raster_path=f_reg['stream']),
            target_path_list=[f_reg['stream']],
            dependent_task_list=[flow_accum_task],
            task_name='stream extraction')

        s_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_accumulation_d8,
            args=((f_reg['flow_direction'], 1), f_reg['s_accumulation']),
            kwargs={
                'weight_raster_path_band': (f_reg['thresholded_slope'], 1)},
            target_path_list=[f_reg['s_accumulation']],
            dependent_task_list=[flow_dir_task, threshold_slope_task],
            task_name='route s')

    runoff_proxy_index_task = task_graph.add_task(
        func=_normalize_raster,
        args=((f_reg['masked_runoff_proxy'], 1),
              f_reg['runoff_proxy_index']),
        kwargs={'user_provided_mean': runoff_proxy_av},
        target_path_list=[f_reg['runoff_proxy_index']],
        dependent_task_list=[align_raster_task, mask_runoff_proxy_task],
        task_name='runoff proxy mean')

    s_bar_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=numpy.divide,  # s_bar = s_accum / flow_accum
            rasters=[f_reg['s_accumulation'], f_reg['flow_accumulation']],
            target_path=f_reg['s_bar'],
            target_dtype=numpy.float32,
            target_nodata=_TARGET_NODATA),
        target_path_list=[f_reg['s_bar']],
        dependent_task_list=[s_task, flow_accum_task],
        task_name='calculate s bar')

    d_up_task = task_graph.add_task(
        func=d_up_calculation,
        args=(f_reg['s_bar'], f_reg['flow_accumulation'],
              f_reg['d_up']),
        target_path_list=[f_reg['d_up']],
        dependent_task_list=[s_bar_task, flow_accum_task],
        task_name='d up')

    s_inv_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=_inverse_op,
            rasters=[f_reg['thresholded_slope']],
            target_path=f_reg['s_factor_inverse'],
            target_nodata=_TARGET_NODATA),
        target_path_list=[f_reg['s_factor_inverse']],
        dependent_task_list=[threshold_slope_task],
        task_name='s inv')

    if args['flow_dir_algorithm'] == 'mfd':
        d_dn_task = task_graph.add_task(
            func=pygeoprocessing.routing.distance_to_channel_mfd,
            args=(
                (f_reg['flow_direction'], 1),
                (f_reg['stream'], 1),
                f_reg['d_dn']),
            kwargs={'weight_raster_path_band': (
                f_reg['s_factor_inverse'], 1)},
            dependent_task_list=[stream_extraction_task, s_inv_task],
            target_path_list=[f_reg['d_dn']],
            task_name='d dn')

        dist_to_channel_task = task_graph.add_task(
            func=pygeoprocessing.routing.distance_to_channel_mfd,
            args=(
                (f_reg['flow_direction'], 1),
                (f_reg['stream'], 1),
                f_reg['dist_to_channel']),
            dependent_task_list=[stream_extraction_task],
            target_path_list=[f_reg['dist_to_channel']],
            task_name='dist to channel')
    else: # D8
        d_dn_task = task_graph.add_task(
            func=pygeoprocessing.routing.distance_to_channel_d8,
            args=(
                (f_reg['flow_direction'], 1),
                (f_reg['stream'], 1),
                f_reg['d_dn']),
            kwargs={'weight_raster_path_band': (
                f_reg['s_factor_inverse'], 1)},
            dependent_task_list=[stream_extraction_task, s_inv_task],
            target_path_list=[f_reg['d_dn']],
            task_name='d dn')

        dist_to_channel_task = task_graph.add_task(
            func=pygeoprocessing.routing.distance_to_channel_d8,
            args=(
                (f_reg['flow_direction'], 1),
                (f_reg['stream'], 1),
                f_reg['dist_to_channel']),
            dependent_task_list=[stream_extraction_task],
            target_path_list=[f_reg['dist_to_channel']],
            task_name='dist to channel')

    _ = task_graph.add_task(
        func=sdr._calculate_what_drains_to_stream,
        args=(f_reg['flow_direction'],
              f_reg['dist_to_channel'],
              f_reg['what_drains_to_stream']),
        target_path_list=[f_reg['what_drains_to_stream']],
        dependent_task_list=[flow_dir_task, dist_to_channel_task],
        task_name='write mask of what drains to stream')

    ic_task = task_graph.add_task(
        func=calculate_ic,
        args=(
            f_reg['d_up'], f_reg['d_dn'], f_reg['ic_factor']),
        target_path_list=[f_reg['ic_factor']],
        dependent_task_list=[d_dn_task, d_up_task],
        task_name='calc ic')

    for nutrient in nutrients_to_process:
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
                f_reg['masked_lulc'],
                biophysical_df[
                    [f'load_{nutrient}', f'eff_{nutrient}',
                     f'load_type_{nutrient}']].to_dict('index'),
                nutrient,
                f_reg[f'load_{nutrient}']),
            dependent_task_list=[align_raster_task, mask_lulc_task],
            target_path_list=[f_reg[f'load_{nutrient}']],
            task_name=f'{nutrient} load')

        modified_load_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=_mult_op,
                rasters=[f_reg[f'load_{nutrient}'], f_reg['runoff_proxy_index']],
                target_path=f_reg[f'modified_load_{nutrient}'],
                target_nodata=_TARGET_NODATA),
            target_path_list=[f_reg[f'modified_load_{nutrient}']],
            dependent_task_list=[load_task, runoff_proxy_index_task],
            task_name=f'modified load {nutrient}')

        surface_load_task = task_graph.add_task(
            func=_map_surface_load,
            args=(f_reg[f'modified_load_{nutrient}'], f_reg['masked_lulc'],
                  subsurface_proportion_map, f_reg[f'surface_load_{nutrient}']),
            target_path_list=[f_reg[f'surface_load_{nutrient}']],
            dependent_task_list=[modified_load_task, align_raster_task],
            task_name=f'map surface load {nutrient}')

        eff_task = task_graph.add_task(
            func=_map_lulc_to_val_mask_stream,
            args=(
                f_reg['masked_lulc'], f_reg['stream'],
                biophysical_df[f'eff_{nutrient}'].to_dict(),
                f_reg[f'eff_{nutrient}']),
            target_path_list=[f_reg[f'eff_{nutrient}']],
            dependent_task_list=[align_raster_task, stream_extraction_task],
            task_name=f'ret eff {nutrient}')

        crit_len_task = task_graph.add_task(
            func=_map_lulc_to_val_mask_stream,
            args=(
                f_reg['masked_lulc'], f_reg['stream'],
                biophysical_df[f'crit_len_{nutrient}'].to_dict(),
                f_reg[f'crit_len_{nutrient}']),
            target_path_list=[f_reg[f'crit_len_{nutrient}']],
            dependent_task_list=[align_raster_task, stream_extraction_task],
            task_name=f'ret eff {nutrient}')

        ndr_eff_task = task_graph.add_task(
            func=ndr_core.ndr_eff_calculation,
            args=(
                f_reg['flow_direction'],
                f_reg['stream'], f_reg[f'eff_{nutrient}'],
                f_reg[f'crit_len_{nutrient}'],
                f_reg[f'effective_retention_{nutrient}'],
                args['flow_dir_algorithm']),
            target_path_list=[f_reg[f'effective_retention_{nutrient}']],
            dependent_task_list=[
                stream_extraction_task, eff_task, crit_len_task],
            task_name=f'eff ret {nutrient}')

        ndr_task = task_graph.add_task(
            func=_calculate_ndr,
            args=(
                f_reg[f'effective_retention_{nutrient}'], f_reg['ic_factor'],
                float(args['k_param']), f_reg[f'ndr_{nutrient}']),
            target_path_list=[f_reg[f'ndr_{nutrient}']],
            dependent_task_list=[ndr_eff_task, ic_task],
            task_name=f'calc ndr {nutrient}')

        surface_export_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=numpy.multiply,  # export = load * ndr
                rasters=[
                    f_reg[f'surface_load_{nutrient}'],
                    f_reg[f'ndr_{nutrient}']],
                target_path=f_reg[f'{nutrient}_surface_export'],
                target_nodata=_TARGET_NODATA),
            target_path_list=[f_reg[f'{nutrient}_surface_export']],
            dependent_task_list=[
                load_task, ndr_task, surface_load_task],
            task_name=f'surface export {nutrient}')

        field_pickle_map[f'{nutrient}_surface_load'] = (
            f_reg[f'surface_load_{nutrient}_pickle'])
        field_pickle_map[f'{nutrient}_surface_export'] = (
            f_reg[f'surface_export_{nutrient}_pickle'])

        # only calculate subsurface things for nitrogen
        if nutrient == 'n':
            proportion_subsurface_map = (
                biophysical_df['proportion_subsurface_n'].to_dict())
            subsurface_load_task = task_graph.add_task(
                func=_map_subsurface_load,
                args=(f_reg[f'modified_load_{nutrient}'], f_reg['masked_lulc'],
                      proportion_subsurface_map, f_reg['sub_load_n']),
                target_path_list=[f_reg['sub_load_n']],
                dependent_task_list=[modified_load_task, align_raster_task],
                task_name='map subsurface load n')

            subsurface_ndr_task = task_graph.add_task(
                func=_calculate_sub_ndr,
                args=(
                    float(args['subsurface_eff_n']),
                    float(args['subsurface_critical_length_n']),
                    f_reg['dist_to_channel'], f_reg['sub_ndr_n']),
                target_path_list=[f_reg['sub_ndr_n']],
                dependent_task_list=[dist_to_channel_task],
                task_name='sub ndr n')

            subsurface_export_task = task_graph.add_task(
                func=pygeoprocessing.raster_map,
                kwargs=dict(
                    op=numpy.multiply,  # export = load * ndr
                    rasters=[f_reg['sub_load_n'], f_reg['sub_ndr_n']],
                    target_path=f_reg['n_subsurface_export'],
                    target_nodata=_TARGET_NODATA),
                target_path_list=[f_reg['n_subsurface_export']],
                dependent_task_list=[
                    subsurface_load_task, subsurface_ndr_task],
                task_name='subsurface export n')

            # only need to calculate total for nitrogen because
            # phosphorus only has surface export
            total_export_task = task_graph.add_task(
                func=pygeoprocessing.raster_map,
                kwargs=dict(
                    op=_sum_op,
                    rasters=[f_reg[f'{nutrient}_surface_export'], f_reg['n_subsurface_export']],
                    target_path=f_reg['n_total_export'],
                    target_nodata=_TARGET_NODATA),
                target_path_list=[f_reg['n_total_export']],
                dependent_task_list=[
                    surface_export_task, subsurface_export_task],
                task_name='total export n')

            _ = task_graph.add_task(
                func=_aggregate_and_pickle_total,
                args=(
                    (f_reg['n_subsurface_export'], 1),
                    f_reg['watershed_results_ndr'],
                    f_reg['subsurface_export_n_pickle']),
                target_path_list=[f_reg['subsurface_export_n_pickle']],
                dependent_task_list=[
                    subsurface_export_task, create_vector_task],
                task_name='aggregate n subsurface export')

            _ = task_graph.add_task(
                func=_aggregate_and_pickle_total,
                args=(
                    (f_reg['n_total_export'], 1),
                    f_reg['watershed_results_ndr'],
                    f_reg['total_export_n_pickle']),
                target_path_list=[
                    f_reg[f'total_export_{nutrient}_pickle']],
                dependent_task_list=[total_export_task, create_vector_task],
                task_name='aggregate n total export')

            _ = task_graph.add_task(
                func=_aggregate_and_pickle_total,
                args=(
                    (f_reg['sub_load_n'], 1),
                    f_reg['watershed_results_ndr'],
                    f_reg[f'subsurface_load_{nutrient}_pickle']),
                target_path_list=[
                    f_reg[f'subsurface_load_{nutrient}_pickle']],
                dependent_task_list=[subsurface_load_task, create_vector_task],
                task_name=f'aggregate {nutrient} subsurface load')

            field_pickle_map['n_subsurface_export'] = f_reg[
                'subsurface_export_n_pickle']
            field_pickle_map['n_total_export'] = f_reg[
                'total_export_n_pickle']
            field_pickle_map['n_subsurface_load'] = f_reg[
                'subsurface_load_n_pickle']

        _ = task_graph.add_task(
            func=_aggregate_and_pickle_total,
            args=(
                (f_reg[f'{nutrient}_surface_export'], 1), f_reg['watershed_results_ndr'],
                f_reg[f'surface_export_{nutrient}_pickle']),
            target_path_list=[f_reg[f'surface_export_{nutrient}_pickle']],
            dependent_task_list=[surface_export_task, create_vector_task],
            task_name=f'aggregate {nutrient} export')

        _ = task_graph.add_task(
            func=_aggregate_and_pickle_total,
            args=(
                (f_reg[f'surface_load_{nutrient}'], 1), f_reg['watershed_results_ndr'],
                f_reg[f'surface_load_{nutrient}_pickle']),
            target_path_list=[f_reg[f'surface_load_{nutrient}_pickle']],
            dependent_task_list=[surface_load_task, create_vector_task],
            task_name=f'aggregate {nutrient} surface load')

    task_graph.close()
    task_graph.join()

    LOGGER.info('Writing summaries to output shapefile')
    _add_fields_to_shapefile(
        field_pickle_map, f_reg['watershed_results_ndr'])

    LOGGER.info(r'NDR complete!')
    LOGGER.info(r'  _   _    ____    ____     ')
    LOGGER.info(r' | \ |"|  |  _"\U |  _"\ u  ')
    LOGGER.info(r'<|  \| |>/| | | |\| |_) |/  ')
    LOGGER.info(r'U| |\  |uU| |_| |\|  _ <    ')
    LOGGER.info(r' |_| \_|  |____/ u|_| \_\   ')
    LOGGER.info(r' ||   \\,-.|||_   //   \\_  ')
    LOGGER.info(r' (_")  (_/(__)_) (__)  (__) ')

    return f_reg.registry


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
                 target_masked_raster_path, target_nodata, target_dtype):
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
        target_nodata (int, float): The target nodata value that should match
            ``target_dtype``.
        target_dtype (int): The ``gdal.GDT_*`` datatype of the target raster.

    Returns:
        ``None``
    """
    source_raster_info = pygeoprocessing.get_raster_info(source_raster_path)
    source_nodata = source_raster_info['nodata'][0]
    target_numpy_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(target_dtype)

    def _mask_op(mask, raster):
        result = numpy.full(mask.shape, target_nodata,
                            dtype=target_numpy_dtype)
        valid_pixels = (
            ~pygeoprocessing.array_equals_nodata(raster, source_nodata) &
            (mask == 1))
        result[valid_pixels] = raster[valid_pixels]
        return result

    pygeoprocessing.raster_calculator(
        [(mask_raster_path, 1), (source_raster_path, 1)], _mask_op,
        target_masked_raster_path, target_dtype, target_nodata)


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
            # Since pixel values are kg/(ha•yr), raster sum is (kg•px)/(ha•yr).
            # To convert to kg/yr, multiply by ha/px.
            pixel_area = field_summaries[field_name]['pixel_area']
            ha_per_px = pixel_area / 10000
            feature.SetField(
                field_name, float(
                    field_summaries[field_name][fid]['sum']) * ha_per_px)
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
    spec_copy = copy.deepcopy(MODEL_SPEC)
    # Check required fields given the state of ``calc_n`` and ``calc_p``
    nutrients_selected = []
    for nutrient_letter in ('n', 'p'):
        if f'calc_{nutrient_letter}' in args and args[f'calc_{nutrient_letter}']:
            nutrients_selected.append(nutrient_letter)

    for param in ['load', 'eff', 'crit_len']:
        for nutrient in nutrients_selected:
            spec_copy.get_input('biophysical_table_path').get_column(
                f'{param}_{nutrient}').required = True

    if 'n' in nutrients_selected:
        spec_copy.get_input('biophysical_table_path').get_column(
            'proportion_subsurface_n').required = True

    validation_warnings = validation.validate(args, spec_copy)

    if not nutrients_selected:
        validation_warnings.append(
            (['calc_n', 'calc_p'], MISSING_NUTRIENT_MSG))

    return validation_warnings


def _normalize_raster(base_raster_path_band, target_normalized_raster_path,
                      user_provided_mean=None):
    """Calculate normalize raster by dividing by the mean value.

    Args:
        base_raster_path_band (tuple): raster path/band tuple to calculate
            mean.
        target_normalized_raster_path (string): path to target normalized
            raster from base_raster_path_band.
        user_provided_mean (float, optional): user-provided average.
            If provided, this value will be used instead of computing
            the mean from the raster.

    Returns:
        None.

    """
    if user_provided_mean is None:
        value_sum, value_count = pygeoprocessing.raster_reduce(
            function=lambda sum_count, block: (  # calculate both in one pass
                sum_count[0] + numpy.sum(block), sum_count[1] + block.size),
            raster_path_band=base_raster_path_band,
            initializer=(0, 0))
        value_mean = value_sum
        if value_count > 0:
            value_mean /= value_count
        LOGGER.info(f"Normalizing raster ({base_raster_path_band[0]}) using "
                    f"auto-calculated mean: {value_mean}")
    else:
        value_mean = user_provided_mean

    pygeoprocessing.raster_map(
        op=lambda array: array if value_mean == 0 else array / value_mean,
        rasters=[base_raster_path_band[0]],
        target_path=target_normalized_raster_path,
        target_dtype=numpy.float32)


def _calculate_load(
        lulc_raster_path, lucode_to_load, nutrient_type, target_load_raster):
    """Calculate load raster by mapping landcover.

    If load type is 'application-rate' adjust by ``1 - efficiency``.

    Args:
        lulc_raster_path (string): path to integer landcover raster.
        lucode_to_load (dict): a mapping of landcover IDs to nutrient load,
            efficiency, and load type. The load type value can be one of:
            [ 'measured-runoff' | 'appliation-rate' ].
        nutrient_type (str): the nutrient type key ('p' | 'n').
        target_load_raster (string): path to target raster that will have
            load values (kg/ha) mapped to pixels based on LULC.

    Returns:
        None.

    """
    app_rate = 'application-rate'
    measured_runoff = 'measured-runoff'
    load_key = f'load_{nutrient_type}'
    eff_key = f'eff_{nutrient_type}'
    load_type_key = f'load_type_{nutrient_type}'

    # Raise ValueError if unknown load_type
    for key, value in lucode_to_load.items():
        load_type = value[load_type_key]
        if not load_type in [app_rate, measured_runoff]:
            # unknown load type, raise ValueError
            raise ValueError(
                'nutrient load type must be: '
                f'"{app_rate}" | "{measured_runoff}". Instead '
                f'found value of: "{load_type}".')

    def _map_load_op(lucode_array):
        """Convert unit load to total load."""
        result = numpy.empty(lucode_array.shape)
        for lucode in numpy.unique(lucode_array):
            try:
                if lucode_to_load[lucode][load_type_key] == measured_runoff:
                    result[lucode_array == lucode] = (
                        lucode_to_load[lucode][load_key])
                elif lucode_to_load[lucode][load_type_key] == app_rate:
                    result[lucode_array == lucode] = (
                        lucode_to_load[lucode][load_key] * (
                            1 - lucode_to_load[lucode][eff_key]))
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
        modified_load_path (string): path to modified load raster.
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

    # Write pixel area to pickle file so that _add_fields_to_shapefile
    # can adjust totals as needed.
    raster_info = pygeoprocessing.get_raster_info(base_raster_path_band[0])
    pixel_area = abs(numpy.prod(raster_info['pixel_size']))
    result['pixel_area'] = pixel_area

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
