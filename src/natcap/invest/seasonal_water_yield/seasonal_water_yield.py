"""InVEST Seasonal Water Yield Model."""
import fractions
import logging
import os
import re
import warnings

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import scipy.special
import taskgraph
from osgeo import gdal
from osgeo import ogr

from .. import gettext
from .. import spec
from .. import utils
from .. import validation
from ..unit_registry import u
from . import seasonal_water_yield_core

LOGGER = logging.getLogger(__name__)

TARGET_NODATA = -1
N_MONTHS = 12
MONTH_ID_TO_LABEL = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
    'nov', 'dec']

MODEL_SPEC = spec.ModelSpec(
    model_id="seasonal_water_yield",
    model_title=gettext("Seasonal Water Yield"),
    userguide="seasonal_water_yield.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("swy",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["lulc_raster_path", "biophysical_table_path"],
        ["dem_raster_path", "aoi_path"],
        ["flow_dir_algorithm", "threshold_flow_accumulation", "beta_i", "gamma"],
        ["user_defined_local_recharge", "l_path", "et0_raster_table",
         "precip_raster_table", "soil_group_path"],
        ["monthly_alpha", "alpha_m", "monthly_alpha_path"],
        ["user_defined_climate_zones", "rain_events_table_path",
         "climate_zone_table_path", "climate_zone_raster_path"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.THRESHOLD_FLOW_ACCUMULATION,
        spec.CSVInput(
            id="et0_raster_table",
            name=gettext("ET0 table"),
            about=gettext(
                "Table mapping month indexes (1-12) to reference evapotranspiration"
                " raster paths. The paths may be either absolute or relative to the"
                " location of the ET0 table itself. Required if User-Defined Local"
                " Recharge is not selected."
            ),
            required="not user_defined_local_recharge",
            allowed="not user_defined_local_recharge",
            columns=[
                spec.IntegerInput(
                    id="month",
                    about=gettext("The month index (1-12)."),
                    expression="(value >= 1) & (value <= 12)"
                ),
                spec.SingleBandRasterInput(
                    id="path",
                    about=gettext(
                        "Path to a reference evapotranspiration raster for the month."
                    ),
                    data_type=float,
                    units=u.millimeter / u.month,
                    projected=None
                )
            ]
        ),
        spec.CSVInput(
            id="precip_raster_table",
            name=gettext("precipitation table"),
            about=gettext(
                "Table mapping month indexes (1-12) to monthly precipitation raster"
                " paths. The paths may be either absolute or relative to the location"
                " of the precipitation table itself. Required if User-Defined Local"
                " Recharge is not selected."
            ),
            required="not user_defined_local_recharge",
            allowed="not user_defined_local_recharge",
            columns=[
                spec.IntegerInput(
                    id="month",
                    about=gettext("The month index (1-12)."),
                    expression="(value >= 1) & (value <= 12)"
                ),
                spec.SingleBandRasterInput(
                    id="path",
                    about=gettext("Path to a precipitation raster for the month."),
                    data_type=float,
                    units=u.millimeter / u.month,
                    projected=None
                )
            ]
        ),
        spec.DEM.model_copy(update=dict(id="dem_raster_path")),
        spec.SingleBandRasterInput(
            id="lulc_raster_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes. Each land use/land cover type must be"
                " assigned a unique integer code. All values in this raster MUST have"
                " corresponding entries in the Biophysical Table."
            ),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.SOIL_GROUP.model_copy(update=dict(
            projected=True,
            required="not user_defined_local_recharge",
            allowed="not user_defined_local_recharge"
        )),
        spec.AOI.model_copy(update=dict(
            id="aoi_path",
            projected=True
        )),
        spec.CSVInput(
            id="biophysical_table_path",
            name=gettext("biophysical table"),
            about=gettext(
                "A table mapping each LULC code to biophysical properties of the"
                " corresponding LULC class. All values in the LULC raster must have"
                " corresponding entries in this table."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.NumberInput(
                    id="cn_[SOIL_GROUP]",
                    about=gettext(
                        "Curve number values for each combination of soil group and LULC"
                        " class. Replace [SOIL_GROUP] with each soil group code A, B, C,"
                        " D so that there is one column for each soil group. Curve number"
                        " values must be greater than 0 and less than or equal to 100."
                    ),
                    expression="0 < value <= 100",
                    units=u.none
                ),
                spec.NumberInput(
                    id="kc_[MONTH]",
                    about=gettext(
                        "Crop/vegetation coefficient (Kc) values for this LULC class in"
                        " each month. Replace [MONTH] with the numbers 1 to 12 so that"
                        " there is one column for each month."
                    ),
                    units=u.none
                )
            ],
            index_col="lucode"
        ),
        spec.CSVInput(
            id="rain_events_table_path",
            name=gettext("rain events table"),
            about=gettext(
                "A table containing the number of rain events for each month. Required if"
                " neither User-Defined Local Recharge nor User-Defined Climate Zones is"
                " selected."
            ),
            required=(
                "(not user_defined_local_recharge) & (not user_defined_climate_zones)"
            ),
            allowed="not user_defined_climate_zones",
            columns=[
                spec.NumberInput(
                    id="month",
                    about=gettext(
                        "Values are the numbers 1-12 corresponding to each month, January"
                        " (1) through December (12)."
                    ),
                    units=u.none
                ),
                spec.NumberInput(
                    id="events",
                    about=gettext("The number of rain events in that month."),
                    units=u.none
                )
            ],
            index_col="month"
        ),
        spec.StringInput(
            id="alpha_m",
            name=gettext("alpha_m parameter"),
            about=gettext(
                "The proportion of upslope annual available local recharge that is"
                " available in each month. Required if Use Monthly Alpha Table is not"
                " selected."
            ),
            required="not monthly_alpha",
            allowed="not monthly_alpha",
            regexp=None
        ),
        spec.RatioInput(
            id="beta_i",
            name=gettext("beta_i parameter"),
            about=gettext(
                "The proportion of the upgradient subsidy that is available for"
                " downgradient evapotranspiration."
            ),
            units=None
        ),
        spec.RatioInput(
            id="gamma",
            name=gettext("gamma parameter"),
            about=gettext(
                "The proportion of pixel local recharge that is available to downgradient"
                " pixels."
            ),
            units=None
        ),
        spec.BooleanInput(
            id="user_defined_local_recharge",
            name=gettext("user-defined recharge layer (advanced)"),
            about=gettext(
                "Use user-defined local recharge data instead of calculating local"
                " recharge from the other provided data."
            )
        ),
        spec.SingleBandRasterInput(
            id="l_path",
            name=gettext("local recharge"),
            about=gettext(
                "Map of local recharge data. Required if User-Defined Local Recharge is"
                " selected."
            ),
            required="user_defined_local_recharge",
            allowed="user_defined_local_recharge",
            data_type=float,
            units=u.millimeter,
            projected=True
        ),
        spec.BooleanInput(
            id="user_defined_climate_zones",
            name=gettext("climate zones (advanced)"),
            about=gettext(
                "Use user-defined climate zone data in lieu of a global rain events"
                " table."
            )
        ),
        spec.CSVInput(
            id="climate_zone_table_path",
            name=gettext("climate zone table"),
            about=gettext(
                "Table of monthly precipitation events for each climate zone. Required if"
                " User-Defined Climate Zones is selected."
            ),
            required="user_defined_climate_zones",
            allowed="user_defined_climate_zones",
            columns=[
                spec.IntegerInput(
                    id="cz_id",
                    about=gettext(
                        "Climate zone ID numbers, corresponding to the values in the"
                        " Climate Zones map."
                    )
                ),
                spec.NumberInput(
                    id="[MONTH]",
                    about=gettext(
                        "The number of rain events that occur in each month in this"
                        " climate zone. Replace [MONTH] with the month abbreviations:"
                        " jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec, so"
                        " that there is a column for each month."
                    ),
                    units=u.none
                )
            ],
            index_col="cz_id"
        ),
        spec.SingleBandRasterInput(
            id="climate_zone_raster_path",
            name=gettext("climate zone map"),
            about=gettext(
                "Map of climate zones. All values in this raster must have corresponding"
                " entries in the Climate Zone Table."
            ),
            required="user_defined_climate_zones",
            allowed="user_defined_climate_zones",
            data_type=int,
            units=None,
            projected=True
        ),
        spec.BooleanInput(
            id="monthly_alpha",
            name=gettext("use monthly alpha table (advanced)"),
            about=gettext(
                "Use montly alpha values instead of a single value for the whole year."
            )
        ),
        spec.CSVInput(
            id="monthly_alpha_path",
            name=gettext("monthly alpha table"),
            about=gettext(
                "Table of alpha values for each month. Required if Use Monthly Alpha"
                " Table is selected."
            ),
            required="monthly_alpha",
            allowed="monthly_alpha",
            columns=[
                spec.NumberInput(
                    id="month",
                    about=gettext(
                        "Values are the numbers 1-12 corresponding to each month."
                    ),
                    units=u.none
                ),
                spec.NumberInput(
                    id="alpha",
                    about=gettext("The alpha value for that month."),
                    units=u.none
                )
            ],
            index_col="month"
        ),
        spec.FLOW_DIR_ALGORITHM
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="b",
            path="B.tif",
            about=gettext(
                "Map of baseflow values, the contribution of a pixel to slow release flow"
                " (which is not evapotranspired before it reaches the stream)."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="b_sum",
            path="B_sum.tif",
            about=gettext(
                "Map of B_sum values, the flow through a pixel, contributed by all"
                " upslope pixels, that is not evapotranspirated before it reaches the"
                " stream."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="cn",
            path="CN.tif",
            about=gettext("Map of curve number values."),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="l_avail",
            path="L_avail.tif",
            about=gettext("Map of available local recharge"),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="l",
            path="L.tif",
            about=gettext("Map of local recharge values"),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="l_sum_avail",
            path="L_sum_avail.tif",
            about=gettext(
                "Map of total available water, contributed by all upslope pixels, that is"
                " available for evapotranspiration by this pixel."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="l_sum",
            path="L_sum.tif",
            about=gettext(
                "Map of cumulative upstream recharge: the flow through a pixel,"
                " contributed by all upslope pixels, that is available for"
                " evapotranspiration to downslope pixels."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="qf",
            path="QF.tif",
            about=gettext("Map of quickflow"),
            data_type=float,
            units=u.millimeter / u.year
        ),
        spec.STREAM,
        spec.SingleBandRasterOutput(
            id="annual_precip",
            path="P.tif",
            about=gettext("The total precipitation across all months on this pixel."),
            data_type=float,
            units=u.millimeter / u.year
        ),
        spec.SingleBandRasterOutput(
            id="vri",
            path="Vri.tif",
            about=gettext(
                "Map of the values of recharge (contribution, positive or negative), to"
                " the total recharge."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.VectorOutput(
            id="aggregate_vector",
            path="aggregated_results_swy.shp",
            about=gettext("Table of biophysical values for each watershed"),
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[
                spec.NumberOutput(
                    id="qb",
                    about=gettext("Mean local recharge value within the watershed"),
                    units=u.millimeter
                ),
                spec.NumberOutput(
                    id="vri_sum",
                    about=gettext(
                        "Total recharge contribution, (positive or negative) within the"
                        " watershed."
                    ),
                    units=u.millimeter
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="aet",
            path="intermediate_outputs/aet.tif",
            about=gettext("Map of actual evapotranspiration"),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="aetm_[MONTH]",
            path="intermediate_outputs/aetm_[MONTH].tif",
            about=gettext("Maps of monthly evapotranspiration"),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="flow_dir",
            path="intermediate_outputs/flow_dir.tif",
            about=gettext(
                "Map of flow direction, in either D8 or MFD format according to"
                " the option selected."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="qf_[MONTH]",
            path="intermediate_outputs/qf_[MONTH].tif",
            about=gettext(
                "Maps of monthly quickflow (1 = January… 12 = December)"
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="si",
            path="intermediate_outputs/Si.tif",
            about=gettext("Map of the S_i factor derived from CN"),
            data_type=float,
            units=u.inch
        ),
        spec.SingleBandRasterOutput(
            id="lulc_aligned",
            path="intermediate_outputs/lulc_aligned.tif",
            about=gettext(
                "Copy of LULC input, aligned and clipped to match the other"
                " spatial inputs"
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="dem_aligned",
            path="intermediate_outputs/dem_aligned.tif",
            about=gettext(
                "Copy of DEM input, aligned and clipped to match the other"
                " spatial inputs"
            ),
            data_type=float,
            units=u.meter
        ),
        spec.SingleBandRasterOutput(
            id="pit_filled_dem",
            path="intermediate_outputs/pit_filled_dem.tif",
            about=gettext("Pit filled DEM"),
            data_type=float,
            units=u.meter
        ),
        spec.SingleBandRasterOutput(
            id="soil_group_aligned",
            path="intermediate_outputs/soil_group_aligned.tif",
            about=gettext(
                "Copy of soil groups input, aligned and clipped to match the"
                " other spatial inputs"
            ),
            data_type=int,
            units=None
        ),
        spec.FLOW_ACCUMULATION.model_copy(update=dict(
            id="flow_accum",
            path="intermediate_outputs/flow_accum.tif")),
        spec.SingleBandRasterOutput(
            id="prcp_a[MONTH]",
            path="intermediate_outputs/prcp_a[MONTH].tif",
            about=gettext(
                "Monthly precipitation rasters, aligned and clipped to match the"
                " other spatial inputs"
            ),
            data_type=float,
            units=u.millimeter / u.year
        ),
        spec.SingleBandRasterOutput(
            id="n_events[MONTH]",
            path="intermediate_outputs/n_events[MONTH].tif",
            about=gettext("Map of monthly rain events"),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="et0_a[MONTH]",
            path="intermediate_outputs/et0_a[MONTH].tif",
            about=gettext(
                "Monthly ET0 rasters, aligned and clipped to match the other"
                " spatial inputs"
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="kc_[MONTH]",
            path="intermediate_outputs/kc_[MONTH].tif",
            about=gettext("Map of monthly KC values"),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="l_aligned",
            path="intermediate_outputs/l_aligned.tif",
            about=gettext(
                "Copy of user-defined local recharge input, aligned and clipped"
                " to match the other spatial inputs"
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="cz_aligned",
            path="intermediate_outputs/cz_aligned.tif",
            about=gettext(
                "Copy of user-defined climate zones raster, aligned and clipped"
                " to match the other spatial inputs"
            ),
            data_type=int,
            units=None
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def execute(args):
    """Seasonal Water Yield.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['threshold_flow_accumulation'] (number): used when classifying
            stream pixels from the DEM by thresholding the number of upslope
            cells that must flow into a cell before it's considered
            part of a stream.
        args['et0_raster_table'] (string): required if
            args['user_defined_local_recharge'] is False. Path to a CSV table
            that has headers 'month' (1-12) and 'path', mapping months to
            monthly reference evapotranspiration rasters. The rasters should
            have units in mm.
        args['precip_raster_table'] (string): required if
            args['user_defined_local_recharge'] is False. Path to a CSV table
            that has headers 'month' (1-12) and 'path', mapping months to
            monthly precipitation rasters. The rasters should have units in mm.
        args['dem_raster_path'] (string): a path to a digital elevation raster
        args['lulc_raster_path'] (string): a path to a land cover raster used
            to classify biophysical properties of pixels.
        args['soil_group_path'] (string): required if
            args['user_defined_local_recharge'] is  False. A path to a raster
            indicating SCS soil groups where integer values are mapped to soil
            types
        args['aoi_path'] (string): path to a vector that indicates the area
            over which the model should be run, as well as the area in which to
            aggregate over when calculating the output Qb.
        args['biophysical_table_path'] (string): path to a CSV table that maps
            landcover codes paired with soil group types to curve numbers as
            well as Kc values.  Headers must include 'lucode', 'CN_A', 'CN_B',
            'CN_C', 'CN_D', 'Kc_1', 'Kc_2', 'Kc_3', 'Kc_4', 'Kc_5', 'Kc_6',
            'Kc_7', 'Kc_8', 'Kc_9', 'Kc_10', 'Kc_11', 'Kc_12'.
        args['rain_events_table_path'] (string): Not required if
            args['user_defined_local_recharge'] is True or
            args['user_defined_climate_zones'] is True.  Path to a CSV table
            that has headers 'month' (1-12) and 'events' (int >= 0) that
            indicates the number of rain events per month
        args['alpha_m'] (float or string): required if args['monthly_alpha'] is
            false.  Is the proportion of upslope annual available local
            recharge that is available in month m.
        args['beta_i'] (float or string): is the fraction of the upgradient
            subsidy that is available for downgradient evapotranspiration.
        args['gamma'] (float or string): is the fraction of pixel local
            recharge that is available to downgradient pixels.
        args['user_defined_local_recharge'] (boolean): if True, indicates user
            will provide pre-defined local recharge raster layer
        args['l_path'] (string): required if
            args['user_defined_local_recharge'] is True.  If provided pixels
            indicate the amount of local recharge; units in mm.
        args['user_defined_climate_zones'] (boolean): if True, user provides
            a climate zone rain events table and a climate zone raster map in
            lieu of a global rain events table.
        args['climate_zone_table_path'] (string): required if
            args['user_defined_climate_zones'] is True. Contains monthly
            precipitation events per climate zone.  Fields must be
            "cz_id", "jan", "feb", "mar", "apr", "may", "jun", "jul",
            "aug", "sep", "oct", "nov", "dec".
        args['climate_zone_raster_path'] (string): required if
            args['user_defined_climate_zones'] is True, pixel values correspond
            to the "cz_id" values defined in args['climate_zone_table_path']
        args['monthly_alpha'] (boolean): if True, use the alpha
        args['monthly_alpha_path'] (string): required if args['monthly_alpha']
            is True. A CSV file.
        args['n_workers'] (int): (optional) indicates the number of processes
            to devote to potential parallel task execution. A value < 0 will
            use a single process, 0 will be non-blocking scheduling but
            single process, and >= 1 will make additional processes for
            parallel execution.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    LOGGER.info('prepare and test inputs for common errors')

    # fail early on a missing required rain events table
    if (not args['user_defined_local_recharge'] and
            not args['user_defined_climate_zones']):
        rain_events_df = MODEL_SPEC.get_input(
            'rain_events_table_path').get_validated_dataframe(
            args['rain_events_table_path'])

    biophysical_df = MODEL_SPEC.get_input(
        'biophysical_table_path').get_validated_dataframe(
        args['biophysical_table_path'])

    if args['monthly_alpha']:
        # parse out the alpha lookup table of the form (month_id: alpha_val)
        alpha_month_map = MODEL_SPEC.get_input(
            'monthly_alpha_path').get_validated_dataframe(
            args['monthly_alpha_path'])['alpha'].to_dict()
    else:
        # make all 12 entries equal to args['alpha_m']
        alpha_m = float(fractions.Fraction(args['alpha_m']))
        alpha_month_map = dict(
            (month_index+1, alpha_m) for month_index in range(N_MONTHS))

    beta_i = float(fractions.Fraction(args['beta_i']))
    gamma = float(fractions.Fraction(args['gamma']))
    threshold_flow_accumulation = args['threshold_flow_accumulation']
    pixel_size = pygeoprocessing.get_raster_info(
        args['dem_raster_path'])['pixel_size']

    LOGGER.info('Checking that the AOI is not the output aggregate vector')
    LOGGER.debug("aoi_path: %s", args['aoi_path'])
    LOGGER.debug("aggregate_vector_path: %s",
                 os.path.normpath(file_registry['aggregate_vector']))
    if (os.path.normpath(args['aoi_path']) ==
            os.path.normpath(file_registry['aggregate_vector'])):
        raise ValueError(
            "The input AOI is the same as the output aggregate vector, "
            "please choose a different workspace or move the AOI file "
            "out of the current workspace %s" %
            file_registry['aggregate_vector'])

    LOGGER.info('Aligning and clipping dataset list')
    input_align_list = [args['lulc_raster_path'], args['dem_raster_path']]
    output_align_list = [
        file_registry['lulc_aligned'], file_registry['dem_aligned']]
    if not args['user_defined_local_recharge']:
        month_indexes = [m+1 for m in range(N_MONTHS)]

        precip_df = MODEL_SPEC.get_input(
            'precip_raster_table').get_validated_dataframe(
            args['precip_raster_table']).sort_values('month')
        if not month_indexes == precip_df['month'].tolist():
            raise ValueError(
                'Precipitation table "month" column must include exactly the '
                'values 1 - 12 and no other values.')

        et0_df = MODEL_SPEC.get_input(
            'et0_raster_table').get_validated_dataframe(
            args['et0_raster_table']).sort_values('month')
        if not month_indexes == et0_df['month'].tolist():
            raise ValueError(
                'ET0 table "month" column must include exactly the values '
                '1 - 12 and no other values.')

        precip_path_list = precip_df['path'].tolist()
        et0_path_list = et0_df['path'].tolist()

        input_align_list = (
            precip_path_list + [args['soil_group_path']] + et0_path_list +
            input_align_list)
        output_align_list = (
            [file_registry['prcp_a[MONTH]', month] for month in range(12)] +
            [file_registry['soil_group_aligned']] +
            [file_registry['et0_a[MONTH]', month] for month in range(12)] +
            output_align_list)

    align_index = len(input_align_list) - 1  # this aligns with the DEM
    if args['user_defined_local_recharge']:
        input_align_list.append(args['l_path'])
        output_align_list.append(file_registry['l_aligned'])
    elif args['user_defined_climate_zones']:
        input_align_list.append(args['climate_zone_raster_path'])
        output_align_list.append(file_registry['cz_aligned'])
    interpolate_list = ['near'] * len(input_align_list)

    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            input_align_list, output_align_list, interpolate_list,
            pixel_size, 'intersection'),
        kwargs={
            'base_vector_path_list': (args['aoi_path'],),
            'raster_align_index': align_index},
        target_path_list=output_align_list,
        task_name='align rasters')

    fill_pit_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (file_registry['dem_aligned'], 1),
            file_registry['pit_filled_dem']),
        kwargs={'working_dir': args['workspace_dir']},
        target_path_list=[file_registry['pit_filled_dem']],
        dependent_task_list=[align_task],
        task_name='fill dem pits')

    if args['flow_dir_algorithm'] == 'mfd':
        flow_dir_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_dir_mfd,
            args=(
                (file_registry['pit_filled_dem'], 1),
                file_registry['flow_dir']),
            kwargs={'working_dir': args['workspace_dir']},
            target_path_list=[file_registry['flow_dir']],
            dependent_task_list=[fill_pit_task],
            task_name='flow direction - MFD')

        flow_accum_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_accumulation_mfd,
            args=(
                (file_registry['flow_dir'], 1),
                file_registry['flow_accum']),
            target_path_list=[file_registry['flow_accum']],
            dependent_task_list=[flow_dir_task],
            task_name='flow accumulation - MFD')

        stream_threshold_task = task_graph.add_task(
            func=pygeoprocessing.routing.extract_streams_mfd,
            args=(
                (file_registry['flow_accum'], 1),
                (file_registry['flow_dir'], 1),
                threshold_flow_accumulation,
                file_registry['stream']),
            target_path_list=[file_registry['stream']],
            dependent_task_list=[flow_accum_task],
            task_name='stream threshold - MFD')
    else: # D8
        flow_dir_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_dir_d8,
            args=(
                (file_registry['pit_filled_dem'], 1),
                file_registry['flow_dir']),
            kwargs={'working_dir': args['workspace_dir']},
            target_path_list=[file_registry['flow_dir']],
            dependent_task_list=[fill_pit_task],
            task_name='flow direction - D8')

        flow_accum_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_accumulation_d8,
            args=(
                (file_registry['flow_dir'], 1),
                file_registry['flow_accum']),
            target_path_list=[file_registry['flow_accum']],
            dependent_task_list=[flow_dir_task],
            task_name='flow accumulation - D8')

        stream_threshold_task = task_graph.add_task(
            func=pygeoprocessing.routing.extract_streams_d8,
            kwargs=dict(
                flow_accum_raster_path_band=(file_registry['flow_accum'], 1),
                flow_threshold=threshold_flow_accumulation,
                target_stream_raster_path=file_registry['stream']),
            target_path_list=[file_registry['stream']],
            dependent_task_list=[flow_accum_task],
            task_name='stream threshold - D8')



    LOGGER.info('quick flow')
    if args['user_defined_local_recharge']:
        l_avail_task = task_graph.add_task(
            func=_calculate_l_avail,
            args=(
                file_registry['l_aligned'], gamma,
                file_registry['l_avail']),
            target_path_list=[file_registry['l_avail']],
            dependent_task_list=[align_task],
            task_name='l avail task')
    else:
        # user didn't predefine local recharge so calculate it
        LOGGER.info('loading number of monthly events')
        reclassify_n_events_task_list = []
        reclass_error_details = {
            'raster_name': 'Climate Zone', 'column_name': 'cz_id',
            'table_name': 'Climate Zone'}
        for month_id in range(N_MONTHS):
            if args['user_defined_climate_zones']:
                cz_rain_events_df = MODEL_SPEC.get_input(
                    'climate_zone_table_path').get_validated_dataframe(
                    args['climate_zone_table_path'])
                climate_zone_rain_events_month = (
                    cz_rain_events_df[MONTH_ID_TO_LABEL[month_id]].to_dict())
                n_events_task = task_graph.add_task(
                    func=utils.reclassify_raster,
                    args=(
                        (file_registry['cz_aligned'], 1),
                        climate_zone_rain_events_month,
                        file_registry['n_events[MONTH]', month_id],
                        gdal.GDT_Float32, TARGET_NODATA,
                        reclass_error_details),
                    target_path_list=[
                        file_registry['n_events[MONTH]', month_id]],
                    dependent_task_list=[align_task],
                    task_name='n_events for month %d' % month_id)
                reclassify_n_events_task_list.append(n_events_task)
            else:
                n_events_task = task_graph.add_task(
                    func=pygeoprocessing.new_raster_from_base,
                    args=(
                        file_registry['dem_aligned'],
                        file_registry['n_events[MONTH]', month_id],
                        gdal.GDT_Float32, [TARGET_NODATA]),
                    kwargs={'fill_value_list': (
                        rain_events_df['events'][month_id+1],)},
                    target_path_list=[
                        file_registry['n_events[MONTH]', month_id]],
                    dependent_task_list=[align_task],
                    task_name=(
                        'n_events as a constant raster month %d' % month_id))
                reclassify_n_events_task_list.append(n_events_task)

        curve_number_task = task_graph.add_task(
            func=_calculate_curve_number_raster,
            args=(
                file_registry['lulc_aligned'],
                file_registry['soil_group_aligned'],
                biophysical_df,
                file_registry['cn']),
            target_path_list=[file_registry['cn']],
            dependent_task_list=[align_task],
            task_name='calculate curve number')

        si_task = task_graph.add_task(
            func=_calculate_si_raster,
            args=(
                file_registry['cn'], file_registry['stream'],
                file_registry['si']),
            target_path_list=[file_registry['si']],
            dependent_task_list=[curve_number_task, stream_threshold_task],
            task_name='calculate Si raster')

        quick_flow_task_list = []
        for month_index in range(N_MONTHS):
            LOGGER.info('calculate quick flow for month %d', month_index+1)
            monthly_quick_flow_task = task_graph.add_task(
                func=_calculate_monthly_quick_flow,
                args=(
                    file_registry['prcp_a[MONTH]', month_index],
                    file_registry['n_events[MONTH]', month_index],
                    file_registry['stream'],
                    file_registry['si'],
                    file_registry['qf_[MONTH]', month_index + 1]),
                target_path_list=[
                    file_registry['qf_[MONTH]', month_index + 1]],
                dependent_task_list=[
                    align_task, reclassify_n_events_task_list[month_index],
                    si_task, stream_threshold_task],
                task_name='calculate quick flow for month %d' % (
                    month_index+1))
            quick_flow_task_list.append(monthly_quick_flow_task)

        qf_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=qfi_sum_op,
                rasters=[file_registry['qf_[MONTH]', month] for month in range(1, 13)],
                target_path=file_registry['qf']),
            target_path_list=[file_registry['qf']],
            dependent_task_list=quick_flow_task_list,
            task_name='calculate QFi')

        LOGGER.info('calculate local recharge')
        kc_task_list = []
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Biophysical'}
        for month_index in range(N_MONTHS):
            kc_lookup = biophysical_df['kc_%d' % (month_index+1)].to_dict()
            kc_task = task_graph.add_task(
                func=utils.reclassify_raster,
                args=(
                    (file_registry['lulc_aligned'], 1), kc_lookup,
                    file_registry['kc_[MONTH]', month_index],
                    gdal.GDT_Float32, TARGET_NODATA, reclass_error_details),
                target_path_list=[file_registry['kc_[MONTH]', month_index]],
                dependent_task_list=[align_task],
                task_name='classify kc month %d' % month_index)
            kc_task_list.append(kc_task)

        # call through to a cython function that does the necessary routing
        # between AET and L.sum.avail in equation [7], [4], and [3]
        calculate_local_recharge_task = task_graph.add_task(
            func=seasonal_water_yield_core.calculate_local_recharge,
            args=(
                [file_registry['prcp_a[MONTH]', month] for month in range(12)],
                [file_registry['et0_a[MONTH]', month] for month in range(12)],
                [file_registry['qf_[MONTH]', month] for month in range(1, 13)],
                file_registry['flow_dir'],
                [file_registry['kc_[MONTH]', month] for month in range(12)],
                alpha_month_map,
                beta_i, gamma, file_registry['stream'],
                file_registry['l_aligned'],
                file_registry['l_avail'],
                file_registry['l_sum_avail'],
                file_registry['aet'],
                file_registry['annual_precip'],
                args['flow_dir_algorithm']),
            target_path_list=[
                file_registry['l_aligned'],
                file_registry['l_avail'],
                file_registry['l_sum_avail'],
                file_registry['aet'],
                file_registry['annual_precip']
            ],
            dependent_task_list=[
                align_task, flow_dir_task, stream_threshold_task,
                fill_pit_task] + quick_flow_task_list,
            task_name='calculate local recharge')

    # calculate Qb as the sum of local_recharge_avail over the AOI, Eq [9]
    if args['user_defined_local_recharge']:
        vri_dependent_task_list = [l_avail_task]
    else:
        vri_dependent_task_list = [calculate_local_recharge_task]

    vri_task = task_graph.add_task(
        func=_calculate_vri,
        args=(file_registry['l_aligned'], file_registry['vri']),
        target_path_list=[file_registry['vri']],
        dependent_task_list=vri_dependent_task_list,
        task_name='calculate vri')

    aggregate_recharge_task = task_graph.add_task(
        func=_aggregate_recharge,
        args=(
            args['aoi_path'], file_registry['l_aligned'],
            file_registry['vri'],
            file_registry['aggregate_vector']),
        target_path_list=[file_registry['aggregate_vector']],
        dependent_task_list=[vri_task],
        task_name='aggregate recharge')

    LOGGER.info('calculate L_sum')  # Eq. [12]
    if args['flow_dir_algorithm'] == 'mfd':
        l_sum_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_accumulation_mfd,
            args=(
                (file_registry['flow_dir'], 1),
                file_registry['l_sum']),
            kwargs={'weight_raster_path_band': (file_registry['l_aligned'], 1)},
            target_path_list=[file_registry['l_sum']],
            dependent_task_list=vri_dependent_task_list + [
                fill_pit_task, flow_dir_task, stream_threshold_task],
            task_name='calculate l sum - MFD')
    else:  # D8
        l_sum_task = task_graph.add_task(
            func=pygeoprocessing.routing.flow_accumulation_d8,
            args=(
                (file_registry['flow_dir'], 1),
                file_registry['l_sum']),
            kwargs={'weight_raster_path_band': (file_registry['l_aligned'], 1)},
            target_path_list=[file_registry['l_sum']],
            dependent_task_list=vri_dependent_task_list + [
                fill_pit_task, flow_dir_task, stream_threshold_task],
            task_name='calculate l sum - D8')

    if args['user_defined_local_recharge']:
        b_sum_dependent_task_list = [l_avail_task]
    else:
        b_sum_dependent_task_list = [calculate_local_recharge_task]

    b_sum_task = task_graph.add_task(
        func=seasonal_water_yield_core.route_baseflow_sum,
        args=(
            file_registry['flow_dir'],
            file_registry['l_aligned'],
            file_registry['l_avail'],
            file_registry['l_sum'],
            file_registry['stream'],
            file_registry['b'],
            file_registry['b_sum'],
            args['flow_dir_algorithm']),
        target_path_list=[
            file_registry['b_sum'], file_registry['b']],
        dependent_task_list=b_sum_dependent_task_list + [l_sum_task],
        task_name='calculate B_sum')

    task_graph.close()
    task_graph.join()

    LOGGER.info('  (\\w/)  SWY Complete!')
    LOGGER.info('  (..  \\ ')
    LOGGER.info(' _/  )  \\______')
    LOGGER.info('(oo /\'\\        )`,')
    LOGGER.info(' `--\' (v  __( / ||')
    LOGGER.info('       |||  ||| ||')
    LOGGER.info('      //_| //_|')

    return file_registry.registry


# raster_map equation: sum the monthly qfis
def qfi_sum_op(*qf_values): return numpy.sum(qf_values, axis=0)


def _calculate_l_avail(l_path, gamma, target_l_avail_path):
    """l avail = l * gamma."""
    pygeoprocessing.raster_map(
        op=lambda l: numpy.min(numpy.stack((gamma * l, l)), axis=0),
        rasters=[l_path],
        target_path=target_l_avail_path)


def _calculate_vri(l_path, target_vri_path):
    """Calculate VRI as li_array / qb_sum.

    Args:
        l_path (str): path to L raster.
        target_vri_path (str): path to output Vri raster.

    Returns:
        None.

    """
    qb_sum = 0
    qb_valid_count = 0
    l_nodata = pygeoprocessing.get_raster_info(l_path)['nodata'][0]

    for _, block in pygeoprocessing.iterblocks((l_path, 1)):
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(block, l_nodata) &
            (~numpy.isinf(block)))
        qb_sum += numpy.sum(block[valid_mask])
        qb_valid_count += numpy.count_nonzero(valid_mask)
    li_nodata = pygeoprocessing.get_raster_info(l_path)['nodata'][0]

    def vri_op(li_array):
        """Calculate vri index [Eq 10]."""
        result = numpy.empty_like(li_array)
        result[:] = li_nodata
        if qb_sum > 0:
            valid_mask = ~pygeoprocessing.array_equals_nodata(li_array, li_nodata)
            try:
                result[valid_mask] = li_array[valid_mask] / qb_sum
            except RuntimeWarning:
                LOGGER.exception(qb_sum)
                raise
        return result

    pygeoprocessing.raster_calculator(
        [(l_path, 1)], vri_op, target_vri_path, gdal.GDT_Float32,
        li_nodata)


def _calculate_monthly_quick_flow(precip_path, n_events_path, stream_path,
        si_path, qf_monthly_path):
    """Calculate quick flow for a month.

    Args:
        precip_path (string): path to monthly precipitation raster
        n_events_path (string): a path to a raster where each pixel
            indicates the number of rain events.
        stream_path (string): path to stream mask raster where 1 indicates a
            stream pixel, 0 is a non-stream but otherwise valid area from the
            original DEM, and nodata indicates areas outside the valid DEM.
        si_path (string): path to raster that has potential maximum retention
        qf_monthly_path (string): path to output monthly QF raster.

    Returns:
        None
    """
    p_nodata = pygeoprocessing.get_raster_info(precip_path)['nodata'][0]
    n_nodata = pygeoprocessing.get_raster_info(n_events_path)['nodata'][0]
    stream_nodata = pygeoprocessing.get_raster_info(stream_path)['nodata'][0]
    si_nodata = pygeoprocessing.get_raster_info(si_path)['nodata'][0]

    def qf_op(p_im, s_i, n_m, stream):
        """Calculate quick flow as in Eq [1] in user's guide.

        Args:
            p_im (numpy.array): precipitation at pixel i on month m
            s_i (numpy.array): factor that is 1000/CN_i - 10
            n_m (numpy.array): number of rain events on pixel i in month m
            stream (numpy.array): 1 if stream, otherwise not a stream pixel.

        Returns:
            quick flow (numpy.array)
        """
        valid_p_mask = ~pygeoprocessing.array_equals_nodata(p_im, p_nodata)
        valid_n_mask = ~pygeoprocessing.array_equals_nodata(n_m, n_nodata)
        # precip mask: both p_im and n_m are defined and greater than 0
        precip_ok = valid_p_mask & valid_n_mask & (p_im > 0) & (n_m > 0)
        on_stream = stream == 1
        # stream_nodata is the only input that carries over nodata values from
        # the aligned DEM.
        valid_mask = (
          valid_p_mask &
          valid_n_mask &
          ~pygeoprocessing.array_equals_nodata(stream, stream_nodata) &
          ~pygeoprocessing.array_equals_nodata(s_i, si_nodata))

        # QF is defined in terms of three cases:
        #
        # 1. Where precipitation <=0, QF = 0
        #    (even if stream or s_i is undefined)
        #
        # 2. Where there is precipitation and we're on a stream, QF = P
        #    (even if s_i is undefined)
        #
        # 3. Where there is precipitation and we're not on a stream, use the
        #    quickflow equation (only if all four inputs are defined):
        #    QF_im = 25.4 * n_m * (
        #       (a_im - s_i) * exp(-0.2 * s_i / a_im) +
        #       s_i^2 / a_im * exp(0.8 * s_i / a_im) * E1(s_i / a_im)
        #    )
        #
        # When evaluating the QF equation, there are a few edge cases:
        #
        # 3a. Where s_i = 0, you get NaN and a warning from numpy because
        #     E1(0 / a_im) = infinity. In this case, per conversation with
        #     Rafa, the final term of the equation should evaluate to 0, and
        #     the equation can be simplified to QF_im = P_im
        #     (which makes sense because if s_i = 0, no water is retained).
        #
        #     Solution: Preemptively set QF_im equal to P_im where s_i = 0 in
        #     order to avoid calculations with infinity.
        #
        # 3b. When the ratio s_i / a_im becomes large, QF approaches 0.
        #     [NOTE: I don't know how to prove this mathematically, but it
        #     holds true when I tested with reasonable values of s_i and a_im].
        #     The exp() term becomes very large, while the E1() term becomes
        #     very small.
        #
        #     Per conversation with Rafa and Lisa, large s_i / a_im ratios
        #     shouldn't happen often with real world data. But if they did, it
        #     would be a situation where there is very little precipitation
        #     spread out over relatively many rain events and the soil is very
        #     absorbent, so logically, QF should be effectively zero.
        #
        #     To avoid overflow, we set a threshold of 100 for the s_i / a_im
        #     ratio. Where s_i / a_im > 100, we set QF to 0. 100 was chosen
        #     because it's a nice whole number that gets us close to the
        #     float32 max without surpassing it (exp(0.8*100) = 5e34). When
        #     s_i / a_im = 100, the actual result of the QF equation is on the
        #     order of 1e-6, so it should be rounded down to 0 anyway.
        #
        # 3c. Otherwise, evaluate the QF equation as usual.
        #
        # 3d. With certain inputs [for example: n_m = 10, CN = 50, p_im = 30],
        #     it's possible that the QF equation evaluates to a very small
        #     negative value. Per conversation with Lisa and Rafa, this is an
        #     edge case that the equation was not designed for. Negative QF
        #     doesn't make sense, so we set any negative QF values to 0.

        # qf_im is the quickflow at pixel i on month m
        qf_im = numpy.full(p_im.shape, TARGET_NODATA, dtype=numpy.float32)

        # case 1: where precip or n are <=0 but not nodata
        case_1_mask = ~precip_ok & valid_p_mask & valid_n_mask
        qf_im[case_1_mask] = 0

        # case 2: where there is precipitation and we're on a stream
        qf_im[precip_ok & on_stream] = p_im[precip_ok & on_stream]

        # case 3: where there is precipitation and we're not on a stream
        case_3_mask = valid_mask & precip_ok & ~on_stream

        # for consistent indexing, make a_im the same shape as the other
        # arrays even though we only use it in case 3
        a_im = numpy.full(p_im.shape, numpy.nan, dtype=numpy.float32)
        # a_im is the mean rain depth on a rainy day at pixel i on month m
        # the 25.4 converts inches to mm since s_i is in inches
        a_im[case_3_mask] = p_im[case_3_mask] / (n_m[case_3_mask] * 25.4)

        # case 3a: when s_i = 0, qf = p
        case_3a_mask = case_3_mask & (s_i == 0)
        qf_im[case_3a_mask] = p_im[case_3a_mask]

        # case 3b: set quickflow to 0 when the s_i/a_im ratio is too large
        case_3b_mask = case_3_mask & (s_i / a_im > 100)
        qf_im[case_3b_mask] = 0

        # case 3c: evaluate the equation as usual
        case_3c_mask = case_3_mask & ~(case_3a_mask | case_3b_mask)
        qf_im[case_3c_mask] = (
            25.4 * n_m[case_3c_mask] * (
                ((a_im[case_3c_mask] - s_i[case_3c_mask]) *
                 numpy.exp(-0.2 * s_i[case_3c_mask] / a_im[case_3c_mask])) +
                (s_i[case_3c_mask] ** 2 / a_im[case_3c_mask] *
                 numpy.exp(0.8 * s_i[case_3c_mask] / a_im[case_3c_mask]) *
                 scipy.special.exp1(s_i[case_3c_mask] / a_im[case_3c_mask]))
            )
        )

        # case 3d: set any negative values to 0
        qf_im[valid_mask & (qf_im < 0)] = 0

        return qf_im

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            precip_path, si_path, n_events_path, stream_path]],
        qf_op, qf_monthly_path, gdal.GDT_Float32, TARGET_NODATA)


def _calculate_curve_number_raster(
        lulc_raster_path, soil_group_path, biophysical_df, cn_path):
    """Calculate the CN raster from the landcover and soil group rasters.

    Args:
        lulc_raster_path (string): path to landcover raster
        soil_group_path (string): path to raster indicating soil group where
            pixel values are in [1,2,3,4]
        biophysical_df (pandas.DataFrame): table mapping landcover IDs to the
            columns 'cn_a', 'cn_b', 'cn_c', 'cn_d', that contain
            the curve number values for that landcover and soil type.
        cn_path (string): path to output curve number raster to be output
            which will be the dimensions of the intersection of
            `lulc_raster_path` and `soil_group_path` the cell size of
            `lulc_raster_path`.

    Returns:
        None
    """
    map_soil_type_to_header = {
        1: 'cn_a',
        2: 'cn_b',
        3: 'cn_c',
        4: 'cn_d',
    }
    lulc_to_soil = {}
    lucodes = biophysical_df.index.to_list()
    for soil_id, soil_column in map_soil_type_to_header.items():
        lulc_to_soil[soil_id] = {
            'lulc_values': [],
            'cn_values': []
        }

        for lucode in sorted(lucodes):
            lulc_to_soil[soil_id]['cn_values'].append(
                biophysical_df[soil_column][lucode])
            lulc_to_soil[soil_id]['lulc_values'].append(lucode)

        # Making the landcover array a float32 in case the user provides a
        # float landcover map like Kate did.
        lulc_to_soil[soil_id]['lulc_values'] = (
            numpy.array(lulc_to_soil[soil_id]['lulc_values'],
                        dtype=numpy.float32))
        lulc_to_soil[soil_id]['cn_values'] = (
            numpy.array(lulc_to_soil[soil_id]['cn_values'],
                        dtype=numpy.float32))

    # Use set of table lucodes in cn_op
    lucodes_set = set(lucodes)
    valid_soil_groups = set(map_soil_type_to_header.keys())

    def cn_op(lulc_array, soil_group_array):
        """Map lulc code and soil to a curve number."""
        cn_result = numpy.empty(lulc_array.shape)
        cn_result[:] = TARGET_NODATA

        # if lulc_array value not in lulc_to_soil[soil_group_id]['lulc_values']
        # then numpy.digitize will not bin properly and cause an IndexError
        # during the reshaping call
        lulc_unique = set(i.item() for i in numpy.unique(lulc_array))
        if not lulc_unique.issubset(lucodes_set):
            # cast to list to conform with similar error messages in InVEST
            missing_lulc_values = sorted(lulc_unique.difference(lucodes_set))
            error_message = (
                "Values in the LULC raster were found that are not"
                " represented under the 'lucode' key column of the"
                " Biophysical table. The missing values found in the LULC"
                f" raster but not the table are: {missing_lulc_values}.")
            raise ValueError(error_message)

        unique_soil_groups = numpy.unique(soil_group_array)
        invalid_soil_groups = set(unique_soil_groups) - valid_soil_groups
        if invalid_soil_groups:
            invalid_soil_groups = [str(group) for group in invalid_soil_groups]
            raise ValueError(
                "The soil group raster must only have groups 1, 2, 3 or 4. "
                f"Invalid group(s) {', '.join(invalid_soil_groups)} were "
                f"found in soil group raster {soil_group_path} "
                "(nodata value: "
                f"{pygeoprocessing.get_raster_info(soil_group_path)['nodata'][0]})")

        for soil_group_id in unique_soil_groups:
            current_soil_mask = (soil_group_array == soil_group_id)
            index = numpy.digitize(
                lulc_array.ravel(),
                lulc_to_soil[soil_group_id]['lulc_values'], right=True)
            cn_values = (
                lulc_to_soil[soil_group_id]['cn_values'][index]).reshape(
                    lulc_array.shape)
            cn_result[current_soil_mask] = cn_values[current_soil_mask]
        return cn_result

    pygeoprocessing.raster_map(
        op=cn_op,
        rasters=[lulc_raster_path, soil_group_path],
        target_path=cn_path)


def _calculate_si_raster(cn_path, stream_path, si_path):
    """Calculate the S factor of the quickflow equation [1].

    Args:
        cn_path (string): path to curve number raster
        stream_path (string): path to a stream raster (0, 1)
        si_path (string): path to output s_i raster

    Returns:
        None
    """
    cn_nodata = pygeoprocessing.get_raster_info(cn_path)['nodata'][0]

    def si_op(ci_factor, stream_mask):
        """Calculate si factor."""
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(ci_factor, cn_nodata) &
            (ci_factor > 0))
        si_array = numpy.empty(ci_factor.shape)
        si_array[:] = TARGET_NODATA
        # multiply by the stream mask != 1 so we get 0s on the stream and
        # unaffected results everywhere else
        si_array[valid_mask] = (
            (1000 / ci_factor[valid_mask] - 10) * (
                stream_mask[valid_mask] != 1))
        return si_array

    pygeoprocessing.raster_calculator(
        [(cn_path, 1), (stream_path, 1)], si_op, si_path, gdal.GDT_Float32,
        TARGET_NODATA)


def _aggregate_recharge(
        aoi_path, l_path, vri_path, aggregate_vector_path):
    """Aggregate recharge values for the provided watersheds/AOIs.

    Generates a new shapefile that's a copy of 'aoi_path' in sum values from L
    and Vri.

    Args:
        aoi_path (string): path to shapefile that will be used to
            aggregate rasters
        l_path (string): path to (L) local recharge raster
        vri_path (string): path to Vri raster
        aggregate_vector_path (string): path to shapefile that will be created
            by this function as the aggregating output.  will contain fields
            'l_sum' and 'vri_sum' per original feature in `aoi_path`.  If this
            file exists on disk prior to the call it is overwritten with
            the result of this call.

    Returns:
        None
    """
    if os.path.exists(aggregate_vector_path):
        LOGGER.warning(
            '%s exists, deleting and writing new output',
            aggregate_vector_path)
        os.remove(aggregate_vector_path)

    original_aoi_vector = gdal.OpenEx(aoi_path, gdal.OF_VECTOR)

    driver = gdal.GetDriverByName('ESRI Shapefile')
    driver.CreateCopy(aggregate_vector_path, original_aoi_vector)
    gdal.Dataset.__swig_destroy__(original_aoi_vector)
    original_aoi_vector = None
    aggregate_vector = gdal.OpenEx(aggregate_vector_path, 1)
    aggregate_layer = aggregate_vector.GetLayer()

    for raster_path, aggregate_field_id, op_type in [
            (l_path, 'qb', 'mean'), (vri_path, 'vri_sum', 'sum')]:

        # aggregate carbon stocks by the new ID field
        aggregate_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregate_vector_path)

        aggregate_field = ogr.FieldDefn(aggregate_field_id, ogr.OFTReal)
        aggregate_field.SetWidth(24)
        aggregate_field.SetPrecision(11)
        aggregate_layer.CreateField(aggregate_field)

        aggregate_layer.ResetReading()
        for poly_index, poly_feat in enumerate(aggregate_layer):
            if op_type == 'mean':
                pixel_count = aggregate_stats[poly_index]['count']
                if pixel_count != 0:
                    value = (aggregate_stats[poly_index]['sum'] / pixel_count)
                else:
                    LOGGER.warning(
                        "no coverage for polygon %s", ', '.join(
                            [str(poly_feat.GetField(_)) for _ in range(
                                poly_feat.GetFieldCount())]))
                    value = 0
            elif op_type == 'sum':
                value = aggregate_stats[poly_index]['sum']
            poly_feat.SetField(aggregate_field_id, float(value))
            aggregate_layer.SetFeature(poly_feat)

    aggregate_layer.SyncToDisk()
    aggregate_layer = None
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    aggregate_vector = None


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
    return validation.validate(args, MODEL_SPEC)
