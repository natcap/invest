"""RouteDEM for exposing the natcap.invest's routing package to UI."""
import logging
import os

import pygeoprocessing
import pygeoprocessing.routing
import taskgraph

from . import gettext
from . import spec
from . import utils
from . import validation
from .file_registry import FileRegistry
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

INVALID_BAND_INDEX_MSG = gettext('Must be between 1 and {maximum}')

MODEL_SPEC = spec.ModelSpec(
    model_id="routedem",
    model_title=gettext("RouteDEM"),
    userguide="routedem.html",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=(),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["dem_path", "dem_band_index"],
        ["calculate_slope"],
        ["algorithm"],
        ["calculate_flow_direction"],
        ["calculate_flow_accumulation"],
        ["calculate_stream_threshold", "threshold_flow_accumulation",
         "calculate_downslope_distance", "calculate_stream_order",
         "calculate_subwatersheds"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.DEM.model_copy(update=dict(id="dem_path")),
        spec.IntegerInput(
            id="dem_band_index",
            name=gettext("band index"),
            about=gettext("Index of the raster band to use, for multi-band rasters."),
            required=False,
            units=u.none,
            expression="value >= 1"
        ),
        spec.OptionStringInput(
            id="algorithm",
            name=gettext("routing algorithm"),
            about=gettext("The routing algorithm to use."),
            options=[
                spec.Option(
                    key="D8",
                    about=(
                        "All water on a pixel flows into the most downhill of its 8"
                        " surrounding pixels")),
                spec.Option(
                    key="MFD",
                    about=(
                        "Flow off a pixel is modeled fractionally so that water is split"
                        " among multiple downslope pixels"))
            ]
        ),
        spec.BooleanInput(
            id="calculate_flow_direction",
            name=gettext("calculate flow direction"),
            about=gettext("Calculate flow direction from the provided DEM."),
            required=False
        ),
        spec.BooleanInput(
            id="calculate_flow_accumulation",
            name=gettext("calculate flow accumulation"),
            about=gettext("Calculate flow accumulation from the flow direction output."),
            required=False,
            allowed="calculate_flow_direction"
        ),
        spec.BooleanInput(
            id="calculate_stream_threshold",
            name=gettext("calculate streams"),
            about=gettext("Calculate streams from the flow accumulation output. "),
            required=False,
            allowed="calculate_flow_accumulation"
        ),
        spec.THRESHOLD_FLOW_ACCUMULATION.model_copy(update=dict(
            required="calculate_stream_threshold",
            allowed="calculate_stream_threshold",
            about=(
                spec.THRESHOLD_FLOW_ACCUMULATION.about + " " +
                gettext("Required if Calculate Streams is selected."))
        )),
        spec.BooleanInput(
            id="calculate_downslope_distance",
            name=gettext("calculate distance to stream"),
            about=gettext(
                "Calculate flow distance from each pixel to a stream as defined in the"
                " Calculate Streams output."
            ),
            required=False,
            allowed="calculate_stream_threshold"
        ),
        spec.BooleanInput(
            id="calculate_slope",
            name=gettext("calculate slope"),
            about=gettext("Calculate percent slope from the provided DEM."),
            required=False
        ),
        spec.BooleanInput(
            id="calculate_stream_order",
            name=gettext("calculate strahler stream orders (D8 only)"),
            about=gettext("Calculate the Strahler Stream order."),
            required=False,
            allowed="calculate_stream_threshold and algorithm == 'D8'"
        ),
        spec.BooleanInput(
            id="calculate_subwatersheds",
            name=gettext("calculate subwatersheds (D8 only)"),
            about=gettext("Determine subwatersheds from the stream order."),
            required=False,
            allowed="calculate_stream_order and algorithm == 'D8'"
        )
    ],
    outputs=[
        spec.TASKGRAPH_CACHE,
        spec.FILLED_DEM.model_copy(update=dict(
            id="filled",
            path="filled.tif")),
        spec.FLOW_ACCUMULATION,
        spec.FLOW_DIRECTION,
        spec.SLOPE,
        spec.STREAM.model_copy(update=dict(
            id="stream_mask",
            path="stream_mask.tif")),
        spec.VectorOutput(
            id="strahler_stream_order",
            path="strahler_stream_order.gpkg",
            about=gettext(
                "A vector of line segments indicating the Strahler stream order and other"
                " properties of each stream segment."
            ),
            geometry_types={"LINESTRING"},
            fields=[
                spec.NumberOutput(
                    id="order", about=gettext("The Strahler stream order."), units=u.none
                ),
                spec.NumberOutput(
                    id="river_id",
                    about=gettext(
                        "A unique identifier used by all stream segments that connect to"
                        " the same outlet."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="drop_distance",
                    about=gettext(
                        "The drop distance in DEM elevation units from the upstream to"
                        " downstream component of this stream segment."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="outlet",
                    about=gettext("1 if this segment is an outlet, 0 if it is not."),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="us_fa",
                    about=gettext(
                        "The flow accumulation value at the upstream end of the stream"
                        " segment."
                    ),
                    units=u.pixel
                ),
                spec.NumberOutput(
                    id="ds_fa",
                    about=gettext(
                        "The flow accumulation value at the downstream end of the stream"
                        " segment."
                    ),
                    units=u.pixel
                ),
                spec.NumberOutput(
                    id="thresh_fa",
                    about=gettext(
                        "The final threshold flow accumulation value used to determine"
                        " the river segments."
                    ),
                    units=u.pixel
                ),
                spec.NumberOutput(
                    id="upstream_d8_dir",
                    about=gettext("The direction of flow immediately upstream."),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="ds_x",
                    about=gettext(
                        "The DEM X coordinate for the outlet in pixels from the origin."
                    ),
                    units=u.pixel
                ),
                spec.NumberOutput(
                    id="ds_y",
                    about=gettext(
                        "The DEM Y coordinate for the outlet in pixels from the origin."
                    ),
                    units=u.pixel
                ),
                spec.NumberOutput(
                    id="ds_x_1",
                    about=gettext(
                        "The DEM X coordinate that is 1 pixel upstream from the outlet."
                    ),
                    units=u.pixel
                ),
                spec.NumberOutput(
                    id="ds_y_1",
                    about=gettext(
                        "The DEM Y coordinate that is 1 pixel upstream from the outlet."
                    ),
                    units=u.pixel
                ),
                spec.NumberOutput(
                    id="us_x",
                    about=gettext("The DEM X coordinate for the upstream inlet."),
                    units=u.pixel
                ),
                spec.NumberOutput(
                    id="us_y",
                    about=gettext("The DEM Y coordinate for the upstream inlet."),
                    units=u.pixel
                )
            ]
        ),
        spec.VectorOutput(
            id="subwatersheds",
            path="subwatersheds.gpkg",
            about=gettext(
                "A GeoPackage with polygon features representing subwatersheds.  A new"
                " subwatershed is created for each tributary of a stream and is"
                " influenced greatly by your choice of Threshold Flow Accumulation value."
            ),
            geometry_types={"POLYGON"},
            fields=[
                spec.NumberOutput(
                    id="stream_id",
                    about=gettext(
                        "A unique stream id, matching the one in the Strahler stream"
                        " order vector."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="terminated_early",
                    about=gettext(
                        "Indicates whether generation of this subwatershed terminated"
                        " early (1) or completed as expected (0). If you encounter a (1),"
                        " please let us know via the forums,"
                        " community.naturalcapitalproject.org."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="outlet_x",
                    about=gettext(
                        "The X coordinate in pixels from the origin of the outlet of the"
                        " watershed. This can be useful when determining other properties"
                        " of the watershed when indexing with the underlying raster data."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="outlet_y",
                    about=gettext(
                        "The X coordinate in pixels from the origin of the outlet of the"
                        " watershed. This can be useful when determining other properties"
                        " of the watershed when indexing with the underlying raster data."
                    ),
                    units=u.none
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="downslope_distance",
            path="downslope_distance.tif",
            about=gettext("Flow distance from each pixel to a stream."),
            data_type=float,
            units=u.pixel
        )
    ]
)


_ROUTING_FUNCS = {
    'd8': {
        'flow_accumulation': pygeoprocessing.routing.flow_accumulation_d8,
        'flow_direction': pygeoprocessing.routing.flow_dir_d8,
        'threshold_flow': pygeoprocessing.routing.extract_streams_d8,
        'distance_to_channel': pygeoprocessing.routing.distance_to_channel_d8,
    },
    'mfd': {
        'flow_accumulation': pygeoprocessing.routing.flow_accumulation_mfd,
        'flow_direction': pygeoprocessing.routing.flow_dir_mfd,
        'threshold_flow': pygeoprocessing.routing.extract_streams_mfd,
        'distance_to_channel': pygeoprocessing.routing.distance_to_channel_mfd,
    }
}


def execute(args):
    """RouteDEM: Hydrological routing.

    This model exposes the pygeoprocessing D8 and Multiple Flow Direction
    routing functionality as an InVEST model.

    This tool will always fill pits on the input DEM.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['dem_band_index'] (int): Optional. The band index to operate on.
            If not provided, band index 1 is assumed.
        args['algorithm'] (string): The routing algorithm to use.  Must be
            one of 'D8' or 'MFD' (case-insensitive). Required when calculating
            flow direction, flow accumulation, stream threshold, and downslope
            distance.
        args['calculate_flow_direction'] (bool): If True, model will calculate
            flow direction for the filled DEM.
        args['calculate_flow_accumulation'] (bool): If True, model will
            calculate a flow accumulation raster. Only applies when
            args['calculate_flow_direction'] is True.
        args['calculate_stream_threshold'] (bool): if True, model will
            calculate a stream classification layer by thresholding flow
            accumulation to the provided value in
            ``args['threshold_flow_accumulation']``.  Only applies when
            args['calculate_flow_accumulation'] and
            args['calculate_flow_direction'] are True.
        args['threshold_flow_accumulation'] (int): The number of upslope
            cells that must flow into a cell before it's classified as a
            stream.
        args['calculate_downslope_distance'] (bool): If True, and a stream
            threshold is calculated, model will calculate a downslope
            distance raster in units of pixels. Only applies when
            args['calculate_flow_accumulation'],
            args['calculate_flow_direction'], and
            args['calculate_stream_threshold'] are all True.
        args['calculate_slope'] (bool): If True, model will calculate a
            slope raster from the DEM.
        args['calculate_stream_order']: If True, model will create a vector of
            the Strahler stream order.
        args['calculate_subwatersheds']: If True, the model will create a
            vector of subwatersheds.
        args['n_workers'] (int): The ``n_workers`` parameter to pass to
            the task graph.  The default is ``-1`` if not provided.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    args, file_registry, graph = MODEL_SPEC.setup(args)

    if args['calculate_flow_direction']:
        routing_funcs = _ROUTING_FUNCS[args['algorithm']]

    band_index = args['dem_band_index'] if args['dem_band_index'] else 1

    LOGGER.info('Using DEM band index %s', band_index)

    dem_raster_path_band = (args['dem_path'], band_index)

    # Calculate slope.  This is intentionally on the original DEM, not
    # on the pitfilled DEM.  If the user really wants the slop of the filled
    # DEM, they can pass it back through RouteDEM.
    if args['calculate_slope']:
        graph.add_task(
            pygeoprocessing.calculate_slope,
            args=(dem_raster_path_band, file_registry['slope']),
            task_name='calculate_slope',
            target_path_list=[file_registry['slope']])

    filled_pits_task = graph.add_task(
        pygeoprocessing.routing.fill_pits,
        args=(dem_raster_path_band,
              file_registry['filled'],
              args['workspace_dir']),
        task_name='fill_pits',
        target_path_list=[file_registry['filled']])

    if args['calculate_flow_direction']:
        LOGGER.info("calculating flow direction")
        flow_direction_task = graph.add_task(
            routing_funcs['flow_direction'],
            args=((file_registry['filled'], 1),  # PGP>1.9.0 creates 1-band fills
                  file_registry['flow_direction'],
                  args['workspace_dir']),
            target_path_list=[file_registry['flow_direction']],
            dependent_task_list=[filled_pits_task],
            task_name=f'flow_dir_{args["algorithm"]}')

        if args['calculate_flow_accumulation']:
            LOGGER.info("calculating flow accumulation")
            flow_accum_task = graph.add_task(
                routing_funcs['flow_accumulation'],
                args=((file_registry['flow_direction'], 1), file_registry['flow_accumulation']),
                target_path_list=[file_registry['flow_accumulation']],
                task_name=f'flow_accumulation_{args["algorithm"]}',
                dependent_task_list=[flow_direction_task])

            if args['calculate_stream_threshold']:
                stream_extraction_kwargs = {
                    'flow_accum_raster_path_band': (file_registry['flow_accumulation'], 1),
                    'flow_threshold': args['threshold_flow_accumulation'],
                    'target_stream_raster_path': file_registry['stream_mask'],
                }
                if args['algorithm'] == 'mfd':
                    stream_extraction_kwargs['flow_dir_mfd_path_band'] = (
                        file_registry['flow_direction'], 1)
                stream_threshold_task = graph.add_task(
                    routing_funcs['threshold_flow'],
                    kwargs=stream_extraction_kwargs,
                    target_path_list=[file_registry['stream_mask']],
                    dependent_task_list=[flow_accum_task],
                    task_name=f'stream_thresholding_{args["algorithm"]}')

                if args['calculate_downslope_distance']:
                    graph.add_task(
                        routing_funcs['distance_to_channel'],
                        args=((file_registry['flow_direction'], 1),
                              (file_registry['stream_mask'], 1),
                              file_registry['downslope_distance']),
                        target_path_list=[file_registry['downslope_distance']],
                        task_name=f'downslope_distance_{args["algorithm"]}',
                        dependent_task_list=[stream_threshold_task])

                if args['calculate_stream_order'] and args['algorithm'] == 'd8':
                    stream_order_task = graph.add_task(
                        pygeoprocessing.routing.extract_strahler_streams_d8,
                        kwargs={
                            "flow_dir_d8_raster_path_band":
                                (file_registry['flow_direction'], 1),
                            "flow_accum_raster_path_band":
                                (file_registry['flow_accumulation'], 1),
                            "dem_raster_path_band":
                                (file_registry['filled'], 1),
                            "target_stream_vector_path": file_registry['strahler_stream_order'],
                            "min_flow_accum_threshold": args['threshold_flow_accumulation'],
                            "river_order": 5,  # the default
                        },
                        target_path_list=[file_registry['strahler_stream_order']],
                        task_name='Calculate D8 stream order',
                        dependent_task_list=[
                            filled_pits_task,
                            flow_direction_task,
                            flow_accum_task
                        ])

                    if args['calculate_subwatersheds']:
                        graph.add_task(
                            pygeoprocessing.routing.calculate_subwatershed_boundary,
                            kwargs={
                                'd8_flow_dir_raster_path_band':
                                    (file_registry['flow_direction'], 1),
                                'strahler_stream_vector_path':
                                    file_registry['strahler_stream_order'],
                                'target_watershed_boundary_vector_path':
                                    file_registry['subwatersheds'],
                                'outlet_at_confluence': False,  # The default
                            },
                            target_path_list=[file_registry['subwatersheds']],
                            task_name=(
                                'Calculate subwatersheds from stream order'),
                            dependent_task_list=[flow_direction_task,
                                                 stream_order_task])

    graph.close()
    graph.join()
    return file_registry.registry


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in ``execute`` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire ``args`` dictionary.

    Returns:
        list of ([invalid key_a, invalid key_b, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.
    """
    validation_warnings = validation.validate(args, MODEL_SPEC)

    invalid_keys = validation.get_invalid_keys(validation_warnings)
    sufficient_keys = validation.get_sufficient_keys(args)

    if ('dem_band_index' not in invalid_keys and
            'dem_band_index' in sufficient_keys and
            'dem_path' not in invalid_keys and
            'dem_path' in sufficient_keys):
        raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
        if int(args['dem_band_index']) > raster_info['n_bands']:
            validation_warnings.append((
                ['dem_band_index'],
                INVALID_BAND_INDEX_MSG.format(maximum=raster_info['n_bands'])))

    return validation_warnings
