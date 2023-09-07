"""RouteDEM for exposing the natcap.invest's routing package to UI."""
import logging
import os

import pygeoprocessing
import pygeoprocessing.routing
import taskgraph

from . import gettext
from . import spec_utils
from . import utils
from . import validation
from .model_metadata import MODEL_METADATA
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

INVALID_BAND_INDEX_MSG = gettext('Must be between 1 and {maximum}')

MODEL_SPEC = {
    "model_name": MODEL_METADATA["routedem"].model_title,
    "pyname": MODEL_METADATA["routedem"].pyname,
    "userguide": MODEL_METADATA["routedem"].userguide,
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "dem_path": spec_utils.DEM,
        "dem_band_index": {
            "type": "number",
            "expression": "value >= 1",
            "units": u.none,
            "required": False,
            "about": gettext(
                "Index of the raster band to use, for multi-band rasters."),
            "name": gettext("band index")
        },
        "algorithm": {
            "type": "option_string",
            "options": {
                "D8": {
                    "display_name": "D8",
                    "description": gettext(
                        "All water on a pixel flows into the most downhill of "
                        "its 8 surrounding pixels")},
                "MFD": {
                    "display_name": "MFD",
                    "description": gettext(
                        "Flow off a pixel is modeled fractionally so that "
                        "water is split among multiple downslope pixels")}
            },
            "about": gettext("The routing algorithm to use."),
            "name": gettext("routing algorithm")
        },
        "calculate_flow_direction": {
            "type": "boolean",
            "required": False,
            "about": gettext("Calculate flow direction from the provided DEM."),
            "name": gettext("calculate flow direction")
        },
        "calculate_flow_accumulation": {
            "type": "boolean",
            "required": False,
            "about": gettext(
                "Calculate flow accumulation from the flow direction output."),
            "name": gettext("calculate flow accumulation")
        },
        "calculate_stream_threshold": {
            "type": "boolean",
            "required": False,
            "about": gettext(
                "Calculate streams from the flow accumulation output. "),
            "name": gettext("calculate streams")
        },
        "threshold_flow_accumulation": {
            **spec_utils.THRESHOLD_FLOW_ACCUMULATION,
            "required": "calculate_stream_threshold",
            "about": (
                spec_utils.THRESHOLD_FLOW_ACCUMULATION['about'] + " " +
                gettext("Required if Calculate Streams is selected."))
        },
        "calculate_downslope_distance": {
            "type": "boolean",
            "required": False,
            "about": gettext(
                "Calculate flow distance from each pixel to a stream as "
                "defined in the Calculate Streams output."),
            "name": gettext("calculate distance to stream")
        },
        "calculate_slope": {
            "type": "boolean",
            "required": False,
            "about": gettext("Calculate percent slope from the provided DEM."),
            "name": gettext("calculate slope")
        },
        "calculate_stream_order": {
            "type": "boolean",
            "required": False,
            "about": gettext("Calculate the Strahler Stream order."),
            "name": gettext("calculate strahler stream orders (D8 only)"),
        },
        "calculate_subwatersheds": {
            "type": "boolean",
            "required": False,
            "about": gettext("Determine subwatersheds from the stream order."),
            "name": gettext("calculate subwatersheds (D8 only)"),
        },
    },
    "outputs": {
        "taskgraph_cache": spec_utils.TASKGRAPH_DIR,
        "filled.tif": spec_utils.FILLED_DEM,
        "flow_accumulation.tif": spec_utils.FLOW_ACCUMULATION,
        "flow_direction.tif": spec_utils.FLOW_DIRECTION,
        "slope.tif": spec_utils.SLOPE,
        "stream_mask.tif": spec_utils.STREAM,
        "strahler_stream_order.gpkg": {
            "about": (
                "A vector of line segments indicating the Strahler stream "
                "order and other properties of each stream segment."),
            "geometries": spec_utils.LINESTRING,
            "fields": {
                "order": {
                    "about": "The Strahler stream order.",
                    "type": "number",
                    "units": u.none,
                },
                "river_id": {
                    "about": (
                        "A unique identifier used by all stream segments that "
                        "connect to the same outlet."),
                    "type": "number",
                    "units": u.none,
                },
                "drop_distance": {
                    "about": (
                        "The drop distance in DEM elevation units from the "
                        "upstream to downstream component of this stream "
                        "segment."),
                    "type": "number",
                    "units": u.none,
                },
                "outlet": {
                    "about": (
                        "1 if this segment is an outlet, 0 if it is not."),
                    "type": "number",
                    "units": u.none,
                },
                "us_fa": {
                    "about": (
                        "The flow accumulation value at the upstream end of "
                        "the stream segment."),
                    "type": "number",
                    "units": u.pixels,
                },
                "ds_fa": {
                    "about": (
                        "The flow accumulation value at the downstream end of "
                        "the stream segment."),
                    "type": "number",
                    "units": u.pixels,
                },
                "thresh_fa": {
                    "about": (
                        "The final threshold flow accumulation value used to "
                        "determine the river segments."),
                    "type": "number",
                    "units": u.pixels,
                },
                "upstream_d8_dir": {
                    "about": (
                        "The direction of flow immediately upstream."),
                    "type": "number",
                    "units": u.none,
                },
                "ds_x": {
                    "about": (
                        "The DEM X coordinate for the outlet in pixels from "
                        "the origin."),
                    "type": "number",
                    "units": u.pixels,
                },
                "ds_y": {
                    "about": (
                        "The DEM Y coordinate for the outlet in pixels from "
                        "the origin."),
                    "type": "number",
                    "units": u.pixels,
                },
                "ds_x_1": {
                    "about": (
                        "The DEM X coordinate that is 1 pixel upstream "
                        "from the outlet."),
                    "type": "number",
                    "units": u.pixels,
                },
                "ds_y_1": {
                    "about": (
                        "The DEM Y coordinate that is 1 pixel upstream "
                        "from the outlet."),
                    "type": "number",
                    "units": u.pixels,
                },
                "us_x": {
                    "about": (
                        "The DEM X coordinate for the upstream inlet."),
                    "type": "number",
                    "units": u.pixels,
                },
                "us_y": {
                    "about": (
                        "The DEM Y coordinate for the upstream inlet."),
                    "type": "number",
                    "units": u.pixels,
                },
            },
        },
        "subwatersheds.gpkg": {
            "about": (
                "A GeoPackage with polygon features representing "
                "subwatersheds.  A new subwatershed is created for each "
                "tributary of a stream and is influenced greatly by "
                "your choice of Threshold Flow Accumulation value."),
            "geometries": spec_utils.POLYGON,
            "fields": {
                "stream_id": {
                    "about": (
                        "A unique stream id, matching the one in the Strahler "
                        "stream order vector."),
                    "type": "number",
                    "units": u.none,
                },
                "terminated_early": {
                    "about": (
                        "Indicates whether generation of this subwatershed "
                        "terminated early (1) or completed as expected (0). "
                        "If you encounter a (1), please let us know via the "
                        "forums, community.naturalcapitalproject.org."),
                    "type": "number",
                    "units": u.none,
                },
                "outlet_x": {
                    "about": (
                        "The X coordinate in pixels from the origin of the "
                        "outlet of the watershed. This can be useful when "
                        "determining other properties of the watershed when "
                        "indexing with the underlying raster data."),
                    "type": "number",
                    "units": u.none,
                },
                "outlet_y": {
                    "about": (
                        "The X coordinate in pixels from the origin of the "
                        "outlet of the watershed. This can be useful when "
                        "determining other properties of the watershed when "
                        "indexing with the underlying raster data."),
                    "type": "number",
                    "units": u.none,
                },
            },
        },
    }
}


# replace %s with file suffix
_TARGET_FILLED_PITS_FILED_PATTERN = 'filled%s.tif'
_TARGET_SLOPE_FILE_PATTERN = 'slope%s.tif'
_TARGET_FLOW_DIRECTION_FILE_PATTERN = 'flow_direction%s.tif'
_FLOW_ACCUMULATION_FILE_PATTERN = 'flow_accumulation%s.tif'
_STREAM_MASK_FILE_PATTERN = 'stream_mask%s.tif'
_DOWNSLOPE_DISTANCE_FILE_PATTERN = 'downslope_distance%s.tif'
_STRAHLER_STREAM_ORDER_PATTERN = 'strahler_stream_order%s.gpkg'
_SUBWATERSHEDS_PATTERN = 'subwatersheds%s.gpkg'

_ROUTING_FUNCS = {
    'D8': {
        'flow_accumulation': pygeoprocessing.routing.flow_accumulation_d8,
        'flow_direction': pygeoprocessing.routing.flow_dir_d8,
        'threshold_flow': pygeoprocessing.routing.extract_streams_d8,
        'distance_to_channel': pygeoprocessing.routing.distance_to_channel_d8,
    },
    'MFD': {
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
        ``None``
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    utils.make_directories([args['workspace_dir']])

    if ('calculate_flow_direction' in args and
            bool(args['calculate_flow_direction'])):
        algorithm = args['algorithm'].upper()
        routing_funcs = _ROUTING_FUNCS[algorithm]

    if 'dem_band_index' in args and args['dem_band_index'] not in (None, ''):
        band_index = int(args['dem_band_index'])
    else:
        band_index = 1
    LOGGER.info('Using DEM band index %s', band_index)

    dem_raster_path_band = (args['dem_path'], band_index)

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.

    graph = taskgraph.TaskGraph(
        os.path.join(args['workspace_dir'], 'taskgraph_cache'), n_workers=n_workers)

    # Calculate slope.  This is intentionally on the original DEM, not
    # on the pitfilled DEM.  If the user really wants the slop of the filled
    # DEM, they can pass it back through RouteDEM.
    if bool(args.get('calculate_slope', False)):
        target_slope_path = os.path.join(
            args['workspace_dir'], _TARGET_SLOPE_FILE_PATTERN % file_suffix)
        graph.add_task(
            pygeoprocessing.calculate_slope,
            args=(dem_raster_path_band,
                  target_slope_path),
            task_name='calculate_slope',
            target_path_list=[target_slope_path])

    dem_filled_pits_path = os.path.join(
        args['workspace_dir'],
        _TARGET_FILLED_PITS_FILED_PATTERN % file_suffix)
    filled_pits_task = graph.add_task(
        pygeoprocessing.routing.fill_pits,
        args=(dem_raster_path_band,
              dem_filled_pits_path,
              args['workspace_dir']),
        task_name='fill_pits',
        target_path_list=[dem_filled_pits_path])

    if bool(args.get('calculate_flow_direction', False)):
        LOGGER.info("calculating flow direction")
        flow_dir_path = os.path.join(
            args['workspace_dir'],
            _TARGET_FLOW_DIRECTION_FILE_PATTERN % file_suffix)
        flow_direction_task = graph.add_task(
            routing_funcs['flow_direction'],
            args=((dem_filled_pits_path, 1),  # PGP>1.9.0 creates 1-band fills
                  flow_dir_path,
                  args['workspace_dir']),
            target_path_list=[flow_dir_path],
            dependent_task_list=[filled_pits_task],
            task_name='flow_dir_%s' % algorithm)

        if bool(args.get('calculate_flow_accumulation', False)):
            LOGGER.info("calculating flow accumulation")
            flow_accumulation_path = os.path.join(
                args['workspace_dir'],
                _FLOW_ACCUMULATION_FILE_PATTERN % file_suffix)
            flow_accum_task = graph.add_task(
                routing_funcs['flow_accumulation'],
                args=((flow_dir_path, 1),
                      flow_accumulation_path),
                target_path_list=[flow_accumulation_path],
                task_name='flow_accumulation_%s' % algorithm,
                dependent_task_list=[flow_direction_task])

            if bool(args.get('calculate_stream_threshold', False)):
                stream_mask_path = os.path.join(
                        args['workspace_dir'],
                        _STREAM_MASK_FILE_PATTERN % file_suffix)
                stream_threshold = float(args['threshold_flow_accumulation'])
                stream_extraction_kwargs = {
                    'flow_accum_raster_path_band': (flow_accumulation_path, 1),
                    'flow_threshold': stream_threshold,
                    'target_stream_raster_path': stream_mask_path,
                }
                if algorithm == 'MFD':
                    stream_extraction_kwargs['flow_dir_mfd_path_band'] = (
                        flow_dir_path, 1)
                stream_threshold_task = graph.add_task(
                    routing_funcs['threshold_flow'],
                    kwargs=stream_extraction_kwargs,
                    target_path_list=[stream_mask_path],
                    dependent_task_list=[flow_accum_task],
                    task_name=f'stream_thresholding_{algorithm}')

                if bool(args.get('calculate_downslope_distance', False)):
                    distance_path = os.path.join(
                        args['workspace_dir'],
                        _DOWNSLOPE_DISTANCE_FILE_PATTERN % file_suffix)
                    graph.add_task(
                        routing_funcs['distance_to_channel'],
                        args=((flow_dir_path, 1),
                              (stream_mask_path, 1),
                              distance_path),
                        target_path_list=[distance_path],
                        task_name='downslope_distance_%s' % algorithm,
                        dependent_task_list=[stream_threshold_task])

                # We are only doing stream order for D8 flow direction.
                if (bool(args.get('calculate_stream_order', False)
                         and algorithm == 'D8')):
                    stream_order_path = os.path.join(
                        args['workspace_dir'],
                        _STRAHLER_STREAM_ORDER_PATTERN % file_suffix)
                    stream_order_task = graph.add_task(
                        pygeoprocessing.routing.extract_strahler_streams_d8,
                        kwargs={
                            "flow_dir_d8_raster_path_band":
                                (flow_dir_path, 1),
                            "flow_accum_raster_path_band":
                                (flow_accumulation_path, 1),
                            "dem_raster_path_band":
                                (dem_filled_pits_path, 1),
                            "target_stream_vector_path": stream_order_path,
                            "min_flow_accum_threshold": stream_threshold,
                            "river_order": 5,  # the default
                        },
                        target_path_list=[stream_order_path],
                        task_name='Calculate D8 stream order',
                        dependent_task_list=[
                            filled_pits_task,
                            flow_direction_task,
                            flow_accum_task,
                        ])

                    if bool(args.get('calculate_subwatersheds', False)):
                        subwatersheds_path = os.path.join(
                            args['workspace_dir'],
                            _SUBWATERSHEDS_PATTERN % file_suffix)
                        graph.add_task(
                            pygeoprocessing.routing.calculate_subwatershed_boundary,
                            kwargs={
                                'd8_flow_dir_raster_path_band':
                                    (flow_dir_path, 1),
                                'strahler_stream_vector_path':
                                    stream_order_path,
                                'target_watershed_boundary_vector_path':
                                    subwatersheds_path,
                                'outlet_at_confluence': False,  # The default
                            },
                            target_path_list=[subwatersheds_path],
                            task_name=(
                                'Calculate subwatersheds from stream order'),
                            dependent_task_list=[flow_direction_task,
                                                 stream_order_task])

    graph.close()
    graph.join()


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
    validation_warnings = validation.validate(args, MODEL_SPEC['args'])

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
