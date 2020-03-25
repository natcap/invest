"""RouteDEM for exposing the natcap.invest's routing package to UI."""
import os
import logging

from osgeo import gdal
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph
import numpy

from . import utils
from . import validation

LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": "RouteDEM",
    "module": __name__,
    "userguide_html": "routedem.html",
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "dem_path": {
            "type": "raster",
            "required": True,
            "about": (
                "A GDAL-supported raster file containing a base Digital "
                "Elevation Model to execute the routing functionality "
                "across."),
            "name": "Digital Elevation Model"
        },
        "dem_band_index": {
            "validation_options": {
                "expression": "value >= 1",
            },
            "type": "number",
            "required": False,
            "about": (
                "The band index to use from the raster. This positive "
                "integer is 1-based. Default: 1"),
            "name": "Band Index"
        },
        "algorithm": {
            "validation_options": {
                "options": ["D8", "MFD"],
            },
            "type": "option_string",
            "required": True,
            "about": (
                "The routing algorithm to use. "
                "<ul><li>D8: all water flows directly into the most downhill "
                "of each of the 8 neighbors of a cell.</li>"
                "<li>MFD: Multiple Flow Direction. Fractional flow is "
                "modeled between pixels.</li></ul>"),
            "name": "Routing Algorithm"
        },
        "calculate_flow_direction": {
            "type": "boolean",
            "required": False,
            "about": "Select to calculate flow direction",
            "name": "Calculate Flow Direction"
        },
        "calculate_flow_accumulation": {
            "validation_options": {},
            "type": "boolean",
            "required": False,
            "about": "Select to calculate flow accumulation.",
            "name": "Calculate Flow Accumulation"
        },
        "calculate_stream_threshold": {
            "type": "boolean",
            "required": False,
            "about": "Select to calculate a stream threshold to flow accumulation.",
            "name": "Calculate Stream Thresholds"
        },
        "threshold_flow_accumulation": {
            "validation_options": {},
            "type": "number",
            "required": "calculate_stream_threshold",
            "about": (
                "The number of upstream cells that must flow into a cell "
                "before it's classified as a stream."),
            "name": "Threshold Flow Accumulation Limit"
        },
        "calculate_downstream_distance": {
            "type": "boolean",
            "required": False,
            "about": (
                "If selected, creates a downstream distance raster based "
                "on the thresholded flow accumulation stream "
                "classification."),
            "name": "Calculate Distance to stream"
        },
        "calculate_slope": {
            "type": "boolean",
            "required": False,
            "about": "If selected, calculates slope from the provided DEM.",
            "name": "Calculate Slope"
        }
    }
}


# replace %s with file suffix
_TARGET_FILLED_PITS_FILED_PATTERN = 'filled%s.tif'
_TARGET_SLOPE_FILE_PATTERN = 'slope%s.tif'
_TARGET_FLOW_DIRECTION_FILE_PATTERN = 'flow_direction%s.tif'
_FLOW_ACCUMULATION_FILE_PATTERN = 'flow_accumulation%s.tif'
_STREAM_MASK_FILE_PATTERN = 'stream_mask%s.tif'
_DOWNSTREAM_DISTANCE_FILE_PATTERN = 'downstream_distance%s.tif'

_ROUTING_FUNCS = {
    'D8': {
        'flow_accumulation': pygeoprocessing.routing.flow_accumulation_d8,
        'flow_direction': pygeoprocessing.routing.flow_dir_d8,
        'threshold_flow': None,  # Defined in source code as _threshold_flow
        'distance_to_channel': pygeoprocessing.routing.distance_to_channel_d8,
    },
    'MFD': {
        'flow_accumulation': pygeoprocessing.routing.flow_accumulation_mfd,
        'flow_direction': pygeoprocessing.routing.flow_dir_mfd,
        'threshold_flow': pygeoprocessing.routing.extract_streams_mfd,
        'distance_to_channel': pygeoprocessing.routing.distance_to_channel_mfd,
    }
}


def _threshold_flow(flow_accum_pixels, threshold, in_nodata, out_nodata):
    """Raster_calculator local_op to threshold D8 stream flow.

    Parameters:
        flow_accum_pixels (numpy.ndarray): Array representing the number of
            pixels upstream of a given pixel.
        threshold (int or float): The threshold above which we have a stream.
        in_nodata (int or float): The nodata value of the flow accumulation
            raster.
        out_nodata (int): The nodata value of the target stream mask raster.

    Returns:
        A numpy.ndarray (dtype is numpy.uint8) with pixel values of 1 where
        flow accumulation > threshold, 0 where flow accumulation < threshold
        and out_nodata where flow accumulation is equal to in_nodata.

    """
    out_matrix = numpy.empty(flow_accum_pixels.shape, dtype=numpy.uint8)
    out_matrix[:] = out_nodata
    valid_pixels = ~numpy.isclose(flow_accum_pixels, in_nodata)
    stream_mask = (flow_accum_pixels > threshold)
    out_matrix[valid_pixels & stream_mask] = 1
    out_matrix[valid_pixels & ~stream_mask] = 0
    return out_matrix


def execute(args):
    """RouteDEM: Hydrological routing.

    This model exposes the pygeoprocessing D8 and Multiple Flow Direction
    routing functionality as an InVEST model.

    This tool will always fill pits on the input DEM.

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['dem_band_index'] (int): Optional. The band index to operate on.
            If not provided, band index 1 is assumed.
        args['algorithm'] (string): The routing algorithm to use.  Must be
            one of 'D8' or 'MFD' (case-insensitive). Required when calculating
            flow direction, flow accumulation, stream threshold, and downstream
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
        args['threshold_flow_accumulation'] (int): The number of upstream
            cells that must flow into a cell before it's classified as a
            stream.
        args['calculate_downstream_distance'] (bool): If True, and a stream
            threshold is calculated, model will calculate a downstream
            distance raster in units of pixels. Only applies when
            args['calculate_flow_accumulation'],
            args['calculate_flow_direction'], and
            args['calculate_stream_threshold'] are all True.
        args['calculate_slope'] (bool):  If True, model will calculate a
            slope raster from the DEM.
        args['n_workers'] (int): The ``n_workers`` parameter to pass to
            the task graph.  The default is ``-1`` if not provided.

    Returns:
        ``None``
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    task_cache_dir = os.path.join(args['workspace_dir'], '_taskgraph_working_dir')
    utils.make_directories([args['workspace_dir'], task_cache_dir])

    if ('calculate_flow_direction' in args and
            bool(args['calculate_flow_direction'])):
        # All routing functions depend on this one task.
        # Check the algorithm early so we can fail quickly, but only if we're
        # doing some sort of hydological routing
        algorithm = args['algorithm'].upper()
        try:
            routing_funcs = _ROUTING_FUNCS[algorithm]
        except KeyError:
            raise RuntimeError(
                'Invalid algorithm specified (%s). Must be one of %s' % (
                    args['algorithm'], ', '.join(sorted(_ROUTING_FUNCS.keys()))))

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

    graph = taskgraph.TaskGraph(task_cache_dir, n_workers=n_workers)

    # Calculate slope.  This is intentionally on the original DEM, not
    # on the pitfilled DEM.  If the user really wants the slop of the filled
    # DEM, they can pass it back through RouteDEM.
    if 'calculate_slope' in args and bool(args['calculate_slope']):
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

    if ('calculate_flow_direction' in args and
            bool(args['calculate_flow_direction'])):
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

        if ('calculate_flow_accumulation' in args and
                bool(args['calculate_flow_accumulation'])):
            LOGGER.info("calculating flow accumulation")
            flow_accumulation_path = os.path.join(
                args['workspace_dir'],
                _FLOW_ACCUMULATION_FILE_PATTERN % file_suffix)
            flow_accum_task = graph.add_task(
                routing_funcs['flow_accumulation'],
                args=((flow_dir_path, 1),
                      flow_accumulation_path
                ),
                target_path_list=[flow_accumulation_path],
                task_name='flow_accumulation_%s' % algorithm,
                dependent_task_list=[flow_direction_task])

            if ('calculate_stream_threshold' in args and
                    bool(args['calculate_stream_threshold'])):
                stream_mask_path = os.path.join(
                        args['workspace_dir'],
                    _STREAM_MASK_FILE_PATTERN % file_suffix)
                if algorithm == 'D8':
                    flow_accum_task.join()
                    flow_accum_info = pygeoprocessing.get_raster_info(
                        flow_accumulation_path)
                    stream_threshold_task = graph.add_task(
                        pygeoprocessing.raster_calculator,
                        args=(((flow_accumulation_path, 1),
                               (float(args['threshold_flow_accumulation']), 'raw'),
                               (flow_accum_info['nodata'][0], 'raw'),
                               (255, 'raw')),
                              _threshold_flow,
                              stream_mask_path,
                              gdal.GDT_Byte,
                              255),
                        target_path_list=[stream_mask_path],
                        task_name='stream_thresholding_D8',
                        dependent_task_list=[flow_accum_task])
                else:  # MFD
                    stream_threshold_task = graph.add_task(
                        routing_funcs['threshold_flow'],
                        args=((flow_accumulation_path, 1),
                              (flow_dir_path, 1),
                              float(args['threshold_flow_accumulation']),
                              stream_mask_path),
                        target_path_list=[stream_mask_path],
                        task_name=['stream_extraction_MFD'],
                        dependent_task_list=[flow_accum_task])

                if ('calculate_downstream_distance' in args and
                        bool(args['calculate_downstream_distance'])):
                    distance_path = os.path.join(
                        args['workspace_dir'],
                        _DOWNSTREAM_DISTANCE_FILE_PATTERN % file_suffix)
                    graph.add_task(
                        routing_funcs['distance_to_channel'],
                        args=((flow_dir_path, 1),
                              (stream_mask_path, 1),
                              distance_path),
                        target_path_list=[distance_path],
                        task_name='downstream_distance_%s' % algorithm,
                        dependent_task_list=[stream_threshold_task])
    graph.join()


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Parameters:
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
    validation_warnings = validation.validate(args, ARGS_SPEC['args'])

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
                'Must be between 1 and %s' % raster_info['n_bands']))

    return validation_warnings
