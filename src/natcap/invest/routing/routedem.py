"""RouteDEM for exposing the natcap.invest's routing package to UI."""
import os
import logging

import pygeoprocessing
import pygeoprocessing.routing
import natcap.invest.pygeoprocessing_0_3_3.routing
import natcap.invest.pygeoprocessing_0_3_3.routing.routing_core

from .. import utils

LOGGER = logging.getLogger('natcap.invest.routing.routedem')

# replace %s with file suffix
_TARGET_SLOPE_FILE_PATTERN = 'slope%s.tif'
_TARGET_FLOW_DIRECTION_FILE_PATTERN = 'flow_direction%s.tif'
_FLOW_ACCUMULATION_FILE_PATTERN = 'flow_accumulation%s.tif'
_STREAM_MASK_FILE_PATTERN = 'stream_mask%s.tif'
_DOWNSTREAM_DISTANCE_FILE_PATTERN = 'downstream_distance%s.tif'


def execute(args):
    """RouteDEM: D-Infinity Routing.

    This model exposes the pygeoprocessing_0_3_3 d-infinity routing
    functionality as an InVEST model.

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['calculate_flow_accumulation'] (bool): If True, model will
            calculate a flow accumulation raster.
        args['calculate_stream_threshold'] (bool): if True, model will
            calculate a stream classification layer by thresholding flow
            accumulation to the provided value in
            args['threshold_flow_accumulation'].
        args['threshold_flow_accumulation'] (int): The number of upstream
            cells that must flow into a cell before it's classified as a
            stream.
        args['calculate_downstream_distance'] (bool): If True, and a stream
            threshold is calculated, model will calculate a downstream
            distance raster in units of pixels.
        args['calculate_slope'] (bool):  If True, model will calculate a
            slope raster.

    Returns:
        ``None``
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    utils.make_directories([args['workspace_dir']])
    dem_info = pygeoprocessing.get_raster_info(args['dem_path'])
    if dem_info['n_bands'] > 1:
        LOGGER.warn(
            "There are more than 1 bands in %s.  RouteDEM will only operate "
            "on band 1.", args['dem_path'])
    dem_raster_path_band = (args['dem_path'], 1)

    # Calculate slope
    if 'calculate_slope' in args and bool(args['calculate_slope']):
        LOGGER.info("Calculating slope")
        target_slope_path = os.path.join(
            args['workspace_dir'], _TARGET_SLOPE_FILE_PATTERN % file_suffix)
        pygeoprocessing.calculate_slope(
            dem_raster_path_band, target_slope_path)

    # Calculate flow accumulation
    LOGGER.info("calculating flow direction")
    flow_direction_path = os.path.join(
        args['workspace_dir'],
        _TARGET_FLOW_DIRECTION_FILE_PATTERN % file_suffix)
    natcap.invest.pygeoprocessing_0_3_3.routing.flow_direction_d_inf(
        args['dem_path'], flow_direction_path)

    if ('calculate_flow_accumulation' in args and
            bool(args['calculate_flow_accumulation'])):
        LOGGER.info("calculating flow accumulation")
        flow_accumulation_path = os.path.join(
            args['workspace_dir'],
            _FLOW_ACCUMULATION_FILE_PATTERN % file_suffix)
        natcap.invest.pygeoprocessing_0_3_3.routing.flow_accumulation(
            flow_direction_path, args['dem_path'], flow_accumulation_path)

        if ('calculate_stream_threshold' in args and
                bool(args['calculate_stream_threshold'])):
            LOGGER.info("Classifying streams from flow accumulation raster")

            flow_accumulation_threshold = float(
                args['threshold_flow_accumulation'])

            LOGGER.info(
                "Calculating stream threshold at %s pixels",
                flow_accumulation_threshold)
            stream_mask_path = os.path.join(
                args['workspace_dir'],
                _STREAM_MASK_FILE_PATTERN % file_suffix)

            natcap.invest.pygeoprocessing_0_3_3.routing.stream_threshold(
                flow_accumulation_path, flow_accumulation_threshold,
                stream_mask_path)

            if ('calculate_downstream_distance' in args and
                    bool(args['calculate_downstream_distance'])):
                distance_path = os.path.join(
                    args['workspace_dir'],
                    _DOWNSTREAM_DISTANCE_FILE_PATTERN % file_suffix)
                natcap.invest.pygeoprocessing_0_3_3.routing.distance_to_stream(
                    flow_direction_path, stream_mask_path, distance_path)
