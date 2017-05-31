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


def execute(args):
    """RouteDEM: D-Infinity Routing.

    This model exposes the pygeoprocessing_0_3_3 d-infinity routing functionality in
    the InVEST model API.

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['calculate_stream_threshold'] (bool): if True, model will
            calculate a stream classification layer by thresholding flow
            accumulation to the provided value in
            args['threshold_flow_accumulation'].
        args['threshold_flow_accumulation'] (int): The number of upstream
            cells that must flow into a cell before it's classified as a
            stream.
        args['calculate_downstream_distance'] (bool): If True, model will
        args['calculate_slope'] (bool):  Set to ``True`` to output a slope
            raster.

    Returns:
        ``None``
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    utils.make_directories([args['workspace_dir']])

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
    target_flow_direction_path = os.path.join(
        args['workspace_dir'],
        _TARGET_FLOW_DIRECTION_FILE_PATTERN % file_suffix)
    pygeoprocessing.routing.flow_direction_d_inf(
        dem_raster_path_band, target_flow_direction_path)

    sys.exit()

    LOGGER.info("calculating flow accumulation")
    flow_accumulation_path = os.path.join(
        output_directory, args['flow_accumulation_filename'])
    natcap.invest.pygeoprocessing_0_3_3.routing.flow_accumulation(
        flow_direction_path, dem_path, flow_accumulation_path)

    # classify streams from the flow accumulation raster
    LOGGER.info("Classifying streams from flow accumulation raster")

    if args['multiple_stream_thresholds']:
        lower_threshold = int(args['threshold_flow_accumulation'])
        upper_threshold = int(args['threshold_flow_accumulation_upper'])
        threshold_step = int(args['threshold_flow_accumulation_stepsize'])

        for threshold_amount in range(
                lower_threshold, upper_threshold+1, threshold_step):
            LOGGER.info(
                "Calculating stream threshold at %s pixels",
                threshold_amount)
            v_stream_path = os.path.join(
                output_directory, 'v_stream_%s.tif' %
                (str(threshold_amount),))

            natcap.invest.pygeoprocessing_0_3_3.routing.stream_threshold(
                flow_accumulation_path, threshold_amount, v_stream_path)
    else:
        v_stream_path = os.path.join(output_directory, 'v_stream.tif')
        natcap.invest.pygeoprocessing_0_3_3.routing.stream_threshold(
            flow_accumulation_path, float(args['threshold_flow_accumulation']),
            v_stream_path)

    if args['calculate_downstream_distance']:
        distance_path = os.path.join(
            output_directory, args['downstream_distance_filename'])
        natcap.invest.pygeoprocessing_0_3_3.routing.distance_to_stream(
            flow_direction_path, v_stream_path, distance_path)
