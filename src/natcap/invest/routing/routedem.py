"""RouteDEM entry point for exposing the natcap.invest's routing package
    to a UI."""

import os
import logging


import pygeoprocessing.geoprocessing
import pygeoprocessing.routing
import pygeoprocessing.routing.routing_core


logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.routing.routedem')

def execute(args):

    output_directory = args['workspace_dir']
    LOGGER.info('creating directory %s', output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    file_suffix = ''
    dem_uri = args['dem_uri']

    LOGGER.info('resolving filling pits')

    prefix, suffix = os.path.splitext(args['pit_filled_filename'])
    dem_tiled_uri = os.path.join(
        output_directory, 'dem_tiled' + file_suffix + '.tif')
    pygeoprocessing.geoprocessing.tile_dataset_uri(dem_uri, dem_tiled_uri, 256)
    dem_pit_filled_uri = os.path.join(
        output_directory, prefix + file_suffix + suffix)
    pygeoprocessing.routing.fill_pits(dem_tiled_uri, dem_pit_filled_uri)
    dem_uri = dem_pit_filled_uri

    #Calculate slope
    if args['calculate_slope']:
        LOGGER.info("Calculating slope")
        prefix, suffix = os.path.splitext(args['slope_filename'])
        slope_uri = os.path.join(
            output_directory, prefix + file_suffix + suffix)
        pygeoprocessing.geoprocessing.calculate_slope(dem_uri, slope_uri)

    #Calculate flow accumulation
    LOGGER.info("calculating flow direction")
    prefix, suffix = os.path.splitext(args['flow_direction_filename'])
    flow_direction_uri = os.path.join(
        output_directory, prefix + file_suffix + suffix)
    pygeoprocessing.routing.flow_direction_d_inf(dem_uri, flow_direction_uri)

    LOGGER.info("calculating flow accumulation")
    prefix, suffix = os.path.splitext(args['flow_accumulation_filename'])
    flow_accumulation_uri = os.path.join(
        output_directory, prefix + file_suffix + suffix)
    pygeoprocessing.routing.flow_accumulation(
        flow_direction_uri, dem_uri, flow_accumulation_uri)

    #classify streams from the flow accumulation raster
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
            v_stream_uri = os.path.join(
                output_directory, 'v_stream%s_%s.tif' %
                (file_suffix, str(threshold_amount)))

            pygeoprocessing.routing.stream_threshold(
                flow_accumulation_uri, threshold_amount, v_stream_uri)
    else:
        v_stream_uri = os.path.join(
            output_directory, 'v_stream%s.tif' % file_suffix)
        pygeoprocessing.routing.stream_threshold(
            flow_accumulation_uri, float(args['threshold_flow_accumulation']),
            v_stream_uri)

    if args['calculate_downstream_distance']:
        prefix, suffix = os.path.splitext(args['downstream_distance_filename'])
        distance_uri = os.path.join(
            output_directory, prefix + file_suffix + suffix)
        pygeoprocessing.routing.distance_to_stream(
            flow_direction_uri, v_stream_uri, distance_uri)
