"""RouteDEM for exposing the natcap.invest's routing package to UI."""
import os
import logging

import pygeoprocessing.geoprocessing
import pygeoprocessing.routing
import pygeoprocessing.routing.routing_core

LOGGER = logging.getLogger('natcap.invest.routing.routedem')


def execute(args):
    """RouteDEM: D-Infinity Routing.

    This model exposes the pygeoprocessing d-infinity routing functionality in
    the InVEST model API.

    Parameters:
        workspace_dir (string):  The selected folder is used as the workspace
            where all intermediate and output files will be written. If the
            selected folder does not exist, it will be created. If
            datasets already exist in the selected folder, they will be
            overwritten. (required)
        dem_uri (string):  A GDAL-supported raster file containing a base
            Digital Elevation Model to execute the routing functionality
            across. (required)
        pit_filled_filename (string):  The filename of the output raster
            with pits filled in. It will go in the project workspace.
            (required)
        flow_direction_filename (string):  The filename of the flow direction
            raster. It will go in the project workspace. (required)
        flow_accumulation_filename (string):  The filename of the flow
            accumulation raster. It will go in the project workspace.
            (required)
        threshold_flow_accumulation (int):  The number of upstream cells
            that must flow into a cell before it's classified as a stream.
            (required)
        multiple_stream_thresholds (bool):  Set to ``True`` to calculate
            multiple maps. If enabled, set stream threshold to the lowest
            amount, then set upper and step size thresholds. (optional)
        threshold_flow_accumulation_upper (int):  The number of upstream
            pixels that must flow into a cell before it's classified as a
            stream. (required)
        threshold_flow_accumulation_stepsize (int):  The number of cells
            to step up from lower to upper threshold range. (required)
        calculate_slope (bool):  Set to ``True`` to output a slope raster.
            (optional)
        slope_filename (string):  The filename of the output slope raster.
            This will go in the project workspace. (required)
        calculate_downstream_distance (bool):  Select to calculate a distance
            stream raster, based on uppper threshold limit. (optional)
        downstream_distance_filename (string):  The filename of the output
            raster. It will go in the project workspace. (required)

    Returns:
        ``None``
    """
    output_directory = args['workspace_dir']
    LOGGER.info('creating directory %s', output_directory)
    pygeoprocessing.create_directories([output_directory])
    dem_uri = args['dem_uri']

    LOGGER.info('resolving filling pits')

    dem_tiled_uri = os.path.join(
        output_directory, 'dem_tiled.tif')
    pygeoprocessing.geoprocessing.tile_dataset_uri(
        dem_uri, dem_tiled_uri, 256)
    dem_pit_filled_uri = os.path.join(
        output_directory, args['pit_filled_filename'])
    pygeoprocessing.routing.fill_pits(dem_tiled_uri, dem_pit_filled_uri)
    dem_uri = dem_pit_filled_uri

    # Calculate slope
    if args['calculate_slope']:
        LOGGER.info("Calculating slope")
        slope_uri = os.path.join(output_directory, args['slope_filename'])
        pygeoprocessing.geoprocessing.calculate_slope(dem_uri, slope_uri)

    # Calculate flow accumulation
    LOGGER.info("calculating flow direction")
    flow_direction_uri = os.path.join(
        output_directory, args['flow_direction_filename'])
    pygeoprocessing.routing.flow_direction_d_inf(dem_uri, flow_direction_uri)

    LOGGER.info("calculating flow accumulation")
    flow_accumulation_uri = os.path.join(
        output_directory, args['flow_accumulation_filename'])
    pygeoprocessing.routing.flow_accumulation(
        flow_direction_uri, dem_uri, flow_accumulation_uri)

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
            v_stream_uri = os.path.join(
                output_directory, 'v_stream_%s.tif' %
                (str(threshold_amount),))

            pygeoprocessing.routing.stream_threshold(
                flow_accumulation_uri, threshold_amount, v_stream_uri)
    else:
        v_stream_uri = os.path.join(output_directory, 'v_stream.tif')
        pygeoprocessing.routing.stream_threshold(
            flow_accumulation_uri, float(args['threshold_flow_accumulation']),
            v_stream_uri)

    if args['calculate_downstream_distance']:
        distance_uri = os.path.join(
            output_directory, args['downstream_distance_filename'])
        pygeoprocessing.routing.distance_to_stream(
            flow_direction_uri, v_stream_uri, distance_uri)
