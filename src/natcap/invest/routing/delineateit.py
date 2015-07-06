"""DelineateIt entry point for exposing pygeoprocessing's watershed delineation
    routine to a UI."""

import os
import logging

import pygeoprocessing.routing


logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.routing.delineateit')

def execute(args):

    output_directory = args['workspace_dir']
    LOGGER.info('creating directory %s', output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    dem_uri = args['dem_uri']
    outlet_shapefile_uri = args['outlet_shapefile_uri']
    snap_distance = int(args['snap_distance'])
    flow_threshold = int(args['flow_threshold'])

    snapped_outlet_points_uri = os.path.join(
        output_directory, 'snapped_outlets%s.shp' % file_suffix)
    watershed_out_uri = os.path.join(
        output_directory, 'watersheds%s.shp' % file_suffix)
    stream_out_uri = os.path.join(
        output_directory, 'stream%s.tif' % file_suffix)

    pygeoprocessing.routing.delineate_watershed(
        dem_uri, outlet_shapefile_uri, snap_distance,
        flow_threshold, watershed_out_uri,
        snapped_outlet_points_uri, stream_out_uri)
