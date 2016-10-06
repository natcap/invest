"""DelineateIt wrapper for pygeoprocessing's watershed delineation routine."""

import os
import logging

import pygeoprocessing.routing

from .. import utils

LOGGER = logging.getLogger('natcap.invest.routing.delineateit')


def execute(args):
    """Delineateit: Watershed Delineation.

    This 'model' provides an InVEST-based wrapper around the pygeoprocessing
    routing API for watershed delineation.

    Upon successful completion, the following files are written to the
    output workspace:

        * ``snapped_outlets.shp`` - an ESRI shapefile with the points snapped
          to a nearby stream.
        * ``watersheds.shp`` - an ESRI shapefile of watersheds determined
          by the d-infinity routing algorithm.
        * ``stream.tif`` - a GeoTiff representing detected streams based on
          the provided ``flow_threshold`` parameter.  Values of 1 are
          streams, values of 0 are not.

    Parameters:
        workspace_dir (string):  The selected folder is used as the workspace
            all intermediate and output files will be written.If the
            selected folder does not exist, it will be created. If
            datasets already exist in the selected folder, they will be
            overwritten. (required)
        results_suffix (string):  This text will be appended to the end of
            output files to help separate multiple runs. (optional)
        dem_uri (string):  A GDAL-supported raster file with an elevation
            for each cell. Make sure the DEM is corrected by filling in sinks,
            and if necessary burning hydrographic features into the elevation
            model (recommended when unusual streams are observed.) See the
            'Working with the DEM' section of the InVEST User's Guide for more
            information. (required)
        outlet_shapefile_uri (string):  This is a vector of points representing
            points that the watersheds should be built around. (required)
        flow_threshold (int):  The number of upstream cells that must
            into a cell before it's considered part of a stream such that
            retention stops and the remaining export is exported to the stream.
            Used to define streams from the DEM. (required)
        snap_distance (int):  Pixel Distance to Snap Outlet Points (required)

    Returns:
        None
    """
    output_directory = args['workspace_dir']
    LOGGER.info('creating directory %s', output_directory)
    pygeoprocessing.create_directories([output_directory])
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

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
