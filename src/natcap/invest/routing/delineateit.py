"""DelineateIt wrapper for natcap.invest.pygeoprocessing_0_3_3's watershed delineation routine."""
from __future__ import absolute_import
import os
import logging

from osgeo import gdal
from osgeo import ogr
import natcap.invest.pygeoprocessing_0_3_3.routing

from .. import utils
from .. import validation


LOGGER = logging.getLogger('natcap.invest.routing.delineateit')


def execute(args):
    """Delineateit: Watershed Delineation.

    This 'model' provides an InVEST-based wrapper around the natcap.invest.pygeoprocessing_0_3_3
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
        suffix (string):  This text will be appended to the end of
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
    natcap.invest.pygeoprocessing_0_3_3.create_directories([output_directory])
    file_suffix = utils.make_suffix_string(args, 'suffix')

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

    natcap.invest.pygeoprocessing_0_3_3.routing.delineate_watershed(
        dem_uri, outlet_shapefile_uri, snap_distance,
        flow_threshold, watershed_out_uri,
        snapped_outlet_points_uri, stream_out_uri)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Parameters:
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
    missing_key_list = []
    no_value_list = []
    validation_error_list = []

    required_keys = [
        'workspace_dir',
        'dem_uri',
        'outlet_shapefile_uri',
        'flow_threshold',
        'snap_distance']

    for key in required_keys:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ['', None]:
                no_value_list.append(key)

    if len(missing_key_list) > 0:
        # if there are missing keys, we have raise KeyError to stop hard
        raise KeyError(
            "The following keys were expected in `args` but were missing " +
            ', '.join(missing_key_list))

    if len(no_value_list) > 0:
        validation_error_list.append(
            (no_value_list, 'parameter has no value'))

    file_type_list = [
        ('dem_uri', 'raster'),
        ('outlet_shapefile_uri', 'vector')]

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in file_type_list:
            if (limit_to in [None, key]) and key in required_keys:
                if not os.path.exists(args[key]):
                    validation_error_list.append(
                        ([key], 'not found on disk'))
                    continue
                if key_type == 'raster':
                    raster = gdal.OpenEx(args[key])
                    if raster is None:
                        validation_error_list.append(
                            ([key], 'not a raster'))
                    del raster
                elif key_type == 'vector':
                    vector = gdal.OpenEx(args[key])
                    if vector is None:
                        validation_error_list.append(
                            ([key], 'not a vector'))
                    del vector

    return validation_error_list
