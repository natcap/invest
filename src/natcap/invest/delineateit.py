"""DelineateIt wrapper for pygeoprocessing's watershed delineation routine."""
from __future__ import absolute_import
import os
import logging

from osgeo import gdal
from osgeo import ogr
import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph

from . import utils
from . import validation


LOGGER = logging.getLogger(__name__)

_OUTPUT_FILES = {
    'filled_dem': 'filled_dem.tif',
    'flow_dir_d8': 'flow_direction.tif',
    'flow_accumulation': 'flow_accumulation.tif',
    'streams': 'streams.tif',
    'snapped_outlets': 'snapped_outlets.gpkg',
    'watershed_fragments': 'watershed_fragments.gpkg',
    'watersheds': 'watersheds.gpkg',
}


def execute(args):
    """DelineateIt: Watershed Delineation.

    This 'model' provides an InVEST-based wrapper around the pygeoprocessing
    routing API for watershed delineation.

    Upon successful completion, the following files are written to the
    output workspace:

        * ``snapped_outlets.gpkg`` - A GeoPackage with the points snapped
          to a nearby stream.
        * ``watersheds.gpkg`` - a GeoPackage of watersheds determined
          by the D8 routing algorithm.
        * ``stream.tif`` - a GeoTiff representing detected streams based on
          the provided ``flow_threshold`` parameter.  Values of 1 are
          streams, values of 0 are not.

    Parameters:
        args['workspace_dir'] (string):  The selected folder is used as the
            workspace all intermediate and output files will be written.If the
            selected folder does not exist, it will be created. If datasets
            already exist in the selected folder, they will be overwritten.
            (required)
        args['results_suffix'] (string):  This text will be appended to the end of
            output files to help separate multiple runs. (optional)
        args['dem_path'] (string):  A GDAL-supported raster file with an elevation
            for each cell. Make sure the DEM is corrected by filling in sinks,
            and if necessary burning hydrographic features into the elevation
            model (recommended when unusual streams are observed.) See the
            'Working with the DEM' section of the InVEST User's Guide for more
            information. (required)
        args['outlet_vector_path'] (string):  This is a vector representing
            geometries that the watersheds should be built around. (required)
        args['snap_points'] (bool): Whether to snap point geometries to the
            nearest stream pixel.  If ``True``, ``args['flow_threshold']``
            and ``args['snap_distance']`` must also be defined.
        args['flow_threshold'] (int):  The number of upstream cells that must
            into a cell before it's considered part of a stream such that
            retention stops and the remaining export is exported to the stream.
            Used to define streams from the DEM.
        args['snap_distance'] (int):  Pixel Distance to Snap Outlet Points

    Returns:
        ``None``
    """
    output_directory = args['workspace_dir']
    utils.make_directories([output_directory])

    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_FILES, output_directory)], file_suffix)

    work_token_dir = os.path.join(output_directory, '_work_tokens')

    # Manually setting n_workers to be -1 so that everything happens in the
    # same thread.  In the current implementation of delineateit, all
    # tasks are perfectly sequential ... there's no opportunity to parallelize
    # computational work.
    graph = taskgraph.TaskGraph(work_token_dir, n_workers=-1)

    fill_pits_task = graph.add_task(
        pygeoprocessing.routing.fill_pits,
        args=((args['dem_path'], 1),
              file_registry['filled_dem']),
        kwargs={'working_dir': output_directory},
        target_path_list=[file_registry['filled_dem']],
        task_name='fill_pits')

    flow_dir_task = graph.add_task(
        pygeoprocessing.routing.flow_dir_d8,
        args=((file_registry['filled_dem'], 1),
              file_registry['flow_dir_d8']),
        kwargs={'working_dir': output_directory},
        target_path_list=[file_registry['flow_dir_d8']],
        dependent_task_list=[fill_pits_task],
        task_name='flow_direction')

    flow_accumulation_task = graph.add_task(
        pygeoprocessing.routing.flow_accumulation_d8,
        args=((file_registry['flow_dir_d8'], 1),
              file_registry['flow_accumulation']),
        target_path_list=[file_registry['flow_accumulation']],
        dependent_task_list=[flow_dir_task],
        task_name='flow_accumulation')

    flow_accumulation_task.join()
    delineation_dependent_tasks = [flow_accumulation_task]
    outflow_vector = args['outlet_vector_path']
    if 'snap_points' in args and args['snap_points']:
        snap_distance = int(args['snap_distance'])
        flow_threshold = int(args['flow_threshold'])

        out_nodata = 255
        flow_accumulation_nodata = pygeoprocessing.get_raster_info(
            file_registry['flow_accumulation'])['nodata']
        streams_task = graph.add_task(
            pygeoprocessing.raster_calculator,
            args=([(file_registry['flow_accumulation'], 1),
                   (flow_accumulation_nodata, 'raw'),
                   (out_nodata, 'raw'),
                   (flow_threshold, 'raw')],
                  _threshold_streams,
                  file_registry['streams'],
                  gdal.GDT_Byte,
                  out_nodata),
            target_path_list=[file_registry['streams']],
            dependent_task_list=[flow_accumulation_task],
            task_name='threshold_streams')

        snapped_outflow_points_task = graph.add_task(
            snap_points_to_nearest_stream,
            args=(args['outlet_vector_path'],
                  (file_registry['streams'], 1),
                  snap_distance,
                  file_registry['snapped_outlets']),
            target_path_list=[file_registry['snapped_outlets']],
            dependent_task_list=[streams_task],
            task_name='snapped_outflow_points')
        delineation_dependent_tasks.append(snapped_outflow_points_task)
        outflow_vector = file_registry['snapped_outlets']

    watershed_delineation_task = graph.add_task(
        pygeoprocessing.routing.delineate_watersheds_trivial_d8,
        args=((file_registry['flow_dir_d8'], 1),
              outflow_vector,
              file_registry['watersheds']),
        kwargs={'working_dir': output_directory},
        target_path_list=[file_registry['watersheds']],
        dependent_task_list=delineation_dependent_tasks,
        task_name='delineate_watersheds')

    graph.join()


def _vector_may_contain_points(vector_path):
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    if vector is None:
        return False

    layer = vector.GetLayer()
    if layer.GetGeomType() in (ogr.wkbPoint, ogr.wkbUnknown):
        return True
    return False


def _threshold_streams(flow_accum, src_nodata, out_nodata, threshold):
    out_matrix = numpy.empty(flow_accum.shape, dtype=numpy.int8)
    out_matrix[:] = out_nodata
    valid_pixels = ~numpy.isclose(src_nodata, flow_accum)
    over_threshold = flow_accum > threshold
    out_matrix[valid_pixels & over_threshold] = 1
    out_matrix[valid_pixels & ~over_threshold] = 0
    return out_matrix


# TODO: if two streams are the same distance from a pixel, pick the one with the higher flow accumulation.
def snap_points_to_nearest_stream(points_vector_path, stream_raster_path_band,
                                  snap_distance, snapped_points_vector_path):
    """Adjust the location of points to the nearest stream pixel.

    The new point layer will have all fields and field values copied over from
    the source vector.  Any points that are outside of the stream raster will
    not be included in the output vector.

    Parameters:
        points_vector_path (string): A path to a vector on disk containing
            point geometries.  Must be in the same projection as the stream
            raster.
        stream_raster_path_band (tuple): A tuple of (path, band index), where
            pixel values are ``1`` (indicating a stream pixel) or ``0``
            (indicating a non-stream pixel).
        snap_distance (number): The maximum distance (in pixels) to search
            for stream pixels for each point.  This must be a positive, nonzero
            value.
        snapped_points_vector_path (string): A path to where the output
            points will be written.

    Returns:
        ``None``

    Raises:
        ``ValueError`` when snap_distance is less than or equal to 0.

    """
    if snap_distance <= 0:
        raise ValueError('Snap_distance must be >= 0, not %s' % snap_distance)

    points_vector = gdal.OpenEx(points_vector_path, gdal.OF_VECTOR)
    points_layer = points_vector.GetLayer()

    stream_raster_info = pygeoprocessing.get_raster_info(
        stream_raster_path_band[0])
    geotransform = stream_raster_info['geotransform']
    n_cols, n_rows = stream_raster_info['raster_size']
    stream_raster = gdal.OpenEx(stream_raster_path_band[0], gdal.OF_RASTER)
    stream_band = stream_raster.GetRasterBand(stream_raster_path_band[1])

    driver = gdal.GetDriverByName('GPKG')
    snapped_vector = driver.Create(snapped_points_vector_path, 0, 0, 0,
                                   gdal.GDT_Unknown)
    snapped_layer = snapped_vector.CreateLayer(
        'snapped', points_layer.GetSpatialRef(), points_layer.GetGeomType())
    snapped_layer.CreateFields(points_layer.schema)
    snapped_layer_defn = snapped_layer.GetLayerDefn()

    snapped_layer.StartTransaction()
    # TODO: add time-based logging
    for point_feature in points_layer:
        point_geometry = point_feature.GetGeometryRef()

        # If the geometry is not a primitive point, just create the new feature
        # as it is now in the new vector.
        if point_geometry.GetGeometryName() != 'POINT':
            snapped_layer.CreateFeature(point_feature.Clone())
            continue

        point = point_geometry.GetPoint()
        x_index = (point[0] - geotransform[0]) // geotransform[1]
        y_index = (point[1] - geotransform[3]) // geotransform[5]
        if (x_index < 0 or x_index >= n_cols or
                y_index < 0 or y_index > n_rows):
            LOGGER.warn('Encountered a point that was outside the bounds of '
                        'the stream raster.  FID:%s at %s',
                        point_feature.GetFID(), point)
            continue

        x_center = x_index
        y_center = y_index
        x_left = max(x_index - snap_distance, 0)
        y_top = max(y_index - snap_distance, 0)
        x_right = min(x_index + snap_distance, n_cols)
        y_bottom = min(y_index + snap_distance, n_rows)

        # snap to the nearest stream pixel out to the snap distance
        stream_window = stream_band.ReadAsArray(
            int(x_left), int(y_top), int(x_right - x_left),
            int(y_bottom - y_top))
        row_indexes, col_indexes = numpy.nonzero(
            stream_window == 1)
        if row_indexes.size > 0:
            # Calculate euclidean distance for sorting
            distance_array = numpy.hypot(
                row_indexes - y_center - y_top,
                col_indexes - x_center - x_left,
                dtype=numpy.float32)

            # Find the closest stream pixel that meets the distance
            # requirement.
            min_index = numpy.argmin(distance_array)
            min_row = row_indexes[min_index]
            min_col = col_indexes[min_index]
            offset_row = min_row - (y_center - y_top)
            offset_col = min_col - (x_center - x_left)

            y_index += offset_row
            x_index += offset_col

        point_geometry = ogr.Geometry(ogr.wkbPoint)
        point_geometry.AddPoint(
            geotransform[0] + (x_index + 0.5) * geotransform[1],
            geotransform[3] + (y_index + 0.5) * geotransform[5])

        # Get the output Layer's Feature Definition
        snapped_point_feature = ogr.Feature(snapped_layer_defn)
        for field_name, field_value in point_feature.items().items():
            snapped_point_feature.SetField(field_name, field_value)
        snapped_point_feature.SetGeometry(point_geometry)

        snapped_layer.CreateFeature(snapped_point_feature)
    snapped_layer.CommitTransaction()
    snapped_layer = None
    snapped_vector = None

    points_layer = None
    points_vector = None


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
        'dem_path',
        'outlet_vector_path',
        'snap_points',
    ]
    if 'snap_points' in args and args['snap_points']:
        required_keys += ['flow_threshold', 'snap_distance']

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
        ('dem_path', 'raster'),
        ('outlet_vector_path', 'vector')]

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in file_type_list:
            if (limit_to in [None, key]) and key in required_keys:
                if not os.path.exists(args[key]):
                    validation_error_list.append(
                        ([key], 'not found on disk'))
                    continue
                if key_type == 'raster':
                    raster = gdal.OpenEx(args[key], gdal.OF_RASTER)
                    if raster is None:
                        validation_error_list.append(
                            ([key], 'not a raster'))
                    del raster
                elif key_type == 'vector':
                    vector = gdal.OpenEx(args[key], gdal.OF_VECTOR)
                    if vector is None:
                        validation_error_list.append(
                            ([key], 'not a vector'))
                    del vector

    if 'snap_points' in args and args['snap_points']:
        for key in ('flow_threshold', 'snap_distance'):
            if limit_to in (None, key):
                try:
                    value = int(args[key])
                    if value < 0:
                        validation_error_list.append(
                            ([key], 'must be a positive integer'))
                except (TypeError, ValueError):
                    validation_error_list.append(
                        ([key], 'must be an integer'))

    return validation_error_list
