"""DelineateIt wrapper for pygeoprocessing's watershed delineation routine."""
import os
import logging
import time
import math

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import shapely.errors
import shapely.geometry
import shapely.wkb
import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph

from . import utils
from . import validation


LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": "DelineateIt: Watershed Delineation",
    "module": __name__,
    "userguide_html": "delineateit.html",
    "args_with_spatial_overlap": {
        "spatial_keys": ["dem_path", "outlet_vector_path"],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "dem_path": {
            "validation_options": {
                "projected": True,
            },
            "type": "raster",
            "required": True,
            "about": (
                "A GDAL-supported raster file with an elevation value for "
                "each cell."),
            "name": "Digital Elevation Model"
        },
        "outlet_vector_path": {
            "type": "vector",
            "required": True,
            "about": (
                "This is a layer of geometries representing watershed "
                "outlets such as municipal water intakes or lakes."),
            "name": "Outlet Features"
        },
        "snap_points": {
            "type": "boolean",
            "required": False,
            "about": (
                "Whether to snap point geometries to the nearest stream "
                "pixel.  If ``True``, ``args['flow_threshold']`` and "
                "``args['snap_distance']`` must also be defined."),
            "name": "Snap points to the nearest stream"
        },
        "flow_threshold": {
            "validation_options": {
                "expression": "value > 0",
            },
            "type": "number",
            "required": "snap_points",
            "about": (
                "The number of upstream cells that must flow into a cell "
                "before it's considered part of a stream such that retention "
                "stops and the remaining export is exported to the stream.  "
                "Used to define streams from the DEM."),
            "name": "Threshold Flow Accumulation"
        },
        "snap_distance": {
            "validation_options": {
                "expression": "value > 0",
            },
            "type": "number",
            "required": "snap_points",
            "about": (
                "If provided, the maximum search radius in pixels to look "
                "for stream pixels.  If a stream pixel is found within the "
                "snap distance, the outflow point will be snapped to the "
                "center of the nearest stream pixel.  Geometries that are "
                "not points (such as Lines and Polygons) will not be "
                "snapped.  MultiPoint geoemtries will also not be snapped."),
            "name": "Pixel Distance to Snap Outlet Points"
        },
        "skip_invalid_geometry": {
            "type": "boolean",
            "required": False,
            "about": (
                "If ``True``, any invalid geometries encountered "
                "in the outlet vector will not be included in the "
                "delineation.  If ``False``, an invalid geometry "
                "will cause DelineateIt to crash."),
            "name": "Crash on invalid geometries"
        }
    }
}

_OUTPUT_FILES = {
    'preprocessed_geometries': 'preprocessed_geometries.gpkg',
    'filled_dem': 'filled_dem.tif',
    'flow_dir_d8': 'flow_direction.tif',
    'flow_accumulation': 'flow_accumulation.tif',
    'streams': 'streams.tif',
    'snapped_outlets': 'snapped_outlets.gpkg',
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
        args['skip_invalid_geometry'] (bool): Whether to crash when an
            invalid geometry is passed or skip it, including all valid
            geometries in the vector to be passed to delineation.
            If ``False``, this tool will crash if an invalid geometry is
            found.  If ``True``, invalid geometries will be left out of
            the vector to be delineated.  Default: True
        args['n_workers'] (int): The number of worker processes to use with
            taskgraph. Defaults to -1 (no parallelism).

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
    # same thread.
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, TypeError, ValueError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1
    graph = taskgraph.TaskGraph(work_token_dir, n_workers=n_workers)

    fill_pits_task = graph.add_task(
        pygeoprocessing.routing.fill_pits,
        args=((args['dem_path'], 1),
              file_registry['filled_dem']),
        kwargs={'working_dir': output_directory},
        target_path_list=[file_registry['filled_dem']],
        task_name='fill_pits',
        hash_algorithm='md5',
        copy_duplicate_artifact=True)

    flow_dir_task = graph.add_task(
        pygeoprocessing.routing.flow_dir_d8,
        args=((file_registry['filled_dem'], 1),
              file_registry['flow_dir_d8']),
        kwargs={'working_dir': output_directory},
        target_path_list=[file_registry['flow_dir_d8']],
        dependent_task_list=[fill_pits_task],
        task_name='flow_direction',
        hash_algorithm='md5',
        copy_duplicate_artifact=True)

    check_geometries_task = graph.add_task(
        check_geometries,
        args=(args['outlet_vector_path'],
              file_registry['filled_dem'],
              file_registry['preprocessed_geometries'],
              args.get('skip_invalid_geometry', True)),
        dependent_task_list=[fill_pits_task],
        target_path_list=[file_registry['preprocessed_geometries']],
        task_name='check_geometries',
        hash_algorithm='md5',
        copy_duplicate_artifact=True)
    outlet_vector_path = file_registry['preprocessed_geometries']

    delineation_dependent_tasks = [flow_dir_task, check_geometries_task]
    if 'snap_points' in args and args['snap_points']:
        flow_accumulation_task = graph.add_task(
            pygeoprocessing.routing.flow_accumulation_d8,
            args=((file_registry['flow_dir_d8'], 1),
                  file_registry['flow_accumulation']),
            target_path_list=[file_registry['flow_accumulation']],
            dependent_task_list=[flow_dir_task],
            task_name='flow_accumulation')
        delineation_dependent_tasks.append(flow_accumulation_task)

        snap_distance = int(args['snap_distance'])
        flow_threshold = int(args['flow_threshold'])

        out_nodata = 255
        flow_accumulation_task.join()  # wait so we can read the nodata value
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
            task_name='threshold_streams',
            hash_algorithm='md5',
            copy_duplicate_artifact=True)

        snapped_outflow_points_task = graph.add_task(
            snap_points_to_nearest_stream,
            args=(outlet_vector_path,
                  (file_registry['streams'], 1),
                  snap_distance,
                  file_registry['snapped_outlets']),
            target_path_list=[file_registry['snapped_outlets']],
            dependent_task_list=[streams_task, check_geometries_task],
            task_name='snapped_outflow_points',
            hash_algorithm='md5',
            copy_duplicate_artifact=True)
        delineation_dependent_tasks.append(snapped_outflow_points_task)
        outlet_vector_path = file_registry['snapped_outlets']

    watershed_delineation_task = graph.add_task(
        pygeoprocessing.routing.delineate_watersheds_d8,
        args=((file_registry['flow_dir_d8'], 1),
              outlet_vector_path,
              file_registry['watersheds']),
        kwargs={'working_dir': output_directory},
        target_path_list=[file_registry['watersheds']],
        dependent_task_list=delineation_dependent_tasks,
        task_name='delineate_watersheds_single_worker',
        hash_algorithm='md5',
        copy_duplicate_artifact=True)

    graph.close()
    graph.join()


def _vector_may_contain_points(vector_path, layer_id=0):
    """Test if a vector layer may contain points.

    This function is intended to be used by the InVEST UI only.

    Parameters:
        vector_path (string): The path to a vector on disk.
        layer_id=0 (int or string): The ID or name of the layer to check.

    Returns:
        A ``bool`` indicating whether a vector contains points.  ``False`` if
        the vector cannot be opened at all or if the layer is invalid.

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    if vector is None:
        return False

    layer = vector.GetLayer(layer_id)
    if layer is None:
        return False

    if layer.GetGeomType() in (ogr.wkbPoint, ogr.wkbUnknown):
        return True
    return False


def _threshold_streams(flow_accum, src_nodata, out_nodata, threshold):
    """Identify stream pixels based on a user-defined threshold.

    This is an ``op`` to ``pygeoprocessing.raster_calculator``.  Any non-nodata
    pixels in ``flow_accum`` greater than the value of ``threshold`` are
    marked as stream pixels.  Any non-nodata pixels below ``threshold`` are
    marked as non-stream pixels.

    Parameters:
        flow_accum (numpy array): A numpy array of flow accumulation values.
        src_nodata (number): A number indicating the nodata value of the
            flow accumulation array.
        out_nodata (number): A number indicating the nodata value of the
            target array.
        threshold (number): A numeric threshold over which a flow
            accumulation pixel will be marked as a stream.

    Returns:
        A ``numpy.uint8`` array with values of 0, 1 or ``out_nodata``.

    """
    out_matrix = numpy.empty(flow_accum.shape, dtype=numpy.uint8)
    out_matrix[:] = out_nodata
    valid_pixels = ~numpy.isclose(src_nodata, flow_accum)
    over_threshold = flow_accum > threshold
    out_matrix[valid_pixels & over_threshold] = 1
    out_matrix[valid_pixels & ~over_threshold] = 0
    return out_matrix


def check_geometries(outlet_vector_path, dem_path, target_vector_path,
                     skip_invalid_geometry=False):
    """Perform reasonable checks and repairs on the incoming vector.

    This function will iterate through the vector at ``outlet_vector_path``
    and validate geometries, putting the geometries into a new geopackage
    at ``target_vector_path``.

    The vector at ``target_vector_path`` will include features that:

        * Have valid geometries
        * Are simplified to 1/2 the DEM pixel size
        * Intersect the bounding box of the DEM

    Any geometries that are empty or do not intersect the DEM will not be
    included in ``target_vector_path``.

    Parameters:
        outlet_vector_path (string): The path to an outflow vector.  The first
            layer of the vector only will be inspected.
        dem_path (string): The path to a DEM on disk.
        target_vector_path (string): The target path to where the output
            geopackage should be written.
        skip_invalid_geometry (bool): Whether to raise an exception
            when invalid geometry is found.  If ``False``, an exception
            will be raised when the first invalid geometry is found.
            If ``True``, the invalid geometry will be not be included
            in the output vector but any other valid geometries will.

    Returns:
        ``None``

    """
    if os.path.exists(target_vector_path):
        LOGGER.debug('Target vector path %s exists on disk; removing.',
                     target_vector_path)
        os.remove(target_vector_path)

    dem_info = pygeoprocessing.get_raster_info(dem_path)
    dem_bbox = shapely.prepared.prep(
        shapely.geometry.box(*dem_info['bounding_box']))
    nyquist_limit = numpy.mean(numpy.abs(dem_info['pixel_size'])) / 2.

    dem_srs = osr.SpatialReference()
    dem_srs.ImportFromWkt(dem_info['projection'])

    gpkg_driver = gdal.GetDriverByName('GPKG')
    target_vector = gpkg_driver.Create(target_vector_path, 0, 0, 0,
                                       gdal.GDT_Unknown)
    target_layer = target_vector.CreateLayer(
        'verified_geometries', dem_srs, ogr.wkbUnknown)  # Use source layer type?

    outflow_vector = gdal.OpenEx(outlet_vector_path, gdal.OF_VECTOR)
    outflow_layer = outflow_vector.GetLayer()
    LOGGER.info('Checking %s geometries from source vector',
                outflow_layer.GetFeatureCount())
    target_layer.CreateFields(outflow_layer.schema)

    target_layer.StartTransaction()
    for feature in outflow_layer:
        original_geometry = feature.GetGeometryRef()

        try:
            shapely_geom = shapely.wkb.loads(original_geometry.ExportToWkb())

            # The classic bowtie polygons will load but require a separate
            # check for validity.
            if not shapely_geom.is_valid:
                raise ValueError('Shapely geom is invalid.')
        except (shapely.errors.ReadingError, ValueError):
            # Parent class for shapely GEOS errors
            # Raised when the geometry is invalid.
            if not skip_invalid_geometry:
                outflow_layer = None
                outflow_vector = None
                target_layer = None
                target_vector = None
                raise ValueError(
                    "The geometry at feature %s is invalid.  Check the logs "
                    "for details and try re-running with repaired geometry."
                    % feature.GetFID())
            else:
                LOGGER.warn(
                    "The geometry at feature %s is invalid and will not be "
                    "included in the set of features to be delineated.",
                    feature.GetFID())
                continue

        if shapely_geom.is_empty:
            LOGGER.warn('Feature %s has no geometry. Skipping', feature.GetFID())
            continue

        shapely_bbox = shapely.geometry.box(*shapely_geom.bounds)
        if not dem_bbox.intersects(shapely_bbox):
            LOGGER.warn('Feature %s does not intersect the DEM. Skipping.',
                         feature.GetFID())
            continue

        simplified_geometry = shapely_geom.simplify(nyquist_limit)

        new_feature = ogr.Feature(target_layer.GetLayerDefn())
        new_feature.SetGeometry(ogr.CreateGeometryFromWkb(
            simplified_geometry.wkb))
        for field_name, field_value in feature.items().items():
            new_feature.SetField(field_name, field_value)
        target_layer.CreateFeature(new_feature)

    target_layer.CommitTransaction()

    LOGGER.info('%s features copied to %s from the original %s features',
                target_layer.GetFeatureCount(),
                os.path.basename(target_vector_path),
                outflow_layer.GetFeatureCount())
    outflow_layer = None
    outflow_vector = None
    target_layer = None
    target_vector = None


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
    n_features = points_layer.GetFeatureCount()
    last_time = time.time()
    for index, point_feature in enumerate(points_layer, 1):
        if time.time() - last_time > 5.0:
            LOGGER.info('Snapped %s of %s points', index, n_features)
            last_time = time.time()

        source_geometry = point_feature.GetGeometryRef()

        # If the geometry is not a primitive point, just create the new feature
        # as it is now in the new vector.
        if source_geometry.GetGeometryName() != 'POINT':
            new_feature = ogr.Feature(snapped_layer.GetLayerDefn())
            new_feature.SetGeometry(source_geometry)
            for field_name, field_value in point_feature.items().items():
                new_feature.SetField(field_name, field_value)
            snapped_layer.CreateFeature(new_feature)
            continue

        point = shapely.wkb.loads(source_geometry.ExportToWkb())
        x_index = (point.x - geotransform[0]) // geotransform[1]
        y_index = (point.y - geotransform[3]) // geotransform[5]
        if (x_index < 0 or x_index > n_cols or
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
                y_center - y_top - row_indexes,
                x_center - x_left - col_indexes,
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
    return validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])
