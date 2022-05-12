"""DelineateIt wrapper for pygeoprocessing's watershed delineation routine."""
import logging
import os
import time

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import shapely.errors
import shapely.geometry
import shapely.wkb
import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from .. import gettext
from .. import spec_utils
from .. import utils
from .. import validation
from ..model_metadata import MODEL_METADATA
from ..spec_utils import u
from . import delineateit_core

LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": MODEL_METADATA["delineateit"].model_title,
    "pyname": MODEL_METADATA["delineateit"].pyname,
    "userguide": MODEL_METADATA["delineateit"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["dem_path", "outlet_vector_path"],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "dem_path": {
            **spec_utils.DEM,
            "projected": True
        },
        "detect_pour_points": {
            "type": "boolean",
            "required": False,
            "about": gettext(
                "Detect pour points (watershed outlets) based on "
                "the DEM, and use these instead of a user-provided outlet "
                "features vector."),
            "name": gettext("detect pour points")
        },
        "outlet_vector_path": {
            "type": "vector",
            "fields": {},
            "geometries": spec_utils.ALL_GEOMS,
            "required": "not detect_pour_points",
            "about": gettext(
                "A map of watershed outlets from which to delineate the "
                "watersheds. Required if Detect Pour Points is not checked."),
            "name": gettext("watershed outlets")
        },
        "snap_points": {
            "type": "boolean",
            "required": False,
            "about": gettext(
                "Whether to snap point geometries to the nearest stream "
                "pixel.  If ``True``, ``args['flow_threshold']`` and "
                "``args['snap_distance']`` must also be defined. If a point "
                "is equally near to more than one stream pixel, it will be "
                "snapped to the stream pixel with the highest flow "
                "accumulation value. This has no effect if Detect Pour Points "
                "is selected."),
            "name": gettext("snap points to the nearest stream")
        },
        "flow_threshold": {
            **spec_utils.THRESHOLD_FLOW_ACCUMULATION,
            "required": "snap_points",
            "about": gettext(
                spec_utils.THRESHOLD_FLOW_ACCUMULATION["about"] +
                " Required if Snap Points is selected."),
        },
        "snap_distance": {
            "expression": "value > 0",
            "type": "number",
            "units": u.pixels,
            "required": "snap_points",
            "about": gettext(
                "Maximum distance to relocate watershed outlet points in "
                "order to snap them to a stream. Required if Snap Points "
                "is selected."),
            "name": gettext("snap distance")
        },
        "skip_invalid_geometry": {
            "type": "boolean",
            "required": False,
            "about": gettext(
                "Skip delineation for any invalid geometries found in the "
                "Outlet Features. Otherwise, an invalid geometry will cause "
                "the model to crash."),
            "name": gettext("skip invalid geometries")
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
    'pour_points': 'pour_points.gpkg'
}
_WS_ID_OVERWRITE_WARNING = (
    'Layer {layer_name} of vector {vector_basename} already has a feature '
    'named "ws_id". Field values will be overwritten.')


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

    Args:
        args['workspace_dir'] (string):  The selected folder is used as the
            workspace all intermediate and output files will be written.If the
            selected folder does not exist, it will be created. If datasets
            already exist in the selected folder, they will be overwritten.
            (required)
        args['results_suffix'] (string):  This text will be appended to the end
            of output files to help separate multiple runs. (optional)
        args['dem_path'] (string):  A GDAL-supported raster file with an
            elevation for each cell. Make sure the DEM is corrected by filling
            in sinks, and if necessary burning hydrographic features into the
            elevation model (recommended when unusual streams are observed.)
            See the 'Working with the DEM' section of the InVEST User's Guide
            for more information. (required)
        args['outlet_vector_path'] (string):  This is a vector representing
            geometries that the watersheds should be built around. Required if
            ``args['detect_pour_points']`` is False; not used otherwise.
        args['snap_points'] (bool): Whether to snap point geometries to the
            nearest stream pixel.  If ``True``, ``args['flow_threshold']``
            and ``args['snap_distance']`` must also be defined.
        args['flow_threshold'] (int):  The number of upslope cells that must
            flow into a cell before it's considered part of a stream such that
            retention stops and the remaining export is exported to the stream.
            Used to define streams from the DEM.
        args['snap_distance'] (int):  Pixel Distance to Snap Outlet Points
        args['skip_invalid_geometry'] (bool): Whether to crash when an
            invalid geometry is passed or skip it, including all valid
            geometries in the vector to be passed to delineation.
            If ``False``, this tool will crash if an invalid geometry is
            found.  If ``True``, invalid geometries will be left out of
            the vector to be delineated.  Default: True
        args['detect_pour_points'] (bool): Whether to run the pour point
            detection algorithm. If True, detected pour points are used instead
            of outlet_vector_path geometries. Default: False
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
        task_name='fill_pits')

    flow_dir_task = graph.add_task(
        pygeoprocessing.routing.flow_dir_d8,
        args=((file_registry['filled_dem'], 1),
              file_registry['flow_dir_d8']),
        kwargs={'working_dir': output_directory},
        target_path_list=[file_registry['flow_dir_d8']],
        dependent_task_list=[fill_pits_task],
        task_name='flow_direction')

    if 'detect_pour_points' in args and args['detect_pour_points']:
        # Detect pour points automatically and use them instead of
        # user-provided geometries
        pour_points_task = graph.add_task(
            detect_pour_points,
            args=((file_registry['flow_dir_d8'], 1),
                  file_registry['pour_points']),
            dependent_task_list=[flow_dir_task],
            target_path_list=[file_registry['pour_points']],
            task_name='detect_pour_points')
        outlet_vector_path = file_registry['pour_points']
        geometry_task = pour_points_task
    else:
        preprocess_geometries_task = graph.add_task(
            preprocess_geometries,
            args=(args['outlet_vector_path'],
                  file_registry['filled_dem'],
                  file_registry['preprocessed_geometries'],
                  args.get('skip_invalid_geometry', True)),
            dependent_task_list=[fill_pits_task],
            target_path_list=[file_registry['preprocessed_geometries']],
            task_name='preprocess_geometries')
        outlet_vector_path = file_registry['preprocessed_geometries']
        geometry_task = preprocess_geometries_task

    delineation_dependent_tasks = [flow_dir_task, geometry_task]
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
            task_name='threshold_streams')

        snapped_outflow_points_task = graph.add_task(
            snap_points_to_nearest_stream,
            args=(outlet_vector_path,
                  file_registry['streams'],
                  file_registry['flow_accumulation'],
                  snap_distance,
                  file_registry['snapped_outlets']),
            target_path_list=[file_registry['snapped_outlets']],
            dependent_task_list=[streams_task, geometry_task],
            task_name='snapped_outflow_points')
        delineation_dependent_tasks.append(snapped_outflow_points_task)
        outlet_vector_path = file_registry['snapped_outlets']

    _ = graph.add_task(
        pygeoprocessing.routing.delineate_watersheds_d8,
        args=((file_registry['flow_dir_d8'], 1),
              outlet_vector_path,
              file_registry['watersheds']),
        kwargs={'working_dir': output_directory,
                'target_layer_name':
                    os.path.splitext(
                        os.path.basename(file_registry['watersheds']))[0]},
        target_path_list=[file_registry['watersheds']],
        dependent_task_list=delineation_dependent_tasks,
        task_name='delineate_watersheds_single_worker')

    graph.close()
    graph.join()


def _threshold_streams(flow_accum, src_nodata, out_nodata, threshold):
    """Identify stream pixels based on a user-defined threshold.

    This is an ``op`` to ``pygeoprocessing.raster_calculator``.  Any non-nodata
    pixels in ``flow_accum`` greater than the value of ``threshold`` are
    marked as stream pixels.  Any non-nodata pixels below ``threshold`` are
    marked as non-stream pixels.

    Args:
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

    valid_pixels = slice(None)
    if src_nodata is not None:
        valid_pixels = ~utils.array_equals_nodata(flow_accum, src_nodata)

    over_threshold = flow_accum > threshold
    out_matrix[valid_pixels & over_threshold] = 1
    out_matrix[valid_pixels & ~over_threshold] = 0
    return out_matrix


def preprocess_geometries(outlet_vector_path, dem_path, target_vector_path,
                          skip_invalid_geometry=False):
    """Preprocess geometries in the incoming vector.

    This function will iterate through the vector at ``outlet_vector_path``
    and validate geometries, putting the geometries into a new geopackage
    at ``target_vector_path``.  All output features will also have a `ws_id`
    column created, containing a unique integer ID.

    The vector at ``target_vector_path`` will include features that:

        * Have valid geometries
        * Are simplified to 1/2 the DEM pixel size
        * Intersect the bounding box of the DEM

    Any geometries that are empty or do not intersect the DEM will not be
    included in ``target_vector_path``.

    Args:
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
    dem_srs.ImportFromWkt(dem_info['projection_wkt'])

    gpkg_driver = gdal.GetDriverByName('GPKG')
    target_vector = gpkg_driver.Create(target_vector_path, 0, 0, 0,
                                       gdal.GDT_Unknown)
    layer_name = os.path.splitext(os.path.basename(target_vector_path))[0]
    target_layer = target_vector.CreateLayer(
        layer_name, dem_srs, ogr.wkbUnknown)  # Use source layer type?

    outflow_vector = gdal.OpenEx(outlet_vector_path, gdal.OF_VECTOR)
    outflow_layer = outflow_vector.GetLayer()
    if 'ws_id' in set([field.GetName() for field in outflow_layer.schema]):
        LOGGER.warning(_WS_ID_OVERWRITE_WARNING.format(
            layer_name=outflow_layer.GetName(),
            vector_basename=os.path.basename(outlet_vector_path)))
    else:
        target_layer.CreateField(ogr.FieldDefn('ws_id', ogr.OFTInteger64))

    target_layer.CreateFields(outflow_layer.schema)

    LOGGER.info('Checking %s geometries from source vector',
                outflow_layer.GetFeatureCount())
    target_layer.StartTransaction()
    ws_id = 0
    for feature in outflow_layer:
        original_geometry = feature.GetGeometryRef()

        try:
            shapely_geom = shapely.wkb.loads(
                bytes(original_geometry.ExportToWkb()))

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
                LOGGER.warning(
                    "The geometry at feature %s is invalid and will not be "
                    "included in the set of features to be delineated.",
                    feature.GetFID())
                continue

        if shapely_geom.is_empty:
            LOGGER.warning(
                'Feature %s has no geometry. Skipping', feature.GetFID())
            continue

        shapely_bbox = shapely.geometry.box(*shapely_geom.bounds)
        if not dem_bbox.intersects(shapely_bbox):
            LOGGER.warning('Feature %s does not intersect the DEM. Skipping.',
                           feature.GetFID())
            continue

        simplified_geometry = shapely_geom.simplify(nyquist_limit)

        new_feature = ogr.Feature(target_layer.GetLayerDefn())
        new_feature.SetGeometry(ogr.CreateGeometryFromWkb(
            simplified_geometry.wkb))
        for field_name, field_value in feature.items().items():
            new_feature.SetField(field_name, field_value)

        # In case we're skipping features, ws_id field won't include any gaps
        # in the numbers in this field.  I suspect this may save us some time
        # on the forums later.
        new_feature.SetField('ws_id', ws_id)
        ws_id += 1

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


def snap_points_to_nearest_stream(points_vector_path, stream_raster_path,
                                  flow_accum_raster_path, snap_distance,
                                  snapped_points_vector_path):
    """Adjust the location of points to the nearest stream pixel.

    The new point layer will have all fields and field values copied over from
    the source vector.  Any points that are outside of the stream raster will
    not be included in the output vector.

    Args:
        points_vector_path (string): A path to a vector on disk containing
            point geometries.  Must be in the same projection as the stream
            raster.
        stream_raster_path (string): A path to a stream raster, where
            pixel values are ``1`` (indicating a stream pixel) or ``0``
            (indicating a non-stream pixel).
        flow_accum_raster_path (string): A path to a flow accumulation raster
            that is aligned with the stream raster. Used to break ties between
            equally-near stream pixels.
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

    stream_raster_info = pygeoprocessing.get_raster_info(stream_raster_path)
    geotransform = stream_raster_info['geotransform']
    n_cols, n_rows = stream_raster_info['raster_size']
    stream_raster = gdal.OpenEx(stream_raster_path, gdal.OF_RASTER)
    stream_band = stream_raster.GetRasterBand(1)

    flow_accum_raster = gdal.OpenEx(flow_accum_raster_path, gdal.OF_RASTER)
    flow_accum_band = flow_accum_raster.GetRasterBand(1)

    driver = gdal.GetDriverByName('GPKG')
    snapped_vector = driver.Create(snapped_points_vector_path, 0, 0, 0,
                                   gdal.GDT_Unknown)
    layer_name = os.path.splitext(
        os.path.basename(snapped_points_vector_path))[0]
    snapped_layer = snapped_vector.CreateLayer(
        layer_name, points_layer.GetSpatialRef(), points_layer.GetGeomType())
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
        geom_name = source_geometry.GetGeometryName()
        geom_count = source_geometry.GetGeometryCount()

        if source_geometry.IsEmpty():
            LOGGER.warning(
                f"FID {point_feature.GetFID()} is missing a defined geometry. "
                "Skipping.")
            continue

        # If the geometry is not a primitive point, just create the new feature
        # as it is now in the new vector. MULTIPOINT geometries with a single
        # component point count as primitive points.
        # OGR's wkbMultiPoint, wkbMultiPointM, wkbMultiPointZM and
        # wkbMultiPoint25D all use the MULTIPOINT geometry name.
        if ((geom_name not in ('POINT', 'MULTIPOINT')) or
                (geom_name == 'MULTIPOINT' and geom_count > 1)):
            LOGGER.warning(
                f"FID {point_feature.GetFID()} ({geom_name}, n={geom_count}) "
                "Geometry cannot be snapped.")
            new_feature = ogr.Feature(snapped_layer.GetLayerDefn())
            new_feature.SetGeometry(source_geometry)
            for field_name, field_value in point_feature.items().items():
                new_feature.SetField(field_name, field_value)
            snapped_layer.CreateFeature(new_feature)
            continue

        point = shapely.wkb.loads(bytes(source_geometry.ExportToWkb()))
        if geom_name == 'MULTIPOINT':
            # We already checked (above) that there's only one component point
            point = point.geoms[0]

        x_index = (point.x - geotransform[0]) // geotransform[1]
        y_index = (point.y - geotransform[3]) // geotransform[5]
        if (x_index < 0 or x_index > n_cols or
                y_index < 0 or y_index > n_rows):
            LOGGER.warning(
                'Encountered a point that was outside the bounds of the '
                f'stream raster.  FID:{point_feature.GetFID()} at {point}')
            continue

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

        # Find the closest stream pixel that meets the distance
        # requirement. If there is a tie, snap to the stream pixel with
        # a higher flow accumulation value.
        if row_indexes.size > 0:  # there are streams within the snap distance
            # Calculate euclidean distance from the point to each stream pixel
            distance_array = numpy.hypot(
                # distance along y axis from the point to each stream pixel
                y_index - y_top - row_indexes,
                # distance along x axis from the point to each stream pixel
                x_index - x_left - col_indexes,
                dtype=numpy.float32)

            is_nearest = distance_array == distance_array.min()
            # if > 1 stream pixel is nearest, break tie with flow accumulation
            if is_nearest.sum() > 1:
                flow_accum_array = flow_accum_band.ReadAsArray(
                    int(x_left), int(y_top), int(x_right - x_left),
                    int(y_bottom - y_top))
                # weight by flow accum
                is_nearest = is_nearest * flow_accum_array[row_indexes, col_indexes]

            # 1d index of max value in flattened array
            nearest_stream_index_1d = numpy.argmax(is_nearest)

            # convert 1d index back to coordinates relative to window
            nearest_stream_row = row_indexes[nearest_stream_index_1d]
            nearest_stream_col = col_indexes[nearest_stream_index_1d]

            offset_row = nearest_stream_row - (y_index - y_top)
            offset_col = nearest_stream_col - (x_index - x_left)

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


def detect_pour_points(flow_dir_raster_path_band, target_vector_path):
    """
    Create a pour point vector from D8 flow direction raster.

    A pour point is the center point of a pixel which:
        - flows off of the raster, or
        - flows into a nodata pixel

    Args:
        flow_dir_raster_path_band (tuple): tuple of (raster path, band index)
            indicating the flow direction raster to use. Pixel values are D8
            values [0 - 7] in this order:

                321
                4x0
                567

        target_vector_path (string): path to save pour point vector to.

    Returns:
        None
    """
    raster_info = pygeoprocessing.get_raster_info(flow_dir_raster_path_band[0])
    pour_point_set = _find_raster_pour_points(flow_dir_raster_path_band)

    # use same spatial reference as the input
    aoi_spatial_reference = osr.SpatialReference()
    aoi_spatial_reference.ImportFromWkt(raster_info['projection_wkt'])

    gpkg_driver = ogr.GetDriverByName("GPKG")
    target_vector = gpkg_driver.CreateDataSource(target_vector_path)
    layer_name = os.path.splitext(
        os.path.basename(target_vector_path))[0]
    target_layer = target_vector.CreateLayer(
        layer_name, aoi_spatial_reference, ogr.wkbPoint)
    target_defn = target_layer.GetLayerDefn()

    # It's important to have a user-facing unique ID field for post-processing
    # (e.g. table-joins) that is not the FID. FIDs are not stable across file
    # conversions that users might do.
    target_layer.CreateField(
        ogr.FieldDefn('ws_id', ogr.OFTInteger64))

    # Add a feature to the layer for each point
    target_layer.StartTransaction()
    for idx, (x, y) in enumerate(pour_point_set):
        geometry = ogr.Geometry(ogr.wkbPoint)
        geometry.AddPoint(x, y)
        feature = ogr.Feature(target_defn)
        feature.SetGeometry(geometry)
        feature.SetField('ws_id', idx)
        target_layer.CreateFeature(feature)
    target_layer.CommitTransaction()

    target_layer = None
    target_vector = None


def _find_raster_pour_points(flow_dir_raster_path_band):
    """
    Memory-safe pour point calculation from a flow direction raster.

    Args:
        flow_dir_raster_path_band (tuple): tuple of (raster path, band index)
            indicating the flow direction raster to use.

    Returns:
        set of (x, y) coordinate tuples of pour points, in the same coordinate
        system as the input raster.
    """
    flow_dir_raster_path, band_index = flow_dir_raster_path_band
    raster_info = pygeoprocessing.get_raster_info(flow_dir_raster_path)
    # Open the flow direction raster band
    raster = gdal.OpenEx(flow_dir_raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(band_index)
    width, height = raster_info['raster_size']

    pour_points = set()
    # Read in flow direction data and find pour points one block at a time
    for offsets in pygeoprocessing.iterblocks(
            (flow_dir_raster_path, band_index), offset_only=True):
        # Expand each block by a one-pixel-wide margin, if possible.
        # This way the blocks will overlap so the watershed
        # calculation will be continuous.
        if offsets['xoff'] > 0:
            offsets['xoff'] -= 1
            offsets['win_xsize'] += 1
        if offsets['yoff'] > 0:
            offsets['yoff'] -= 1
            offsets['win_ysize'] += 1
        if offsets['xoff'] + offsets['win_xsize'] < width:
            offsets['win_xsize'] += 1
        if offsets['yoff'] + offsets['win_ysize'] < height:
            offsets['win_ysize'] += 1

        # Keep track of which block edges are raster edges
        edges = numpy.empty(4, dtype=numpy.intc)
        # edges order: top, left, bottom, right
        edges[0] = (offsets['yoff'] == 0)
        edges[1] = (offsets['xoff'] == 0)
        edges[2] = (offsets['yoff'] + offsets['win_ysize'] == height)
        edges[3] = (offsets['xoff'] + offsets['win_xsize'] == width)

        flow_dir_block = band.ReadAsArray(**offsets)
        pour_points = pour_points.union(
            delineateit_core.calculate_pour_point_array(
                # numpy.intc is equivalent to an int in C (normally int32 or
                # int64). This way it can be passed directly into a memoryview
                # (int[:, :]) in the Cython function.
                flow_dir_block.astype(numpy.intc),
                edges,
                nodata=raster_info['nodata'][band_index - 1],
                offset=(offsets['xoff'], offsets['yoff']),
                origin=(raster_info['geotransform'][0],
                        raster_info['geotransform'][3]),
                pixel_size=raster_info['pixel_size']))

    raster = None
    band = None

    return pour_points


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
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
