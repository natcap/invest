"""InVEST Recreation Client"""

import os
import zipfile
import time
import logging
import math

import Pyro4
from osgeo import ogr
import shapely
import shapely.wkt
import shapely.prepared
import pygeoprocessing

from .. import utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_client')

#this serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'

# These are the expected extensions associated with an ESRI Shapefile
# as part of the ESRI Shapefile driver standard, but some extensions
# like .prj, .sbn, and .sbx, are optional depending on versions of the
# format: http://www.gdal.org/drv_shapefile.html
_ESRI_SHAPEFILE_EXTENSIONS = ['.prj', '.shp', '.shx', '.dbf', '.sbn', '.sbx']

_OUTPUT_BASE_FILES = {
    'pud_results_path': 'pud_results.shp',
    }

_TMP_BASE_FILES = {
    'local_aoi_path': 'aoi.shp',
    'compressed_aoi_path': 'aoi.zip',
    'compressed_pud_path': 'pud.zip',
    }


def execute(args):
    """Execute recreation client model on remote server.

    Parameters:
        args['workspace_dir'] (string): path to workspace directory
        args['aoi_path'] (string): path to AOI vector
        args['hostname'] (string): FQDN to recreation server
        args['port'] (string or int): port on hostname for recreation server
        args['start_date'] (string): start date in form YYYY-MM-DD this date
            is the inclusive lower bound to consider points in the PUD and
            regression
        args['end_date'] (string): end date in form YYYY-MM-DD this date
            is the inclusive upper bound to consider points in the PUD and
            regression
        args['aggregating_metric'] (string): one of 'daily', 'monthly', or
            'yearly'.
        args['grid_aoi'] (boolean): if true the polygon vector in
            `args['aoi_path']` should be gridded into a new vector and the
            recreation model should be executed on that
        args['grid_type'] (string): optional, but must exist if
            args['grid_aoi'] is True.  Is one of 'hexagon' or 'square' and
            indicates the style of gridding.
        args['cell_size'] (string/float): optional, but must exist if
            `args['grid_aoi']` is True.  Indicates the long axis size of the
            grid cells.
        args['results_suffix'] (string): optional, if exists is appended to
            any output file paths.

    Returns:
        None."""

    date_range = (args['start_date'], args['end_date'])

    file_suffix = utils.make_suffix_string(
        args, 'results_suffix')

    output_dir = args['workspace_dir']
    pygeoprocessing.create_directories([output_dir])

    file_registry = _build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    recmodel_server = Pyro4.Proxy(
        "PYRO:natcap.invest.recreation@%s:%d" % (
            args['hostname'], int(args['port'])))

    if args['grid_aoi']:
        grid_vector(
            args['aoi_path'], args['grid_type'], args['cell_size'],
            file_registry['local_aoi_path'])
    else:
        aoi_vector = ogr.Open(args['aoi_path'])
        driver = ogr.GetDriverByName('ESRI Shapefile')
        driver.CopyDataSource(aoi_vector, file_registry['local_aoi_path'])
        ogr.DataSource.__swig_destroy__(aoi_vector)
        aoi_vector = None

    basename = os.path.splitext(file_registry['local_aoi_path'])[0]
    with zipfile.ZipFile(file_registry['compressed_aoi_path'], 'w') as aoizip:
        for suffix in _ESRI_SHAPEFILE_EXTENSIONS:
            filename = basename + suffix
            if os.path.exists(filename):
                LOGGER.info('archiving %s', filename)
                aoizip.write(filename, os.path.basename(filename))

    #convert shapefile to binary string for serialization
    zip_file_binary = open(file_registry['compressed_aoi_path'], 'rb').read()

    #transfer zipped file to server
    start_time = time.time()
    LOGGER.info('server version is %s', recmodel_server.get_version())

    result_zip_file_binary = (
        recmodel_server.calc_aggregated_points_in_aoi(
            zip_file_binary, date_range, args['aggregating_metric'],
            os.path.basename(file_registry['pud_results_path'])))
    LOGGER.info('received result, took %f seconds', time.time() - start_time)

    #unpack result
    open(file_registry['compressed_pud_path'], 'wb').write(
        result_zip_file_binary)
    zipfile.ZipFile(file_registry['compressed_pud_path'], 'r').extractall(
        output_dir)

    LOGGER.info('deleting temporary files')
    for file_id in _TMP_BASE_FILES:
        file_path = file_registry[file_id]
        try:
            if file_path.endswith('.shp'):
                #delete like a vector
                driver = ogr.GetDriverByName('ESRI Shapefile')
                driver.DeleteDataSource(file_path)
            else:
                os.remove(file_path)
        except OSError:
            # Let it go.
            pass


def grid_vector(vector_path, grid_type, cell_size, out_grid_vector_path):
    """Convert a vector to a regular grid where grids are contained within
        the original vector.

    Parameters:
        vector_path (string): path to an OGR compatable polygon vector type
        grid_type (string): one of "square" or "hexagon"
        cell_size (float): dimensions of the grid cell in the projected units
            of `vector_path`
        out_grid_vector_path (string): path to the output ESRI shapefile
            vector that contains a gridded version of `vector_path`

    Returns:
        None"""

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_grid_vector_path):
        driver.DeleteDataSource(out_grid_vector_path)

    vector = ogr.Open(vector_path)
    vector_layer = vector.GetLayer()
    spat_ref = vector_layer.GetSpatialRef()

    original_vector_shapes = []
    for feature_id in xrange(vector_layer.GetFeatureCount()):
        feature = vector_layer.GetFeature(feature_id)
        wkt_feat = shapely.wkt.loads(feature.geometry().ExportToWkt())
        original_vector_shapes.append(wkt_feat)
    original_polygon = shapely.prepared.prep(
        shapely.ops.cascaded_union(original_vector_shapes))

    out_grid_vector = driver.CreateDataSource(out_grid_vector_path)
    grid_layer = out_grid_vector.CreateLayer(
        'grid', spat_ref, ogr.wkbPolygon)
    grid_layer_defn = grid_layer.GetLayerDefn()

    extent = vector_layer.GetExtent()  # minx maxx miny maxy

    if grid_type == 'hexagon':
        # calculate the inner domensions of the hexagons
        grid_width = extent[1] - extent[0]
        grid_height = extent[3] - extent[2]
        delta_short_x = cell_size * 0.25
        delta_long_x = cell_size * 0.5
        delta_y = cell_size * 0.25 * (3 ** 0.5)

        #Since the grid is hexagonal it's not obvious how many rows and
        #columns there should be just based on the number of squares that
        #could fit into it.  The solution is to calculate the width and height
        #of the largest row and column.
        n_cols = int(math.floor(grid_width/(3 * delta_long_x)) + 1)
        n_rows = int(math.floor(grid_height/delta_y) + 1)

        def _generate_polygon(col_index, row_index):
            """Generate a points for a closed hexagon given row and col
            index."""
            if (row_index + 1) % 2:
                centroid = (
                    extent[0] + (delta_long_x * (1 + (3 * col_index))),
                    extent[2] + (delta_y * (row_index + 1)))
            else:
                centroid = (
                    extent[0] + (delta_long_x * (2.5 + (3 * col_index))),
                    extent[2] + (delta_y * (row_index + 1)))
            x_coordinate, y_coordinate = centroid
            hexagon = [(x_coordinate - delta_long_x, y_coordinate),
                       (x_coordinate - delta_short_x, y_coordinate + delta_y),
                       (x_coordinate + delta_short_x, y_coordinate + delta_y),
                       (x_coordinate + delta_long_x, y_coordinate),
                       (x_coordinate + delta_short_x, y_coordinate - delta_y),
                       (x_coordinate - delta_short_x, y_coordinate - delta_y),
                       (x_coordinate - delta_long_x, y_coordinate)]
            return hexagon
    elif grid_type == 'square':
        def _generate_polygon(col_index, row_index):
            """Generate points for a closed square given row and col index."""
            square = [
                (extent[0] + col_index * cell_size + x,
                 extent[2] + row_index * cell_size + y)
                for x, y in [
                    (0, 0), (cell_size, 0), (cell_size, cell_size),
                    (0, cell_size), (0, 0)]]
            return square
        n_rows = int((extent[3] - extent[2]) / cell_size)
        n_cols = int((extent[1] - extent[0]) / cell_size)
    else:
        raise ValueError('Unknown polygon type: %s' % grid_type)

    for row_index in xrange(n_rows):
        for col_index in xrange(n_cols):
            polygon_points = _generate_polygon(col_index, row_index)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for xoff, yoff in polygon_points:
                ring.AddPoint(xoff, yoff)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            if original_polygon.contains(
                    shapely.wkt.loads(poly.ExportToWkt())):
                poly_feature = ogr.Feature(grid_layer_defn)
                poly_feature.SetGeometry(poly)
                grid_layer.CreateFeature(poly_feature)
    ogr.DataSource.__swig_destroy__(vector)


def _build_file_registry(base_file_path_list, file_suffix):
    """Construct a file registry by combining file suffixes with file key
    names, base filenames, and directories.

    Parameters:
        base_file_tuple_list (list): a list of (dict, path) tuples where
            the dictionaries have a 'file_key': 'basefilename' pair, or
            'file_key': list of 'basefilename's.  'path'
            indicates the file directory path to prepend to the basefile name.
        file_suffix (string): a string to append to every filename, can be
            empty string

    Returns:
        dictionary of 'file_keys' from the dictionaries in
        `base_file_tuple_list` mapping to full file paths with suffixes or
        lists of file paths with suffixes depending on the original type of
        the 'basefilename' pair.

    Raises:
        ValueError if there are duplicate file keys or duplicate file paths.
    """

    all_paths = set()
    duplicate_keys = set()
    duplicate_paths = set()
    file_registry = {}

    def _build_path(base_filename, path):
        """Internal helper to avoid code duplication"""
        pre, post = os.path.splitext(base_filename)
        full_path = os.path.join(path, pre+file_suffix+post)

        # Check for duplicate keys or paths
        if full_path in all_paths:
            duplicate_paths.add(full_path)
        else:
            all_paths.add(full_path)
        return full_path

    # foo
    for base_file_dict, path in base_file_path_list:
        for file_key, file_payload in base_file_dict.iteritems():
            # check for duplicate keys
            if file_key in file_registry:
                duplicate_keys.add(file_key)
            else:
                # handle the case whether it's a filename or a list of strings
                if isinstance(file_payload, basestring):
                    full_path = _build_path(file_payload, path)
                    file_registry[file_key] = full_path
                elif isinstance(file_payload, list):
                    file_registry[file_key] = []
                    for filename in file_payload:
                        full_path = _build_path(filename, path)
                        file_registry[file_key].append(full_path)

    if len(duplicate_paths) > 0 or len(duplicate_keys):
        raise ValueError(
            "Cannot consolidate because of duplicate paths or keys: "
            "duplicate_keys: %s duplicate_paths: %s" % (
                str(duplicate_keys), str(duplicate_paths)))

    return file_registry


def build_regression_coefficients(
        response_vector_path, predictor_table_path,
        out_coefficient_vector_path):
    """Build a least squares fit for the polygons in the response vector
    dataset and the spatial predictor datasets in `predictor_table_path`.

    Parameters:
        predictor_vector_path (string): path to a single layer polygon vector.
        predictor_table_path (string): path to a CSV file with three columns
            'id', 'path' and 'type' where 'path' indicates the full or
            relative path to the `predictor_table_path` table for the spatial
            predictor dataset and 'type' is one of
                'point_count': count # of points per response polygon
                'point_nearest_distance': distance from nearest point to the
                    centroid of the response polygon
                'line_intersect_length': length of line segments intersecting
                    each response polygon
                'polygon_area': area of predictor polygon intersecting the
                    response polygon
                'raster_sum': sum of predictor raster under the response
                    polygon
                'raster_mean': average of predictor raster under the
                    response polygon
        out_coefficient_vector_path (string): path to a copy of
            `response_vector_path` with the modified predictor variable
            responses. Overwritten if exists.

    Returns:
        None."""

    response_vector = ogr.Open(response_vector_path)
    response_layer = response_vector.GetLayer()
    response_polygons_lookup = {}  # maps FID to prepared geometry
    for response_feature in response_layer:
        feature_geometry = response_feature.GetGeometryRef()
        feature_polygon = shapely.prepared.prep(
            shapely.wkt.loads(feature_geometry.ExportToWkt()))
        feature_geometry = None
        response_polygons_lookup[response_feature.GetFID()] = feature_polygon
    response_layer = None

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_coefficient_vector_path):
        driver.DeleteDataSource(out_coefficient_vector_path)

    out_coefficent_vector = driver.CopyDataSource(
        response_vector, out_coefficient_vector_path)
    ogr.DataSource.__swig_destroy__(response_vector)
    response_vector = None

    out_coefficent_layer = out_coefficent_vector.GetLayer()

    # lookup functions for response types
    predictor_functions = {
        'point_count': _point_count,
        'point_nearest_distance': lambda x, y: {},
        'line_intersect_length': lambda x, y: {},
        'polygon_area': lambda x, y: {},
        'raster_sum': lambda x, y: {},
        'raster_mean': lambda x, y: {}
        }

    predictor_table = pygeoprocessing.get_lookup_from_csv(
        predictor_table_path, 'id')
    for predictor_id in predictor_table:
        LOGGER.info("Building predictor %s", predictor_id)
        predictor_field = ogr.FieldDefn(str(predictor_id), ogr.OFTReal)
        out_coefficent_layer.CreateField(predictor_field)

        raw_path = predictor_table[predictor_id]['path']
        if os.path.isabs(raw_path):
            response_vector_path = raw_path
        else:
            # assume relative path w.r.t. the response table
            response_vector_path = os.path.join(
                os.path.dirname(predictor_table_path), raw_path)

        predictor_type = predictor_table[predictor_id]['type']
        #TODO: worry about a difference in projection between the predictor data and the response polygons
        predictor_results = predictor_functions[predictor_type](
            response_polygons_lookup, response_vector_path)
        for feature_id, value in predictor_results.iteritems():
            feature = out_coefficent_layer.GetFeature(feature_id)
            feature.SetField(str(predictor_id), value)
            out_coefficent_layer.SetFeature(feature)


def _point_count(response_polygons_lookup, point_vector_path):
    """Append number of points that intersect polygons on the
    `response_polygons_lookup`.

    Parameters:
        response_polygons_lookup (dictionary): maps feature ID to
            prepared shapely.Polygon.

        point_vector_path (string): path to a single layer point vector
            object.

    Returns:
        None."""

    points = _ogr_to_geometry_list(point_vector_path)
    point_count_lookup = {}  # map FID to point count
    for feature_id, geometry in response_polygons_lookup.iteritems():
        point_count = len([
            point for point in points if geometry.contains(point)])
        point_count_lookup[feature_id] = point_count
    return point_count_lookup


def _ogr_to_geometry_list(vector_path):
    """Convert an OGR type with one layer to a list of shapely geometry"""

    vector = ogr.Open(vector_path)
    layer = vector.GetLayer()
    geometry_list = []
    for feature in layer:
        feature_geometry = feature.GetGeometryRef()
        shapely_geometry = shapely.wkt.loads(feature_geometry.ExportToWkt())
        geometry_list.append(shapely_geometry)
        feature_geometry = None
    layer = None
    ogr.DataSource.__swig_destroy__(vector)
    return geometry_list
