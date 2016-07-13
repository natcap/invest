"""InVEST Recreation Client."""

import uuid
import os
import zipfile
import time
import logging
import math
import urllib
import tempfile
import shutil

import Pyro4
from osgeo import ogr
from osgeo import gdal
from osgeo import osr
import shapely
import shapely.geometry
import shapely.wkt
import shapely.prepared
import pygeoprocessing
import numpy
import numpy.linalg

# prefer to do intrapackage imports to avoid case where global package is
# installed and we import the global version of it rather than the local
from .. import utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_client')
# This URL is a NatCap global constant
RECREATION_SERVER_URL = 'http://data.naturalcapitalproject.org/server_registry/invest_recreation_model/'  # pylint: disable=line-too-long

# 'marshal' serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'

# These are the expected extensions associated with an ESRI Shapefile
# as part of the ESRI Shapefile driver standard, but some extensions
# like .prj, .sbn, and .sbx, are optional depending on versions of the
# format: http://www.gdal.org/drv_shapefile.html
_ESRI_SHAPEFILE_EXTENSIONS = ['.prj', '.shp', '.shx', '.dbf', '.sbn', '.sbx']

# Have 5.0 seconds between timed progress outputs
LOGGER_TIME_DELAY = 5.0

# For now, this is the field name we use to mark the photo user "days"
RESPONSE_ID = 'PUD_YR_AVG'
SCENARIO_RESPONSE_ID = 'PUD_EST'

_OUTPUT_BASE_FILES = {
    'pud_results_path': 'pud_results.shp',
    'coefficent_vector_path': 'regression_coefficients.shp',
    'scenario_results_path': 'scenario_results.shp',
    'regression_coefficients': 'regression_coefficients.txt',
    }

_TMP_BASE_FILES = {
    'local_aoi_path': 'aoi.shp',
    'compressed_aoi_path': 'aoi.zip',
    'compressed_pud_path': 'pud.zip',
    'tmp_indexed_vector_path': 'indexed_vector.shp',
    'tmp_fid_raster_path': 'vector_fid_raster.tif',
    'tmp_scenario_indexed_vector_path': 'scenario_indexed_vector.shp',
    }


def execute(args):
    """Recreation.

    Execute recreation client model on remote server.

    Parameters:
        args['workspace_dir'] (string): path to workspace directory
        args['aoi_path'] (string): path to AOI vector
        args['hostname'] (string): FQDN to recreation server
        args['port'] (string or int): port on hostname for recreation server
        args['start_year'] (string): start year in form YYYY.  This year
            is the inclusive lower bound to consider points in the PUD and
            regression.
        args['end_year'] (string): end year in form YYYY.  This year
            is the inclusive upper bound to consider points in the PUD and
            regression.
        args['grid_aoi'] (boolean): if true the polygon vector in
            `args['aoi_path']` should be gridded into a new vector and the
            recreation model should be executed on that
        args['grid_type'] (string): optional, but must exist if
            args['grid_aoi'] is True.  Is one of 'hexagon' or 'square' and
            indicates the style of gridding.
        args['cell_size'] (string/float): optional, but must exist if
            `args['grid_aoi']` is True.  Indicates the cell size of square
            pixels and the width of the horizontal axis for the hexagonal
            cells.
        args['compute_regression'] (boolean): if True, then process the
            predictor table and scenario table (if present).
        args['predictor_table_path'] (string): required if
            args['compute_regression'] is True.  Path to a table that
            describes the regression predictors, their IDs and types.  Must
            contain the fields 'id', 'path', and 'type' where:

                * 'id': is a <=10 character length ID that is used to uniquely
                  describe the predictor.  It will be added to the output
                  result shapefile attribute table which is an ESRI
                  Shapefile, thus limited to 10 characters.
                * 'path': an absolute or relative (to this table) path to the
                  predictor dataset, either a vector or raster type.
                * 'type': one of the following,

                    * 'raster_mean': mean of values in the raster under the
                      response polygon
                    * 'raster_sum': sum of values in the raster under the
                      response polygon
                    * 'point_count': count of the points contained in the
                      response polygon
                    * 'point_nearest_distance': distance to the nearest point
                      from the response polygon
                    * 'line_intersect_length': length of lines that intersect
                      with the response polygon in projected units of AOI
                    * 'polygon_area': area of the polygon contained within
                      response polygon in projected units of AOI

        args['scenario_predictor_table_path'] (string): (optional) if
            present runs the scenario mode of the recreation model with the
            datasets described in the table on this path.  Field headers
            are identical to `args['predictor_table_path']` and ids in the
            table are required to be identical to the predictor list.
        args['results_suffix'] (string): optional, if exists is appended to
            any output file paths.

    Returns:
        None
    """
    if ('predictor_table_path' in args and
            args['predictor_table_path'] != ''):
        _validate_same_id_lengths(args['predictor_table_path'])
        _validate_same_projection(
            args['aoi_path'], args['predictor_table_path'])
    if ('predictor_table_path' in args and
            'scenario_predictor_table_path' in args and
            args['predictor_table_path'] != '' and
            args['scenario_predictor_table_path'] != ''):
        _validate_same_ids_and_types(
            args['predictor_table_path'],
            args['scenario_predictor_table_path'])
        _validate_same_projection(
            args['aoi_path'], args['scenario_predictor_table_path'])

    if int(args['end_year']) < int(args['start_year']):
        raise ValueError(
            "Start year must be less than or equal to end year.\n"
            "start_year: %s\nend_year: %s" % (
                args['start_year'], args['end_year']))

    # in case the user defines a hostname
    if 'hostname' in args:
        path = "PYRO:natcap.invest.recreation@%s:%s" % (
            args['hostname'], args['port'])
    else:
        # else use a well known path to get active server
        path = urllib.urlopen(RECREATION_SERVER_URL).read().rstrip()

    LOGGER.info('Contacting server, please wait.')
    recmodel_server = Pyro4.Proxy(path)
    server_version = recmodel_server.get_version()
    LOGGER.info('Server online, version: %s', server_version)

    # validate available year range
    min_year, max_year = recmodel_server.get_valid_year_range()
    LOGGER.info(
        "Server supports year queries between %d and %d", min_year, max_year)
    if not min_year <= int(args['start_year']) <= max_year:
        raise ValueError(
            "Start year must be between %d and %d.\n"
            " User input: (%s)" % (min_year, max_year, args['start_year']))
    if not min_year <= int(args['end_year']) <= max_year:
        raise ValueError(
            "End year must be between %d and %d.\n"
            " User input: (%s)" % (min_year, max_year, args['end_year']))

    # append jan 1 to start and dec 31 to end
    date_range = (args['start_year']+'-01-01', args['end_year']+'-12-31')
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    output_dir = args['workspace_dir']
    pygeoprocessing.create_directories([output_dir])

    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    if args['grid_aoi']:
        LOGGER.info("gridding aoi")
        _grid_vector(
            args['aoi_path'], args['grid_type'], args['cell_size'],
            file_registry['local_aoi_path'])
    else:
        aoi_vector = ogr.Open(args['aoi_path'])
        driver = ogr.GetDriverByName('ESRI Shapefile')
        driver.CopyDataSource(aoi_vector, file_registry['local_aoi_path'])
        # hard destroying the object just in case.  during testing I sometimes
        # had issues with shapefiles being unable to delete, __swig_destroy__
        # alleviated that
        ogr.DataSource.__swig_destroy__(aoi_vector)
        aoi_vector = None

    basename = os.path.splitext(file_registry['local_aoi_path'])[0]
    with zipfile.ZipFile(file_registry['compressed_aoi_path'], 'w') as aoizip:
        for suffix in _ESRI_SHAPEFILE_EXTENSIONS:
            filename = basename + suffix
            if os.path.exists(filename):
                LOGGER.info('archiving %s', filename)
                aoizip.write(filename, os.path.basename(filename))

    # convert shapefile to binary string for serialization
    zip_file_binary = open(file_registry['compressed_aoi_path'], 'rb').read()

    # transfer zipped file to server
    start_time = time.time()
    LOGGER.info('Please wait for server to calculate PUD...')

    result_zip_file_binary, workspace_id = (
        recmodel_server.calc_photo_user_days_in_aoi(
            zip_file_binary, date_range,
            os.path.basename(file_registry['pud_results_path'])))
    LOGGER.info(
        'received result, took %f seconds, workspace_id: %s',
        time.time() - start_time, workspace_id)

    # unpack result
    open(file_registry['compressed_pud_path'], 'wb').write(
        result_zip_file_binary)
    temporary_output_dir = tempfile.mkdtemp()
    zipfile.ZipFile(file_registry['compressed_pud_path'], 'r').extractall(
        temporary_output_dir)
    monthly_table_path = os.path.join(
        temporary_output_dir, 'monthly_table.csv')
    if os.path.exists(monthly_table_path):
        os.rename(
            monthly_table_path,
            os.path.splitext(monthly_table_path)[0] + file_suffix + '.csv')
    for filename in os.listdir(temporary_output_dir):
        shutil.copy(os.path.join(temporary_output_dir, filename), output_dir)
    shutil.rmtree(temporary_output_dir)

    if 'compute_regression' in args and args['compute_regression']:
        LOGGER.info('Calculating regression')
        predictor_id_list = []
        _build_regression_coefficients(
            file_registry['pud_results_path'], args['predictor_table_path'],
            file_registry['tmp_indexed_vector_path'],
            file_registry['coefficent_vector_path'], predictor_id_list)

        coefficents, ssreg, r_sq, r_sq_adj, std_err, dof, se_est = (
            _build_regression(
                file_registry['coefficent_vector_path'], RESPONSE_ID,
                predictor_id_list))

        # the last coefficient is the y intercept and has no id, thus
        # the [:-1] on the coefficients list
        coefficents_string = '               estimate     stderr    t value\n'
        coefficents_string += '%-12s %+.3e %+.3e %+.3e\n' % (
            '(Intercept)', coefficents[-1], se_est[-1],
            coefficents[-1] / se_est[-1])
        coefficents_string += '\n'.join(
            '%-12s %+.3e %+.3e %+.3e' % (
                p_id, coefficent, se_est_factor, coefficent / se_est_factor)
            for p_id, coefficent, se_est_factor in zip(
                predictor_id_list, coefficents[:-1], se_est[:-1]))

        # generate a nice looking regression result and write to log and file
        report_string = (
            '\n******************************\n'
            '%s\n'
            '---\n\n'
            'Residual standard error: %.4f on %d degrees of freedom\n'
            'Multiple R-squared: %.4f\n'
            'Adjusted R-squared: %.4f\n'
            'SSreg: %.4f\n'
            'server id hash: %s\n'
            '********************************\n' % (
                coefficents_string, std_err, dof, r_sq, r_sq_adj, ssreg,
                server_version))
        LOGGER.info(report_string)
        with open(file_registry['regression_coefficients'], 'w') as \
                regression_log:
            regression_log.write(report_string + '\n')

        if ('scenario_predictor_table_path' in args and
                args['scenario_predictor_table_path'] != ''):
            LOGGER.info('Calculating scenario')
            _calculate_scenario(
                file_registry['pud_results_path'], SCENARIO_RESPONSE_ID,
                coefficents, predictor_id_list,
                args['scenario_predictor_table_path'],
                file_registry['tmp_scenario_indexed_vector_path'],
                file_registry['scenario_results_path'])

    LOGGER.info('deleting temporary files')
    for file_id in _TMP_BASE_FILES:
        file_path = file_registry[file_id]
        try:
            if file_path.endswith('.shp') and os.path.exists(file_path):
                driver = ogr.GetDriverByName('ESRI Shapefile')
                driver.DeleteDataSource(file_path)
            else:
                os.remove(file_path)
        except OSError:
            pass  # let it go


def _grid_vector(vector_path, grid_type, cell_size, out_grid_vector_path):
    """Convert vector to a regular grid.

    Here the vector is gridded such that all cells are contained within the
    original vector.  Cells that would intersect with the boundary are not
    produced.

    Parameters:
        vector_path (string): path to an OGR compatible polygon vector type
        grid_type (string): one of "square" or "hexagon"
        cell_size (float): dimensions of the grid cell in the projected units
            of `vector_path`; if "square" then this indicates the side length,
            if "hexagon" indicates the width of the horizontal axis.
        out_grid_vector_path (string): path to the output ESRI shapefile
            vector that contains a gridded version of `vector_path`, this file
            should not exist before this call

    Returns:
        None
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_grid_vector_path):
        driver.DeleteDataSource(out_grid_vector_path)

    vector = ogr.Open(vector_path)
    vector_layer = vector.GetLayer()
    spat_ref = vector_layer.GetSpatialRef()

    original_vector_shapes = []
    for feature in vector_layer:
        wkt_feat = shapely.wkt.loads(feature.geometry().ExportToWkt())
        original_vector_shapes.append(wkt_feat)
    vector_layer.ResetReading()
    original_polygon = shapely.prepared.prep(
        shapely.ops.cascaded_union(original_vector_shapes))

    out_grid_vector = driver.CreateDataSource(out_grid_vector_path)
    grid_layer = out_grid_vector.CreateLayer(
        'grid', spat_ref, ogr.wkbPolygon)
    grid_layer_defn = grid_layer.GetLayerDefn()

    extent = vector_layer.GetExtent()  # minx maxx miny maxy
    if grid_type == 'hexagon':
        # calculate the inner dimensions of the hexagons
        grid_width = extent[1] - extent[0]
        grid_height = extent[3] - extent[2]
        delta_short_x = cell_size * 0.25
        delta_long_x = cell_size * 0.5
        delta_y = cell_size * 0.25 * (3 ** 0.5)

        # Since the grid is hexagonal it's not obvious how many rows and
        # columns there should be just based on the number of squares that
        # could fit into it.  The solution is to calculate the width and
        # height of the largest row and column.
        n_cols = int(math.floor(grid_width / (3 * delta_long_x)) + 1)
        n_rows = int(math.floor(grid_height / delta_y) + 1)

        def _generate_polygon(col_index, row_index):
            """Generate a points for a closed hexagon."""
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
            """Generate points for a closed square."""
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


def _build_regression_coefficients(
        response_vector_path, predictor_table_path,
        tmp_indexed_vector_path, out_coefficient_vector_path,
        out_predictor_id_list):
    """Calculate least squares fit for the polygons in the response vector.

    Build a least squares regression from the log normalized response vector,
    spatial predictor datasets in `predictor_table_path`, and a column of 1s
    for the y intercept.

    Parameters:
        response_vector_path (string): path to a single layer polygon vector.
        predictor_table_path (string): path to a CSV file with three columns
            'id', 'path' and 'type'.  'id' is the unique ID for that predictor
            and must be less than 10 characters long. 'path' indicates the
            full or relative path to the `predictor_table_path` table for the
            spatial predictor dataset. 'type' is one of:
                'point_count': count # of points per response polygon
                'point_nearest_distance': distance from nearest point to the
                    centroid of the response polygon
                'line_intersect_length': length of line segments intersecting
                    each response polygon
                'polygon_area': area of predictor polygon intersecting the
                    response polygon
                'polygon_percent_coverage': percent of predictor polygon
                    intersecting the response polygon
                'raster_sum': sum of predictor raster under the response
                    polygon
                'raster_mean': average of predictor raster under the
                    response polygon
        tmp_indexed_vector_path (string): path to temporary working file in
            case the response vector needs an index added
        out_coefficient_vector_path (string): path to a copy of
            `response_vector_path` with the modified predictor variable
            responses. Overwritten if exists.
        out_predictor_id_list (list): a list that is overwritten with the
            predictor ids that are added to the coefficient vector attribute
            table.

    Returns:
        None
    """
    response_vector = ogr.Open(response_vector_path)
    response_layer = response_vector.GetLayer()
    response_polygons_lookup = {}  # maps FID to prepared geometry
    for response_feature in response_layer:
        feature_geometry = response_feature.GetGeometryRef()
        feature_polygon = shapely.wkt.loads(feature_geometry.ExportToWkt())
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
        'point_nearest_distance': _point_nearest_distance,
        'line_intersect_length': _line_intersect_length,
        'polygon_area_coverage': lambda x, y: _polygon_area('area', x, y),
        'polygon_percent_coverage': lambda x, y: _polygon_area(
            'percent', x, y),
        }

    predictor_table = pygeoprocessing.get_lookup_from_csv(
        predictor_table_path, 'id')
    out_predictor_id_list[:] = predictor_table.keys()

    for predictor_id in predictor_table:
        LOGGER.info("Building predictor %s", predictor_id)
        # Delete the field if it already exists
        field_index = out_coefficent_layer.FindFieldIndex(
            str(predictor_id), 1)
        if field_index >= 0:
            out_coefficent_layer.DeleteField(field_index)
        predictor_field = ogr.FieldDefn(str(predictor_id), ogr.OFTReal)
        predictor_field.SetWidth(24)
        predictor_field.SetPrecision(11)
        out_coefficent_layer.CreateField(predictor_field)

        predictor_path = _sanitize_path(
            predictor_table_path, predictor_table[predictor_id]['path'])

        predictor_type = predictor_table[predictor_id]['type']

        if predictor_type.startswith('raster'):
            # type must be one of raster_sum or raster_mean
            raster_type = predictor_type.split('_')[1]
            raster_sum_mean_results = _raster_sum_mean(
                response_vector_path, predictor_path,
                tmp_indexed_vector_path)
            predictor_results = raster_sum_mean_results[raster_type]
        else:
            predictor_results = predictor_functions[predictor_type](
                response_polygons_lookup, predictor_path)
        for feature_id, value in predictor_results.iteritems():
            feature = out_coefficent_layer.GetFeature(int(feature_id))
            feature.SetField(str(predictor_id), value)
            out_coefficent_layer.SetFeature(feature)
    out_coefficent_layer = None
    out_coefficent_vector.SyncToDisk()
    ogr.DataSource.__swig_destroy__(out_coefficent_vector)
    out_coefficent_vector = None


def _build_temporary_indexed_vector(vector_path, out_fid_index_vector_path):
    """Copy single layer vector and add a field to map feature indexes.

    Parameters:
        vector_path (string): path to OGR vector
        out_fid_index_vector_path (string): desired path to the copied vector
            that has an index field added to it

    Returns:
        fid_field (string): name of FID field added to output vector_path
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vector = ogr.Open(vector_path)
    if os.path.exists(out_fid_index_vector_path):
        os.remove(out_fid_index_vector_path)

    fid_indexed_vector = driver.CopyDataSource(
        vector, out_fid_index_vector_path)
    fid_indexed_layer = fid_indexed_vector.GetLayer()

    # make a random field name
    fid_field = str(uuid.uuid4())[-8:]
    fid_field_defn = ogr.FieldDefn(str(fid_field), ogr.OFTInteger)
    fid_indexed_layer.CreateField(fid_field_defn)
    for feature in fid_indexed_layer:
        fid = feature.GetFID()
        feature.SetField(fid_field, fid)
        fid_indexed_layer.SetFeature(feature)

    fid_indexed_vector.SyncToDisk()
    fid_indexed_layer = None
    ogr.DataSource.__swig_destroy__(fid_indexed_vector)

    return fid_field


def _raster_sum_mean(
        response_vector_path, raster_path, tmp_indexed_vector_path):
    """Sum all non-nodata values in the raster under each polygon.

    Parameters:
        response_vector_path (string): path to response polygons
        raster_path (string): path to a raster.
        tmp_indexed_vector_path (string): desired path to a vector that will
            be used to add unique indexes to `response_vector_path`
        tmp_fid_raster_path (string): desired path to raster that will be used
            to aggregate `raster_path` values by unique response IDs.

    Returns:
        A dictionary indexing 'sum', 'mean', and 'count', to dictionaries
        mapping feature IDs from `response_polygons_lookup` to those values
        of the raster under the polygon.
    """
    fid_field = _build_temporary_indexed_vector(
        response_vector_path, tmp_indexed_vector_path)

    aggregate_results = pygeoprocessing.aggregate_raster_values_uri(
        raster_path, tmp_indexed_vector_path, shapefile_field=fid_field)

    fid_raster_values = {
        'sum': aggregate_results.total,
        'mean': aggregate_results.pixel_mean,
        'count': aggregate_results.n_pixels,
        }
    return fid_raster_values


def _polygon_area(mode, response_polygons_lookup, polygon_vector_path):
    """Calculate polygon area overlap.

    Calculates the amount of projected area overlap from `polygon_vector_path`
    with `response_polygons_lookup`.

    Parameters:
        mode (string): one of 'area' or 'percent'.  How this is set affects
            the metric that's output.  'area' is the area covered in projected
            units while 'percent' is percent of the total response area
            covered
        response_polygons_lookup (dictionary): maps feature ID to
            prepared shapely.Polygon.
        polygon_vector_path (string): path to a single layer polygon vector
            object.

    Returns:
        A dictionary mapping feature IDs from `response_polygons_lookup`
        to polygon area coverage.
    """
    start_time = time.time()
    polygons = _ogr_to_geometry_list(polygon_vector_path)
    prepared_polygons = [
        shapely.prepared.prep(polygon) for polygon in polygons
        if polygon.is_valid]
    polygon_coverage_lookup = {}  # map FID to point count
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        if time.time() - start_time > 5.0:
            LOGGER.info(
                "%s polygon area: %.2f%% complete",
                os.path.basename(polygon_vector_path),
                (100.0*index)/len(response_polygons_lookup))
            start_time = time.time()

        polygon_area_coverage = sum([
            (polygon.intersection(geometry)).area for polygon, prep_poly in
            zip(polygons, prepared_polygons) if
            prep_poly.intersects(geometry)])
        if mode == 'area':
            polygon_coverage_lookup[feature_id] = polygon_area_coverage
        elif mode == 'percent':
            polygon_coverage_lookup[feature_id] = (
                polygon_area_coverage / geometry.area * 100.0)
    LOGGER.info(
        "%s polygon area: 100.00%% complete",
        os.path.basename(polygon_vector_path))
    return polygon_coverage_lookup


def _line_intersect_length(response_polygons_lookup, line_vector_path):
    """Calculate the length of the intersecting lines on the response polygon.

    Parameters:
        response_polygons_lookup (dictionary): maps feature ID to
            prepared shapely.Polygon.

        line_vector_path (string): path to a single layer point vector
            object.

    Returns:
        A dictionary mapping feature IDs from `response_polygons_lookup`
        to line intersect length.
    """
    last_time = time.time()
    lines = _ogr_to_geometry_list(line_vector_path)
    line_length_lookup = {}  # map FID to intersecting line length

    index = None
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        last_time = delay_op(
            last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                "%s line intersect length: %.2f%% complete",
                os.path.basename(line_vector_path),
                (100.0 * index)/len(response_polygons_lookup)))
        line_length = sum([
            (line.intersection(geometry)).length for line in lines])
        line_length_lookup[feature_id] = line_length
    LOGGER.info(
        "%s line intersect length: 100.00%% complete",
        os.path.basename(line_vector_path))
    return line_length_lookup


def _point_nearest_distance(response_polygons_lookup, point_vector_path):
    """Calculate distance to nearest point for all polygons.

    Parameters:
        response_polygons_lookup (dictionary): maps feature ID to
            prepared shapely.Polygon.

        point_vector_path (string): path to a single layer point vector
            object.

    Returns:
        A dictionary mapping feature IDs from `response_polygons_lookup`
        to distance to nearest point.
    """
    last_time = time.time()
    points = _ogr_to_geometry_list(point_vector_path)
    point_distance_lookup = {}  # map FID to point count
    index = None
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        last_time = delay_op(
            last_time, 5.0, lambda: LOGGER.info(
                "%s point distance: %.2f%% complete",
                os.path.basename(point_vector_path),
                (100.0*index)/len(response_polygons_lookup)))

        point_distance_lookup[feature_id] = min([
            geometry.distance(point) for point in points])
    LOGGER.info(
        "%s point distance: 100.00%% complete",
        os.path.basename(point_vector_path))
    return point_distance_lookup


def _point_count(response_polygons_lookup, point_vector_path):
    """Calculate number of points that intersect the response polygons.

    Parameters:
        response_polygons_lookup (dictionary): maps feature ID to
            prepared shapely.Polygon.

        point_vector_path (string): path to a single layer point vector
            object.

    Returns:
        A dictionary mapping feature IDs from `response_polygons_lookup`
        to number of points in that polygon.
    """
    last_time = time.time()
    points = _ogr_to_geometry_list(point_vector_path)
    point_count_lookup = {}  # map FID to point count
    index = None
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        last_time = delay_op(
            last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                "%s point count: %.2f%% complete",
                os.path.basename(point_vector_path),
                (100.0*index)/len(response_polygons_lookup)))
        point_count = len([
            point for point in points if geometry.contains(point)])
        point_count_lookup[feature_id] = point_count
    LOGGER.info(
        "%s point count: 100.00%% complete",
        os.path.basename(point_vector_path))
    return point_count_lookup


def _ogr_to_geometry_list(vector_path):
    """Convert an OGR type with one layer to a list of shapely geometry.

    Iterates through the features in the `vector_path`'s first layer and
    converts them to `shapely` geometry objects.  if the objects are not
    valid geometry, an attempt is made to buffer the object by 0 units
    before adding to the list.

    Parameters:
        vector_path (string): path to an OGR datasource

    Returns:
        list of shapely geometry objects representing the features in the
        `vector_path` layer.
    """
    vector = ogr.Open(vector_path)
    layer = vector.GetLayer()
    geometry_list = []
    for feature in layer:
        feature_geometry = feature.GetGeometryRef()
        shapely_geometry = shapely.wkt.loads(feature_geometry.ExportToWkt())
        if not shapely_geometry.is_valid:
            shapely_geometry = shapely_geometry.buffer(0)
        geometry_list.append(shapely_geometry)
        feature_geometry = None
    layer = None
    ogr.DataSource.__swig_destroy__(vector)
    return geometry_list


def _build_regression(coefficient_vector_path, response_id, predictor_id_list):
    """Multiple regression for log response of the coefficient vector table.

    The regression is built such that each feature in the single layer vector
    pointed to by `coefficent_vector_path` corresponds to one data point.
    The coefficients are defined in the vector's attribute table such that
    `response_id` is the response coefficient, and `predictor_id_list` is a
    list of the predictor ids.

    Parameters:
        coefficient_vector_path (string): path to a shapefile that contains
            at least the fields described in `response_id` and
            `predictor_id_list`.
        response_id (string): field ID in `coefficient_vector_path` whose
            values correspond to the regression response variable.
        predictor_id_list (list): a list of field IDs in
            `coefficient_vector_path` that correspond to the predictor
            variables in the regression.  The order of this list also
            determines the order of the regression coefficients returned
            by this function.

    Returns:
        X: A list of coefficients in the least-squares solution including
            the y intercept as the last element
        ssreg: sums of squared residuals
        r_sq: R^2 value
        r_sq_adj: adjusted R^2 value
        std_err: residual standard error
        dof: degrees of freedom
        se_est: standard error estimate on coefficients
    """
    coefficent_vector = ogr.Open(coefficient_vector_path)
    coefficent_layer = coefficent_vector.GetLayer()

    # Loop through each feature and build up the dictionary representing the
    # attribute table
    n_features = coefficent_layer.GetFeatureCount()
    coefficient_matrix = numpy.empty((n_features, len(predictor_id_list)+2))
    for row_index, feature in enumerate(coefficent_layer):
        coefficient_matrix[row_index, :] = numpy.array(
            [feature.GetField(str(response_id))] + [
                feature.GetField(str(key)) for key in predictor_id_list] +
            [1])  # add the 1s for the y intercept

    y_factors = numpy.log1p(coefficient_matrix[:, 0])

    coefficents, _, _, _ = numpy.linalg.lstsq(
        coefficient_matrix[:, 1:], y_factors)
    ssreg = numpy.sum((
        y_factors -
        numpy.sum(coefficient_matrix[:, 1:] * coefficents, axis=1)) ** 2)
    sstot = numpy.sum((
        numpy.average(y_factors) -
        numpy.log1p(coefficient_matrix[:, 0])) ** 2)
    dof = n_features - len(predictor_id_list) - 1

    if sstot == 0.0 or dof <= 0.0:
        # this can happen if there is only one sample
        r_sq = 1.0
        r_sq_adj = 1.0
    else:
        r_sq = 1. - ssreg / sstot
        r_sq_adj = 1 - (1 - r_sq) * (n_features - 1) / dof

    if dof > 0:
        std_err = numpy.sqrt(ssreg / dof)
        sigma2 = numpy.sum((
            y_factors - numpy.sum(
                coefficient_matrix[:, 1:] * coefficents, axis=1)) ** 2) / dof
        var_est = sigma2 * numpy.diag(numpy.linalg.pinv(
            numpy.dot(
                coefficient_matrix[:, 1:].T, coefficient_matrix[:, 1:])))
        se_est = numpy.sqrt(var_est)
    else:
        LOGGER.warn("Linear model is under constrained with DOF=%d", dof)
        std_err = sigma2 = numpy.nan
        se_est = var_est = [numpy.nan] * coefficient_matrix.shape[0]

    return coefficents, ssreg, r_sq, r_sq_adj, std_err, dof, se_est


def _calculate_scenario(
        base_aoi_path, response_id, predictor_coefficents, predictor_id_list,
        scenario_predictor_table_path, tmp_indexed_vector_path,
        scenario_results_path):
    """Calculate the PUD of a scenario given an existing regression.

    It is expected that the predictor coefficients have been derived from a
    log normal distribution.

    Parameters:
        base_aoi_path (string): path to the a polygon vector that was used
            to build the original regression.  Geometry will be copied for
            `scenario_results_path` output vector.
        response_id (string): text ID of response variable to write to
            the scenario result
        predictor_coefficents (numpy.ndarray): 1D array of regression
            coefficients that are parallel to `predictor_id_list` except the
            last element is the y-intercept.
        predictor_id_list (list of string): list of text ID predictor
            variables that correspond with `coefficients`
        scenario_predictor_table_path (string): path to a CSV table of
            regression predictors, their IDs and types.  Must contain the
            fields 'id', 'path', and 'type' where:
                'id': is a <=10 character length ID that is used to uniquely
                    describe the predictor.  It will be added to the output
                    result shapefile attribute table which is an ESRI
                    Shapefile, thus limited to 10 characters.
                'path': an absolute or relative (to this table) path to the
                    predictor dataset, either a vector or raster type.
                'type': one of the following,
                    'raster_mean': mean of values in the raster under the
                        response polygon
                    'raster_sum': sum of values in the raster under the
                        response polygon
                    'point_count': count of the points contained in the
                        response polygon
                    'point_nearest_distance': distance to the nearest point
                        from the response polygon
                    'line_intersect_length': length of lines that intersect
                        with the response polygon in projected units of AOI
                    'polygon_area': area of the polygon contained within
                        response polygon in projected units of AOI
                Note also that each ID in the table must have a corresponding
                entry in `response_id` else the input is invalid.
        tmp_indexed_vector_path (string): path to temporary working file in
            case the response vector needs an index added
        scenario_results_path (string): path to desired output scenario
            vector result which will be geometrically a copy of the input
            AOI but contain the base regression fields as well as the scenario
            derived response.

    Returns:
        None
    """
    scenario_predictor_id_list = []
    _build_regression_coefficients(
        base_aoi_path, scenario_predictor_table_path,
        tmp_indexed_vector_path, scenario_results_path,
        scenario_predictor_id_list)

    id_to_coefficient = dict(
        (p_id, coeff) for p_id, coeff in zip(
            predictor_id_list, predictor_coefficents))

    # Open for writing
    scenario_coefficent_vector = ogr.Open(scenario_results_path, 1)
    scenario_coefficent_layer = scenario_coefficent_vector.GetLayer()

    # delete the response ID if it's already there because it must have been
    # copied from the base layer
    response_index = scenario_coefficent_layer.FindFieldIndex(response_id, 1)
    if response_index >= 0:
        scenario_coefficent_layer.DeleteField(response_index)

    response_field = ogr.FieldDefn(response_id, ogr.OFTReal)
    response_field.SetWidth(24)
    response_field.SetPrecision(11)

    scenario_coefficent_layer.CreateField(response_field)

    for feature_id in xrange(scenario_coefficent_layer.GetFeatureCount()):
        feature = scenario_coefficent_layer.GetFeature(feature_id)
        response_value = 0.0
        for scenario_predictor_id in scenario_predictor_id_list:
            response_value += (
                id_to_coefficient[scenario_predictor_id] *
                feature.GetField(str(scenario_predictor_id)))
        response_value += predictor_coefficents[-1]  # y-intercept
        # recall the coefficients are log normal, so expm1 inverses it
        feature.SetField(response_id, numpy.expm1(response_value))
        scenario_coefficent_layer.SetFeature(feature)

    scenario_coefficent_layer = None
    scenario_coefficent_vector.SyncToDisk()
    ogr.DataSource.__swig_destroy__(scenario_coefficent_vector)
    scenario_coefficent_vector = None


def _validate_same_id_lengths(table_path):
    """Ensure both table has ids of length less than 10.

    Parameter:
        table_path (string):  path to a csv table that has at least
            the field 'id'

    Raises:
        ValueError if any of the fields in 'id' and 'type' don't match between
        tables.
    """
    predictor_table = pygeoprocessing.get_lookup_from_csv(table_path, 'id')
    too_long = set()
    for p_id in predictor_table:
        if len(p_id) > 10:
            too_long.add(p_id)
    if len(too_long) > 0:
        raise ValueError(
            "The following IDs are more than 10 characters long: %s" %
            str(too_long))


def _validate_same_ids_and_types(
        predictor_table_path, scenario_predictor_table_path):
    """Ensure both tables have same ids and types.

    Assert that both the elements of the 'id' and 'type' fields of each table
    contain the same elements and that their values are the same.  This
    ensures that a user won't get an accidentally incorrect simulation result.

    Parameters:
        predictor_table_path (string): path to a csv table that has at least
            the fields 'id' and 'type'
        scenario_predictor_table_path (string):  path to a csv table that has
            at least the fields 'id' and 'type'

    Returns:
        None

    Raises:
        ValueError if any of the fields in 'id' and 'type' don't match between
        tables.
    """
    predictor_table = pygeoprocessing.get_lookup_from_csv(
        predictor_table_path, 'id')

    scenario_predictor_table = pygeoprocessing.get_lookup_from_csv(
        scenario_predictor_table_path, 'id')

    predictor_table_pairs = set([
        (p_id, predictor_table[p_id]['type']) for p_id in predictor_table])
    scenario_predictor_table_pairs = set([
        (p_id, scenario_predictor_table[p_id]['type']) for p_id in
        scenario_predictor_table])
    if predictor_table_pairs != scenario_predictor_table_pairs:
        raise ValueError(
            'table pairs unequal.\n\tpredictor: %s\n\tscenario:%s' % (
                str(predictor_table_pairs),
                str(scenario_predictor_table_pairs)))
    LOGGER.info('tables validate correctly')


def _validate_same_projection(base_vector_path, table_path):
    """Assert the GIS data in the table are in the same projection as the AOI.

    Parameters:
        base_vector_path (string): path to a GIS vector
        table_path (string): path to a csv table that has at least
            the field 'path'

    Returns:
        None

    Raises:
        ValueError if the projections in each of the GIS types in the table
            are not identical to the projection in base_vector_path
    """
    # This will load the table as paths which we can iterate through without
    # bothering the rest of the table structure
    data_paths = pygeoprocessing.get_lookup_from_csv(
        table_path, 'path')

    base_vector = ogr.Open(base_vector_path)
    base_layer = base_vector.GetLayer()
    base_ref = osr.SpatialReference(base_layer.GetSpatialRef().ExportToWkt())
    base_layer = None
    base_vector = None

    invalid_projections = False
    for raw_path in data_paths:
        path = _sanitize_path(table_path, raw_path)

        def error_handler(err_level, err_no, err_msg):
            """Empty error handler to avoid stderr output."""
            pass
        gdal.PushErrorHandler(error_handler)
        raster = gdal.Open(path)
        gdal.PopErrorHandler()
        if raster is not None:
            projection_as_str = raster.GetProjection()
            ref = osr.SpatialReference()
            ref.ImportFromWkt(projection_as_str)
            raster = None
        else:
            vector = ogr.Open(path)
            if vector is None:
                raise ValueError("%s did not load", path)
            layer = vector.GetLayer()
            ref = osr.SpatialReference(layer.GetSpatialRef().ExportToWkt())
            layer = None
            vector = None
        if not base_ref.IsSame(ref):
            LOGGER.warn(
                "%s might have a different projection than the base AOI "
                "\nbase:%s\ncurrent:%s", path, base_ref.ExportToPrettyWkt(),
                ref.ExportToPrettyWkt())
            invalid_projections = True
    if invalid_projections:
        raise ValueError(
            "One or more of the projections in the table did not match the "
            "projection of the base vector")


def delay_op(last_time, time_delay, func):
    """Execute `func` if last_time + time_delay >= current time.

    Parameters:
        last_time (float): last time in seconds that `func` was triggered
        time_delay (float): time to wait in seconds since last_time before
            triggering `func`
        func (function): parameterless function to invoke if
         current_time >= last_time + time_delay

    Returns:
        If `func` was triggered, return the time which it was triggered in
        seconds, otherwise return `last_time`.
    """
    if time.time() - last_time > time_delay:
        func()
        return time.time()
    return last_time


def _sanitize_path(base_path, raw_path):
    """Return `path` if absolute, or make absolute local to `base_path`."""
    if os.path.isabs(raw_path):
        return raw_path
    else:  # assume relative path w.r.t. the response table
        return os.path.join(os.path.dirname(base_path), raw_path)
