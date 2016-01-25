"""InVEST Recreation Client."""

import collections
import uuid
import tempfile
import os
import zipfile
import time
import logging
import math
import urllib

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

from .. import utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_client')
# This URL is a NatCap global constant
RECREATION_SERVER_URL = 'http://data.naturalcapitalproject.org/server_registry/invest_recreation_model/'

#this serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'

# These are the expected extensions associated with an ESRI Shapefile
# as part of the ESRI Shapefile driver standard, but some extensions
# like .prj, .sbn, and .sbx, are optional depending on versions of the
# format: http://www.gdal.org/drv_shapefile.html
_ESRI_SHAPEFILE_EXTENSIONS = ['.prj', '.shp', '.shx', '.dbf', '.sbn', '.sbx']

# For now, this is the field name we use to mark the photo user "days"
RESPONSE_ID = 'PUD_YR_AVG'
SCENARIO_RESPONSE_ID = 'PUD_EST'

_OUTPUT_BASE_FILES = {
    'pud_results_path': 'pud_results.shp',
    'coefficent_vector_path': 'regression_coeffiicents.shp',
    'scenario_results_path': 'scenario_results.shp',
    'regression_coefficients': 'regression_coefficients.txt',
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
            `args['grid_aoi']` is True.  Indicates the long axis size of the
            grid cells.
        args['compute_regression'] (boolean): if True, then process the
            predictor table and scenario table (if present).
        args['predictor_table_path'] (string): required if
            args['compute_regression'] is True.  Path to a table that
            describes the regression predictors, their IDs and types.  Must
            contain the fields 'id', 'path', and 'type' where:
                'id': is a 10 character of less ID that is used to uniquely
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

    # append jan 1 to start and dec 31 to end
    if args['end_year'] < args['start_year']:
        raise ValueError(
            "Start year must be less than or equal to end year.\n"
            "start_year: %s\nend_year: %s" % (
                args['start_year'], args['end_year']))
    date_range = (args['start_year']+'-01-01', args['end_year']+'-12-31')
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    output_dir = args['workspace_dir']
    pygeoprocessing.create_directories([output_dir])

    file_registry = _build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    # in case the user defines a hostname
    if 'hostname' in args:
        path = "PYRO:natcap.invest.recreation@%s:%s" % (
            args['hostname'], args['port'])
    else:
        # else use a well known path to get active server
        path = urllib.urlopen(RECREATION_SERVER_URL).read().rstrip()
    recmodel_server = Pyro4.Proxy(path)

    if args['grid_aoi']:
        LOGGER.info("gridding aoi")
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
    LOGGER.info('Contacting server, please wait.')
    server_version = recmodel_server.get_version()
    LOGGER.info('Server online, version: %s', server_version)
    LOGGER.info('Please wait for server to calculate PUD...')

    result_zip_file_binary, workspace_id = (
        recmodel_server.calc_photo_user_days_in_aoi(
            zip_file_binary, date_range,
            os.path.basename(file_registry['pud_results_path'])))
    LOGGER.info(
        'received result, took %f seconds, workspace_id: %s',
        time.time() - start_time, workspace_id)

    #unpack result
    open(file_registry['compressed_pud_path'], 'wb').write(
        result_zip_file_binary)
    zipfile.ZipFile(file_registry['compressed_pud_path'], 'r').extractall(
        output_dir)

    if 'compute_regression' in args and args['compute_regression']:
        LOGGER.info('Calculating regression')
        predictor_id_list = []
        build_regression_coefficients(
            file_registry['pud_results_path'], args['predictor_table_path'],
            file_registry['coefficent_vector_path'], predictor_id_list)

        coefficents, residual, r_sq, std_err = build_regression(
            file_registry['coefficent_vector_path'], RESPONSE_ID,
            predictor_id_list)

        # the last coefficient is the y intercept and has no id, thus
        # the [:-1] on the coefficients list
        regression_string = ' +\n      '.join(
            '%+.2e * %s' % (coefficent, p_id)
            for p_id, coefficent in zip(predictor_id_list, coefficents[:-1]))
        regression_string += ' +\n      %+.2e' % coefficents[-1]  # y intercept

        # generate a nice looking regression result and write to log and file
        report_string = (
            '\nRegression:\n%s = %s\nR^2: %s\nstd_err: %s\n'
            'residuals: %s\nserver id hash: %s' % (
                RESPONSE_ID, regression_string, r_sq, std_err, residual,
                server_version))
        LOGGER.info(report_string)
        with open(file_registry['regression_coefficients'], 'w') as \
                regression_log:
            regression_log.write(report_string + '\n')

        if ('scenario_predictor_table_path' in args and
                args['scenario_predictor_table_path'] != ''):
            LOGGER.info('Calculating scenario')
            calculate_scenario(
                file_registry['pud_results_path'], SCENARIO_RESPONSE_ID,
                coefficents, predictor_id_list,
                args['scenario_predictor_table_path'],
                file_registry['scenario_results_path'])

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
    """Convert vector to a regular grid.

    Here the vector is gridded such that all cells are contained within the
    original vector.

    Parameters:
        vector_path (string): path to an OGR compatable polygon vector type
        grid_type (string): one of "square" or "hexagon"
        cell_size (float): dimensions of the grid cell in the projected units
            of `vector_path`
        out_grid_vector_path (string): path to the output ESRI shapefile
            vector that contains a gridded version of `vector_path`

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


def _build_file_registry(base_file_path_list, file_suffix):
    """Combine file suffixes with base names and directories.

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
        """Internal helper to avoid code duplication."""
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
        out_coefficient_vector_path, out_predictor_id_list):
    """Calculate least squares fit for the polygons in the response vector.

    Build a least squares regression from the response vector, spatial
    predictor datasets in `predictor_table_path`, and a column of 1s for the
    y intercept.

    Parameters:
        response_vector_path (string): path to a single layer polygon vector.
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
                'polygon_percent_coverage': percent of predictor polygon
                    intersecting the response polygon
                'raster_sum': sum of predictor raster under the response
                    polygon
                'raster_mean': average of predictor raster under the
                    response polygon

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
        'raster_sum': lambda x, y: _raster_sum_mean(x, y)['sum'],
        'raster_mean': lambda x, y: _raster_sum_mean(x, y)['mean'],
        }

    predictor_table = pygeoprocessing.get_lookup_from_csv(
        predictor_table_path, 'id')
    del out_predictor_id_list[:]  # prepare for appending
    for predictor_id in predictor_table.keys():
        out_predictor_id_list.append(predictor_id)
        if len(predictor_id) > 10:
            short_predictor_id = predictor_id[:10]
            LOGGER.warn(
                '%s is too long for shapefile, truncating to %s',
                predictor_id, short_predictor_id)
            if short_predictor_id in predictor_table:
                raise ValueError(
                    "predictor_table id collision because we had to shorten "
                    "a long id.")
            predictor_table[short_predictor_id] = predictor_table[
                predictor_id]
            del predictor_table[predictor_id]

    # see if we need to compute raster results, just do both at once

    for predictor_id in predictor_table:
        LOGGER.info("Building predictor %s", predictor_id)
        predictor_field = ogr.FieldDefn(str(predictor_id), ogr.OFTReal)
        out_coefficent_layer.CreateField(predictor_field)

        raw_path = predictor_table[predictor_id]['path']
        if os.path.isabs(raw_path):
            predictor_path = raw_path
        else:
            # assume relative path w.r.t. the response table
            predictor_path = os.path.join(
                os.path.dirname(predictor_table_path), raw_path)

        predictor_type = predictor_table[predictor_id]['type']

        if predictor_type.startswith('raster'):
            # type must be one of raster_sum or raster_mean
            raster_type = predictor_type.split('_')[1]
            raster_sum_mean_results = _raster_sum_mean(
                response_vector_path, predictor_path)
            predictor_results = raster_sum_mean_results[raster_type]
        else:
            predictor_results = predictor_functions[predictor_type](
                response_polygons_lookup, predictor_path)
        for feature_id, value in predictor_results.iteritems():
            feature = out_coefficent_layer.GetFeature(feature_id)
            feature.SetField(str(predictor_id), value)
            out_coefficent_layer.SetFeature(feature)
    out_coefficent_layer = None
    out_coefficent_vector.SyncToDisk()
    ogr.DataSource.__swig_destroy__(out_coefficent_vector)
    out_coefficent_vector = None


def _build_temporary_indexed_vector(vector_path):
    """Copy single layer vector and add a field to map feature indexes.

    Parameters:
        vector_path (string): path to OGR vector

    Returns:
        fid_field (string): name of FID field added to output vector_path
        fid_indexed_vector_path (string): path to copy of `vector_path` with
            additional FID field added to it.
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vector = ogr.Open(vector_path)
    tmp_dir = tempfile.mkdtemp()
    fid_indexed_path = os.path.join(tmp_dir, 'copy.shp')
    fid_indexed_vector = driver.CopyDataSource(
        vector, fid_indexed_path)
    fid_indexed_layer = fid_indexed_vector.GetLayer()

    # make a random field name
    fid_name = str(uuid.uuid4())[-8:]
    fid_field = ogr.FieldDefn(str(fid_name), ogr.OFTInteger)
    fid_indexed_layer.CreateField(fid_field)
    for feature in fid_indexed_layer:
        fid = feature.GetFID()
        feature.SetField(fid_name, fid)
        fid_indexed_layer.SetFeature(feature)
    fid_indexed_vector.SyncToDisk()

    return fid_name, fid_indexed_path


def _raster_sum_mean(response_vector_path, raster_path):
    """Sum all non-nodata values in the raster under each polygon.

    Parameters:
        response_vector_path (string): path to response polygons
        raster_path (string): path to a raster.

    Returns:
        A dictionary indexing 'sum', 'mean', and 'count', to dictionaries
        mapping feature IDs from `response_polygons_lookup` to those values
        of the raster under the polygon.
    """
    fid_field, fid_indexed_path = _build_temporary_indexed_vector(
        response_vector_path)

    raster_nodata = pygeoprocessing.get_nodata_from_uri(raster_path)
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(raster_path)
    fid_raster_path = 'fid.tif'#pygeoprocessing.temporary_filename(suffix='.tif')
    fid_nodata = -1
    pygeoprocessing.vectorize_datasets(
        [raster_path], lambda x: x*0+fid_nodata, fid_raster_path, gdal.GDT_Int32,
        fid_nodata, out_pixel_size, "union",
        dataset_to_align_index=0, aoi_uri=fid_indexed_path,
        vectorize_op=False)

    fid_vector = ogr.Open(fid_indexed_path)
    fid_layer = fid_vector.GetLayer()
    fid_raster = gdal.Open(fid_raster_path, gdal.GA_Update)
    gdal.RasterizeLayer(
        fid_raster, [1], fid_layer, options=['ATTRIBUTE=%s' % fid_field])
    fid_raster.FlushCache()

    raster = gdal.Open(raster_path)
    band = raster.GetRasterBand(1)
    fid_raster_values = {
        'sum': collections.defaultdict(float),
        'mean': collections.defaultdict(float),
        'count': collections.defaultdict(int),
        }
    for offset_dict, fid_block in pygeoprocessing.iterblocks(fid_raster_path):
        raster_array = band.ReadAsArray(**offset_dict)

        unique_ids = numpy.unique(fid_block)
        for attribute_id in unique_ids:
            # ignore masked values
            if attribute_id == fid_nodata:
                continue
            masked_values = raster_array[fid_block == attribute_id]
            if raster_nodata is not None:
                valid_mask = masked_values != raster_nodata
            else:
                valid_mask = numpy.empty(
                    masked_values.shape, dtype=numpy.bool)
                valid_mask[:] = True
            fid_raster_values['sum'][attribute_id] += numpy.sum(
                masked_values[valid_mask])
            fid_raster_values['count'][attribute_id] += numpy.count_nonzero(
                valid_mask)

    for attribute_id in fid_raster_values['count']:
        if fid_raster_values['count'][attribute_id] != 0.0:
            fid_raster_values['mean'][attribute_id] = (
                fid_raster_values['sum'][attribute_id] /
                fid_raster_values['count'][attribute_id])
        else:
            fid_raster_values['mean'][attribute_id] = 0.0

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
    start_time = time.time()
    lines = _ogr_to_geometry_list(line_vector_path)
    line_length_lookup = {}  # map FID to intersecting line length

    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        if time.time() - start_time > 5.0:
            LOGGER.info(
                "%s line intersect length: %.2f%% complete",
                os.path.basename(line_vector_path),
                (100.0*index)/len(response_polygons_lookup))
            start_time = time.time()

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
    start_time = time.time()
    points = _ogr_to_geometry_list(point_vector_path)
    point_distance_lookup = {}  # map FID to point count
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        if time.time() - start_time > 5.0:
            LOGGER.info(
                "%s point distance: %.2f%% complete",
                os.path.basename(point_vector_path),
                (100.0*index)/len(response_polygons_lookup))
            start_time = time.time()

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
    start_time = time.time()
    points = _ogr_to_geometry_list(point_vector_path)
    point_count_lookup = {}  # map FID to point count
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        if time.time() - start_time > 5.0:
            LOGGER.info(
                "%s point count: %.2f%% complete",
                os.path.basename(point_vector_path),
                (100.0*index)/len(response_polygons_lookup))
            start_time = time.time()

        point_count = len([
            point for point in points if geometry.contains(point)])
        point_count_lookup[feature_id] = point_count
    LOGGER.info(
        "%s point count: 100.00%% complete",
        os.path.basename(point_vector_path))
    return point_count_lookup


def _ogr_to_geometry_list(vector_path):
    """Convert an OGR type with one layer to a list of shapely geometry."""
    vector = ogr.Open(vector_path)
    layer = vector.GetLayer()
    geometry_list = []
    for feature in layer:
        feature_geometry = feature.GetGeometryRef()
        shapely_geometry = shapely.wkt.loads(feature_geometry.ExportToWkt())
        if not shapely_geometry.is_valid:
            shapely_geometry = shapely_geometry.buffer(0)
        if shapely_geometry.is_valid:
            geometry_list.append(shapely_geometry)
        else:
            LOGGER.error(
                "Unable to fix broken geometry on FID %d in %s", 
                feature.GetFID(), vector_path)
        feature_geometry = None
    layer = None
    ogr.DataSource.__swig_destroy__(vector)
    return geometry_list


def build_regression(coefficient_vector_path, response_id, predictor_id_list):
    """Build multiple regression for response in the coefficient vector table.

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
        X: A list of coefficents in the least-squares solution including
            the y intercept as the last element
        residual_sum: sums of resisuals
        r_sq: R^2 value
        std_err: residual standard error
    """
    # Pull apart the datasource
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

    coefficents, _, _, _ = numpy.linalg.lstsq(
        coefficient_matrix[:, 1:], coefficient_matrix[:, 0])
    residual_sum = numpy.sum(numpy.sum(
        (coefficient_matrix[:, 1:] * coefficents), axis=1) ** 2)
    r_sq = 1 - residual_sum / (n_features * coefficient_matrix[:, 0].var())
    std_err = numpy.sqrt(r_sq / (n_features - 2))
    LOGGER.debug(residual_sum)
    return coefficents, residual_sum, r_sq, std_err


def calculate_scenario(
        base_aoi_path, response_id, predictor_coefficents, predictor_id_list,
        scenario_predictor_table_path, scenario_results_path):
    """Calculate the PUD of a scenario given an existing regression.

    Parameters:
        base_aoi_path (string): path to the a polygon vector that was used
            to build the original regresssion.  Geometry will be copied for
            `scenario_results_path` output vector.
        response_id (string): text ID of response variable to write to
            the scenario result
        predictor_coefficents (numpy.ndarray): 1D array of regression
            coefficents
        predictor_id_list (list of string): list of text ID predictor
            variables that correspond with `coefficients`
        scenario_predictor_table_path (string): path to a CSV table of
            regression predictors, their IDs and types.  Must contain the
            fields 'id', 'path', and 'type' where:
                'id': is a 10 character of less ID that is used to uniquely
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
        scenario_results_path (string): path to desired output scenario
            vector result which will be geometrically a copy of the input
            AOI but contain the base regression fields as well as the scenario
            derived response.

    Returns:
        None
    """
    scenario_predictor_id_list = []
    build_regression_coefficients(
        base_aoi_path, scenario_predictor_table_path,
        scenario_results_path, scenario_predictor_id_list)

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
    scenario_coefficent_layer.CreateField(response_field)

    for feature_id in xrange(scenario_coefficent_layer.GetFeatureCount()):
        feature = scenario_coefficent_layer.GetFeature(feature_id)
        response_value = 0.0
        for scenario_predictor_id in scenario_predictor_id_list:
            response_value += (
                id_to_coefficient[scenario_predictor_id] *
                feature.GetField(str(scenario_predictor_id)))
        feature.SetField(response_id, response_value)
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

    Assert that both the elements of the 'id' and 'type' fields of each tables
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
        if os.path.isabs(raw_path):
            path = raw_path
        else:
            # assume relative path
            path = os.path.join(os.path.dirname(table_path), raw_path)

        raster = gdal.Open(path)
        if raster is not None:
            projection_as_str = raster.GetProjection()
            ref = osr.SpatialReference()
            ref.ImportFromWkt(projection_as_str)
            raster = None
        else:
            vector = ogr.Open(path)
            if vector is None:
                LOGGER.error("%s did not load", path)
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
