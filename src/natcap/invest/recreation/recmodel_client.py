"""InVEST Recreation Client."""
from __future__ import absolute_import

import json
import uuid
import os
import zipfile
import time
import logging
import math
import pickle
import urllib
import tempfile
import shutil

import rtree
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
import shapely.speedups
import taskgraph

if shapely.speedups.available:
    shapely.speedups.enable()

# prefer to do intrapackage imports to avoid case where global package is
# installed and we import the global version of it rather than the local
from .. import utils
from .. import validation

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
    'monthly_table_path': 'monthly_table.csv',
    'coefficient_vector_path': 'regression_coefficients.shp',
    'scenario_results_path': 'scenario_results.shp',
    'regression_coefficients': 'regression_coefficients.txt',
    }

_TMP_BASE_FILES = {
    'local_aoi_path': 'aoi.shp',
    'compressed_aoi_path': 'aoi.zip',
    'compressed_pud_path': 'pud.zip',
    'server_version': 'server_version.pickle',
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
        server_path = "PYRO:natcap.invest.recreation@%s:%s" % (
            args['hostname'], args['port'])
    else:
        # else use a well known path to get active server
        server_path = urllib.urlopen(RECREATION_SERVER_URL).read().rstrip()

    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    output_dir = args['workspace_dir']
    intermediate_dir = os.path.join(output_dir, 'intermediate')
    scenario_dir = os.path.join(intermediate_dir, 'scenario')
    utils.make_directories([output_dir, intermediate_dir, scenario_dir])

    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    # Initialize a TaskGraph
    taskgraph_db_dir = os.path.join(output_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    task_graph = taskgraph.TaskGraph(taskgraph_db_dir, n_workers)

    prep_aoi_task = []
    if args['grid_aoi']:
        LOGGER.info("gridding aoi")
        prep_aoi_task.append(task_graph.add_task(
            func=_grid_vector,
            args=(args['aoi_path'], args['grid_type'], float(args['cell_size']),
                  file_registry['local_aoi_path']),
            target_path_list=[file_registry['local_aoi_path']],
            task_name='grid_aoi'))
    else:
        prep_aoi_task.append(task_graph.add_task(
            func=_copy_aoi_no_grid,
            args=(args['aoi_path'], file_registry['local_aoi_path']),
            target_path_list=[file_registry['local_aoi_path']],
            task_name='copy_aoi'))

    photo_user_days_task = task_graph.add_task(
        func=_retrieve_photo_user_days,
        args=(file_registry['local_aoi_path'],
              file_registry['compressed_aoi_path'], args['start_year'], args['end_year'],
              os.path.basename(file_registry['pud_results_path']),
              file_registry['compressed_pud_path'],
              output_dir, server_path, file_registry['server_version']),
        ignore_path_list=[file_registry['compressed_aoi_path'],
                          file_registry['compressed_pud_path']],
        target_path_list=[file_registry['pud_results_path'],
                          file_registry['monthly_table_path'],
                          file_registry['server_version']],
        dependent_task_list=prep_aoi_task,
        task_name='photo-user-day-calculation')
    
    if 'compute_regression' in args and args['compute_regression']:
        LOGGER.info('Calculating regression')
        # predictor_json_list = []  # tracks predictor files to add to shp
        # because unwanted files from previous runs could be present.
        build_regression_data_task = task_graph.add_task(
            func=_build_regression_coefficients,
            args=(file_registry['pud_results_path'],
                  args['predictor_table_path'],
                  file_registry['coefficient_vector_path'],
                  intermediate_dir, task_graph),
            target_path_list=[file_registry['coefficient_vector_path']],
            dependent_task_list=[photo_user_days_task],
            task_name='build predictors')
        # task_graph.close()
        task_graph.join()

        # predictor_id_list = [
        #     os.path.basename(x).replace('.json', '') for x in predictor_json_list
        #     if x.endswith('.json')]
        # _build_regression_coefficients(
        #     file_registry['pud_results_path'], args['predictor_table_path'],
        #     file_registry['coefficient_vector_path'], predictor_id_list,
        #     intermediate_dir)
        predictor_id_list, coefficients, ssres, r_sq, r_sq_adj, std_err, dof, se_est = (
            _build_regression(file_registry['pud_results_path'],
                file_registry['coefficient_vector_path'], RESPONSE_ID))
        # LOGGER.warn(id_to_coefficient)
        # coefficients = id_to_coefficient.values()
        # predictor_id_list = id_to_coefficient.keys()

        # the last coefficient is the y intercept and has no id, thus
        # the [:-1] on the coefficients list
        coefficients_string = '               estimate     stderr    t value\n'
        coefficients_string += '%-12s %+.3e %+.3e %+.3e\n' % (
            predictor_id_list[-1], coefficients[-1], se_est[-1],
            coefficients[-1] / se_est[-1])
        coefficients_string += '\n'.join(
            '%-12s %+.3e %+.3e %+.3e' % (
                p_id, coefficient, se_est_factor, coefficient / se_est_factor)
            for p_id, coefficient, se_est_factor in zip(
                predictor_id_list[:-1], coefficients[:-1], se_est[:-1]))

        # generate a nice looking regression result and write to log and file
        with open(file_registry['server_version'], 'rb') as f:
            server_version = pickle.load(f)
        report_string = (
            '\n******************************\n'
            '%s\n'
            '---\n\n'
            'Residual standard error: %.4f on %d degrees of freedom\n'
            'Multiple R-squared: %.4f\n'
            'Adjusted R-squared: %.4f\n'
            'SSres: %.4f\n'
            'server id hash: %s\n'
            '********************************\n' % (
                coefficients_string, std_err, dof, r_sq, r_sq_adj, ssres,
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
                # coefficients, predictor_id_list,
                id_to_coefficient,
                args['scenario_predictor_table_path'],
                file_registry['scenario_results_path'], task_graph, scenario_dir)
        task_graph.close()
        task_graph.join()

    # LOGGER.info('deleting temporary files')
    # shutil.rmtree(temporary_output_dir, ignore_errors=True)


def _copy_aoi_no_grid(source_aoi_path, dest_aoi_path):
    aoi_vector = gdal.OpenEx(source_aoi_path, gdal.OF_VECTOR)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    local_aoi_vector = driver.CreateCopy(
        dest_aoi_path, aoi_vector)
    gdal.Dataset.__swig_destroy__(local_aoi_vector)
    local_aoi_vector = None
    gdal.Dataset.__swig_destroy__(aoi_vector)
    aoi_vector = None


def _retrieve_photo_user_days(
    local_aoi_path, compressed_aoi_path, start_year, end_year, pud_results_filename,
    compressed_pud_path, output_dir, server_path, server_version_pickle):

    LOGGER.info('Contacting server, please wait.')
    recmodel_server = Pyro4.Proxy(server_path)
    server_version = recmodel_server.get_version()
    LOGGER.info('Server online, version: %s', server_version)
    # store server info in a file because with taskgraph, we won't always connect
    # to the server, but still want to report version info in results txt file
    with open(server_version_pickle, 'wb') as f:
        pickle.dump(server_version, f)

    # validate available year range
    min_year, max_year = recmodel_server.get_valid_year_range()
    LOGGER.info(
        "Server supports year queries between %d and %d", min_year, max_year)
    if not min_year <= int(start_year) <= max_year:
        raise ValueError(
            "Start year must be between %d and %d.\n"
            " User input: (%s)" % (min_year, max_year, start_year))
    if not min_year <= int(end_year) <= max_year:
        raise ValueError(
            "End year must be between %d and %d.\n"
            " User input: (%s)" % (min_year, max_year, end_year))

    # append jan 1 to start and dec 31 to end
    date_range = (str(start_year)+'-01-01',
                  str(end_year)+'-12-31')

    basename = os.path.splitext(local_aoi_path)[0]
    with zipfile.ZipFile(compressed_aoi_path, 'w') as aoizip:
        for suffix in _ESRI_SHAPEFILE_EXTENSIONS:
            filename = basename + suffix
            if os.path.exists(filename):
                LOGGER.info('archiving %s', filename)
                aoizip.write(filename, os.path.basename(filename))

    # convert shapefile to binary string for serialization
    zip_file_binary = open(compressed_aoi_path, 'rb').read()

    # transfer zipped file to server
    start_time = time.time()
    LOGGER.info('Please wait for server to calculate PUD...')

    result_zip_file_binary, workspace_id = (
        recmodel_server.calc_photo_user_days_in_aoi(
            zip_file_binary, date_range,
            pud_results_filename))
    LOGGER.info(
        'received result, took %f seconds, workspace_id: %s',
        time.time() - start_time, workspace_id)

    # unpack result
    open(compressed_pud_path, 'wb').write(
        result_zip_file_binary)
    temporary_output_dir = tempfile.mkdtemp(dir=output_dir)
    zipfile.ZipFile(compressed_pud_path, 'r').extractall(
        temporary_output_dir)
    # monthly_table_path = os.path.join(
    #     temporary_output_dir, monthly_table_filename)
    # if os.path.exists(monthly_table_path):
    #     os.rename(
    #         monthly_table_path,
    #         os.path.splitext(monthly_table_path)[0] + file_suffix + '.csv')
    for filename in os.listdir(temporary_output_dir):
        shutil.copy(os.path.join(temporary_output_dir, filename), output_dir)
    shutil.rmtree(temporary_output_dir)

    LOGGER.info('connection release')
    recmodel_server._pyroRelease()


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
    driver = gdal.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_grid_vector_path):
        driver.Delete(out_grid_vector_path)

    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    vector_layer = vector.GetLayer()
    spat_ref = vector_layer.GetSpatialRef()

    original_vector_shapes = []
    for feature in vector_layer:
        wkt_feat = shapely.wkt.loads(feature.geometry().ExportToWkt())
        original_vector_shapes.append(wkt_feat)
    vector_layer.ResetReading()
    original_polygon = shapely.prepared.prep(
        shapely.ops.cascaded_union(original_vector_shapes))

    out_grid_vector = driver.Create(
        out_grid_vector_path, 0, 0, 0, gdal.GDT_Unknown)
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


def _build_regression_coefficients(
        response_vector_path, predictor_table_path,
        out_coefficient_vector_path, working_dir, task_graph):
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
        out_coefficient_vector_path (string): path to a copy of
            `response_vector_path` with the modified predictor variable
            responses. Overwritten if exists.

    Returns:
        None
    """
    response_vector = gdal.OpenEx(response_vector_path, gdal.OF_VECTOR)
    response_layer = response_vector.GetLayer()
    response_polygons_lookup = {}  # maps FID to prepared geometry
    for response_feature in response_layer:
        feature_geometry = response_feature.GetGeometryRef()
        feature_polygon = shapely.wkt.loads(feature_geometry.ExportToWkt())
        feature_geometry = None
        response_polygons_lookup[response_feature.GetFID()] = feature_polygon
    response_layer = None

    driver = gdal.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_coefficient_vector_path):
        driver.Delete(out_coefficient_vector_path)
    out_coefficient_vector = driver.CreateCopy(
        out_coefficient_vector_path, response_vector)
    response_vector = None
    out_coefficient_vector = None

    # lookup functions for response types
    predictor_functions = {
        'point_count': _point_count,
        'point_nearest_distance': _point_nearest_distance,
        'line_intersect_length': _line_intersect_length,
        'polygon_area_coverage': lambda x, y, z: _polygon_area(
            'area', x, y, z),
        'polygon_percent_coverage': lambda x, y, z: _polygon_area(
            'percent', x, y, z),
        }

    predictor_table = utils.build_lookup_from_csv(
        predictor_table_path, 'id')
    predictor_task_list = []
    predictor_json_list = []  # tracks predictor files to add to shp
    for predictor_id in predictor_table:
        LOGGER.info("Building predictor %s", predictor_id)

        predictor_path = _sanitize_path(
            predictor_table_path, predictor_table[predictor_id]['path'])
        predictor_type = predictor_table[predictor_id]['type']
        if predictor_type.startswith('raster'):
            # type must be one of raster_sum or raster_mean
            raster_type = predictor_type.split('_')[1]
            predictor_target_path = os.path.join(
                working_dir, predictor_id + '.json')
            predictor_json_list.append(predictor_target_path)
            predictor_task_list.append(task_graph.add_task(
                func=_raster_sum_count,
                args=(predictor_path, raster_type, response_vector_path,
                      predictor_target_path),
                target_path_list=[predictor_target_path],
                task_name='predictor %s' % predictor_id))
        else:
            predictor_target_path = os.path.join(
                working_dir, predictor_id + '.json')
            predictor_json_list.append(predictor_target_path)
            predictor_task_list.append(task_graph.add_task(
                func=predictor_functions[predictor_type],
                args=(response_polygons_lookup, predictor_path,
                      predictor_target_path),
                target_path_list=[predictor_target_path],
                task_name='predictor %s' % predictor_id))
            # predictor_results = predictor_functions[predictor_type](
            #     response_polygons_lookup, predictor_path, 
            #     predictor_target_path)

    # target_path_list is empty because if we've gotten here
    # we always want this task to execute.
    assemble_predictor_data_task = task_graph.add_task(
        func=_json_to_shp_table,
        args=(out_coefficient_vector_path, predictor_json_list),
        target_path_list=[],
        dependent_task_list=predictor_task_list,
        task_name='assemble predictor data')

def _json_to_shp_table(vector_path, predictor_json_list):
    vector = gdal.OpenEx(vector_path, gdal.GA_Update)
    layer = vector.GetLayer()
    layer_defn = layer.GetLayerDefn()

    # TODO: leave this list empty if later on we have _build_regression
    # read PUD_YR_AVG from pud_results.shp instead of regression_coefficients.shp
    # that would enable server and client ops to happen in parallel.
    predictor_id_list = []
    for json_filename in predictor_json_list:
        predictor_id = os.path.basename(json_filename).replace('.json', '')
        predictor_id_list.append(predictor_id)
        # Create a new field for the predictor
        # Delete the field first if it already exists
        field_index = layer.FindFieldIndex(
            str(predictor_id), 1)
        if field_index >= 0:
            layer.DeleteField(field_index)
        predictor_field = ogr.FieldDefn(str(predictor_id), ogr.OFTReal)
        predictor_field.SetWidth(24)
        predictor_field.SetPrecision(11)
        layer.CreateField(predictor_field)

        with open(json_filename, 'r') as file:
            predictor_results = json.load(file)
        for feature_id, value in predictor_results.iteritems():
            feature = layer.GetFeature(int(feature_id))
            feature.SetField(str(predictor_id), value)
            layer.SetFeature(feature)

    # Get all the fieldnames, if they are not in the predictor_id_list,
    # get their index and delete
    n_fields = layer_defn.GetFieldCount()
    schema = []
    for idx in range(n_fields):
        field_defn = layer_defn.GetFieldDefn(idx)
        schema.append(field_defn.GetName())
    for field_name in schema:
        if field_name not in predictor_id_list:
            idx = layer.FindFieldIndex(field_name, 1)
            layer.DeleteField(idx)
    layer_defn = None
    layer = None
    vector.FlushCache()
    vector = None


def _raster_sum_count(
        raster_path, raster_type, response_vector_path,
        predictor_target_path):
    """Sum all non-nodata values in the raster under each polygon.

    Parameters:
        response_vector_path (string): path to response polygons
        raster_path (string): path to a raster.
        tmp_fid_raster_path (string): desired path to raster that will be used
            to aggregate `raster_path` values by unique response IDs.

    Returns:
        A dictionary indexing 'sum', 'mean', and 'count', to dictionaries
        mapping feature IDs from `response_polygons_lookup` to those values
        of the raster under the polygon.
    """
    aggregate_results = pygeoprocessing.zonal_statistics(
        (raster_path, 1), response_vector_path)
    # remove results when the pixel count is 0 (only nodata pixels).
    # we don't have predictor data for those features,
    # so features should be excluded from the linear regression.
    aggregate_results = {
        fid: stats for fid, stats in aggregate_results.iteritems()
        if stats['count'] != 0}
    if not aggregate_results:
        # raise ValueError('raster predictor does not intersect with vector AOI')
        LOGGER.warn('raster predictor does not intersect with vector AOI')
        # Create an empty file so that Taskgraph has its target file.
        predictor_results = {}
        with open(predictor_target_path, 'w') as jsonfile:
            json.dump(predictor_results, jsonfile)
        return None

    fid_raster_values = {
        'fid': aggregate_results.keys(),
        'sum': [fid['sum'] for fid in aggregate_results.values()],
        'count': [fid['count'] for fid in aggregate_results.values()],
        }

    if raster_type == 'mean':
        mean_results = (
            numpy.array(fid_raster_values['sum']) /
            numpy.array(fid_raster_values['count']))
        predictor_results = dict(
            zip(fid_raster_values['fid'], mean_results))
    else:
        predictor_results = dict(
            zip(fid_raster_values['fid'], fid_raster_values['sum']))
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(predictor_results, jsonfile)
    # predictor_json_list.append(predictor_target_path)
    # return fid_raster_values


def _polygon_area(
        mode, response_polygons_lookup, polygon_vector_path,
        predictor_target_path):
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
    prepped_polygons = [shapely.prepared.prep(polygon) for polygon in polygons]
    polygon_spatial_index = rtree.index.Index()
    for polygon_index, polygon in enumerate(polygons):
        polygon_spatial_index.insert(polygon_index, polygon.bounds)
    polygon_coverage_lookup = {}  # map FID to point count
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        if time.time() - start_time > 5.0:
            LOGGER.info(
                "%s polygon area: %.2f%% complete",
                os.path.basename(polygon_vector_path),
                (100.0*index)/len(response_polygons_lookup))
            start_time = time.time()

        potential_intersecting_poly_ids = polygon_spatial_index.intersection(
            geometry.bounds)
        intersecting_polygons = [
            polygons[polygon_index]
            for polygon_index in potential_intersecting_poly_ids
            if prepped_polygons[polygon_index].intersects(geometry)]
        polygon_area_coverage = sum([
            (geometry.intersection(polygon)).area
            for polygon in intersecting_polygons])

        if mode == 'area':
            polygon_coverage_lookup[feature_id] = polygon_area_coverage
        elif mode == 'percent':
            polygon_coverage_lookup[feature_id] = (
                polygon_area_coverage / geometry.area * 100.0)
    LOGGER.info(
        "%s polygon area: 100.00%% complete",
        os.path.basename(polygon_vector_path))
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(polygon_coverage_lookup, jsonfile)
    # return polygon_coverage_lookup


def _line_intersect_length(response_polygons_lookup, line_vector_path, predictor_target_path):
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

    line_spatial_index = rtree.index.Index()
    for line_index, line in enumerate(lines):
        line_spatial_index.insert(line_index, line.bounds)

    feature_count = None
    for feature_count, (feature_id, geometry) in enumerate(
            response_polygons_lookup.iteritems()):
        last_time = delay_op(
            last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                "%s line intersect length: %.2f%% complete",
                os.path.basename(line_vector_path),
                (100.0 * feature_count)/len(response_polygons_lookup)))
        potential_intersecting_lines = line_spatial_index.intersection(
            geometry.bounds)
        line_length = sum([
            (lines[line_index].intersection(geometry)).length
            for line_index in potential_intersecting_lines if
            geometry.intersects(lines[line_index])])
        line_length_lookup[feature_id] = line_length
    LOGGER.info(
        "%s line intersect length: 100.00%% complete",
        os.path.basename(line_vector_path))
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(line_length_lookup, jsonfile)
    # return line_length_lookup


def _point_nearest_distance(response_polygons_lookup, point_vector_path, predictor_target_path):
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
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(point_distance_lookup, jsonfile)
    # return point_distance_lookup


def _point_count(response_polygons_lookup, point_vector_path, predictor_target_path):
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
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(point_count_lookup, jsonfile)
    # return point_count_lookup


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
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
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
    vector = None
    return geometry_list


def _build_regression(
        response_vector_path, coefficient_vector_path,
        response_id):
    """Multiple regression for log response of the coefficient vector table.

    The regression is built such that each feature in the single layer vector
    pointed to by `coefficient_vector_path` corresponds to one data point.
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
        ssres: sums of squared residuals
        r_sq: R^2 value
        r_sq_adj: adjusted R^2 value
        std_err: residual standard error
        dof: degrees of freedom
        se_est: standard error estimate on coefficients
    """
    response_vector = gdal.OpenEx(response_vector_path, gdal.OF_VECTOR)
    response_layer = response_vector.GetLayer()

    coefficient_vector = gdal.OpenEx(coefficient_vector_path, gdal.OF_VECTOR)
    coefficient_layer = coefficient_vector.GetLayer()
    coefficient_layer_defn = coefficient_layer.GetLayerDefn()

    n_features = coefficient_layer.GetFeatureCount()
    # Not sure what would cause this to be untrue, but if it ever is,
    # we sure want to know about it.
    assert(n_features == response_layer.GetFeatureCount())

    # Response data matrix
    response_array = numpy.empty((n_features, 1))
    for row_index, feature in enumerate(response_layer):
        response_array[row_index, :] = feature.GetField(str(response_id))
    response_array = numpy.log1p(response_array)

    # Y-Intercept data matrix
    intercept_array = numpy.ones_like(response_array)

    # Predictor data matrix
    n_predictors = coefficient_layer_defn.GetFieldCount()
    coefficient_matrix = numpy.empty((n_features, n_predictors))
    predictor_names = []
    for idx in range(n_predictors):
        field_defn = coefficient_layer_defn.GetFieldDefn(idx)
        field_name = field_defn.GetName()
        predictor_names.append(field_name)
    for row_index, feature in enumerate(coefficient_layer):
        coefficient_matrix[row_index, :] = numpy.array(
            [feature.GetField(str(key)) for key in predictor_names])
    # for row_index, feature in enumerate(coefficient_layer):
    #     coefficient_matrix[row_index, :] = numpy.array(
    #         [feature.GetField(str(response_id))] + [
    #             feature.GetField(str(key)) for key in predictor_names] +
    #         [1])  # add the 1s for the y intercept
    # predictor_names.append('y-intercept')

    # If some predictor has no data in all features, drop that predictor:
    LOGGER.warn(predictor_names)
    valid_pred = ~numpy.isnan(coefficient_matrix).all(axis=0)
    LOGGER.warn(valid_pred)
    coefficient_matrix = coefficient_matrix[:, valid_pred]
    predictor_names = [
        pred for (pred, valid) in zip(predictor_names, valid_pred)
        if valid]
    n_predictors = coefficient_matrix.shape[1]
    # add columns for response variable and y-intercept
    data_matrix = numpy.concatenate(
        (response_array, coefficient_matrix, intercept_array), axis=1)
    predictor_names.append('(Intercept)')

    # if a variable is missing data for some features, drop those features:
    data_matrix = data_matrix[~numpy.isnan(data_matrix).any(axis=1)]
    n_features = data_matrix.shape[0]
    y_factors = data_matrix[:, 0]  # useful to have this as a 1-D array

    coefficients, _, _, _ = numpy.linalg.lstsq(
        data_matrix[:, 1:], y_factors, rcond=None)
    LOGGER.warn(predictor_names)
    LOGGER.warn(coefficients)
    # id_to_coefficient = dict(
    #     (p_id, coeff) for p_id, coeff in zip(
    #         predictor_names + ['y-intercept'], coefficients))
    ssres = numpy.sum((
        y_factors -
        numpy.sum(data_matrix[:, 1:] * coefficients, axis=1)) ** 2)
    # sstot = numpy.sum((
    #     numpy.average(response_array) -
    #     numpy.log1p(coefficient_matrix[:, 0])) ** 2)
    sstot = numpy.sum((
        numpy.average(y_factors) - y_factors) ** 2)
    dof = n_features - n_predictors - 1
    if sstot == 0.0 or dof <= 0.0:
        # this can happen if there is only one sample
        r_sq = 1.0
        r_sq_adj = 1.0
    else:
        r_sq = 1. - ssres / sstot
        r_sq_adj = 1 - (1 - r_sq) * (n_features - 1) / dof

    if dof > 0:
        std_err = numpy.sqrt(ssres / dof)
        sigma2 = numpy.sum((
            y_factors - numpy.sum(
                data_matrix[:, 1:] * coefficients, axis=1)) ** 2) / dof
        var_est = sigma2 * numpy.diag(numpy.linalg.pinv(
            numpy.dot(
                data_matrix[:, 1:].T, data_matrix[:, 1:])))
        se_est = numpy.sqrt(var_est)
    else:
        LOGGER.warn("Linear model is under constrained with DOF=%d", dof)
        std_err = sigma2 = numpy.nan
        se_est = var_est = [numpy.nan] * data_matrix.shape[1]
    return predictor_names, coefficients, ssres, r_sq, r_sq_adj, std_err, dof, se_est


def _calculate_scenario(
        base_aoi_path, response_id, id_to_coefficient,
        scenario_predictor_table_path, scenario_results_path,
        working_dir, task_graph):
    """Calculate the PUD of a scenario given an existing regression.

    It is expected that the predictor coefficients have been derived from a
    log normal distribution.

    Parameters:
        base_aoi_path (string): path to the a polygon vector that was used
            to build the original regression.  Geometry will be copied for
            `scenario_results_path` output vector.
        response_id (string): text ID of response variable to write to
            the scenario result
        predictor_coefficients (numpy.ndarray): 1D array of regression
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
        scenario_results_path (string): path to desired output scenario
            vector result which will be geometrically a copy of the input
            AOI but contain the base regression fields as well as the scenario
            derived response.

    Returns:
        None
    """
    _build_regression_coefficients(
        base_aoi_path, scenario_predictor_table_path,
        scenario_results_path, task_graph, working_dir)

    # id_to_coefficient = dict(
    #     (p_id, coeff) for p_id, coeff in zip(
    #         predictor_id_list, predictor_coefficients))

    # Open for writing
    scenario_coefficient_vector = gdal.OpenEx(
        scenario_results_path, gdal.OF_VECTOR | gdal.GA_Update)
    scenario_coefficient_layer = scenario_coefficient_vector.GetLayer()

    # delete the response ID if it's already there because it must have been
    # copied from the base layer
    response_index = scenario_coefficient_layer.FindFieldIndex(response_id, 1)
    if response_index >= 0:
        scenario_coefficient_layer.DeleteField(response_index)

    response_field = ogr.FieldDefn(response_id, ogr.OFTReal)
    response_field.SetWidth(24)
    response_field.SetPrecision(11)

    scenario_coefficient_layer.CreateField(response_field)
    y_intercept = id_to_coefficient.pop('y-intercept')

    for feature_id in xrange(scenario_coefficient_layer.GetFeatureCount()):
        feature = scenario_coefficient_layer.GetFeature(feature_id)
        response_value = 0.0
        try:
            for predictor_id, coefficient in id_to_coefficient.iteritems():
                response_value += (
                    coefficient *
                    feature.GetField(str(predictor_id)))
        # TypeError will happen if GetField returned None
        except TypeError as e:
            LOGGER.warn(
                'incomplete predictor data for feature_id %d, \
                not estimating PUD_EST' % feature_id)
            continue
        response_value += y_intercept
        # recall the coefficients are log normal, so expm1 inverses it
        feature.SetField(response_id, numpy.expm1(response_value))
        scenario_coefficient_layer.SetFeature(feature)

    scenario_coefficient_layer = None
    scenario_coefficient_vector.FlushCache()
    scenario_coefficient_vector = None


def _validate_same_id_lengths(table_path):
    """Ensure both table has ids of length less than 10.

    Parameter:
        table_path (string):  path to a csv table that has at least
            the field 'id'

    Raises:
        ValueError if any of the fields in 'id' and 'type' don't match between
        tables.
    """
    predictor_table = utils.build_lookup_from_csv(table_path, 'id')
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
    predictor_table = utils.build_lookup_from_csv(
        predictor_table_path, 'id')

    scenario_predictor_table = utils.build_lookup_from_csv(
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
    data_paths = utils.build_lookup_from_csv(
        table_path, 'path')

    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
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
        raster = gdal.OpenEx(path, gdal.OF_RASTER)
        gdal.PopErrorHandler()
        if raster is not None:
            projection_as_str = raster.GetProjection()
            ref = osr.SpatialReference()
            ref.ImportFromWkt(projection_as_str)
            raster = None
        else:
            vector = gdal.OpenEx(path, gdal.OF_VECTOR)
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
        'aoi_path',
        'start_year',
        'end_year',
        'compute_regression',
        'grid_aoi']

    if limit_to in [None, 'predictor_table_path']:
        if 'compute_regression' in args and args['compute_regression']:
            required_keys.append('predictor_table_path')

    if limit_to in [None, 'grid_type', 'cell_size']:
        if 'grid_aoi' in args and args['grid_aoi']:
            required_keys.append('grid_type')
            required_keys.append('cell_size')

    for key in required_keys:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ['', None]:
                no_value_list.append(key)

    if len(missing_key_list) > 0:
        # if there are missing keys, we have raise KeyError to stop hard
        print missing_key_list
        raise KeyError(
            "The following keys were expected in `args` but were missing " +
            ', '.join(missing_key_list))

    if len(no_value_list) > 0:
        validation_error_list.append(
            (no_value_list, 'parameter has no value'))

    if limit_to in [None, 'scenario_predictor_table_path']:
        if 'compute_regression' in args and args['compute_regression']:
            if (limit_to in [None, 'scenario_predictor_table_path'] and
                    'scenario_predictor_table_path' in args):
                scenario_predictor_table_path = args[
                    'scenario_predictor_table_path']
                if (scenario_predictor_table_path not in [None, ''] and
                        not os.path.exists(scenario_predictor_table_path)):
                    validation_error_list.append(
                        (['scenario_predictor_table_path'],
                         'not found on disk'))

    file_type_list = [
        ('aoi_path', 'vector'),
        ('predictor_table_path', 'table'),
        ('lulc_path', 'raster'),
        ('watersheds_path', 'vector'),
        ('biophysical_table_path', 'table')]

    if limit_to in ['drainage_path', None] and (
            'drainage_path' in args and
            args['drainage_path'] not in ['', None]):
        file_type_list.append(('drainage_path', 'raster'))

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in file_type_list:
            if (limit_to is None or limit_to == key) and key in args:
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

    return validation_error_list
