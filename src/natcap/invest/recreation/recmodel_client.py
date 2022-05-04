"""InVEST Recreation Client."""
import json
import os
import zipfile
import time
import logging
import math
import pickle
import urllib.request
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
from .. import spec_utils
from ..spec_utils import u
from .. import validation
from ..model_metadata import MODEL_METADATA
from .. import gettext


LOGGER = logging.getLogger(__name__)

# This URL is a NatCap global constant
RECREATION_SERVER_URL = 'http://data.naturalcapitalproject.org/server_registry/invest_recreation_model_py36/'  # pylint: disable=line-too-long

# 'marshal' serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'

predictor_table_columns = {
    "id": {
        "type": "freestyle_string",
        "about": gettext("A unique identifier for the predictor (10 "
                   "characters or less).")
    },
    "path": {
        "type": {"raster", "vector"},
        "about": gettext("A spatial file to use as a predictor."),
        "bands": {1: {"type": "number", "units": u.none}},
        "fields": {},
        "geometries": spec_utils.ALL_GEOMS
    },
    "type": {
        "type": "option_string",
        "about": gettext("The type of predictor file provided in the 'path' column."),
        "options": {
            "raster_mean": {
                "description": gettext(
                    "Predictor is a raster. Metric is the mean of values "
                    "within the AOI grid cell or polygon.")},
            "raster_sum": {
                "description": gettext(
                    "Predictor is a raster. Metric is the sum of values "
                    "within the AOI grid cell or polygon.")},
            "point_count": {
                "description": gettext(
                    "Predictor is a point vector. Metric is the number of "
                    "points within each AOI grid cell or polygon.")},
            "point_nearest_distance": {
                "description": gettext(
                    "Predictor is a point vector. Metric is the Euclidean "
                    "distance between the center of each AOI grid cell and "
                    "the nearest point in this layer.")},
            "line_intersect_length": {
                "description": gettext(
                    "Predictor is a line vector. Metric is the total length "
                    "of the lines that fall within each AOI grid cell.")},
            "polygon_area_coverage": {
                "description": gettext(
                    "Predictor is a polygon vector. Metric is the area of "
                    "overlap between the polygon and each AOI grid cell.")},
            "polygon_percent_coverage": {
                "description": gettext(
                    "Predictor is a polygon vector. Metric is the percentage "
                    "(0-100) of overlapping area between the polygon and each "
                    "AOI grid cell.")}
        }
    }
}


ARGS_SPEC = {
    "model_name": MODEL_METADATA["recreation"].model_title,
    "pyname": MODEL_METADATA["recreation"].pyname,
    "userguide": MODEL_METADATA["recreation"].userguide,
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "aoi_path": {
            **spec_utils.AOI,
            "about": gettext("Map of area(s) over which to run the model.")
        },
        "hostname": {
            "type": "freestyle_string",
            "required": False,
            "about": gettext(
                "FQDN to a recreation server.  If not provided, a default is "
                "assumed."),
            "name": gettext("hostname")
        },
        "port": {
            "type": "number",
            "expression": "value >= 0",
            "units": u.none,
            "required": False,
            "about": gettext(
                "the port on ``hostname`` to use for contacting the "
                "recreation server."),
            "name": gettext("port")
        },
        "start_year": {
            "type": "number",
            "expression": "value >= 2005",
            "units": u.year_AD,
            "about": gettext(
                "Year at which to start photo user-day calculations. "
                "Calculations start on the first day of the year. Year "
                "must be in the range 2005 - 2017, and must be less than "
                "or equal to the End Year."),
            "name": gettext("start year")
        },
        "end_year": {
            "type": "number",
            "expression": "value <= 2017",
            "units": u.year_AD,
            "about": gettext(
                "Year at which to end photo user-day calculations. "
                "Calculations continue through the last day of the year. "
                "Year must be in the range 2005 - 2017, and must be "
                "greater than or equal to the Start Year."),
            "name": gettext("end year")
        },
        "grid_aoi": {
            "type": "boolean",
            "required": False,
            "about": gettext(
                "Divide the AOI polygons into equal-sized grid cells, and "
                "compute results for those cells instead of the original "
                "polygons."),
            "name": gettext("grid the AOI")
        },
        "grid_type": {
            "type": "option_string",
            "options": {
                "square": {"display_name": gettext("square")},
                "hexagon": {"display_name": gettext("hexagon")}
            },
            "required": "grid_aoi",
            "about": gettext(
                "The shape of grid cells to make within the AOI polygons. "
                "Required if Grid AOI is selected."),
            "name": gettext("grid type")
        },
        "cell_size": {
            "type": "number",
            "expression": "value > 0",
            "units": u.other,  # any unit of length is ok
            "required": "grid_aoi",
            "about": gettext(
                "Size of grid cells to make, measured in the projection units "
                "of the AOI. If the Grid Type is 'square', this is the length "
                "of each side of the square. If the Grid Type is 'hexagon', "
                "this is the hexagon's maximal diameter."),
            "name": gettext("cell size")
        },
        "compute_regression": {
            "type": "boolean",
            "required": False,
            "about": gettext(
                "Run the regression model using the predictor table and "
                "scenario table, if provided."),
            "name": gettext("compute regression")
        },
        "predictor_table_path": {
            "type": "csv",
            "columns": predictor_table_columns,
            "required": "compute_regression",
            "about": gettext(
                "A table that maps predictor IDs to spatial files and their "
                "predictor metric types. The file paths can be absolute or "
                "relative to the table."),
            "name": gettext("predictor table")
        },
        "scenario_predictor_table_path": {
            "type": "csv",
            "columns": predictor_table_columns,
            "required": False,
            "about": gettext(
                "A table of future or alternative scenario predictors. Maps "
                "IDs to files and their types. The file paths can be absolute "
                "or relative to the table."),
            "name": gettext("scenario predictor table")
        }
    }
}


# These are the expected extensions associated with an ESRI Shapefile
# as part of the ESRI Shapefile driver standard, but some extensions
# like .prj, .sbn, and .sbx, are optional depending on versions of the
# format: http://www.gdal.org/drv_shapefile.html
_ESRI_SHAPEFILE_EXTENSIONS = ['.prj', '.shp', '.shx', '.dbf', '.sbn', '.sbx']

# Have 5 seconds between timed progress outputs
LOGGER_TIME_DELAY = 5

# For now, this is the field name we use to mark the photo user "days"
RESPONSE_ID = 'PUD_YR_AVG'
SCENARIO_RESPONSE_ID = 'PUD_EST'

_OUTPUT_BASE_FILES = {
    'pud_results_path': 'pud_results.shp',
    'monthly_table_path': 'monthly_table.csv',
    'predictor_vector_path': 'predictor_data.shp',
    'scenario_results_path': 'scenario_results.shp',
    'regression_coefficients': 'regression_coefficients.txt',
}

_INTERMEDIATE_BASE_FILES = {
    'local_aoi_path': 'aoi.shp',
    'compressed_aoi_path': 'aoi.zip',
    'compressed_pud_path': 'pud.zip',
    'response_polygons_lookup': 'response_polygons_lookup.pickle',
    'server_version': 'server_version.pickle',
}


def execute(args):
    """Recreation.

    Execute recreation client model on remote server.

    Args:
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
            ``args['aoi_path']`` should be gridded into a new vector and the
            recreation model should be executed on that
        args['grid_type'] (string): optional, but must exist if
            args['grid_aoi'] is True.  Is one of 'hexagon' or 'square' and
            indicates the style of gridding.
        args['cell_size'] (string/float): optional, but must exist if
            ``args['grid_aoi']`` is True.  Indicates the cell size of square
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
                    * 'polygon_percent_coverage': percent (0-100) of area of
                      overlap between the predictor and each AOI grid cell

        args['scenario_predictor_table_path'] (string): (optional) if
            present runs the scenario mode of the recreation model with the
            datasets described in the table on this path.  Field headers
            are identical to ``args['predictor_table_path']`` and ids in the
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
        _validate_predictor_types(args['predictor_table_path'])

    if ('predictor_table_path' in args and
            'scenario_predictor_table_path' in args and
            args['predictor_table_path'] != '' and
            args['scenario_predictor_table_path'] != ''):
        _validate_same_ids_and_types(
            args['predictor_table_path'],
            args['scenario_predictor_table_path'])
        _validate_same_projection(
            args['aoi_path'], args['scenario_predictor_table_path'])
        _validate_predictor_types(args['scenario_predictor_table_path'])

    if int(args['end_year']) < int(args['start_year']):
        raise ValueError(
            "Start year must be less than or equal to end year.\n"
            f"start_year: {args['start_year']}\nend_year: {args['end_year']}")

    # in case the user defines a hostname
    if 'hostname' in args:
        server_url = f"PYRO:natcap.invest.recreation@{args['hostname']}:{args['port']}"
    else:
        # else use a well known path to get active server
        server_url = urllib.request.urlopen(
            RECREATION_SERVER_URL).read().decode('utf-8').rstrip()
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    output_dir = args['workspace_dir']
    intermediate_dir = os.path.join(output_dir, 'intermediate')
    scenario_dir = os.path.join(intermediate_dir, 'scenario')
    utils.make_directories([output_dir, intermediate_dir])

    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_dir)], file_suffix)

    # Initialize a TaskGraph
    taskgraph_db_dir = os.path.join(intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    task_graph = taskgraph.TaskGraph(taskgraph_db_dir, n_workers)

    if args['grid_aoi']:
        prep_aoi_task = task_graph.add_task(
            func=_grid_vector,
            args=(args['aoi_path'], args['grid_type'],
                  float(args['cell_size']), file_registry['local_aoi_path']),
            target_path_list=[file_registry['local_aoi_path']],
            task_name='grid_aoi')
    else:
        # Even if we don't modify the AOI by gridding it, we still need
        # to move it to the expected location.
        prep_aoi_task = task_graph.add_task(
            func=_copy_aoi_no_grid,
            args=(args['aoi_path'], file_registry['local_aoi_path']),
            target_path_list=[file_registry['local_aoi_path']],
            task_name='copy_aoi')
    # All other tasks are dependent on this one, including tasks added
    # within _schedule_predictor_data_processing(). Rather than passing
    # this task to that function, I'm joining here.
    prep_aoi_task.join()

    # All the server communication happens in this task.
    photo_user_days_task = task_graph.add_task(
        func=_retrieve_photo_user_days,
        args=(file_registry['local_aoi_path'],
              file_registry['compressed_aoi_path'],
              args['start_year'], args['end_year'],
              os.path.basename(file_registry['pud_results_path']),
              os.path.basename(file_registry['monthly_table_path']),
              file_registry['compressed_pud_path'],
              output_dir, server_url, file_registry['server_version']),
        target_path_list=[file_registry['compressed_aoi_path'],
                          file_registry['compressed_pud_path'],
                          file_registry['pud_results_path'],
                          file_registry['monthly_table_path'],
                          file_registry['server_version']],
        task_name='photo-user-day-calculation')

    if 'compute_regression' in args and args['compute_regression']:
        # Prepare the AOI for geoprocessing.
        prepare_response_polygons_task = task_graph.add_task(
            func=_prepare_response_polygons_lookup,
            args=(file_registry['local_aoi_path'],
                  file_registry['response_polygons_lookup']),
            target_path_list=[file_registry['response_polygons_lookup']],
            task_name='prepare response polygons for geoprocessing')

        # Build predictor data
        build_predictor_data_task = _schedule_predictor_data_processing(
            file_registry['local_aoi_path'],
            file_registry['response_polygons_lookup'],
            prepare_response_polygons_task,
            args['predictor_table_path'],
            file_registry['predictor_vector_path'],
            intermediate_dir, task_graph)

        # Compute the regression
        coefficient_json_path = os.path.join(
            intermediate_dir, 'predictor_estimates.json')
        compute_regression_task = task_graph.add_task(
            func=_compute_and_summarize_regression,
            args=(file_registry['pud_results_path'],
                  file_registry['predictor_vector_path'],
                  file_registry['server_version'],
                  coefficient_json_path,
                  file_registry['regression_coefficients']),
            target_path_list=[file_registry['regression_coefficients'],
                              coefficient_json_path],
            dependent_task_list=[
                photo_user_days_task, build_predictor_data_task],
            task_name='compute regression')

        if ('scenario_predictor_table_path' in args and
                args['scenario_predictor_table_path'] != ''):
            utils.make_directories([scenario_dir])
            build_scenario_data_task = _schedule_predictor_data_processing(
                file_registry['local_aoi_path'],
                file_registry['response_polygons_lookup'],
                prepare_response_polygons_task,
                args['scenario_predictor_table_path'],
                file_registry['scenario_results_path'],
                scenario_dir, task_graph)

            task_graph.add_task(
                func=_calculate_scenario,
                args=(file_registry['scenario_results_path'],
                      SCENARIO_RESPONSE_ID, coefficient_json_path),
                target_path_list=[file_registry['scenario_results_path']],
                dependent_task_list=[
                    compute_regression_task, build_scenario_data_task],
                task_name='calculate scenario')

    task_graph.close()
    task_graph.join()


def _copy_aoi_no_grid(source_aoi_path, dest_aoi_path):
    """Copy a shapefile from source to destination"""
    aoi_vector = gdal.OpenEx(source_aoi_path, gdal.OF_VECTOR)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    local_aoi_vector = driver.CreateCopy(
        dest_aoi_path, aoi_vector)
    gdal.Dataset.__swig_destroy__(local_aoi_vector)
    local_aoi_vector = None
    gdal.Dataset.__swig_destroy__(aoi_vector)
    aoi_vector = None


def _retrieve_photo_user_days(
        local_aoi_path, compressed_aoi_path, start_year, end_year,
        pud_results_filename, monthly_table_filename, compressed_pud_path,
        output_dir, server_url, server_version_pickle):
    """Calculate photo-user-days (PUD) on the server and send back results.

    All of the client-server communication happens in this scope. The local AOI
    is sent to the server for PUD calculations. PUD results are sent back when
    complete.

    Args:
        local_aoi_path (string): path to polygon vector for PUD aggregation
        compressed_aoi_path (string): path to zip file storing compressed AOI
        start_year (int/string): lower limit of date-range for PUD queries
        end_year (int/string): upper limit of date-range for PUD queries
        pud_results_filename (string): filename for a shapefile to hold results
        monthly_table_filename (string): filename for monthly PUD results CSV
        compressed_pud_path (string): path to zip file storing compressed PUD
            results, including 'pud_results.shp' and 'monthly_table.csv'.
        output_dir (string): path to output workspace where results are
            unpacked.
        server_url (string): URL for connecting to the server
        server_version_pickle (string): path to a pickle that stores server
            version and workspace id info.

    Returns:
        None

    """

    LOGGER.info('Contacting server, please wait.')
    recmodel_server = Pyro4.Proxy(server_url)
    server_version = recmodel_server.get_version()
    LOGGER.info(f'Server online, version: {server_version}')
    # store server version info in a file so we can list it in results summary.
    with open(server_version_pickle, 'wb') as f:
        pickle.dump(server_version, f)

    # validate available year range
    min_year, max_year = recmodel_server.get_valid_year_range()
    LOGGER.info(
        f"Server supports year queries between {min_year} and {max_year}")
    if not min_year <= int(start_year) <= max_year:
        raise ValueError(
            f"Start year must be between {min_year} and {max_year}.\n"
            f" User input: ({start_year})")
    if not min_year <= int(end_year) <= max_year:
        raise ValueError(
            f"End year must be between {min_year} and {max_year}.\n"
            f" User input: ({end_year})")

    # append jan 1 to start and dec 31 to end
    date_range = (str(start_year)+'-01-01',
                  str(end_year)+'-12-31')

    basename = os.path.splitext(local_aoi_path)[0]
    with zipfile.ZipFile(compressed_aoi_path, 'w') as aoizip:
        for suffix in _ESRI_SHAPEFILE_EXTENSIONS:
            filename = basename + suffix
            if os.path.exists(filename):
                LOGGER.info(f'archiving {filename}')
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
    LOGGER.info(f'received result, took {time.time() - start_time} seconds, '
                f'workspace_id: {workspace_id}')

    # unpack result
    open(compressed_pud_path, 'wb').write(
        result_zip_file_binary)
    temporary_output_dir = tempfile.mkdtemp(dir=output_dir)
    zipfile.ZipFile(compressed_pud_path, 'r').extractall(
        temporary_output_dir)

    for filename in os.listdir(temporary_output_dir):
        shutil.copy(os.path.join(temporary_output_dir, filename), output_dir)
    shutil.rmtree(temporary_output_dir)
    # the monthly table is returned from the server without a results_suffix.
    shutil.move(
        os.path.join(output_dir, 'monthly_table.csv'),
        os.path.join(output_dir, monthly_table_filename))

    LOGGER.info('connection release')
    recmodel_server._pyroRelease()


def _grid_vector(vector_path, grid_type, cell_size, out_grid_vector_path):
    """Convert vector to a regular grid.

    Here the vector is gridded such that all cells are contained within the
    original vector.  Cells that would intersect with the boundary are not
    produced.

    Args:
        vector_path (string): path to an OGR compatible polygon vector type
        grid_type (string): one of "square" or "hexagon"
        cell_size (float): dimensions of the grid cell in the projected units
            of ``vector_path``; if "square" then this indicates the side length,
            if "hexagon" indicates the width of the horizontal axis.
        out_grid_vector_path (string): path to the output ESRI shapefile
            vector that contains a gridded version of ``vector_path``, this file
            should not exist before this call

    Returns:
        None

    """
    LOGGER.info("gridding aoi")
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
        shapely.ops.unary_union(original_vector_shapes))

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
        raise ValueError(f'Unknown polygon type: {grid_type}')

    for row_index in range(n_rows):
        for col_index in range(n_cols):
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


def _schedule_predictor_data_processing(
        response_vector_path, response_polygons_pickle_path,
        prepare_response_polygons_task,
        predictor_table_path, out_predictor_vector_path,
        working_dir, task_graph):
    """Summarize spatial predictor data by polygons in the response vector.

    Build a shapefile with geometry from the response vector, and tabular
    data from aggregate metrics of spatial predictor datasets in
    ``predictor_table_path``.

    Args:
        response_vector_path (string): path to a single layer polygon vector.
        response_polygons_pickle_path (string): path to pickle that stores a
            dict which maps each feature FID from ``response_vector_path`` to
            its shapely geometry.
        prepare_response_polygons_task (Taskgraph.Task object):
            A Task needed for dependent_task_lists in this scope.
        predictor_table_path (string): path to a CSV file with three columns
            'id', 'path' and 'type'.  'id' is the unique ID for that predictor
            and must be less than 10 characters long. 'path' indicates the
            full or relative path to the ``predictor_table_path`` table for the
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
        out_predictor_vector_path (string): path to a copy of
            ``response_vector_path`` with a column for each id in
            predictor_table_path. Overwritten if exists.
        working_dir (string): path to an intermediate directory to store json
            files with geoprocessing results.
        task_graph (Taskgraph): the graph that was initialized in execute()

    Returns:
        The ultimate task object from this branch of the taskgraph.

    """
    LOGGER.info('Processing predictor datasets')

    # lookup functions for response types
    # polygon predictor types are a special case because the polygon_area
    # function requires a 'mode' argument that these fucntions do not.
    predictor_functions = {
        'point_count': _point_count,
        'point_nearest_distance': _point_nearest_distance,
        'line_intersect_length': _line_intersect_length,
    }

    predictor_table = utils.build_lookup_from_csv(
        predictor_table_path, 'id')
    predictor_task_list = []
    predictor_json_list = []  # tracks predictor files to add to shp

    for predictor_id in predictor_table:
        LOGGER.info(f"Building predictor {predictor_id}")

        predictor_path = _sanitize_path(
            predictor_table_path, predictor_table[predictor_id]['path'])
        predictor_type = predictor_table[predictor_id]['type'].strip()
        if predictor_type.startswith('raster'):
            # type must be one of raster_sum or raster_mean
            raster_op_mode = predictor_type.split('_')[1]
            predictor_target_path = os.path.join(
                working_dir, predictor_id + '.json')
            predictor_json_list.append(predictor_target_path)
            predictor_task_list.append(task_graph.add_task(
                func=_raster_sum_mean,
                args=(predictor_path, raster_op_mode, response_vector_path,
                      predictor_target_path),
                target_path_list=[predictor_target_path],
                task_name=f'predictor {predictor_id}'))
        # polygon types are a special case because the polygon_area
        # function requires an additional 'mode' argument.
        elif predictor_type.startswith('polygon'):
            predictor_target_path = os.path.join(
                working_dir, predictor_id + '.json')
            predictor_json_list.append(predictor_target_path)
            predictor_task_list.append(task_graph.add_task(
                func=_polygon_area,
                args=(predictor_type, response_polygons_pickle_path,
                      predictor_path, predictor_target_path),
                target_path_list=[predictor_target_path],
                dependent_task_list=[prepare_response_polygons_task],
                task_name=f'predictor {predictor_id}'))
        else:
            predictor_target_path = os.path.join(
                working_dir, predictor_id + '.json')
            predictor_json_list.append(predictor_target_path)
            predictor_task_list.append(task_graph.add_task(
                func=predictor_functions[predictor_type],
                args=(response_polygons_pickle_path, predictor_path,
                      predictor_target_path),
                target_path_list=[predictor_target_path],
                dependent_task_list=[prepare_response_polygons_task],
                task_name=f'predictor {predictor_id}'))

    assemble_predictor_data_task = task_graph.add_task(
        func=_json_to_shp_table,
        args=(response_vector_path, out_predictor_vector_path,
              predictor_json_list),
        target_path_list=[out_predictor_vector_path],
        dependent_task_list=predictor_task_list,
        task_name='assemble predictor data')

    return assemble_predictor_data_task


def _prepare_response_polygons_lookup(
        response_vector_path, target_pickle_path):
    """Translate a shapefile to a dictionary that maps FIDs to geometries."""
    response_vector = gdal.OpenEx(response_vector_path, gdal.OF_VECTOR)
    response_layer = response_vector.GetLayer()
    response_polygons_lookup = {}  # maps FID to prepared geometry
    for response_feature in response_layer:
        feature_geometry = response_feature.GetGeometryRef()
        feature_polygon = shapely.wkt.loads(feature_geometry.ExportToWkt())
        feature_geometry = None
        response_polygons_lookup[response_feature.GetFID()] = feature_polygon
    response_layer = None
    with open(target_pickle_path, 'wb') as pickle_file:
        pickle.dump(response_polygons_lookup, pickle_file)


def _json_to_shp_table(
        response_vector_path, predictor_vector_path,
        predictor_json_list):
    """Create a shapefile and a field with data from each json file.

    Args:
        response_vector_path (string): Path to the response vector polygon
            shapefile.
        predictor_vector_path (string): a copy of ``response_vector_path``.
            One field will be added for each json file, and all other
            fields will be deleted.
        predictor_json_list (list): list of json filenames, one for each
            predictor dataset. A json file will look like this,
            {0: 0.0, 1: 0.0}
            Keys match FIDs of ``response_vector_path``.

    Returns:
        None

    """
    driver = gdal.GetDriverByName('ESRI Shapefile')
    if os.path.exists(predictor_vector_path):
        driver.Delete(predictor_vector_path)
    response_vector = gdal.OpenEx(
        response_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    predictor_vector = driver.CreateCopy(
        predictor_vector_path, response_vector)
    response_vector = None

    layer = predictor_vector.GetLayer()
    layer_defn = layer.GetLayerDefn()

    predictor_id_list = []
    for json_filename in predictor_json_list:
        predictor_id = os.path.basename(os.path.splitext(json_filename)[0])
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
        for feature_id, value in predictor_results.items():
            feature = layer.GetFeature(int(feature_id))
            feature.SetField(str(predictor_id), value)
            layer.SetFeature(feature)

    # Get all the fieldnames. If they are not in the predictor_id_list,
    # get their index and delete
    n_fields = layer_defn.GetFieldCount()
    fieldnames = []
    for idx in range(n_fields):
        field_defn = layer_defn.GetFieldDefn(idx)
        fieldnames.append(field_defn.GetName())
    for field_name in fieldnames:
        if field_name not in predictor_id_list:
            idx = layer.FindFieldIndex(field_name, 1)
            layer.DeleteField(idx)
    layer_defn = None
    layer = None
    predictor_vector.FlushCache()
    predictor_vector = None


def _raster_sum_mean(
        raster_path, op_mode, response_vector_path,
        predictor_target_path):
    """Sum or mean for all non-nodata values in the raster under each polygon.

    Args:
        raster_path (string): path to a raster.
        op_mode (string): either 'mean' or 'sum'.
        response_vector_path (string): path to response polygons
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from ``response_vector_path``
            to values of the raster under the polygon.

    Returns:
        None

    """
    aggregate_results = pygeoprocessing.zonal_statistics(
        (raster_path, 1), response_vector_path,
        polygons_might_overlap=False)
    # remove results for a feature when the pixel count is 0.
    # we don't have non-nodata predictor values for those features.
    aggregate_results = {
        str(fid): stats for fid, stats in aggregate_results.items()
        if stats['count'] != 0}
    if not aggregate_results:
        LOGGER.warning('raster predictor does not intersect with vector AOI')
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

    if op_mode == 'mean':
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


def _polygon_area(
        mode, response_polygons_pickle_path, polygon_vector_path,
        predictor_target_path):
    """Calculate polygon area overlap.

    Calculates the amount of projected area overlap from ``polygon_vector_path``
    with ``response_polygons_lookup``.

    Args:
        mode (string): one of 'area' or 'percent'.  How this is set affects
            the metric that's output.  'area' is the area covered in projected
            units while 'percent' is percent of the total response area
            covered.
        response_polygons_pickle_path (str): path to a pickled dictionary which
            maps response polygon feature ID to prepared shapely.Polygon.
        polygon_vector_path (string): path to a single layer polygon vector
            object.
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from
            ``response_polygons_pickle_path`` to polygon area coverage.

    Returns:
        None

    """
    start_time = time.time()
    with open(response_polygons_pickle_path, 'rb') as pickle_file:
        response_polygons_lookup = pickle.load(pickle_file)
    polygons = _ogr_to_geometry_list(polygon_vector_path)
    prepped_polygons = [shapely.prepared.prep(polygon) for polygon in polygons]
    polygon_spatial_index = rtree.index.Index()
    for polygon_index, polygon in enumerate(polygons):
        polygon_spatial_index.insert(polygon_index, polygon.bounds)
    polygon_coverage_lookup = {}  # map FID to point count

    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.items()):
        if time.time() - start_time > 5:
            LOGGER.info(
                f"{os.path.basename(polygon_vector_path)} polygon area: "
                f"{(100*index)/len(response_polygons_lookup):.2f}% complete")
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

        if mode == 'polygon_area_coverage':
            polygon_coverage_lookup[feature_id] = polygon_area_coverage
        elif mode == 'polygon_percent_coverage':
            polygon_coverage_lookup[str(feature_id)] = (
                polygon_area_coverage / geometry.area * 100)
    LOGGER.info(f"{os.path.basename(polygon_vector_path)} polygon area: "
                f"100.00% complete")

    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(polygon_coverage_lookup, jsonfile)


def _line_intersect_length(
        response_polygons_pickle_path,
        line_vector_path, predictor_target_path):
    """Calculate the length of the intersecting lines on the response polygon.

    Args:
        response_polygons_pickle_path (str): path to a pickled dictionary which
            maps response polygon feature ID to prepared shapely.Polygon.
        line_vector_path (string): path to a single layer line vector
            object.
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from
            ``response_polygons_pickle_path`` to line intersect length.

    Returns:
        None

    """
    last_time = time.time()
    with open(response_polygons_pickle_path, 'rb') as pickle_file:
        response_polygons_lookup = pickle.load(pickle_file)
    lines = _ogr_to_geometry_list(line_vector_path)
    line_length_lookup = {}  # map FID to intersecting line length

    line_spatial_index = rtree.index.Index()
    for line_index, line in enumerate(lines):
        line_spatial_index.insert(line_index, line.bounds)

    feature_count = None
    for feature_count, (feature_id, geometry) in enumerate(
            response_polygons_lookup.items()):
        last_time = delay_op(
            last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                f"{os.path.basename(line_vector_path)} line intersect length: "
                f"{(100 * feature_count)/len(response_polygons_lookup):.2f}% complete"))
        potential_intersecting_lines = line_spatial_index.intersection(
            geometry.bounds)
        line_length = sum([
            (lines[line_index].intersection(geometry)).length
            for line_index in potential_intersecting_lines if
            geometry.intersects(lines[line_index])])
        line_length_lookup[str(feature_id)] = line_length
    LOGGER.info(f"{os.path.basename(line_vector_path)} line intersect length: "
                "100.00% complete")
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(line_length_lookup, jsonfile)


def _point_nearest_distance(
        response_polygons_pickle_path, point_vector_path,
        predictor_target_path):
    """Calculate distance to nearest point for all polygons.

    Args:
        response_polygons_pickle_path (str): path to a pickled dictionary which
            maps response polygon feature ID to prepared shapely.Polygon.
        point_vector_path (string): path to a single layer point vector
            object.
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from
            ``response_polygons_pickle_path`` to distance to nearest point.

    Returns:
        None

    """
    last_time = time.time()
    with open(response_polygons_pickle_path, 'rb') as pickle_file:
        response_polygons_lookup = pickle.load(pickle_file)
    points = _ogr_to_geometry_list(point_vector_path)
    point_distance_lookup = {}  # map FID to point count

    index = None
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.items()):
        last_time = delay_op(
            last_time, 5, lambda: LOGGER.info(
                f"{os.path.basename(point_vector_path)} point distance: "
                f"{(100*index)/len(response_polygons_lookup):.2f}% complete"))

        point_distance_lookup[str(feature_id)] = min([
            geometry.distance(point) for point in points])
    LOGGER.info(f"{os.path.basename(point_vector_path)} point distance: "
                "100.00% complete")
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(point_distance_lookup, jsonfile)


def _point_count(
        response_polygons_pickle_path, point_vector_path,
        predictor_target_path):
    """Calculate number of points contained in each response polygon.

    Args:
        response_polygons_pickle_path (str): path to a pickled dictionary which
            maps response polygon feature ID to prepared shapely.Polygon.
        point_vector_path (string): path to a single layer point vector
            object.
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from
            ``response_polygons_pickle_path`` to the number of points in that
            polygon.

    Returns:
        None

    """
    last_time = time.time()
    with open(response_polygons_pickle_path, 'rb') as pickle_file:
        response_polygons_lookup = pickle.load(pickle_file)
    points = _ogr_to_geometry_list(point_vector_path)
    point_count_lookup = {}  # map FID to point count

    index = None
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.items()):
        last_time = delay_op(
            last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                f"{os.path.basename(point_vector_path)} point count: "
                f"{(100*index)/len(response_polygons_lookup):.2f}% complete"))
        point_count = len([
            point for point in points if geometry.contains(point)])
        point_count_lookup[str(feature_id)] = point_count
    LOGGER.info(f"{os.path.basename(point_vector_path)} point count: "
                "100.00% complete")
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(point_count_lookup, jsonfile)


def _ogr_to_geometry_list(vector_path):
    """Convert an OGR type with one layer to a list of shapely geometry.

    Iterates through the features in the ``vector_path``'s first layer and
    converts them to ``shapely`` geometry objects.  if the objects are not
    valid geometry, an attempt is made to buffer the object by 0 units
    before adding to the list.

    Args:
        vector_path (string): path to an OGR vector

    Returns:
        list of shapely geometry objects representing the features in the
        ``vector_path`` layer.

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


def _compute_and_summarize_regression(
        response_vector_path, predictor_vector_path, server_version_path,
        target_coefficient_json_path, target_regression_summary_path):
    """Compute a regression and summary statistics and generate a report.

    Args:
        response_vector_path (string): path to polygon vector containing the
            RESPONSE_ID field.
        predictor_vector_path (string): path to polygon vector containing
            fields for each predictor variable. Geometry is identical to that
            of 'response_vector_path'.
        server_version_path (string): path to pickle file containing the
            rec server id hash.
        target_coefficient_json_path (string): path to json file to store a dictionary
            that maps a predictor id its coefficient estimate.
            This file is created by this function.
        target_regression_summary_path (string): path to txt file for the report.
            This file is created by this function.

    Returns:
        None

    """
    predictor_id_list, coefficients, ssres, r_sq, r_sq_adj, std_err, dof, se_est = (
        _build_regression(
            response_vector_path, predictor_vector_path, RESPONSE_ID))

    # Generate a nice looking regression result and write to log and file
    coefficients_string = '               estimate     stderr    t value\n'
    # The last coefficient is the y-intercept,
    # but we want it at the top of the report, thus [-1] on lists
    coefficients_string += (
        f'{predictor_id_list[-1]:12} {coefficients[-1]:+.3e} '
        f'{se_est[-1]:+.3e} {coefficients[-1] / se_est[-1]:+.3e}\n')
    # Since the intercept has already been reported, [:-1] on all the lists
    coefficients_string += '\n'.join(
        f'{p_id:12} {coefficient:+.3e} {se_est_factor:+.3e} '
        f'{coefficient / se_est_factor:+.3e}'
        for p_id, coefficient, se_est_factor in zip(
            predictor_id_list[:-1], coefficients[:-1], se_est[:-1]))

    # Include the server version and PUD hash in the report:
    with open(server_version_path, 'rb') as f:
        server_version = pickle.load(f)
    report_string = (
        f'\n******************************\n'
        f'{coefficients_string}\n'
        f'---\n\n'
        f'Residual standard error: {std_err:.4f} on {dof} degrees of freedom\n'
        f'Multiple R-squared: {r_sq:.4f}\n'
        f'Adjusted R-squared: {r_sq_adj:.4f}\n'
        f'SSres: {ssres:.4f}\n'
        f'server id hash: {server_version}\n'
        f'********************************\n')
    LOGGER.info(report_string)
    with open(target_regression_summary_path, 'w') as \
            regression_log:
        regression_log.write(report_string + '\n')

    # Predictor coefficients are needed for _calculate_scenario()
    predictor_estimates = dict(zip(predictor_id_list, coefficients))
    with open(target_coefficient_json_path, 'w') as json_file:
        json.dump(predictor_estimates, json_file)


def _build_regression(
        response_vector_path, predictor_vector_path,
        response_id):
    """Multiple least-squares regression with log-transformed response.

    The regression is built such that each feature in the single layer vector
    pointed to by ``predictor_vector_path`` corresponds to one data point.
    ``response_id`` is the response variable to be log-transformed, and is found
    in ``response_vector_path``. Predictor variables are found in
    ``predictor_vector_path`` and are not transformed. Features with incomplete
    data are dropped prior to computing the regression.

    Args:
        response_vector_path (string): path to polygon vector with PUD
            results, in particular a field named with the ``response_id``.
        predictor_vector_path (string): path to a shapefile that contains
            only the fields to be used as predictor variables.
        response_id (string): field ID in ``response_vector_path`` whose
            values correspond to the regression response variable.

    Asserts:
        ``response_vector_path`` and ``predictor_vector_path`` have an equal
        number of features.

    Returns:
        predictor_names: A list of predictor id strings. Length matches
            coeffiecients.
        coefficients: A list of coefficients in the least-squares solution
            including the y intercept as the last element.
        ssres: sums of squared residuals
        r_sq: R^2 value
        r_sq_adj: adjusted R^2 value
        std_err: residual standard error
        dof: degrees of freedom
        se_est: A list of standard error estimate, length matches coefficients.

    """
    LOGGER.info("Computing regression")
    response_vector = gdal.OpenEx(response_vector_path, gdal.OF_VECTOR)
    response_layer = response_vector.GetLayer()

    predictor_vector = gdal.OpenEx(predictor_vector_path, gdal.OF_VECTOR)
    predictor_layer = predictor_vector.GetLayer()
    predictor_layer_defn = predictor_layer.GetLayerDefn()

    n_features = predictor_layer.GetFeatureCount()
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
    n_predictors = predictor_layer_defn.GetFieldCount()
    predictor_matrix = numpy.empty((n_features, n_predictors))
    predictor_names = []
    for idx in range(n_predictors):
        field_defn = predictor_layer_defn.GetFieldDefn(idx)
        field_name = field_defn.GetName()
        predictor_names.append(field_name)
    for row_index, feature in enumerate(predictor_layer):
        predictor_matrix[row_index, :] = numpy.array(
            [feature.GetField(str(key)) for key in predictor_names])

    # If some predictor has no data across all features, drop that predictor:
    valid_pred = ~numpy.isnan(predictor_matrix).all(axis=0)
    predictor_matrix = predictor_matrix[:, valid_pred]
    predictor_names = [
        pred for (pred, valid) in zip(predictor_names, valid_pred)
        if valid]
    n_predictors = predictor_matrix.shape[1]

    # add columns for response variable and y-intercept
    data_matrix = numpy.concatenate(
        (response_array, predictor_matrix, intercept_array), axis=1)
    predictor_names.append('(Intercept)')

    # if any variable is missing data for some feature, drop that feature:
    data_matrix = data_matrix[~numpy.isnan(data_matrix).any(axis=1)]
    n_features = data_matrix.shape[0]
    y_factors = data_matrix[:, 0]  # useful to have this as a 1-D array
    coefficients, _, _, _ = numpy.linalg.lstsq(
        data_matrix[:, 1:], y_factors, rcond=-1)

    ssres = numpy.sum((
        y_factors -
        numpy.sum(data_matrix[:, 1:] * coefficients, axis=1)) ** 2)
    sstot = numpy.sum((
        numpy.average(y_factors) - y_factors) ** 2)
    dof = n_features - n_predictors - 1
    if sstot == 0 or dof <= 0:
        # this can happen if there is only one sample
        r_sq = 1
        r_sq_adj = 1
    else:
        r_sq = 1 - ssres / sstot
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
        LOGGER.warning(f"Linear model is under constrained with DOF={dof}")
        std_err = sigma2 = numpy.nan
        se_est = var_est = [numpy.nan] * data_matrix.shape[1]
    return predictor_names, coefficients, ssres, r_sq, r_sq_adj, std_err, dof, se_est


def _calculate_scenario(
        scenario_results_path, response_id, coefficient_json_path):
    """Estimate the PUD of a scenario given an existing regression equation.

    It is expected that the predictor coefficients have been derived from a
    log normal distribution.

    Args:
        scenario_results_path (string): path to desired output scenario
            vector result which will be geometrically a copy of the input
            AOI but contain the scenario predictor data fields as well as the
            scenario esimated response.
        response_id (string): text ID of response variable to write to
            the scenario result.
        coefficient_json_path (string): path to json file with the pre-existing
            regression results. It contains a dictionary that maps
            predictor id strings to coefficient values. Includes Y-Intercept.

    Returns:
        None

    """
    LOGGER.info("Calculating scenario results")

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

    # Load the pre-existing predictor coefficients to build the regression
    # equation.
    with open(coefficient_json_path, 'r') as json_file:
        predictor_estimates = json.load(json_file)

    y_intercept = predictor_estimates.pop("(Intercept)")

    for feature in scenario_coefficient_layer:
        feature_id = feature.GetFID()
        response_value = 0
        try:
            for predictor_id, coefficient in predictor_estimates.items():
                response_value += (
                    coefficient *
                    feature.GetField(str(predictor_id)))
        except TypeError:
            # TypeError will happen if GetField returned None
            LOGGER.warning('incomplete predictor data for feature_id '
                           f'{feature_id}, not estimating PUD_EST')
            feature = None
            continue  # without writing to the feature
        response_value += y_intercept
        # recall the coefficients are log normal, so expm1 inverses it
        feature.SetField(response_id, numpy.expm1(response_value))
        scenario_coefficient_layer.SetFeature(feature)
        feature = None

    scenario_coefficient_layer = None
    scenario_coefficient_vector.FlushCache()
    scenario_coefficient_vector = None


def _validate_same_id_lengths(table_path):
    """Ensure a predictor table has ids of length less than 10.

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
            "The following IDs are more than 10 characters long: "
            f"{str(too_long)}")


def _validate_same_ids_and_types(
        predictor_table_path, scenario_predictor_table_path):
    """Ensure both tables have same ids and types.

    Assert that both the elements of the 'id' and 'type' fields of each table
    contain the same elements and that their values are the same.  This
    ensures that a user won't get an accidentally incorrect simulation result.

    Args:
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
        (p_id, predictor_table[p_id]['type'].strip()) for p_id in predictor_table])
    scenario_predictor_table_pairs = set([
        (p_id, scenario_predictor_table[p_id]['type'].strip()) for p_id in
        scenario_predictor_table])
    if predictor_table_pairs != scenario_predictor_table_pairs:
        raise ValueError('table pairs unequal.\n\t'
                         f'predictor: {predictor_table_pairs}\n\t'
                         f'scenario:{scenario_predictor_table_pairs}')
    LOGGER.info('tables validate correctly')


def _validate_same_projection(base_vector_path, table_path):
    """Assert the GIS data in the table are in the same projection as the AOI.

    Args:
        base_vector_path (string): path to a GIS vector
        table_path (string): path to a csv table that has at least
            the field 'path'

    Returns:
        None

    Raises:
        ValueError if the projections in each of the GIS types in the table
            are not identical to the projection in base_vector_path

    """
    # This will load the table as a list of paths which we can iterate through
    # without bothering the rest of the table structure
    data_paths = utils.read_csv_to_dataframe(
        table_path, to_lower=True, squeeze=True)['path'].tolist()

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
                raise ValueError(f"{path} did not load")
            layer = vector.GetLayer()
            ref = osr.SpatialReference(layer.GetSpatialRef().ExportToWkt())
            layer = None
            vector = None
        if not base_ref.IsSame(ref):
            LOGGER.warning(
                f"{path} might have a different projection than the base AOI\n"
                f"base:{base_ref.ExportToPrettyWkt()}\n"
                f"current:{ref.ExportToPrettyWkt()}")
            invalid_projections = True
    if invalid_projections:
        raise ValueError(
            "One or more of the projections in the table did not match the "
            "projection of the base vector")


def _validate_predictor_types(table_path):
    """Validate the type values in a predictor table.

    Args:
        table_path (string): path to a csv table that has at least
            the field 'type'

    Returns:
        None

    Raises:
        ValueError if any value in the ``type`` column does not match a valid
        type, ignoring leading/trailing whitespace.
    """
    df = utils.read_csv_to_dataframe(table_path, to_lower=True)
    # ignore leading/trailing whitespace because it will be removed
    # when the type values are used
    type_list = set([type.strip() for type in df['type']])
    valid_types = set({'raster_mean', 'raster_sum', 'point_count',
                       'point_nearest_distance', 'line_intersect_length',
                       'polygon_area_coverage', 'polygon_percent_coverage'})
    difference = type_list.difference(valid_types)
    if difference:
        raise ValueError('The table contains invalid type value(s): '
                         f'{difference}. The allowed types are: {valid_types}')


def delay_op(last_time, time_delay, func):
    """Execute ``func`` if last_time + time_delay >= current time.

    Args:
        last_time (float): last time in seconds that ``func`` was triggered
        time_delay (float): time to wait in seconds since last_time before
            triggering ``func``
        func (function): parameterless function to invoke if
         current_time >= last_time + time_delay

    Returns:
        If ``func`` was triggered, return the time which it was triggered in
        seconds, otherwise return ``last_time``.

    """
    if time.time() - last_time > time_delay:
        func()
        return time.time()
    return last_time


def _sanitize_path(base_path, raw_path):
    """Return ``path`` if absolute, or make absolute local to ``base_path``."""
    if os.path.isabs(raw_path):
        return raw_path
    else:  # assume relative path w.r.t. the response table
        return os.path.join(os.path.dirname(base_path), raw_path)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in ``execute`` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire ``args`` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    return validation.validate(args, ARGS_SPEC['args'])
