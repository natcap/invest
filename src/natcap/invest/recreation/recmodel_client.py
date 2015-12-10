"""InVEST Recreation Client"""

import os
import glob
import zipfile
import time
import logging
import math

import Pyro4
from osgeo import ogr
import shapely
import shapely.wkt
import shapely.prepared

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_client')

#this serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'


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

    Returns:
        None."""

    date_range = (args['start_date'], args['end_date'])

    if not os.path.exists(args['workspace_dir']):
        os.makedirs(args['workspace_dir'])

    recmodel_server = Pyro4.Proxy(
        "PYRO:natcap.invest.recreation@%s:%d" % (
            args['hostname'], int(args['port'])))

    if args['grid_aoi']:
        out_grid_vector_path = os.path.splitext(
            args['aoi_path'])[0] + '_grid.shp'
        grid_vector(
            args['aoi_path'], args['grid_type'], args['cell_size'],
            out_grid_vector_path)
        aoi_path = out_grid_vector_path
    else:
        aoi_path = args['aoi_path']

    basename = os.path.splitext(aoi_path)[0]
    aoi_archive_path = os.path.join(args['workspace_dir'], 'aoi_zipped.zip')
    with zipfile.ZipFile(aoi_archive_path, 'w') as myzip:
        for filename in glob.glob(basename + '.*'):
            LOGGER.info('archiving %s', filename)
            myzip.write(filename, os.path.basename(filename))

    #convert shapefile to binary string for serialization
    zip_file_binary = open(aoi_archive_path, 'rb').read()

    #transfer zipped file to server
    start_time = time.time()
    LOGGER.info('server version is %s', recmodel_server.get_version())

    result_zip_file_binary = (
        recmodel_server.calc_aggregated_points_in_aoi(
            zip_file_binary, date_range, args['aggregating_metric']))
    LOGGER.info('received result, took %f seconds', time.time() - start_time)

    #unpack result
    result_zip_path = os.path.join(args['workspace_dir'], 'pud_result.zip')
    open(result_zip_path, 'wb').write(result_zip_file_binary)
    zipfile.ZipFile(result_zip_path, 'r').extractall(args['workspace_dir'])
    os.remove(result_zip_path)


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
