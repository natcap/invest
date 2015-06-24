"""This reads in the CSV table of methods/parameters by grid-cell and creates a
    point shapefile of those locations"""

import os

import gdal
import ogr
import osr
import pygeoprocessing
import numpy
import scipy

def main():
    """main entry point"""


    table_uri = r"C:\Users\rich\Documents\invest-natcap.invest-3\test\invest-data\carbon_edge_effect\carbon_edge_regression_coefficients.csv"
    regression_table = pygeoprocessing.get_lookup_from_table(table_uri, 'id')

    raster_uri = r"C:\Users\rich\Documents\Dropbox\globio_sample_data_mgds\lulc_2008.tif"
    raster_ds = gdal.Open(raster_uri)

    ds_projection_wkt = raster_ds.GetProjection()
    output_sr = osr.SpatialReference()
    output_sr.ImportFromWkt(ds_projection_wkt)

    wgs84_sr = osr.SpatialReference()
    wgs84_sr.SetWellKnownGeogCS("WGS84")

    coord_trans = osr.CoordinateTransformation(wgs84_sr, output_sr)

    print regression_table[118]
    print coord_trans.TransformPoint(
        regression_table[118]['meanlon'], regression_table[118]['meanlat'])

    grid_points_uri = 'grid_points.shp'

    if os.path.isfile(grid_points_uri):
        os.remove(grid_points_uri)

    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    grid_points_datasource = output_driver.CreateDataSource(
        grid_points_uri)


    grid_points_layer = grid_points_datasource.CreateLayer(
        'grid_points', output_sr, ogr.wkbPoint)

    field_indexes = {
        0: 'id',
        1: 'method',
        2: 'theta1',
        3: 'theta2',
        4: 'theta3'
    }

    grid_points_layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    grid_points_layer.CreateField(ogr.FieldDefn('method', ogr.OFTInteger))
    grid_points_layer.CreateField(ogr.FieldDefn('theta1', ogr.OFTReal))
    grid_points_layer.CreateField(ogr.FieldDefn('theta2', ogr.OFTReal))
    grid_points_layer.CreateField(ogr.FieldDefn('theta3', ogr.OFTReal))

    ds_geotransform = raster_ds.GetGeoTransform()
    print ds_geotransform
    point_list = []
    value_list = []
    for grid_id in regression_table:
        grid_coords = coord_trans.TransformPoint(
            regression_table[grid_id]['meanlon'],
            regression_table[grid_id]['meanlat'])
        point_geometry = ogr.Geometry(ogr.wkbPoint)
        point_geometry.AddPoint(grid_coords[0], grid_coords[1])

        grid_col = (grid_coords[0] - ds_geotransform[0]) / ds_geotransform[1]
        grid_row = (grid_coords[1] - ds_geotransform[3]) / ds_geotransform[5]

        point_list.append([grid_col, grid_row])
        value_list.append(grid_id)
        if grid_col >= 0 and grid_col < raster_ds.RasterXSize and grid_row >= 0 and grid_row < raster_ds.RasterYSize:
            print grid_col, grid_row, grid_id
        # Get the output Layer's Feature Definition
        feature_def = grid_points_layer.GetLayerDefn()
        grid_point_feature = ogr.Feature(feature_def)
        grid_point_feature.SetGeometry(point_geometry)
        for field_index in field_indexes:
            try:
                grid_point_feature.SetField(
                    field_index,
                    float(regression_table[grid_id][field_indexes[field_index]]))
            except:
                grid_point_feature.SetField(field_index, -1.0)
        grid_points_layer.CreateFeature(grid_point_feature)

    print numpy.array(point_list)

    block_row_size = 256
    block_col_size = 256
    n_rows_blocks = 1
    n_cols_blocks = 1
    for block_row in xrange(n_rows_blocks):
        for block_col in xrange(n_cols_blocks):
            xi = []
            for local_row_index in xrange(block_row_size):
                for local_col_index in xrange(block_col_size):
                    xi.append((
                        block_col * block_col_size + local_col_index,
                        block_row * block_row_size + local_row_index))

            grid = scipy.interpolate.griddata(
                numpy.array(point_list),
                numpy.array(value_list),
                numpy.array(xi), method='nearest', fill_value=-1)
            print grid

    #import scipy
    #scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan)[source]

if __name__ == '__main__':

    main()
