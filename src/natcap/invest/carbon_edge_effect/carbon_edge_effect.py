"""InVEST Carbon Edge Effect Model"""

import os
import logging

import numpy
import gdal
import pygeoprocessing

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.carbon_edge_effect')


def execute(args):
    """InVEST Carbon Edge Model calculates the carbon due to edge effects in
    forest pixels.

    Args:
        args['workspace_dir'] (string): a uri to the directory that will write
            output and other temporary files during calculation. (required)
        args['results_suffix'] (string): a string to append to any output file
            name (optional)
        args['biophysical_table_uri'] (string): a path to a CSV table that has
            at least a header for an 'lucode', 'is_forest', and 'c_above'.
                'lucode': an integer that corresponds to landcover codes in
                    the raster args['lulc_uri']

                'is_forest': either 0 or 1 indicating whether the landcover type
                    is forest (1) or not (0)

                'c_above': floating point number indicating tons of carbon per
                    hectare for that landcover type

                Example:
                    lucode, is_forest, c_above
                    0,0,32.8
                    1,1,n/a
                    2,1,n/a
                    16,0,28.1

        args['lulc_uri'] (string): path to a integer landcover code raster
        args['carbon_model_shape_uri'] (string): path to a shapefile that has
            areas defining carbon edge models.  Has at least the fields
            'method', 'theta1', 'theta2', 'theta3'

    returns None"""

    pygeoprocessing.create_directories([args['workspace_dir']])
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    #TASK: (optional) clip dataset to AOI if it exists

    #classify forest pixels from lulc
    biophysical_table = pygeoprocessing.get_lookup_from_table(
        args['biophysical_table_uri'], 'lucode')

    lucode_to_per_pixel_carbon = {}
    forest_codes = []
    cell_area_ha = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        args['lulc_uri']) ** 2 / 10000.0
    LOGGER.debug("cell area in ha %f", cell_area_ha)

    for lucode in biophysical_table:
        try:
            is_forest = int(biophysical_table[int(lucode)]['is_forest'])
            if is_forest == 1:
                forest_codes.append(lucode)
            lucode_to_per_pixel_carbon[int(lucode)] = float(
                biophysical_table[lucode]['c_above']) * cell_area_ha
        except ValueError:
            #this might be because the c_above parameter is n/a or undefined
            #because of forest
            lucode_to_per_pixel_carbon[int(lucode)] = 0.0

    #map aboveground carbon from table to lulc that is not forest
    carbon_map_nodata = -1
    non_edge_carbon_map_uri = os.path.join(
        args['workspace_dir'], 'non_edge_carbon_map%s.tif' % file_suffix)
    pygeoprocessing.reclassify_dataset_uri(
        args['lulc_uri'], lucode_to_per_pixel_carbon, non_edge_carbon_map_uri,
        gdal.GDT_Float32, carbon_map_nodata)

    #map distance to edge
    forest_mask_uri = os.path.join(
        args['workspace_dir'], 'non_forest_mask%s.tif' % file_suffix)
    forest_mask_nodata = 255
    lulc_nodata = pygeoprocessing.get_nodata_from_uri(args['lulc_uri'])
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(args['lulc_uri'])
    def mask_non_forest_op(lulc_array):
        """converts forest lulc codes to 1"""
        forest_mask = ~numpy.in1d(lulc_array.flatten(), forest_codes).reshape(
            lulc_array.shape)
        nodata_mask = lulc_array == lulc_nodata
        return numpy.where(nodata_mask, forest_mask_nodata, forest_mask)
    pygeoprocessing.vectorize_datasets(
        [args['lulc_uri']], mask_non_forest_op, forest_mask_uri, gdal.GDT_Byte,
        forest_mask_nodata, out_pixel_size, "intersection", vectorize_op=False)

    edge_distance_uri = os.path.join(
        args['workspace_dir'], 'edge_distance%s.tif' % file_suffix)
    pygeoprocessing.distance_transform_edt(forest_mask_uri, edge_distance_uri)

    #TASK rasterize points into a raster!
    carbon_model_reproject_uri = os.path.join(
        args['workspace_dir'], 'local_carbon_shape.shp')

    lulc_dataset = gdal.Open(args['lulc_uri'])
    lulc_projection_wkt = lulc_dataset.GetProjection()

    pygeoprocessing.reproject_datasource_uri(
        args['carbon_model_shape_uri'], lulc_projection_wkt,
        carbon_model_reproject_uri)

    for field_id, datatype, field_raster_nodata in [
            ('method', gdal.GDT_Byte, 255),
            ('theta1', gdal.GDT_Float32, -9999),
            ('theta2', gdal.GDT_Float32, -9999),
            ('theta3', gdal.GDT_Float32, -9999)]:
        raster_uri = os.path.join(args['workspace_dir'], field_id + '.tif')
        pygeoprocessing.new_raster_from_base_uri(
            args['lulc_uri'], raster_uri, 'GTiff', field_raster_nodata,
            datatype, fill_value=field_raster_nodata)
        pygeoprocessing.rasterize_layer_uri(
            raster_uri, carbon_model_reproject_uri,
            option_list=['ATTRIBUTE=%s' % field_id])
    #rasterize: method, theta 1-3

    #TASK: combine maps into output
    carbon_map_uri = os.path.join(
        args['workspace_dir'], 'carbon_map%s.tif' % file_suffix)


    #TASK: generate report (optional) by serviceshed if they exist
