"""InVEST Carbon Edge Effect Model"""

import os

import numpy
import gdal
import pygeoprocessing

def execute(args):
    """InVEST Carbon Edge Model calculates the carbon due to

        args['workspace_dir'] -
        args['results_suffix'] -
        args['biophysical_table_uri'] -
        args['lulc_uri'] -
        args['regression_coefficient_table'] -
        args['servicesheds_uri'] -
    """

    pygeoprocessing.create_directories([args['workspace_dir']])
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    #TASK: (optional) clip dataset to AOI if it exists

    #TASK: classify forest pixels from lulc
    biophysical_table = pygeoprocessing.get_lookup_from_table(
        args['biophysical_table_uri'], 'lucode')

    lucode_to_carbon = {}
    forest_codes = []
    for lucode in biophysical_table:
        try:
            is_forest = int(biophysical_table[int(lucode)]['is_forest'])
            if is_forest == 1:
                forest_codes.append(lucode)
            lucode_to_carbon[int(lucode)] = float(
                biophysical_table[lucode]['c_above'])
        except ValueError:
            #this might be because the c_above parameter is n/a or undefined
            #because of forest
            lucode_to_carbon[int(lucode)] = 0.0

    #TASK: map aboveground carbon from table to lulc that is not forest
    carbon_map_nodata = -1
    non_edge_carbon_map_uri = os.path.join(
        args['workspace_dir'], 'non_edge_carbon_map%s.tif' % file_suffix)
    pygeoprocessing.reclassify_dataset_uri(
        args['lulc_uri'], lucode_to_carbon, non_edge_carbon_map_uri,
        gdal.GDT_Float32, carbon_map_nodata)

    #TASK: map distance to edge
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
    pygeoprocessing.distance_transform_edt(
        forest_mask_uri, edge_distance_uri, process_pool=None)

    #TASK rasterize points into a raster!

    #TASK: combine maps into output
    carbon_map_uri = os.path.join(
        args['workspace_dir'], 'carbon_map%s.tif' % file_suffix)


    #TASK: generate report (optional) by serviceshed if they exist
