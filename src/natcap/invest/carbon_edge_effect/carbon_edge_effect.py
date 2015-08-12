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

    #rasterize: method, theta 1-3 into rasters that can be vectorized
    carbon_model_reproject_uri = os.path.join(
        args['workspace_dir'], 'local_carbon_shape.shp')

    lulc_dataset = gdal.Open(args['lulc_uri'])
    lulc_projection_wkt = lulc_dataset.GetProjection()

    pygeoprocessing.reproject_datasource_uri(
        args['carbon_model_shape_uri'], lulc_projection_wkt,
        carbon_model_reproject_uri)

    model_raster_uris = {}
    for field_id, datatype, field_raster_nodata in [
            ('method', gdal.GDT_Byte, 255),
            ('theta1', gdal.GDT_Float32, -9999),
            ('theta2', gdal.GDT_Float32, -9999),
            ('theta3', gdal.GDT_Float32, -9999)]:
        raster_uri = os.path.join(args['workspace_dir'], field_id + '.tif')
        model_raster_uris[field_id] = raster_uri
        pygeoprocessing.new_raster_from_base_uri(
            args['lulc_uri'], raster_uri, 'GTiff', field_raster_nodata,
            datatype, fill_value=field_raster_nodata)
        pygeoprocessing.rasterize_layer_uri(
            raster_uri, carbon_model_reproject_uri,
            option_list=['ATTRIBUTE=%s' % field_id])

    #TASK: calculate edge effect carbon raster
    carbon_edge_nodata = -9999.0
    cell_size_in_meters = pygeoprocessing.get_cell_size_from_uri(
        args['lulc_uri'])
    def carbon_edge_op(
            edge_distance, method, theta_1, theta_2, theta_3, forest_mask):
        """calculate carbon model
        Args:

            edge_distance (numpy.array): distance from forest edge in pixels
            method (numpy.array): values with 1, 2, or 3 indicating the three
                model regression types
            theta_{1,2,3} (numpy.array): parameters to regression that have
                different meaning depending on which method
            forest_mask (numpy.array): used to determine where this model is
                valid so we can mask out the rest as nodata
            """
        nodata_mask = (forest_mask) | (edge_distance == 0)
        edge_distance_km = edge_distance * (cell_size_in_meters / 1000.0)
        #asymtotic model
        biomass_1 = theta_1 - theta_2 * numpy.exp(-theta_3 * edge_distance_km)
        #logarithmic model
        biomass_2 = theta_1 + theta_2 * numpy.log(edge_distance_km)
        #linear regression
        biomass_3 = theta_1 + theta_2 * edge_distance_km

        result = numpy.where(method == 1, biomass_1, carbon_edge_nodata)
        result = numpy.where(method == 2, biomass_2, result)
        result = numpy.where(method == 3, biomass_3, result)

        return numpy.where(nodata_mask, carbon_edge_nodata, result)

    edge_carbon_map_uri = os.path.join(
        args['workspace_dir'], 'edge_carbon_map%s.tif' % file_suffix)
    pygeoprocessing.vectorize_datasets(
        [edge_distance_uri, model_raster_uris['method'],
         model_raster_uris['theta1'], model_raster_uris['theta2'],
         model_raster_uris['theta3'], forest_mask_uri], carbon_edge_op,
        edge_carbon_map_uri, gdal.GDT_Float32, carbon_edge_nodata,
        cell_size_in_meters, 'intersection', vectorize_op=False,
        datasets_are_pre_aligned=True)


    #TASK: combine maps into output
    carbon_map_uri = os.path.join(
        args['workspace_dir'], 'carbon_map%s.tif' % file_suffix)


    #TASK: generate report (optional) by serviceshed if they exist
