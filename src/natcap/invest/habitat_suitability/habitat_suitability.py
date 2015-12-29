"""Habitat suitability model"""

import os
import logging
import csv

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy
import scipy

import pygeoprocessing.geoprocessing

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.habitat_suitability.habitat_suitability')


def execute(args):
    """
    Calculates habitat suitability scores and patches given biophysical
    rasters and classification curves.

    Args:
        workspace_dir (string): uri to workspace directory for output files
        habitat_threshold (float): (optional) size of output cells
        output_cell_size (float): a value to threshold the habitat score values
            to 0 and 1
        temperature_biophysical_uri (string): uri to temperature raster
        salinity_biophysical_uri (string): uri to salinity raster
        depth_biophysical_uri (string): uri to a depth raster
        oyster_habitat_suitability_temperature_table_uri (string): uri to a csv
            table that has that has columns "Suitability" in (0,1) and
            "Temperature" in range(temperature_biophysical_uri)
        oyster_habitat_suitability_salinity_table_uri (string): uri to a csv
            table that has that has columns "Suitability" in (0,1) and
            "Salinity" in range(salinity_biophysical_uri)
        oyster_habitat_suitability_depth_table_uri (string): uri to a csv
            table that has that has columns "Suitability" in (0,1) and "Depth"
            in range(depth_biophysical_uri)

    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'habitat_threshold': 'example',
            'output_cell_size': 'example',
            'temperature_biophysical_uri': 'path/to/raster',
            'salinity_biophysical_uri': 'path/to/raster',
            'depth_biophysical_uri': 'path/to/raster',
            'oyster_habitat_suitability_temperature_table_uri': 'path/to/csv',
            'oyster_habitat_suitability_salinity_table_uri': 'path/to/csv',
            'oyster_habitat_suitability_depth_table_uri': 'path/to/csv',
        }

    """
    try:
        file_suffix = args['suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    output_dir = os.path.join(args['workspace_dir'], 'output')

    #Sets up the intermediate and output directory structure for the workspace
    for directory in [output_dir, intermediate_dir]:
        if not os.path.exists(directory):
            LOGGER.info('creating directory %s', directory)
            os.makedirs(directory)

    #align the raster lists
    aligned_raster_stack = {
        'salinity_biophysical_uri': os.path.join(
            intermediate_dir, 'aligned_salinity.tif'),
        'temperature_biophysical_uri': os.path.join(
            intermediate_dir, 'aligned_temperature.tif'),
        'depth_biophysical_uri':  os.path.join(
            intermediate_dir, 'algined_depth.tif')
    }
    biophysical_keys = [
        'salinity_biophysical_uri', 'temperature_biophysical_uri',
        'depth_biophysical_uri']
    dataset_uri_list = [args[x] for x in biophysical_keys]
    dataset_out_uri_list = [aligned_raster_stack[x] for x in biophysical_keys]

    out_pixel_size = min(
        [pygeoprocessing.geoprocessing.get_cell_size_from_uri(x) for x in dataset_uri_list])

    pygeoprocessing.geoprocessing.align_dataset_list(
        dataset_uri_list, dataset_out_uri_list,
        ['nearest'] * len(dataset_out_uri_list),
        out_pixel_size, 'intersection', 0)


    #build up the interpolation functions for the habitat
    biophysical_to_table = {
        'salinity_biophysical_uri':
            ('oyster_habitat_suitability_salinity_table_uri', 'salinity'),
        'temperature_biophysical_uri':
            ('oyster_habitat_suitability_temperature_table_uri', 'temperature'),
        'depth_biophysical_uri':
            ('oyster_habitat_suitability_depth_table_uri', 'depth'),
        }
    biophysical_to_interp = {}
    for biophysical_uri_key, (habitat_suitability_table_uri, key) in \
            biophysical_to_table.iteritems():
        csv_dict_reader = csv.DictReader(
            open(args[habitat_suitability_table_uri], 'rU'))
        suitability_list = []
        value_list = []
        for row in csv_dict_reader:
            #convert keys to lowercase
            row = {k.lower().rstrip():v for k, v in row.items()}
            suitability_list.append(float(row['suitability']))
            value_list.append(float(row[key]))
        biophysical_to_interp[biophysical_uri_key] = scipy.interpolate.interp1d(
            value_list, suitability_list, kind='linear',
            bounds_error=False, fill_value=0.0)
    biophysical_to_habitat_quality = {
        'salinity_biophysical_uri': os.path.join(
            intermediate_dir, 'oyster_salinity_suitability.tif'),
        'temperature_biophysical_uri': os.path.join(
            intermediate_dir, 'oyster_temperature_suitability.tif'),
        'depth_biophysical_uri':  os.path.join(
            intermediate_dir, 'oyster_depth_suitability.tif'),
    }
    #classify the biophysical maps into habitat quality maps
    reclass_nodata = -1.0
    for biophysical_uri_key, interpolator in biophysical_to_interp.iteritems():
        biophysical_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
            aligned_raster_stack[biophysical_uri_key])
        LOGGER.debug(aligned_raster_stack[biophysical_uri_key])
        def reclass_op(values):
            """reclasses a value into an interpolated value"""
            nodata_mask = values == biophysical_nodata
            return numpy.where(
                nodata_mask, reclass_nodata,
                interpolator(values))
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [aligned_raster_stack[biophysical_uri_key]], reclass_op,
            biophysical_to_habitat_quality[biophysical_uri_key],
            gdal.GDT_Float32, reclass_nodata, out_pixel_size, "intersection",
            dataset_to_align_index=0, vectorize_op=False)

    #calculate the geometric mean of the suitability rasters
    oyster_suitability_uri = os.path.join(
        output_dir, 'oyster_habitat_suitability.tif')

    def geo_mean(*values):
        """Geometric mean of input values"""
        running_product = values[0]
        running_mask = values[0] == reclass_nodata
        for index in range(1, len(values)):
            running_product *= values[index]
            running_mask = running_mask | (values[index] == reclass_nodata)
        return numpy.where(
            running_mask, reclass_nodata, running_product**(1./len(values)))

    pygeoprocessing.geoprocessing.vectorize_datasets(
        biophysical_to_habitat_quality.values(), geo_mean,
        oyster_suitability_uri, gdal.GDT_Float32, reclass_nodata,
        out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

     #calculate the geometric mean of the suitability rasters
    oyster_suitability_mask_uri = os.path.join(
        output_dir, 'oyster_habitat_suitability_mask.tif')

    def threshold(value):
        """Threshold the values to args['habitat_threshold']"""

        threshold_value = value >= args['habitat_threshold']
        return numpy.where(
            value == reclass_nodata, reclass_nodata, threshold_value)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [oyster_suitability_uri], threshold,
        oyster_suitability_mask_uri, gdal.GDT_Float32, reclass_nodata,
        out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    #polygonalize output mask
    output_mask_ds = gdal.Open(oyster_suitability_mask_uri)
    output_mask_band = output_mask_ds.GetRasterBand(1)
    output_mask_wkt = output_mask_ds.GetProjection()

    output_sr = osr.SpatialReference()
    output_sr.ImportFromWkt(output_mask_wkt)


    oyster_suitability_datasource_uri = os.path.join(
        output_dir, 'oyster_habitat_suitability_mask.shp')

    if os.path.isfile(oyster_suitability_datasource_uri):
        os.remove(oyster_suitability_datasource_uri)


    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    oyster_suitability_datasource = output_driver.CreateDataSource(
        oyster_suitability_datasource_uri)
    oyster_suitability_layer = oyster_suitability_datasource.CreateLayer(
            'oyster', output_sr, ogr.wkbPolygon)

    field = ogr.FieldDefn('pixel_value', ogr.OFTReal)
    oyster_suitability_layer.CreateField(field)

    gdal.Polygonize(
        output_mask_band, output_mask_band, oyster_suitability_layer, 0)
