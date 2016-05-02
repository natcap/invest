"""Habitat suitability model."""
import os
import logging
import csv

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy
import scipy
import pygeoprocessing.geoprocessing

from . import utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.habitat_suitability')

_OUTPUT_BASE_FILES = {
    'suitability_path': 'hsi.tif',
    'threshold_suitability_path': 'hsi_threshold.tif',
    'screened_suitability_path': 'hsi_screened.tif',
    }

_INTERMEDIATE_BASE_FILES = {
    }

_TMP_BASE_FILES = {
    }


def execute(args):
    """
    Calculate habitat suitability indexes given biophysical parameters.

    The objective of a habitat suitability index (HSI) is to help users
    identify areas within their AOI that would be most suitable for habitat
    restoration.  The output is a gridded map of the user's AOI in which each
    grid cell is assigned a suitability rank between 0 (not suitable) and 1
    (most suitable).  The suitability rank is generally calculated as the
    weighted geometric mean of several individual input criteria, which have
    also been ranked by suitability from 0-1.  Habitat types (e.g. marsh,
    mangrove, coral, etc.) are treated separately, and each habitat type will
    have a unique set of relevant input criteria and a resultant habitat
    suitability map.

    Parameters:
        args['workspace_dir'] (string): directory path to workspace directory
            for output files.
        args['results_suffix'] (string): (optional) string to append to any
            output file names.
        args['aoi_path'] (string): file path to an area of interest shapefile.
        args['exclusion_path_list'] (list): (optional) a list of file paths to
            shapefiles which define areas which the HSI should be masked out
            in a final output.
        args['output_cell_size'] (float): (optional) size of output cells.
            If not present, the output size will snap to the smallest cell
            size in the HSI range rasters.
        args['habitat_threshold'] (float): a value to threshold the habitat
            score values to 0 and 1.
        args['hsi_ranges'] (dict): a dictionary that describes the habitat
            biophysical base rasters as well as the ranges for optimal and
            tolerable values.  Each biophysical value has a unique key in the
            dictionary that is used to name the mapping of biophysical to
            local HSI value.  Each value is dictionary with keys:
                'raster_path': path to disk for biophysical raster.
                'range': a 4-tuple in non-decreasing order describing
                    the "tolerable" to "optimal" ranges for those biophysical
                    values.  The endpoints non-inclusively define where the
                    suitability score is 0.0, the two midpoints inclusively
                    define the range where the suitability is 1.0, and the
                    ranges above and below are linearly interpolated between
                    0.0 and 1.0.
                Example:
                    {
                        'depth':
                            {
                                'raster_path': r'C:/path/to/depth.tif',
                                'range': (-50, -30, -10, -10),
                            },
                        'temperature':
                            {
                                'temperature_path': (
                                    r'C:/path/to/temperature.tif'),
                                'range': (5, 7, 12.5, 16),
                            }
                    }
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    pygeoprocessing.create_directories(
        [output_dir, intermediate_output_dir])

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    # determine the minimum cell size
    if 'output_cell_size' in args:
        output_cell_size = args['output_cell_size']
    else:
        # cell size is the min cell size of all the biophysical inputs
        output_cell_size = min(
            [pygeoprocessing.get_cell_size_from_uri(entry['raster_path'])
             for entry in args['hsi_ranges'].itervalues()])

    # align the raster stack
    algined_raster_stack = {}
    out_aligned_raster_list = []
    base_raster_list = []
    for key, entry in args['hsi_ranges'].iteritems():
        aligned_path = os.path.join(intermediate_output_dir, key + '.tif')
        algined_raster_stack[key] = aligned_path
        out_aligned_raster_list.append(aligned_path)
        base_raster_list.append(entry['raster_path'])

    LOGGER.info("Aligning base biophysical raster list.")
    pygeoprocessing.geoprocessing.align_dataset_list(
        base_raster_list, out_aligned_raster_list,
        ['nearest'] * len(base_raster_list),
        output_cell_size, 'intersection', 0, aoi_uri=args['aoi_path'])

    # map biophysical to individual habitat suitability index
    base_nodata = None
    reclass_nodata = -1.0
    suitability_range = None
    suitability_raster_list = []
    for key, entry in args['hsi_ranges'].iteritems():
        LOGGER.info("Mapping biophysical values to HSI on %s", key)
        base_raster_path = algined_raster_stack[key]
        base_nodata = pygeoprocessing.get_nodata_from_uri(base_raster_path)
        suitability_range = entry['suitability_range']
        suitability_raster = os.path.join(
            intermediate_output_dir, key+'_suitability.tif')
        suitability_raster_list.append(suitability_raster)

        def local_map(biophysical_values):
            """Map biophysical values to suitability index values."""
            # the following condition and function lists capture the following
            # ranges in order:
            #   1) range[0] to range[1] noninclusive (linear interp 0-1)
            #   2) range[1] to range[2] inclusive (exactly 1.0)
            #   3) range[2] to range[3] noninclusive (linaer interp 1-0)
            #   4) nodata -> nodata
            #   5) 0.0 everywhere else

            condlist = [
                (suitability_range[0] < biophysical_values) &
                (biophysical_values < suitability_range[1]),
                (suitability_range[1] <= biophysical_values) &
                (biophysical_values <= suitability_range[2]),
                (suitability_range[2] < biophysical_values) &
                (biophysical_values < suitability_range[3]),
                biophysical_values == base_nodata]
            funclist = [
                lambda x: (
                    (x-suitability_range[0]) /
                    (suitability_range[1]-suitability_range[0])),
                1.0,
                lambda x: 1.0 - (
                    (x-suitability_range[2]) /
                    (suitability_range[3]-suitability_range[2])),
                reclass_nodata,
                0.0]
            return numpy.piecewise(biophysical_values, condlist, funclist)

        pygeoprocessing.vectorize_datasets(
            [base_raster_path], local_map, suitability_raster,
            gdal.GDT_Float32, reclass_nodata, output_cell_size, 'intersection',
            vectorize_op=False)

    # calculate geometric mean
    LOGGER.info("Calculate geometric mean of HSIs.")
    def geo_mean(*suitability_values):
        """Geometric mean of input suitability_values."""
        running_product = suitability_values[0].astype(numpy.float32)
        running_mask = suitability_values[0] == reclass_nodata
        for index in range(1, len(suitability_values)):
            running_product *= suitability_values[index]
            running_mask = running_mask | (
                suitability_values[index] == reclass_nodata)
        result = numpy.empty(running_mask.shape, dtype=numpy.float32)
        result[:] = reclass_nodata
        result[~running_mask] = (
            running_product[~running_mask]**(1./len(suitability_values)))
        return result

    pygeoprocessing.geoprocessing.vectorize_datasets(
        suitability_raster_list, geo_mean, f_reg['suitability_path'],
        gdal.GDT_Float32, reclass_nodata, output_cell_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    return

##################
    # align the raster lists
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
