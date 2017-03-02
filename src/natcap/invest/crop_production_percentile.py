"""InVEST Crop Production Percentile Model."""
import os
import logging

import numpy
from osgeo import gdal
import pygeoprocessing

from . import utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.crop_production')

_OUTPUT_BASE_FILES = {
    }

_INTERMEDIATE_BASE_FILES = {
    }

_TMP_BASE_FILES = {
    }

_NODATA_CLIMATE_BIN = 255


def execute(args):
    """Crop Production Percentile Model.

    This model will take a landcover (crop cover?) map and produce yields,
    production, and observed crop yields, a nutrient table, and a clipped
    observed map.

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['landcover_raster_path'] (string): path to landcover raster
        args['landcover_to_crop_table_path'] (string): path to a table that
            converts landcover types to crop names that has two headers:
            * lucode: integer value corresponding to a landcover code in
              `args['landcover_raster_path']`.
            * crop_name: a string that must match one of the crops in
              args['global_data_path']/climate_bin_maps/[cropname]_*
              A ValueError is raised if strings don't match.
        args['global_data_path'] (string): path to the InVEST Crop Production
            global data directory.  This model expects that the following
            directories are subdirectories of this path
            * climate_bin_maps (contains [cropname]_climate_bin.tif files)
            * climate_percentile_yield (contains
              [cropname]_percentile_yield_table.csv files)

    Returns:
        None.
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories(
        [output_dir, intermediate_output_dir])

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    LOGGER.debug("TODO: convert landcover map to lat/lng projection ")

    landcover_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])
    landcover_nodata = landcover_raster_info['nodata'][0]

    crop_to_landcover_table = utils.build_lookup_from_csv(
        args['landcover_to_crop_table_path'], 'crop_name', to_lower=True,
        numerical_cast=True)

    crop_lucode = None
    for crop_name in crop_to_landcover_table:
        crop_lucode = crop_to_landcover_table[crop_name]['lucode']
        print crop_name, crop_lucode
        crop_climate_bin_raster_path = os.path.join(
            args['global_data_path'], 'climate_bin_maps',
            '%s_climate_bin_map.tif' % crop_name)
        climate_percentile_yield_table_path = os.path.join(
            args['global_data_path'], 'climate_percentile_yield',
            '%s_percentile_yield_table.csv' % crop_name)
        if not os.path.exists(crop_climate_bin_raster_path):
            raise ValueError(
                "Expected climate bin map called %s for crop %s "
                "specified in %s", crop_climate_bin_raster_path, crop_name,
                args['landcover_to_crop_table_path'])
        if not os.path.exists(crop_climate_bin_raster_path):
            raise ValueError(
                "Expected climate bin map called %s for crop %s "
                "specified in %s", crop_climate_bin_raster_path, crop_name,
                args['landcover_to_crop_table_path'])

        local_climate_bin_raster_path = os.path.join(
            intermediate_output_dir,
            'local_%s_climate_bin_map%s.tif' % (crop_name, file_suffix))
        pygeoprocessing.warp_raster(
            crop_climate_bin_raster_path,
            landcover_raster_info['pixel_size'],
            local_climate_bin_raster_path, 'mode',
            target_sr_wkt=landcover_raster_info['projection'],
            target_bb=landcover_raster_info['bounding_box'])

        LOGGER.info("Mask out crop %s from landcover map", crop_name)

        masked_crop_raster_path = os.path.join(
            intermediate_output_dir, 'masked_climate_bin_map_%s%s.tif' % (
                crop_name, file_suffix))
        def _mask_climate_bin(lulc_array, climate_bin_array):
            """Mask in climate bins that intersect with `crop_lucode`."""
            result = numpy.empty(lulc_array.shape, dtype=numpy.int8)
            result[:] = _NODATA_CLIMATE_BIN
            valid_mask = lulc_array != landcover_nodata
            result[valid_mask] = 0
            lulc_mask = (lulc_array[valid_mask] == crop_lucode) & valid_mask
            result[lulc_mask] = climate_bin_array[lulc_mask]
            return result

        pygeoprocessing.raster_calculator(
            [(args['landcover_raster_path'], 1)],
            _mask_climate_bin, masked_crop_raster_path, gdal.GDT_Byte, 255)

        crop_climate_percentile_table = utils.build_lookup_from_csv(
            args['global_data_path'], 'climate_bin', to_lower=True,
            numerical_cast=True)

        yield_percentile_headers = [
            x for x in crop_climate_percentile_table.itervalues().first()
            if x is not 'climate_bin']

        LOGGER.debug(yield_percentile_headers)
        for yield_percentile in yield_percentile_headers:
            yield_percentile_raster_path = os.path.join(
                output_dir, '%s_%s%s.tif' % (
                    crop_name, yield_percentile, file_suffix))

            reclassify_raster(
                base_raster_path_band, value_map, target_raster_path, target_datatype,
                target_nodata, exception_flag='values_required')
