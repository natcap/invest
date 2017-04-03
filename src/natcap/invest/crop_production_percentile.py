"""InVEST Crop Production Percentile Model."""
import collections
import re
import os
import sys
import logging

import numpy
from osgeo import gdal
from osgeo import osr
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

_YIELD_PERCENTILE_FIELD_PATTERN = 'yield_([^_]+)'

_EXPECTED_NUTRIENT_TABLE_HEADERS = [
    'Protein', 'Lipid', 'Energy', 'Ca', 'Fe', 'Mg', 'Ph', 'K', 'Na', 'Zn',
    'Cu', 'Fl', 'Mn', 'Se', 'VitA', 'betaC', 'alphaC', 'VitE', 'Crypto',
    'Lycopene', 'Lutein', 'betaT', 'gammaT', 'deltaT', 'VitC', 'Thiamin',
    'Riboflavin', 'Niacin', 'Pantothenic', 'VitB6', 'Folate', 'VitB12',
    'VitK']
_EXPECTED_LUCODE_TABLE_HEADER = 'lucode'
_NODATA_CLIMATE_BIN = 255
_NODATA_YIELD = -1.0


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
              args['model_data_path']/climate_bin_maps/[cropname]_*
              A ValueError is raised if strings don't match.
        args['model_data_path'] (string): path to the InVEST Crop Production
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

    landcover_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])
    pixel_area_ha = numpy.product([
        abs(x) for x in landcover_raster_info['pixel_size']]) / 10000.0
    landcover_nodata = landcover_raster_info['nodata'][0]

    crop_to_landcover_table = utils.build_lookup_from_csv(
        args['landcover_to_crop_table_path'], 'crop_name', to_lower=True,
        numerical_cast=True)

    crop_lucodes = [
        x[_EXPECTED_LUCODE_TABLE_HEADER]
        for x in crop_to_landcover_table.itervalues()]

    LOGGER.info(
        "Calculating total land area and warning if the landcover raster "
        "is missing lucodes")
    unique_lucodes = numpy.array([])
    total_area = 0.0
    for _, lu_band_data in pygeoprocessing.iterblocks(
            args['landcover_raster_path']):
        unique_block = numpy.unique(lu_band_data)
        unique_lucodes = numpy.unique(numpy.concatenate(
            (unique_lucodes, unique_block)))
        total_area += numpy.count_nonzero((lu_band_data != _NODATA_YIELD))

    missing_lucodes = set(crop_lucodes).difference(
        set(unique_lucodes))
    if len(missing_lucodes) > 0:
        LOGGER.warn(
            "The following lucodes are in the landcover to crop table but "
            "aren't in the landcover raster: %s" % missing_lucodes)

    # Calculate lat/lng bounding box for landcover map
    wgs84srs = osr.SpatialReference()
    wgs84srs.ImportFromEPSG(4326)  # EPSG4326 is WGS84 lat/lng
    landcover_wgs84_bounding_box = pygeoprocessing.transform_bounding_box(
        landcover_raster_info['bounding_box'],
        landcover_raster_info['projection'], wgs84srs.ExportToWkt(),
        edge_samples=11)

    crop_lucode = None
    observed_yield_nodata = None
    crop_area = collections.defaultdict(float)
    for crop_name in crop_to_landcover_table:
        crop_lucode = crop_to_landcover_table[crop_name][
            _EXPECTED_LUCODE_TABLE_HEADER]
        print crop_name, crop_lucode
        crop_climate_bin_raster_path = os.path.join(
            args['model_data_path'], 'extended_climate_bin_maps',
            'extendedclimatebins%s.tif' % crop_name)
        climate_percentile_yield_table_path = os.path.join(
            args['model_data_path'], 'climate_percentile_yield',
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

        clipped_climate_bin_raster_path = os.path.join(
            intermediate_output_dir,
            'clipped_%s_climate_bin_map%s.tif' % (crop_name, file_suffix))

        #TODO: at this point could we query for the lat/lng points that
        #      intersect crop/climate bins?  Then for each percentile header
        #      we can calculate a per-pixel yield for the given landcover's
        #      pixel size?  After that we can
        #      pygeoprocessing.interpolate_points onto the raster?
        crop_climate_bin_raster_info = pygeoprocessing.get_raster_info(
            crop_climate_bin_raster_path)
        pygeoprocessing.warp_raster(
            crop_climate_bin_raster_path,
            crop_climate_bin_raster_info['pixel_size'],
            clipped_climate_bin_raster_path, 'nearest',
            target_bb=landcover_wgs84_bounding_box)

        climate_percentile_table_path = os.path.join(
            args['model_data_path'], 'climate_percentile_yield_tables',
            '%s_percentile_yield_table.csv' % crop_name)
        crop_climate_percentile_table = utils.build_lookup_from_csv(
            climate_percentile_table_path, 'climate_bin', to_lower=True,
            numerical_cast=True)

        yield_percentile_headers = [
            x for x in crop_climate_percentile_table.itervalues().next()
            if x != 'climate_bin']

        clipped_climate_bin_raster_path_info = (
            pygeoprocessing.get_raster_info(
                clipped_climate_bin_raster_path))

        for yield_percentile_id in yield_percentile_headers:
            yield_percentile_raster_path = os.path.join(
                intermediate_output_dir, '%s_%s%s.tif' % (
                    crop_name, yield_percentile_id, file_suffix))

            bin_to_percentile_yield = dict([
                (bin_id,
                 crop_climate_percentile_table[bin_id][yield_percentile_id])
                for bin_id in crop_climate_percentile_table])
            bin_to_percentile_yield[
                clipped_climate_bin_raster_path_info['nodata'][0]] = 0.0

            coarse_yield_percentile_raster_path = os.path.join(
                intermediate_output_dir, 'coarse_%s_%s%s.tif' % (
                    crop_name, yield_percentile_id, file_suffix))
            pygeoprocessing.reclassify_raster(
                (clipped_climate_bin_raster_path, 1), bin_to_percentile_yield,
                coarse_yield_percentile_raster_path, gdal.GDT_Float32,
                _NODATA_YIELD, exception_flag='values_required')

            pygeoprocessing.warp_raster(
                coarse_yield_percentile_raster_path,
                landcover_raster_info['pixel_size'],
                yield_percentile_raster_path, 'cubic_spline',
                target_sr_wkt=landcover_raster_info['projection'],
                target_bb=landcover_raster_info['bounding_box'])

            LOGGER.info(
                "Calculate yield for %s at %s", crop_name,
                yield_percentile_id)

            percentile_crop_yield_raster_path = os.path.join(
                intermediate_output_dir, '%s_%s_yield_rate_map%s.tif' % (
                    yield_percentile_id, crop_name, file_suffix))

            def _crop_yield_op(lulc_array, yield_rate_array):
                """Mask in climate bins that intersect with `crop_lucode`."""
                result = numpy.empty(lulc_array.shape, dtype=numpy.float32)
                result[:] = _NODATA_YIELD
                valid_mask = lulc_array != landcover_nodata
                lulc_mask = lulc_array == crop_lucode
                result[valid_mask] = 0
                result[lulc_mask] = (
                    yield_rate_array[lulc_mask] * pixel_area_ha)
                return result

            pygeoprocessing.raster_calculator(
                [(args['landcover_raster_path'], 1),
                 (yield_percentile_raster_path, 1)],
                _crop_yield_op, percentile_crop_yield_raster_path,
                gdal.GDT_Float32, _NODATA_YIELD)

        # calculate the non-zero production area for that crop, okay to use
        # just one of the percentile rasters
        for _, band_values in pygeoprocessing.iterblocks(
                yield_percentile_raster_path):
            crop_area[crop_name] += numpy.count_nonzero(
                (band_values != _NODATA_YIELD) & (band_values > 0.0))

        crop_area[crop_name] *= pixel_area_ha

        LOGGER.info("Calculate observed yield for %s", crop_name)
        global_observed_yield_rate_raster_path = os.path.join(
            args['model_data_path'], 'observed_yield',
            '%s_yield_map.tif' % crop_name)
        observed_yield_rate_raster_path = os.path.join(
            intermediate_output_dir, '%s_observed_yield_rate%s.tif' % (
                crop_name, file_suffix))
        pygeoprocessing.warp_raster(
            global_observed_yield_rate_raster_path,
            landcover_raster_info['pixel_size'],
            observed_yield_rate_raster_path, 'cubic_spline',
            target_sr_wkt=landcover_raster_info['projection'],
            target_bb=landcover_raster_info['bounding_box'])

        observed_yield_rate_nodata = pygeoprocessing.get_raster_info(
            observed_yield_rate_raster_path)['nodata'][0]

        def _observed_yield_op(lulc_array, observed_yield_rate_array):
            """Mask in climate bins that intersect with `crop_lucode`."""
            result = numpy.empty(lulc_array.shape, dtype=numpy.float32)
            result[:] = observed_yield_rate_nodata
            valid_mask = lulc_array != landcover_nodata
            lulc_mask = lulc_array == crop_lucode
            result[valid_mask] = 0
            result[lulc_mask] = observed_yield_rate_array[lulc_mask]
            return result

        observed_yield_raster_path = os.path.join(
            intermediate_output_dir,
            '%s_observed_yield%s.tif' % (crop_name, file_suffix))

        pygeoprocessing.raster_calculator(
            [(args['landcover_raster_path'], 1),
             (observed_yield_rate_raster_path, 1)],
            _observed_yield_op, observed_yield_raster_path,
            gdal.GDT_Float32, observed_yield_rate_nodata)

    nutrient_table = utils.build_lookup_from_csv(
        os.path.join(args['model_data_path'], 'cropNutrient.csv'),
        'filenm', to_lower=False)

    LOGGER.info("Report table")
    result_table_path = os.path.join(
        output_dir, 'result_table%s.csv' % file_suffix)
    production_percentile_headers = [
        'production_' + re.match(
            _YIELD_PERCENTILE_FIELD_PATTERN,
            yield_percentile_id).group(1) for yield_percentile_id in sorted(
                yield_percentile_headers)]
    nutrient_headers = [
        nutrient_id + '_' + re.match(
            _YIELD_PERCENTILE_FIELD_PATTERN,
            yield_percentile_id).group(1)
        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
        for yield_percentile_id in sorted(yield_percentile_headers) + [
            'yield_observed']]
    total_nutrient_table = collections.defaultdict(
        lambda: collections.defaultdict(float))
    with open(result_table_path, 'wb') as result_table:
        result_table.write(
            'crop,area (ha),' + ','.join(production_percentile_headers) +
            ',production_observed' + ',' + ','.join(nutrient_headers) + '\n')
        for crop_name in sorted(crop_to_landcover_table):
            result_table.write(crop_name)
            result_table.write(',%f' % crop_area[crop_name])
            production_lookup = {}
            nutrient_factor = 1e4 * (
                1.0 - nutrient_table[crop_name]['Percentrefuse'] / 100.0)
            for yield_percentile_id in sorted(yield_percentile_headers):
                yield_percentile_raster_path = os.path.join(
                    intermediate_output_dir, '%s_%s%s.tif' % (
                        crop_name, yield_percentile_id, file_suffix))
                yield_sum = 0.0
                for _, yield_block in pygeoprocessing.iterblocks(
                        yield_percentile_raster_path):
                    yield_sum += numpy.sum(
                        yield_block[_NODATA_YIELD != yield_block])
                production = yield_sum * pixel_area_ha
                production_lookup[yield_percentile_id] = production
                result_table.write(",%f" % production)
            yield_sum = 0.0
            observed_yield_raster_path = os.path.join(
                intermediate_output_dir,
                '%s_observed_yield%s.tif' % (
                    crop_name, file_suffix))
            observed_yield_nodata = pygeoprocessing.get_raster_info(
                observed_yield_raster_path)['nodata'][0]
            for _, yield_block in pygeoprocessing.iterblocks(
                    observed_yield_raster_path):
                yield_sum += numpy.sum(
                    yield_block[observed_yield_nodata != yield_block])
            production = yield_sum * pixel_area_ha
            production_lookup['observed'] = production
            result_table.write(",%f" % production)
            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                for yield_percentile_id in sorted(yield_percentile_headers):
                    total_nutrient_table[nutrient_id][yield_percentile_id] += (
                        nutrient_factor *
                        production_lookup[yield_percentile_id] *
                        nutrient_table[crop_name][nutrient_id])
                    result_table.write(",%f" % (
                        total_nutrient_table[nutrient_id][yield_percentile_id]
                        ))
                result_table.write(
                    ",%f" % (
                        production_lookup['observed'] *
                        nutrient_table[crop_name][nutrient_id]))
            result_table.write('\n')

        total_area = 0.0
        for _, band_values in pygeoprocessing.iterblocks(
                args['landcover_raster_path']):
            total_area += numpy.count_nonzero((band_values != _NODATA_YIELD))
        result_table.write(
            '\n,total area (both crop and non-crop)\n,%f\n' % (
                total_area * pixel_area_ha))
