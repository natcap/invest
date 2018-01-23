"""InVEST Crop Production Percentile Model."""
import collections
import logging
import os
import re

import numpy
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pygeoprocessing

from . import utils
from . import validation


LOGGER = logging.getLogger('natcap.invest.crop_production_percentile')

_INTERMEDIATE_OUTPUT_DIR = 'intermediate_output'

_YIELD_PERCENTILE_FIELD_PATTERN = 'yield_([^_]+)'
_GLOBAL_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    'observed_yield', '%s_yield_map.tif')  # crop_name
_EXTENDED_CLIMATE_BIN_FILE_PATTERN = os.path.join(
    'extended_climate_bin_maps', 'extendedclimatebins%s.tif')  # crop_name
_CLIMATE_PERCENTILE_TABLE_PATTERN = os.path.join(
    'climate_percentile_yield_tables',
    '%s_percentile_yield_table.csv')  # crop_name

# crop_name, yield_percentile_id
_INTERPOLATED_YIELD_PERCENTILE_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_%s_interpolated_yield%s.tif')

# crop_name, file_suffix
_CLIPPED_CLIMATE_BIN_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR,
    'clipped_%s_climate_bin_map%s.tif')

# crop_name, yield_percentile_id, file_suffix
_COARSE_YIELD_PERCENTILE_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_%s_coarse_yield%s.tif')

# crop_name, yield_percentile_id, file_suffix
_PERCENTILE_CROP_PRODUCTION_FILE_PATTERN = os.path.join(
    '.', '%s_%s_production%s.tif')

# crop_name, file_suffix
_CLIPPED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_clipped_observed_yield%s.tif')

# crop_name, file_suffix
_ZEROED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_zeroed_observed_yield%s.tif')

# crop_name, file_suffix
_INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_interpolated_observed_yield%s.tif')

# crop_name, file_suffix
_OBSERVED_PRODUCTION_FILE_PATTERN = os.path.join(
    '.', '%s_observed_production%s.tif')

# file_suffix
_AGGREGATE_VECTOR_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, 'aggrgate_vector%s.shp')

# file_suffix
_AGGREGATE_TABLE_FILE_PATTERN = os.path.join(
    '.', 'aggregate_results%s.csv')

_EXPECTED_NUTRIENT_TABLE_HEADERS = [
    'Protein', 'Lipid', 'Energy', 'Ca', 'Fe', 'Mg', 'Ph', 'K', 'Na', 'Zn',
    'Cu', 'Fl', 'Mn', 'Se', 'VitA', 'betaC', 'alphaC', 'VitE', 'Crypto',
    'Lycopene', 'Lutein', 'betaT', 'gammaT', 'deltaT', 'VitC', 'Thiamin',
    'Riboflavin', 'Niacin', 'Pantothenic', 'VitB6', 'Folate', 'VitB12',
    'VitK']
_EXPECTED_LUCODE_TABLE_HEADER = 'lucode'
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
        args['aggregate_polygon_path'] (string): path to polygon shapefile
            that will be used to aggregate crop yields and total nutrient
            value. (optional, if value is None, then skipped)
        args['aggregate_polygon_id'] (string): This is the id field in
            args['aggregate_polygon_path'] to be used to index the final
            aggregate results.  If args['aggregate_polygon_path'] is not
            provided, this value is ignored.
        args['model_data_path'] (string): path to the InVEST Crop Production
            global data directory.  This model expects that the following
            directories are subdirectories of this path
            * climate_bin_maps (contains [cropname]_climate_bin.tif files)
            * climate_percentile_yield (contains
              [cropname]_percentile_yield_table.csv files)
            Please see the InVEST user's guide chapter on crop production for
            details about how to download these data.

    Returns:
        None.
    """
    crop_to_landcover_table = utils.build_lookup_from_csv(
        args['landcover_to_crop_table_path'], 'crop_name', to_lower=True,
        numerical_cast=True)
    bad_crop_name_list = []
    for crop_name in crop_to_landcover_table:
        crop_climate_bin_raster_path = os.path.join(
            args['model_data_path'],
            _EXTENDED_CLIMATE_BIN_FILE_PATTERN % crop_name)
        if not os.path.exists(crop_climate_bin_raster_path):
            bad_crop_name_list.append(crop_name)
    if len(bad_crop_name_list) > 0:
        raise ValueError(
            "The following crop names were provided in %s but no such crops "
            "exist for this model: %s" % (
                args['landcover_to_crop_table_path'], bad_crop_name_list))

    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories([
        output_dir, os.path.join(output_dir, _INTERMEDIATE_OUTPUT_DIR)])

    landcover_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])
    pixel_area_ha = numpy.product([
        abs(x) for x in landcover_raster_info['pixel_size']]) / 10000.0
    landcover_nodata = landcover_raster_info['nodata'][0]

    # Calculate lat/lng bounding box for landcover map
    wgs84srs = osr.SpatialReference()
    wgs84srs.ImportFromEPSG(4326)  # EPSG4326 is WGS84 lat/lng
    landcover_wgs84_bounding_box = pygeoprocessing.transform_bounding_box(
        landcover_raster_info['bounding_box'],
        landcover_raster_info['projection'], wgs84srs.ExportToWkt(),
        edge_samples=11)

    crop_lucode = None
    observed_yield_nodata = None
    production_area = collections.defaultdict(float)
    for crop_name in crop_to_landcover_table:
        crop_lucode = crop_to_landcover_table[crop_name][
            _EXPECTED_LUCODE_TABLE_HEADER]
        LOGGER.info("Processing crop %s", crop_name)
        crop_climate_bin_raster_path = os.path.join(
            args['model_data_path'],
            _EXTENDED_CLIMATE_BIN_FILE_PATTERN % crop_name)

        LOGGER.info(
            "Clipping global climate bin raster to landcover bounding box.")
        clipped_climate_bin_raster_path = os.path.join(
            output_dir, _CLIPPED_CLIMATE_BIN_FILE_PATTERN % (
                crop_name, file_suffix))
        crop_climate_bin_raster_info = pygeoprocessing.get_raster_info(
            crop_climate_bin_raster_path)
        pygeoprocessing.warp_raster(
            crop_climate_bin_raster_path,
            crop_climate_bin_raster_info['pixel_size'],
            clipped_climate_bin_raster_path, 'nearest',
            target_bb=landcover_wgs84_bounding_box)

        climate_percentile_yield_table_path = os.path.join(
            args['model_data_path'],
            _CLIMATE_PERCENTILE_TABLE_PATTERN % crop_name)
        crop_climate_percentile_table = utils.build_lookup_from_csv(
            climate_percentile_yield_table_path, 'climate_bin',
            to_lower=True, numerical_cast=True)
        yield_percentile_headers = [
            x for x in crop_climate_percentile_table.itervalues().next()
            if x != 'climate_bin']

        for yield_percentile_id in yield_percentile_headers:
            LOGGER.info("Map %s to climate bins.", yield_percentile_id)
            interpolated_yield_percentile_raster_path = os.path.join(
                output_dir,
                _INTERPOLATED_YIELD_PERCENTILE_FILE_PATTERN % (
                    crop_name, yield_percentile_id, file_suffix))
            bin_to_percentile_yield = dict([
                (bin_id,
                 crop_climate_percentile_table[bin_id][yield_percentile_id])
                for bin_id in crop_climate_percentile_table])
            bin_to_percentile_yield[
                crop_climate_bin_raster_info['nodata'][0]] = 0.0
            coarse_yield_percentile_raster_path = os.path.join(
                output_dir,
                _COARSE_YIELD_PERCENTILE_FILE_PATTERN % (
                    crop_name, yield_percentile_id, file_suffix))
            pygeoprocessing.reclassify_raster(
                (clipped_climate_bin_raster_path, 1), bin_to_percentile_yield,
                coarse_yield_percentile_raster_path, gdal.GDT_Float32,
                _NODATA_YIELD)

            LOGGER.info(
                "Interpolate %s %s yield raster to landcover resolution.",
                crop_name, yield_percentile_id)
            pygeoprocessing.warp_raster(
                coarse_yield_percentile_raster_path,
                landcover_raster_info['pixel_size'],
                interpolated_yield_percentile_raster_path, 'cubic_spline',
                target_sr_wkt=landcover_raster_info['projection'],
                target_bb=landcover_raster_info['bounding_box'])

            LOGGER.info(
                "Calculate yield for %s at %s", crop_name,
                yield_percentile_id)
            percentile_crop_production_raster_path = os.path.join(
                output_dir,
                _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                    crop_name, yield_percentile_id, file_suffix))

            def _crop_production_op(lulc_array, yield_array):
                """Mask in yields that overlap with `crop_lucode`."""
                result = numpy.empty(lulc_array.shape, dtype=numpy.float32)
                result[:] = _NODATA_YIELD
                valid_mask = lulc_array != landcover_nodata
                lulc_mask = lulc_array == crop_lucode
                result[valid_mask] = 0
                result[lulc_mask] = (
                    yield_array[lulc_mask] * pixel_area_ha)
                return result

            pygeoprocessing.raster_calculator(
                [(args['landcover_raster_path'], 1),
                 (interpolated_yield_percentile_raster_path, 1)],
                _crop_production_op, percentile_crop_production_raster_path,
                gdal.GDT_Float32, _NODATA_YIELD)

        # calculate the non-zero production area for that crop, assuming that
        # all the percentile rasters have non-zero production so it's okay to
        # use just one of the percentile rasters
        LOGGER.info("Calculating production area.")
        for _, band_values in pygeoprocessing.iterblocks(
                percentile_crop_production_raster_path):
            production_area[crop_name] += numpy.count_nonzero(
                (band_values != _NODATA_YIELD) & (band_values > 0.0))
        production_area[crop_name] *= pixel_area_ha

        LOGGER.info("Calculate observed yield for %s", crop_name)
        global_observed_yield_raster_path = os.path.join(
            args['model_data_path'],
            _GLOBAL_OBSERVED_YIELD_FILE_PATTERN % crop_name)
        global_observed_yield_raster_info = (
            pygeoprocessing.get_raster_info(
                global_observed_yield_raster_path))

        clipped_observed_yield_raster_path = os.path.join(
            output_dir, _CLIPPED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))
        pygeoprocessing.warp_raster(
            global_observed_yield_raster_path,
            global_observed_yield_raster_info['pixel_size'],
            clipped_observed_yield_raster_path, 'nearest',
            target_bb=landcover_wgs84_bounding_box)

        observed_yield_nodata = (
            global_observed_yield_raster_info['nodata'][0])

        zeroed_observed_yield_raster_path = os.path.join(
            output_dir, _ZEROED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))

        def _zero_observed_yield_op(observed_yield_array):
            """Calculate observed 'actual' yield."""
            result = numpy.empty(
                observed_yield_array.shape, dtype=numpy.float32)
            result[:] = 0.0
            valid_mask = observed_yield_array != observed_yield_nodata
            result[valid_mask] = observed_yield_array[valid_mask]
            return result

        pygeoprocessing.raster_calculator(
            [(clipped_observed_yield_raster_path, 1)],
            _zero_observed_yield_op, zeroed_observed_yield_raster_path,
            gdal.GDT_Float32, observed_yield_nodata)

        interpolated_observed_yield_raster_path = os.path.join(
            output_dir, _INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))

        LOGGER.info(
            "Interpolating observed %s raster to landcover.", crop_name)
        pygeoprocessing.warp_raster(
            zeroed_observed_yield_raster_path,
            landcover_raster_info['pixel_size'],
            interpolated_observed_yield_raster_path, 'cubic_spline',
            target_sr_wkt=landcover_raster_info['projection'],
            target_bb=landcover_raster_info['bounding_box'])

        def _mask_observed_yield(lulc_array, observed_yield_array):
            """Mask total observed yield to crop lulc type."""
            result = numpy.empty(lulc_array.shape, dtype=numpy.float32)
            result[:] = observed_yield_nodata
            valid_mask = lulc_array != landcover_nodata
            lulc_mask = lulc_array == crop_lucode
            result[valid_mask] = 0
            result[lulc_mask] = (
                observed_yield_array[lulc_mask] * pixel_area_ha)
            return result

        observed_production_raster_path = os.path.join(
            output_dir, _OBSERVED_PRODUCTION_FILE_PATTERN % (
                crop_name, file_suffix))

        pygeoprocessing.raster_calculator(
            [(args['landcover_raster_path'], 1),
             (interpolated_observed_yield_raster_path, 1)],
            _mask_observed_yield, observed_production_raster_path,
            gdal.GDT_Float32, observed_yield_nodata)

    # both 'crop_nutrient.csv' and 'crop' are known data/header values for
    # this model data.
    nutrient_table = utils.build_lookup_from_csv(
        os.path.join(args['model_data_path'], 'crop_nutrient.csv'),
        'crop', to_lower=False)

    LOGGER.info("Generating report table")
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
    with open(result_table_path, 'wb') as result_table:
        result_table.write(
            'crop,area (ha),' + 'production_observed,' +
            ','.join(production_percentile_headers) + ',' + ','.join(
                nutrient_headers) + '\n')
        for crop_name in sorted(crop_to_landcover_table):
            result_table.write(crop_name)
            result_table.write(',%f' % production_area[crop_name])
            production_lookup = {}
            yield_sum = 0.0
            observed_production_raster_path = os.path.join(
                output_dir,
                _OBSERVED_PRODUCTION_FILE_PATTERN % (
                    crop_name, file_suffix))
            observed_yield_nodata = pygeoprocessing.get_raster_info(
                observed_production_raster_path)['nodata'][0]
            for _, yield_block in pygeoprocessing.iterblocks(
                    observed_production_raster_path):
                yield_sum += numpy.sum(
                    yield_block[observed_yield_nodata != yield_block])
            production_lookup['observed'] = yield_sum
            result_table.write(",%f" % yield_sum)

            for yield_percentile_id in sorted(yield_percentile_headers):
                yield_percentile_raster_path = os.path.join(
                    output_dir,
                    _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                        crop_name, yield_percentile_id, file_suffix))
                yield_sum = 0.0
                for _, yield_block in pygeoprocessing.iterblocks(
                        yield_percentile_raster_path):
                    yield_sum += numpy.sum(
                        yield_block[_NODATA_YIELD != yield_block])
                production_lookup[yield_percentile_id] = yield_sum
                result_table.write(",%f" % yield_sum)

            # convert 100g to Mg and fraction left over from refuse
            nutrient_factor = 1e4 * (
                1.0 - nutrient_table[crop_name]['Percentrefuse'] / 100.0)
            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                for yield_percentile_id in sorted(yield_percentile_headers):
                    total_nutrient = (
                        nutrient_factor *
                        production_lookup[yield_percentile_id] *
                        nutrient_table[crop_name][nutrient_id])
                    result_table.write(",%f" % (total_nutrient))
                result_table.write(
                    ",%f" % (
                        nutrient_factor *
                        production_lookup['observed'] *
                        nutrient_table[crop_name][nutrient_id]))
            result_table.write('\n')

        total_area = 0.0
        for _, band_values in pygeoprocessing.iterblocks(
                args['landcover_raster_path']):
            total_area += numpy.count_nonzero(
                (band_values != landcover_nodata))
        result_table.write(
            '\n,total area (both crop and non-crop)\n,%f\n' % (
                total_area * pixel_area_ha))

    if ('aggregate_polygon_path' in args and
            args['aggregate_polygon_path'] is not None):
        LOGGER.info("aggregating result over query polygon")
        # reproject polygon to LULC's projection
        target_aggregate_vector_path = os.path.join(
            output_dir, _AGGREGATE_VECTOR_FILE_PATTERN % (file_suffix))
        pygeoprocessing.reproject_vector(
            args['aggregate_polygon_path'],
            landcover_raster_info['projection'],
            target_aggregate_vector_path, layer_index=0,
            driver_name='ESRI Shapefile')

        # loop over every crop and query with pgp function
        total_yield_lookup = {}
        total_nutrient_table = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(
                float)))
        for crop_name in crop_to_landcover_table:
            # convert 100g to Mg and fraction left over from refuse
            nutrient_factor = 1e4 * (
                1.0 - nutrient_table[crop_name]['Percentrefuse'] / 100.0)
            # loop over percentiles
            for yield_percentile_id in yield_percentile_headers:
                percentile_crop_production_raster_path = os.path.join(
                    output_dir,
                    _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                        crop_name, yield_percentile_id, file_suffix))
                LOGGER.info(
                    "Calculating zonal stats for %s  %s", crop_name,
                    yield_percentile_id)
                total_yield_lookup['%s_%s' % (
                    crop_name, yield_percentile_id)] = (
                        pygeoprocessing.zonal_statistics(
                            (percentile_crop_production_raster_path, 1),
                            target_aggregate_vector_path,
                            str(args['aggregate_polygon_id'])))

                for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                    for id_index in total_yield_lookup['%s_%s' % (
                            crop_name, yield_percentile_id)]:
                        total_nutrient_table[nutrient_id][
                            yield_percentile_id][id_index] += (
                                nutrient_factor *
                                total_yield_lookup['%s_%s' % (
                                    crop_name, yield_percentile_id)][
                                        id_index]['sum'] *
                                nutrient_table[crop_name][nutrient_id])

            # process observed
            observed_yield_path = os.path.join(
                output_dir, _OBSERVED_PRODUCTION_FILE_PATTERN % (
                    crop_name, file_suffix))
            total_yield_lookup['%s_observed' % crop_name] = (
                pygeoprocessing.zonal_statistics(
                    (observed_yield_path, 1),
                    target_aggregate_vector_path,
                    str(args['aggregate_polygon_id'])))
            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                for id_index in total_yield_lookup['%s_observed' % crop_name]:
                    total_nutrient_table[
                        nutrient_id]['observed'][id_index] += (
                            nutrient_factor *
                            total_yield_lookup[
                                '%s_observed' % crop_name][id_index]['sum'] *
                            nutrient_table[crop_name][nutrient_id])

        # use that result to calculate nutrient totals

        # report everything to a table
        aggregate_table_path = os.path.join(
            output_dir, _AGGREGATE_TABLE_FILE_PATTERN % file_suffix)
        with open(aggregate_table_path, 'wb') as aggregate_table:
            # write header
            aggregate_table.write('%s,' % args['aggregate_polygon_id'])
            aggregate_table.write(','.join(sorted(total_yield_lookup)) + ',')
            aggregate_table.write(
                ','.join([
                    '%s_%s' % (nutrient_id, model_type)
                    for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
                    for model_type in sorted(
                        total_nutrient_table.itervalues().next())]))
            aggregate_table.write('\n')

            # iterate by polygon index
            for id_index in total_yield_lookup.itervalues().next():
                aggregate_table.write('%s,' % id_index)
                aggregate_table.write(','.join([
                    str(total_yield_lookup[yield_header][id_index]['sum'])
                    for yield_header in sorted(total_yield_lookup)]))

                for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                    for model_type in sorted(
                            total_nutrient_table.itervalues().next()):
                        aggregate_table.write(
                            ',%s' % total_nutrient_table[
                                nutrient_id][model_type][id_index])
                aggregate_table.write('\n')


# This decorator ensures the input arguments are formatted for InVEST
@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Parameters:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.
    """
    missing_key_list = []
    no_value_list = []
    validation_error_list = []

    required_keys = [
        'workspace_dir',
        'model_data_path',
        'landcover_raster_path',
        'landcover_to_crop_table_path'
        ]

    if limit_to in [None, 'aggregate_polygon_id', 'aggregate_polygon_path']:
        if ('aggregate_polygon_path' in args and
                args['aggregate_polygon_path'] not in ['', None]):
            required_keys.append('aggregate_polygon_id')
            required_keys.append('aggregate_polygon_path')

    for key in required_keys:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ['', None]:
                no_value_list.append(key)

    if len(missing_key_list) > 0:
        # if there are missing keys, we have raise KeyError to stop hard
        raise KeyError(
            "The following keys were expected in `args` but were missing " +
            ', '.join(missing_key_list))

    if len(no_value_list) > 0:
        validation_error_list.append(
            (no_value_list, 'parameter has no value'))

    file_type_list = [
        ('landcover_raster_path', 'raster'),
        ('aggregate_polygon_path', 'vector')]

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in file_type_list:
            if (limit_to in [None, key]) and key in required_keys:
                if not os.path.exists(args[key]):
                    validation_error_list.append(
                        ([key], 'not found on disk'))
                    continue
                if key_type == 'raster':
                    raster = gdal.OpenEx(args[key])
                    if raster is None:
                        validation_error_list.append(
                            ([key], 'not a raster'))
                    del raster
                elif key_type == 'vector':
                    vector = gdal.OpenEx(args[key])
                    if vector is None:
                        validation_error_list.append(
                            ([key], 'not a vector'))
                    del vector
    return validation_error_list
