"""InVEST Carbon Model."""
import collections
import math
import logging
import os
import sys
import shutil

from osgeo import gdal
from osgeo import ogr
import numpy
import pygeoprocessing

from . import utils

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.carbon')

_OUTPUT_BASE_FILES = {
    'tot_c_cur': 'tot_c_cur.tif',
    'tot_c_fut': 'tot_c_fut.tif',
    'tot_c_redd': 'tot_c_redd.tif',
    }

_INTERMEDIATE_BASE_FILES = {
    'c_above_cur': 'c_above_cur.tif',
    'c_below_cur': 'c_below_cur.tif',
    'c_soil_cur': 'c_soil_cur.tif',
    'c_dead_cur': 'c_dead_cur.tif',
    'c_above_fut': 'c_above_fut.tif',
    'c_below_fut': 'c_below_fut.tif',
    'c_soil_fut': 'c_soil_fut.tif',
    'c_dead_fut': 'c_dead_fut.tif',
    'c_above_redd': 'c_above_redd.tif',
    'c_below_redd': 'c_below_redd.tif',
    'c_soil_redd': 'c_soil_redd.tif',
    'c_dead_redd': 'c_dead_redd.tif',
    }

_TMP_BASE_FILES = {
    'aligned_lulc_cur_path': 'aligned_lulc_cur.tif',
    'aligned_lulc_fut_path': 'aligned_lulc_fut.tif',
    'aligned_lulc_redd_path': 'aligned_lulc_redd.tif',
    }

_CARBON_NODATA = -1.0


def execute(args):
    """InVEST Carbon Model.

    Calculate the amount of carbon stocks given a landscape, or the difference
    due to a future change, and/or the tradeoffs between that and a REDD
    scenario, and calculate economic valuation on those scenarios.

    The model can operate on a single scenario, a combined present and future
    scenario, as well as an additional REDD scenario.

    Parameters:
        args['workspace_dir'] (string): a path to the directory that will
            write output and other temporary files during calculation.
        args['results_suffix'] - a string to append to any output file name.
        args['lulc_cur_path'] (string): a path to a raster representing the
            current carbon stocks
        args['lulc_fut_path'] (string): a path to a raster representing future
            landcover scenario.  Optional, but if present and well defined
            will trigger a sequestration calculation.
        args['lulc_redd_path'] (string): a path to a raster representing the
            alternative REDD scenario which is only possible if the
            args['lulc_fut_path'] is present and well defined.
        args['carbon_pools_path'] (string): path to CSV or that indexes carbon
            storage density to lulc codes. (required if 'do_uncertainty' is
            false)
        args['carbon_pools_uncertain_uri'] - as above, but has probability
            distribution data for each lulc type rather than point estimates.
            (required if 'do_uncertainty' is true)
        args['lulc_cur_year'] - An integer representing the year of lulc_cur
            used in HWP calculation (required if args contains a
            'hwp_cur_shape_uri', or 'hwp_fut_shape_uri' key)
        args['lulc_fut_year'] - An integer representing the year of  lulc_fut
            used in HWP calculation (required if args contains a
            'hwp_fut_shape_uri' key)
        args['lulc_redd_path'] (string): a path to a raster that represents
            land cover data for the REDD policy scenario (optional).
    Returns:
        None.
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = args['workspace_dir']
    pygeoprocessing.create_directories([intermediate_output_dir, output_dir])

    LOGGER.info('Building file registry')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    carbon_pool_table = pygeoprocessing.get_lookup_from_table(
        args['carbon_pools_path'], 'lucode')

    cell_sizes = []
    valid_lulc_keys = []
    for scenario_type in ['cur', 'fut', 'redd']:
        lulc_key = "lulc_%s_path" % (scenario_type)
        if lulc_key in args and len(args[lulc_key]) > 0:
            cell_sizes.append(
                pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                    args[lulc_key]))
            valid_lulc_keys.append(lulc_key)
    pixel_size_out = min(cell_sizes)

    # align the input datasets
    pygeoprocessing.align_dataset_list(
        [args[_] for _ in valid_lulc_keys],
        [file_registry['aligned_' + _] for _ in valid_lulc_keys],
        ['nearest'] * len(valid_lulc_keys),
        pixel_size_out, 'intersection', 0, assert_datasets_projected=True)

    keys = None
    nodata = None
    values = None
    aligned_lulc_key = None
    pool_storage_path_lookup = collections.defaultdict(list)
    for pool_type in ['c_above', 'c_below', 'c_soil', 'c_dead']:
        carbon_pool_by_type = dict([
            (lucode, float(carbon_pool_table[lucode][pool_type]))
            for lucode in carbon_pool_table])
        for scenario_type in ['cur', 'fut', 'redd']:
            lulc_key = 'lulc_%s_path' % scenario_type
            if lulc_key not in args or len(args[lulc_key]) == 0:
                continue
            LOGGER.info('Mapping carbon for %s scenario.', lulc_key)
            aligned_lulc_key = 'aligned_' + lulc_key
            nodata = pygeoprocessing.get_nodata_from_uri(
                file_registry[aligned_lulc_key])
            carbon_pool_by_type_copy = carbon_pool_by_type.copy()
            carbon_pool_by_type_copy[nodata] = _CARBON_NODATA

            keys = sorted(numpy.array(carbon_pool_by_type_copy.keys()))
            values = numpy.array([carbon_pool_by_type_copy[x] for x in keys])

            def _map_lulc_to_total_carbon(lulc_array):
                """Convert a block of original values to the lookup values."""
                unique = numpy.unique(lulc_array)
                has_map = numpy.in1d(unique, keys)
                if not all(has_map):
                    raise ValueError(
                        'There was not a value for at least the following'
                        ' codes %s for this file %s.\nNodata value is:'
                        ' %s' % (
                            str(unique[~has_map]),
                            file_registry[aligned_lulc_key], str(nodata)))
                index = numpy.digitize(
                    lulc_array.ravel(), keys, right=True)
                result = numpy.empty(lulc_array.shape, dtype=numpy.float32)
                result[:] = _CARBON_NODATA
                valid_mask = lulc_array != nodata
                result = values[index].reshape(lulc_array.shape)
                # multipy density by area to get storage
                result[valid_mask] *= pixel_size_out**2 / 10**4
                return result

            storage_key = '%s_%s' % (pool_type, scenario_type)
            pygeoprocessing.vectorize_datasets(
                [file_registry[aligned_lulc_key]], _map_lulc_to_total_carbon,
                file_registry[storage_key], gdal.GDT_Float32, _CARBON_NODATA,
                pixel_size_out, "intersection", dataset_to_align_index=0,
                vectorize_op=False,
                assert_datasets_projected=True,
                datasets_are_pre_aligned=True)
            pool_storage_path_lookup[scenario_type].append(
                file_registry[storage_key])

    for scenario_type, storage_paths in pool_storage_path_lookup.iteritems():
        LOGGER.info(
            "Calculate total carbon storage for %s", scenario_type)

        def _sum_op(*storage_arrays):
            valid_mask = reduce(
                lambda x, y: x & y, [
                    _ != _CARBON_NODATA for _ in storage_arrays])
            result = numpy.empty(storage_arrays[0].shape)
            result[:] = _CARBON_NODATA
            result[valid_mask] = numpy.sum([
                _[valid_mask] for _ in storage_arrays], axis=0)
            return result

        pygeoprocessing.vectorize_datasets(
            storage_paths, _sum_op, file_registry['tot_c_' + scenario_type],
            gdal.GDT_Float32, _CARBON_NODATA, pixel_size_out, "intersection",
            dataset_to_align_index=0, vectorize_op=False,
            datasets_are_pre_aligned=True)

    return

    for fut_type in ['fut', 'redd']:
        fut_type_lulc_uri = 'lulc_%s_uri' % fut_type
        if 'lulc_cur_uri' in args and fut_type_lulc_uri in args:
            LOGGER.info('Computing sequestration for %s scenario', fut_type)

            def sub_op(c_cur, c_fut):
                fut_nodata = c_fut == nodata_out
                cur_nodata = c_cur == nodata_out
                cur_clean = numpy.where(cur_nodata, 0, c_cur)
                fut_clean = numpy.where(fut_nodata, 0, c_fut)
                seq = fut_clean - cur_clean
                return numpy.where(fut_nodata & cur_nodata, nodata_out, seq)

            pixel_size_out = pygeoprocessing.geoprocessing.get_cell_size_from_uri(args['lulc_cur_uri'])
            outputs['sequest_%s' % fut_type] = outfile_uri('sequest', fut_type)
            pygeoprocessing.geoprocessing.vectorize_datasets(
                [outputs['tot_C_cur'], outputs['tot_C_%s' % fut_type]], sub_op,
                outputs['sequest_%s' % fut_type], gdal.GDT_Float32, nodata_out,
                pixel_size_out, "intersection", dataset_to_align_index=0,
                process_pool=args['_process_pool'], vectorize_op=False)

            if do_uncertainty:
                LOGGER.info('Computing confident cells for %s scenario.', fut_type)
                confidence_threshold = args['confidence_threshold']

                # Returns 1 if we're confident storage will increase,
                #         -1 if we're confident storage will decrease,
                #         0 if we're not confident either way.
                def confidence_op(c_cur, c_fut, var_cur, var_fut):
                    if nodata_out in [c_cur, c_fut, var_cur, var_fut]:
                        return nodata_out

                    if var_cur == 0 and var_fut == 0:
                        # There's no variance, so we can just compare the mean estimates.
                        if c_fut > c_cur:
                            return 1
                        if c_fut < c_cur:
                            return -1
                        return 0

                    # Given two distributions (one for current storage, one for future storage),
                    # we use the difference distribution (current storage - future storage),
                    # and calculate the probability that the difference is less than 0.
                    # This is equal to the probability that the future storage is greater than
                    # the current storage.
                    # We calculate the standard score by beginning with 0, subtracting the mean
                    # of the difference distribution, and dividing by the standard deviation
                    # of the difference distribution.
                    # The mean of the difference distribution is the difference of the means of cur and fut.
                    # The variance of the difference distribution is the sum of the variances of cur and fut.
                    standard_score = (c_fut - c_cur) / math.sqrt(var_cur + var_fut)

                    # Calculate the cumulative distribution function for the standard normal distribution.
                    # This gives us the probability that future carbon storage is greater than
                    # current carbon storage.
                    # This formula is copied from http://docs.python.org/3.2/library/math.html
                    probability = (1.0 + math.erf(standard_score / math.sqrt(2.0))) / 2.0

                    # Multiply by 100 so we have probability in the same units as the confidence_threshold.
                    confidence = 100 * probability
                    if confidence >= confidence_threshold:
                        # We're confident carbon storage will increase.
                        return 1
                    if confidence <= 100 - confidence_threshold:
                        # We're confident carbon storage will decrease.
                        return -1
                    # We're not confident about whether storage will increase or decrease.
                    return 0

                outputs['conf_%s' % fut_type] = outfile_uri('conf', fut_type)
                pygeoprocessing.geoprocessing.vectorize_datasets(
                    [outputs[name] for name in ['tot_C_cur', 'tot_C_%s' % fut_type,
                                                       'variance_C_cur', 'variance_C_%s' % fut_type]],
                    confidence_op, outputs['conf_%s' % fut_type],
                    gdal.GDT_Float32, nodata_out,
                    pixel_size_out, "intersection", dataset_to_align_index=0,
                    process_pool=args['_process_pool'])

    # Do a Monte Carlo simulation for uncertainty analysis.
    # We only do this if HWP is not enabled, because the simulation
    # computes carbon just by summing carbon across the
    # landscape, which is wrong if we're doing HWP analysis.
    if (do_uncertainty and
        'hwp_cur_shape_uri' not in args and
        'hwp_fut_shape_uri' not in args):
        outputs['uncertainty'] = _compute_uncertainty_data(args, pools)

    return outputs


def _compute_carbon_pools(args):
    """Returns a dict with data on carbon pool totals and variance."""

    if args['do_uncertainty']:
        pool_inputs = pygeoprocessing.geoprocessing.get_lookup_from_table(
            args['carbon_pools_uncertain_uri'], 'lucode')
    else:
        pool_inputs = pygeoprocessing.geoprocessing.get_lookup_from_table(
            args['carbon_pools_uri'], 'lucode')

    cell_area_ha = _compute_cell_area_ha(args)

    pool_estimate_types = ['c_above', 'c_below', 'c_soil', 'c_dead']

    if args['do_uncertainty']:
        # We want the mean and standard deviation columns from the input.
        pool_estimate_sds = [s + '_sd' for s in pool_estimate_types]
        pool_estimate_types = [s + '_mean' for s in pool_estimate_types]

    pools = {}
    for lulc_id in pool_inputs:
        pools[lulc_id] = {}

        # Compute the total carbon per pixel for each lulc type
        pools[lulc_id]['total'] = cell_area_ha * sum(
            pool_inputs[lulc_id][pool_type]
            for pool_type in pool_estimate_types)

        if args['do_uncertainty']:
            # Compute the total variance per pixel for each lulc type.
            # We have a normal distribution for each pool; we assume each is
            # independent, so the variance of the sum is equal to the sum of
            # the variances. Note that we scale by the area squared.
            pools[lulc_id]['variance'] = (cell_area_ha ** 2) * sum(
                pool_inputs[lulc_id][pool_type_sd] ** 2
                for pool_type_sd in pool_estimate_sds)

    return pools


def _compute_cell_area_ha(args):
    cell_area_cur = pygeoprocessing.geoprocessing.get_cell_size_from_uri(args['lulc_cur_uri']) ** 2

    for scenario in ['fut', 'redd']:
        try:
            lulc_uri = args['lulc_%s_uri' % scenario]
        except KeyError:
            continue

        cell_area_in_scenario = pygeoprocessing.geoprocessing.get_cell_size_from_uri(lulc_uri) ** 2

        if abs(cell_area_cur - cell_area_in_scenario) <= sys.float_info.epsilon:
            LOGGER.warn(
                'The LULC map for the %s scenario has a different cell area '
                'than the LULC map for the current scenario. Please '
                'ensure that all LULC maps have the same cell area.' % scenario)

    # Convert to hectares.
    return cell_area_cur / 10000.0


def _compute_uncertainty_data(args, pools):
    """Computes the mean and std dev for carbon storage and sequestration."""

    LOGGER.info("Computing uncertainty data.")

    # Count how many grid cells have each lulc type in each scenario map.
    lulc_counts = {}
    for scenario in ['cur', 'fut', 'redd']:
        try:
            lulc_uri = args['lulc_%s_uri' % scenario]
        except KeyError:
            continue

        lulc_counts[scenario] = pygeoprocessing.geoprocessing.unique_raster_values_count(
            lulc_uri)

    # Do a Monte Carlo simulation for carbon storage.
    monte_carlo_results = {}
    LOGGER.info("Beginning Monte Carlo simulation.")
    for _ in range(NUM_MONTE_CARLO_RUNS):
        run_results = _do_monte_carlo_run(pools, lulc_counts)

        # Note that in this context, 'scenario' could be an actual scenario
        # (e.g. current, future, REDD) or it could be a sequestration
        # (e.g. sequestration under future or sequestration under REDD).
        for scenario, carbon_amount in run_results.items():
            try:
                monte_carlo_results[scenario].append(carbon_amount)
            except KeyError:
                monte_carlo_results[scenario] = [carbon_amount]

    LOGGER.info("Done with Monte Carlo simulation.")

    # Compute the mean and standard deviation for each scenario.
    results = {}
    for scenario in monte_carlo_results:
        results[scenario] = norm.fit(monte_carlo_results[scenario])

    return results


def _do_monte_carlo_run(pools, lulc_counts):
    """Do a single Monte Carlo run for carbon storage.

    Returns a dict with the results, keyed by scenario, and
    # including results for sequestration.
    """

    # Sample carbon-per-grid-cell from the given normal distribution.
    # We sample this independently for each LULC type.
    lulc_carbon_samples = {}
    for lulc_id, distribution in pools.items():
        if not distribution['variance']:
            lulc_carbon_samples[lulc_id] = distribution['total']
        else:
            lulc_carbon_samples[lulc_id] = numpy.random.normal(
                distribution['total'],
                math.sqrt(distribution['variance']))

    # Compute the amount of carbon in each scenario.
    results = {}
    for scenario, counts in lulc_counts.items():
        # Amount of carbon is the sum across all lulc types of:
        # (number of grid cells) x (carbon per grid cell)
        results[scenario] = sum(
            count * lulc_carbon_samples[lulc_id]
            for lulc_id, count in counts.items())

    # Compute sequestration.
    for scenario in ['fut', 'redd']:
        if scenario not in results:
            continue
        results['sequest_%s' % scenario] = results[scenario] - results['cur']

    return results


def _calculate_hwp_storage_cur(
    hwp_shape_uri, base_dataset_uri, c_hwp_uri, bio_hwp_uri, vol_hwp_uri,
    yr_cur):
    """Calculates carbon storage, hwp biomassPerPixel and volumePerPixel due
        to harvested wood products in parcels on current landscape.

        hwp_shape - oal shapefile indicating harvest map of interest
        base_dataset_uri - a gdal dataset to create the output rasters from
        c_hwp - an output GDAL rasterband representing  carbon stored in
            harvested wood products for current calculation
        bio_hwp - an output GDAL rasterband representing carbon stored in
            harvested wood products for land cover under interest
        vol_hwp - an output GDAL rasterband representing carbon stored in
             harvested wood products for land cover under interest
        yr_cur - year of the current landcover map

        No return value"""

    ############### Start
    pixel_area = pygeoprocessing.geoprocessing.get_cell_size_from_uri(base_dataset_uri) ** 2 / 10000.0 #convert to Ha
    hwp_shape = ogr.Open(hwp_shape_uri)
    base_dataset = gdal.Open(base_dataset_uri)
    nodata = -5.0

    #Create a temporary shapefile to hold values of per feature carbon pools
    #HWP biomassPerPixel and volumePerPixel, will be used later to rasterize
    #those values to output rasters
    hwp_shape_copy = ogr.GetDriverByName('Memory').CopyDataSource(hwp_shape, '')
    hwp_shape_layer_copy = hwp_shape_copy.GetLayer()

    #Create fields in the layers to hold hardwood product pools,
    #biomassPerPixel and volumePerPixel
    calculated_attribute_names = ['c_hwp_pool', 'bio_hwp', 'vol_hwp']
    for x in calculated_attribute_names:
        field_def = ogr.FieldDefn(x, ogr.OFTReal)
        hwp_shape_layer_copy.CreateField(field_def)

    #Visit each feature and calculate the carbon pool, biomassPerPixel, and
    #volumePerPixel of that parcel
    for feature in hwp_shape_layer_copy:
        #This makes a helpful dictionary to access fields in the feature
        #later in the code
        field_args = _get_fields(feature)

        #If start date and/or the amount of carbon per cut is zero, it doesn't
        #make sense to do any calculation on carbon pools or
        #biomassPerPixel/volumePerPixel
        if field_args['start_date'] != 0 and field_args['cut_cur'] != 0:

            time_span = yr_cur - field_args['start_date']
            start_years = time_span

            #Calculate the carbon pool due to decaying HWP over the time_span
            feature_carbon_storage_per_pixel = (
                pixel_area * _carbon_pool_in_hwp_from_parcel(
                    field_args['cut_cur'], time_span, start_years,
                    field_args['freq_cur'], field_args['decay_cur']))

            #Next lines caculate biomassPerPixel and volumePerPixel of
            #harvested wood
            number_of_harvests = \
                math.ceil(time_span / float(field_args['freq_cur']))

            biomass_in_feature = field_args['cut_cur'] * number_of_harvests / \
                float(field_args['c_den_cur'])

            biomass_per_pixel = biomass_in_feature * pixel_area

            volume_per_pixel = biomass_per_pixel / field_args['bcef_cur']

            #Copy biomass_per_pixel and carbon pools to the temporary feature
            #for rasterization of the entire layer later
            for field, value in zip(calculated_attribute_names,
                                    [feature_carbon_storage_per_pixel,
                                     biomass_per_pixel, volume_per_pixel]):
                feature.SetField(feature.GetFieldIndex(field), value)

            #This saves the changes made to feature back to the shape layer
            hwp_shape_layer_copy.SetFeature(feature)

    #burn all the attribute values to a raster
    for attribute_name, raster_uri in zip(
        calculated_attribute_names, [c_hwp_uri, bio_hwp_uri, vol_hwp_uri]):

        raster = pygeoprocessing.geoprocessing.new_raster_from_base(
            base_dataset, raster_uri, 'GTiff', nodata, gdal.GDT_Float32,
            fill_value=nodata)
        gdal.RasterizeLayer(raster, [1], hwp_shape_layer_copy,
                            options=['ATTRIBUTE=' + attribute_name])
        raster.FlushCache()
        raster = None


def _calculate_hwp_storage_fut(
    hwp_shapes, base_dataset_uri, c_hwp_uri, bio_hwp_uri, vol_hwp_uri,
    yr_cur, yr_fut, process_pool=None):
    """Calculates carbon storage, hwp biomassPerPixel and volumePerPixel due to
        harvested wood products in parcels on current landscape.

        hwp_shapes - a dictionary containing the current and/or future harvest
            maps (or nothing)
            hwp_shapes['cur'] - oal shapefile indicating harvest map from the
                current landscape
            hwp_shapes['fut'] - oal shapefile indicating harvest map from the
                future landscape
        c_hwp - an output GDAL rasterband representing  carbon stored in
            harvested wood products for current calculation
        bio_hwp - an output GDAL rasterband representing carbon stored in
            harvested wood products for land cover under interest
        vol_hwp - an output GDAL rasterband representing carbon stored in
             harvested wood products for land cover under interest
        yr_cur - year of the current landcover map
        yr_fut - year of the current landcover map
        process_pool - a process pool for parallel processing (can be None)

        No return value"""

    ############### Start
    pixel_area = pygeoprocessing.geoprocessing.get_cell_size_from_uri(base_dataset_uri) ** 2 / 10000.0 #convert to Ha
    nodata = -5.0

    c_hwp_cur_uri = pygeoprocessing.geoprocessing.temporary_filename()
    bio_hwp_cur_uri = pygeoprocessing.geoprocessing.temporary_filename()
    vol_hwp_cur_uri = pygeoprocessing.geoprocessing.temporary_filename()

    pygeoprocessing.geoprocessing.new_raster_from_base_uri(base_dataset_uri, c_hwp_uri, 'GTiff', nodata, gdal.GDT_Float32, fill_value=nodata)
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(base_dataset_uri, bio_hwp_uri, 'GTiff', nodata, gdal.GDT_Float32, fill_value=nodata)
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(base_dataset_uri, vol_hwp_uri, 'GTiff', nodata, gdal.GDT_Float32, fill_value=nodata)

    #Create a temporary shapefile to hold values of per feature carbon pools
    #HWP biomassPerPixel and volumePerPixel, will be used later to rasterize
    #those values to output rasters

    calculatedAttributeNames = ['c_hwp_pool', 'bio_hwp', 'vol_hwp']
    if 'cur' in hwp_shapes:
        hwp_shape = ogr.Open(hwp_shapes['cur'])
        hwp_shape_copy = \
            ogr.GetDriverByName('Memory').CopyDataSource(hwp_shape, '')
        hwp_shape_layer_copy = \
            hwp_shape_copy.GetLayer()

        #Create fields in the layers to hold hardwood product pools,
        #biomassPerPixel and volumePerPixel
        for fieldName in calculatedAttributeNames:
            field_def = ogr.FieldDefn(fieldName, ogr.OFTReal)
            hwp_shape_layer_copy.CreateField(field_def)

        #Visit each feature and calculate the carbon pool, biomassPerPixel,
        #and volumePerPixel of that parcel
        for feature in hwp_shape_layer_copy:
            #This makes a helpful dictionary to access fields in the feature
            #later in the code
            field_args = _get_fields(feature)

            #If start date and/or the amount of carbon per cut is zero, it
            #doesn't make sense to do any calculation on carbon pools or
            #biomassPerPixel/volumePerPixel
            if field_args['start_date'] != 0 and field_args['cut_cur'] != 0:

                time_span = (yr_fut + yr_cur) / 2.0 - field_args['start_date']
                start_years = yr_fut - field_args['start_date']

                #Calculate the carbon pool due to decaying HWP over the
                #time_span
                feature_carbon_storage_per_pixel = (
                    pixel_area * _carbon_pool_in_hwp_from_parcel(
                        field_args['cut_cur'], time_span, start_years,
                        field_args['freq_cur'], field_args['decay_cur']))

                #Claculate biomassPerPixel and volumePerPixel of harvested wood
                numberOfHarvests = \
                    math.ceil(time_span / float(field_args['freq_cur']))
                #The measure of biomass is in terms of Mg/ha
                biomassInFeaturePerArea = field_args['cut_cur'] * \
                    numberOfHarvests / float(field_args['c_den_cur'])


                biomassPerPixel = biomassInFeaturePerArea * pixel_area
                volumePerPixel = biomassPerPixel / field_args['bcef_cur']

                #Copy biomassPerPixel and carbon pools to the temporary
                #feature for rasterization of the entire layer later
                for field, value in zip(calculatedAttributeNames,
                                        [feature_carbon_storage_per_pixel,
                                         biomassPerPixel, volumePerPixel]):
                    feature.SetField(feature.GetFieldIndex(field), value)

                #This saves the changes made to feature back to the shape layer
                hwp_shape_layer_copy.SetFeature(feature)

        #burn all the attribute values to a raster
        for attributeName, raster_uri in zip(calculatedAttributeNames,
                                          [c_hwp_cur_uri, bio_hwp_cur_uri, vol_hwp_cur_uri]):
            nodata = -1.0
            pygeoprocessing.geoprocessing.new_raster_from_base_uri(base_dataset_uri, raster_uri, 'GTiff', nodata, gdal.GDT_Float32, fill_value=nodata)
            raster = gdal.Open(raster_uri, gdal.GA_Update)
            gdal.RasterizeLayer(raster, [1], hwp_shape_layer_copy, options=['ATTRIBUTE=' + attributeName])
            raster.FlushCache()
            raster = None

    #handle the future term
    if 'fut' in hwp_shapes:
        hwp_shape = ogr.Open(hwp_shapes['fut'])
        hwp_shape_copy = \
            ogr.GetDriverByName('Memory').CopyDataSource(hwp_shape, '')
        hwp_shape_layer_copy = \
            hwp_shape_copy.GetLayer()

        #Create fields in the layers to hold hardwood product pools,
        #biomassPerPixel and volumePerPixel
        for fieldName in calculatedAttributeNames:
            field_def = ogr.FieldDefn(fieldName, ogr.OFTReal)
            hwp_shape_layer_copy.CreateField(field_def)

        #Visit each feature and calculate the carbon pool, biomassPerPixel,
        #and volumePerPixel of that parcel
        for feature in hwp_shape_layer_copy:
            #This makes a helpful dictionary to access fields in the feature
            #later in the code
            field_args = _get_fields(feature)

            #If start date and/or the amount of carbon per cut is zero, it
            #doesn't make sense to do any calculation on carbon pools or
            #biomassPerPixel/volumePerPixel
            if field_args['cut_fut'] != 0:

                time_span = yr_fut - (yr_fut + yr_cur) / 2.0
                start_years = time_span

                #Calculate the carbon pool due to decaying HWP over the
                #time_span
                feature_carbon_storage_per_pixel = pixel_area * \
                    _carbon_pool_in_hwp_from_parcel(
                    field_args['cut_fut'], time_span, start_years,
                    field_args['freq_fut'], field_args['decay_fut'])

                #Claculate biomassPerPixel and volumePerPixel of harvested wood
                numberOfHarvests = \
                    math.ceil(time_span / float(field_args['freq_fut']))

                biomassInFeaturePerArea = field_args['cut_fut'] * \
                    numberOfHarvests / float(field_args['c_den_fut'])

                biomassPerPixel = biomassInFeaturePerArea * pixel_area

                volumePerPixel = biomassPerPixel / field_args['bcef_fut']

                #Copy biomassPerPixel and carbon pools to the temporary
                #feature for rasterization of the entire layer later
                for field, value in zip(calculatedAttributeNames,
                                        [feature_carbon_storage_per_pixel,
                                         biomassPerPixel, volumePerPixel]):
                    feature.SetField(feature.GetFieldIndex(field), value)

                #This saves the changes made to feature back to the shape layer
                hwp_shape_layer_copy.SetFeature(feature)

        #burn all the attribute values to a raster
        for attributeName, (raster_uri, cur_raster_uri) in zip(
            calculatedAttributeNames, [(c_hwp_uri, c_hwp_cur_uri), (bio_hwp_uri, bio_hwp_cur_uri), (vol_hwp_uri, vol_hwp_cur_uri)]):

            temp_filename = pygeoprocessing.geoprocessing.temporary_filename()
            pygeoprocessing.geoprocessing.new_raster_from_base_uri(
                base_dataset_uri, temp_filename, 'GTiff',
                nodata, gdal.GDT_Float32, fill_value=nodata)
            temp_raster = gdal.Open(temp_filename, gdal.GA_Update)
            gdal.RasterizeLayer(temp_raster, [1], hwp_shape_layer_copy,
                                options=['ATTRIBUTE=' + attributeName])
            temp_raster.FlushCache()
            temp_raster = None

            #add temp_raster and raster cur raster into the output raster
            nodata = -1.0
            base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
                raster_uri)
            cur_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
                cur_raster_uri)
            def add_op(base, current):
                """add two rasters"""
                nodata_mask = (base == base_nodata) | (current == cur_nodata)
                return numpy.where(nodata_mask, nodata, base+current)

            pixel_size_out = (
                pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                    raster_uri))
            pygeoprocessing.geoprocessing.vectorize_datasets(
                [cur_raster_uri, temp_filename], add_op, raster_uri,
                gdal.GDT_Float32, nodata,
                pixel_size_out, "intersection", dataset_to_align_index=0,
                vectorize_op=False)


def _get_fields(feature):
    """Return a dict with all fields in the given feature.

        feature - an OGR feature.

        Returns an assembled python dict with a mapping of
        fieldname -> fieldvalue"""

    fields = {}
    for i in xrange(feature.GetFieldCount()):
        field_def = feature.GetFieldDefnRef(i)
        name = field_def.GetName().lower()
        value = feature.GetField(i)
        fields[name] = value

    return fields


def _carbon_pool_in_hwp_from_parcel(carbonPerCut, start_years, timeSpan, harvestFreq,
                              decay):
    """This is the summation equation that appears in equations 1, 5, 6, and 7
        from the user's guide

        carbonPerCut - The amount of carbon removed from a parcel during a
            harvest period
        start_years - The number of years ago that the harvest first started
        timeSpan - The number of years to calculate the harvest over
        harvestFreq - How many years between harvests
        decay - the rate at which carbon is decaying from HWP harvested from
            parcels

        returns a float indicating the amount of carbon stored from HWP
            harvested in units of Mg/ha"""

    carbonSum = 0.0
    omega = math.log(2) / decay
    #Recall that xrange is nonexclusive on the upper bound, so it corresponds
    #to the -1 in the summation terms given in the user's manual
    for t in xrange(int(math.ceil(float(start_years) / harvestFreq))):
        carbonSum += (1 - math.exp(-omega)) / (omega *
            math.exp((timeSpan - t * harvestFreq) * omega))
    return carbonSum * carbonPerCut


def execute_storage_seq(args):
    """Carbon Storage and Sequestration.

    This can include the biophysical model, the valuation model, or both.

    Args:
        workspace_dir (string): a uri to the directory that will write output
            and other temporary files during calculation. (required)
        suffix (string): a string to append to any output file name (optional)

        do_biophysical (boolean): whether to run the biophysical model
        lulc_cur_uri (string): a uri to a GDAL raster dataset (required)
        lulc_cur_year (int): An integer representing the year of lulc_cur
            used in HWP calculation (required if args contains a
            'hwp_cur_shape_uri', or 'hwp_fut_shape_uri' key)
        lulc_fut_uri (string): a uri to a GDAL raster dataset (optional
            if calculating sequestration)
        lulc_redd_uri (string): a uri to a GDAL raster dataset that represents
            land cover data for the REDD policy scenario (optional).
        lulc_fut_year (int): An integer representing the year of  lulc_fut
            used in HWP calculation (required if args contains a
            'hwp_fut_shape_uri' key)
        carbon_pools_uri (string): a uri to a CSV or DBF dataset mapping carbon
            storage density to the lulc classifications specified in the
            lulc rasters. (required if 'do_uncertainty' is false)
        hwp_cur_shape_uri (String): Current shapefile uri for harvested wood
            calculation (optional, include if calculating current lulc hwp)
        hwp_fut_shape_uri (String): Future shapefile uri for harvested wood
            calculation (optional, include if calculating future lulc hwp)
        do_uncertainty (boolean): a boolean that indicates whether we should do
            uncertainty analysis. Defaults to False if not present.
        carbon_pools_uncertain_uri (string): as above, but has probability
            distribution data for each lulc type rather than point estimates.
            (required if 'do_uncertainty' is true)
        confidence_threshold (float): a number between 0 and 100 that indicates
            the minimum threshold for which we should highlight regions in the
            output raster. (required if 'do_uncertainty' is True)
        sequest_uri (string): uri to a GDAL raster dataset describing the
            amount of carbon sequestered.
        yr_cur (int): the year at which the sequestration measurement started
        yr_fut (int): the year at which the sequestration measurement ended
        do_valuation (boolean): whether to run the valuation model
        carbon_price_units (string): indicates whether the price is
            in terms of carbon or carbon dioxide. Can value either as
            'Carbon (C)' or 'Carbon Dioxide (CO2)'.
        V (string): value of a sequestered ton of carbon or carbon dioxide in
        dollars per metric ton
        r (int): the market discount rate in terms of a percentage
        c (float): the annual rate of change in the price of carbon


    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir/',
            'suffix': '_results',
            'do_biophysical': True,
            'lulc_cur_uri': 'path/to/lulc_cur',
            'lulc_cur_year': 2014,
            'lulc_fut_uri': 'path/to/lulc_fut',
            'lulc_redd_uri': 'path/to/lulc_redd',
            'lulc_fut_year': 2025,
            'carbon_pools_uri': 'path/to/carbon_pools',
            'hwp_cur_shape_uri': 'path/to/hwp_cur_shape',
            'hwp_fut_shape_uri': 'path/to/hwp_fut_shape',
            'do_uncertainty': True,
            'carbon_pools_uncertain_uri': 'path/to/carbon_pools_uncertain',
            'confidence_threshold': 50.0,
            'sequest_uri': 'path/to/sequest_uri',
            'yr_cur': 2014,
            'yr_fut': 2025,
            'do_valuation': True,
            'carbon_price_units':, 'Carbon (C)',
            'V': 43.0,
            'r': 7,
            'c': 0,
        }

    Returns:
        outputs (dictionary): contains names of all output files

    """

    if args['do_biophysical']:
        LOGGER.info('Executing biophysical model.')
        biophysical_outputs = carbon_biophysical.execute(args)
    else:
        biophysical_outputs = None

        # We can't do uncertainty analysis if only the valuation model is run.
        args['do_uncertainty'] = False

    if args['do_valuation']:
        if not args['do_biophysical'] and not args.get('sequest_uri'):
            raise Exception(
                'In order to perform valuation, you must either run the '
                'biophysical model, or provide a sequestration raster '
                'mapping carbon sequestration for a landscape. Neither '
                'was provided in this case, so valuation cannot run.')
        LOGGER.info('Executing valuation model.')
        valuation_args = _package_valuation_args(args, biophysical_outputs)
        valuation_outputs = carbon_valuation.execute(valuation_args)
    else:
        valuation_outputs = None

    _create_HTML_report(args, biophysical_outputs, valuation_outputs)

def _package_valuation_args(args, biophysical_outputs):
    if not biophysical_outputs:
        return args

    if 'sequest_fut' not in biophysical_outputs:
        raise Exception(
            'Both biophysical and valuation models were requested, '
            'but sequestration was not calculated. In order to calculate '
            'valuation data, please run the biophysical model with '
            'sequestration analysis enabled. This requires a future LULC map '
            'in addition to the current LULC map.')

    args['sequest_uri'] = biophysical_outputs['sequest_fut']
    args['yr_cur'] = args['lulc_cur_year']
    args['yr_fut'] = args['lulc_fut_year']

    if args['yr_cur'] >= args['yr_fut']:
        raise Exception(
            'The current year must be earlier than the future year. '
            'The values for current/future year are: %d/%d' %
            (args['yr_cur'], args['yr_fut']))

    biophysical_to_valuation = {
        'uncertainty': 'uncertainty_data',
        'sequest_redd': 'sequest_redd_uri',
        'conf_fut': 'conf_uri',
        'conf_redd': 'conf_redd_uri'
        }

    for biophysical_key, valuation_key in biophysical_to_valuation.items():
        try:
            args[valuation_key] = biophysical_outputs[biophysical_key]
        except KeyError:
            continue

    return args

def _create_HTML_report(args, biophysical_outputs, valuation_outputs):
    html_uri = os.path.join(
        args['workspace_dir'], 'output',
        'summary%s.html' % carbon_utils.make_suffix(args))

    doc = html.HTMLDocument(html_uri, 'Carbon Results',
                            'InVEST Carbon Model Results')

    doc.write_paragraph(_make_report_intro(args))

    doc.insert_table_of_contents()

    if args['do_biophysical']:
        doc.write_header('Biophysical Results')
        doc.add(_make_biophysical_table(biophysical_outputs))
        if 'uncertainty' in biophysical_outputs:
            doc.write_header('Uncertainty Results', level=3)
            for paragraph in _make_biophysical_uncertainty_intro():
                doc.write_paragraph(paragraph)
            doc.add(_make_biophysical_uncertainty_table(
                    biophysical_outputs['uncertainty']))

    if args['do_valuation']:
        doc.write_header('Valuation Results')
        for paragraph in _make_valuation_intro(args):
            doc.write_paragraph(paragraph)
        for table in _make_valuation_tables(valuation_outputs):
            doc.add(table)
        if 'uncertainty_data' in valuation_outputs:
            doc.write_header('Uncertainty Results', level=3)
            for paragraph in _make_valuation_uncertainty_intro():
                doc.write_paragraph(paragraph)
            doc.add(_make_valuation_uncertainty_table(
                    valuation_outputs['uncertainty_data']))

    doc.write_header('Output Files')
    doc.write_paragraph(
        'This run of the carbon model produced the following output files.')
    doc.add(_make_outfile_table(
            args, biophysical_outputs, valuation_outputs, html_uri))

    doc.flush()

def _make_report_intro(args):
    models = []
    for model in 'biophysical', 'valuation':
        if args['do_%s' % model]:
            models.append(model)

    return ('This document summarizes the results from running the InVEST '
            'carbon model. This run of the model involved the %s %s.' %
            (' and '.join(models),
             'model' if len(models) == 1 else 'models'))

def _make_biophysical_uncertainty_intro():
    return [
        'This data was computed by doing a Monte Carlo '
        'simulation, which involved %d runs of the model.' %
        carbon_biophysical.NUM_MONTE_CARLO_RUNS,
        'For each run of the simulation, the amount of carbon '
        'per grid cell for each LULC type was independently sampled '
        'from the normal distribution given in the input carbon pools. '
        'Given this set of carbon pools, the model computed the amount of '
        'carbon in each scenario, and computed sequestration by subtracting '
        'the carbon storage in different scenarios. ',
        'Results across all Monte Carlo simulation runs were '
        'analyzed to produce the following mean and standard deviation data.',
        'All uncertainty analysis in this model assumes that true carbon pool '
        'values for different LULC types are independently distributed, '
        'with no systematic bias. If there is systematic bias in the carbon '
        'pool estimates, then actual standard deviations for results may be '
        'larger than reported in the following table.']

def _make_biophysical_uncertainty_table(uncertainty_results):
    table = html.Table(id='biophysical_uncertainty')
    table.add_two_level_header(
        outer_headers=['Total carbon (Mg of carbon)',
                       'Sequestered carbon (compared to current scenario)'
                       '<br>(Mg of carbon)'],
        inner_headers=['Mean', 'Standard deviation'],
        row_id_header='Scenario')

    for scenario in ['cur', 'fut', 'redd']:
        if scenario not in uncertainty_results:
            continue

        row = [_make_scenario_name(scenario, 'redd' in uncertainty_results)]
        row += uncertainty_results[scenario]

        if scenario == 'cur':
            row += ['n/a', 'n/a']
        else:
            row += uncertainty_results['sequest_%s' % scenario]

        table.add_row(row)

    return table

def _make_biophysical_table(biophysical_outputs):
    do_uncertainty = 'uncertainty' in biophysical_outputs

    table = html.Table(id='biophysical_table')
    headers = ['Scenario', 'Total carbon<br>(Mg of carbon)',
               'Sequestered carbon<br>(compared to current scenario)'
               '<br>(Mg of carbon)']

    table.add_row(headers, is_header=True)

    for scenario in ['cur', 'fut', 'redd']:
        total_carbon_key = 'tot_C_%s' % scenario
        if total_carbon_key not in biophysical_outputs:
            continue

        row = []
        row.append(
            _make_scenario_name(scenario, 'tot_C_redd' in biophysical_outputs))

        # Append total carbon.
        row.append(carbon_utils.sum_pixel_values_from_uri(
                biophysical_outputs[total_carbon_key]))

        # Append sequestration.
        sequest_key = 'sequest_%s' % scenario
        if sequest_key in biophysical_outputs:
            row.append(carbon_utils.sum_pixel_values_from_uri(
                    biophysical_outputs[sequest_key]))
        else:
            row.append('n/a')

        table.add_row(row)

    return table

def _make_valuation_tables(valuation_outputs):
    scenario_results = {}
    change_table = html.Table(id='change_table')
    change_table.add_row(["Scenario",
                          "Sequestered carbon<br>(Mg of carbon)",
                          "Net present value<br>(USD)"],
                         is_header=True)

    for scenario_type in ['base', 'redd']:
        try:
            sequest_uri = valuation_outputs['sequest_%s' % scenario_type]
        except KeyError:
            # We may not be doing REDD analysis.
            continue

        scenario_name = _make_scenario_name(
            scenario_type, 'sequest_redd' in valuation_outputs)

        total_seq = carbon_utils.sum_pixel_values_from_uri(sequest_uri)
        total_val = carbon_utils.sum_pixel_values_from_uri(
            valuation_outputs['%s_val' % scenario_type])
        scenario_results[scenario_type] = (total_seq, total_val)
        change_table.add_row([scenario_name, total_seq, total_val])

        try:
            seq_mask_uri = valuation_outputs['%s_seq_mask' % scenario_type]
            val_mask_uri = valuation_outputs['%s_val_mask' % scenario_type]
        except KeyError:
            # We may not have confidence-masking data.
            continue

        # Compute output for confidence-masked data.
        masked_seq = carbon_utils.sum_pixel_values_from_uri(seq_mask_uri)
        masked_val = carbon_utils.sum_pixel_values_from_uri(val_mask_uri)
        scenario_results['%s_mask' % scenario_type] = (masked_seq, masked_val)
        change_table.add_row(['%s (confident cells only)' % scenario_name,
                              masked_seq,
                              masked_val])

    yield change_table

    # If REDD scenario analysis is enabled, write the table
    # comparing the baseline and REDD scenarios.
    if 'base' in scenario_results and 'redd' in scenario_results:
        comparison_table = html.Table(id='comparison_table')
        comparison_table.add_row(
            ["Scenario Comparison",
             "Difference in carbon stocks<br>(Mg of carbon)",
             "Difference in net present value<br>(USD)"],
            is_header=True)

        # Add a row with the difference in carbon and in value.
        base_results = scenario_results['base']
        redd_results = scenario_results['redd']
        comparison_table.add_row(
            ['%s vs %s' % (_make_scenario_name('redd'),
                           _make_scenario_name('base')),
             redd_results[0] - base_results[0],
             redd_results[1] - base_results[1]
             ])

        if 'base_mask' in scenario_results and 'redd_mask' in scenario_results:
            # Add a row with the difference in carbon and in value for the
            # uncertainty-masked scenario.
            base_mask_results = scenario_results['base_mask']
            redd_mask_results = scenario_results['redd_mask']
            comparison_table.add_row(
                ['%s vs %s (confident cells only)'
                 % (_make_scenario_name('redd'),
                    _make_scenario_name('base')),
                 redd_mask_results[0] - base_mask_results[0],
                 redd_mask_results[1] - base_mask_results[1]
                 ])

        yield comparison_table


def _make_valuation_uncertainty_intro():
    return [
        'These results were computed by using the uncertainty data from the '
        'Monte Carlo simulation in the biophysical model.'
        ]


def _make_valuation_uncertainty_table(uncertainty_data):
    table = html.Table(id='valuation_uncertainty')

    table.add_two_level_header(
        outer_headers=['Sequestered carbon (Mg of carbon)',
                       'Net present value (USD)'],
        inner_headers=['Mean', 'Standard Deviation'],
        row_id_header='Scenario')

    for fut_type in ['fut', 'redd']:
        if fut_type not in uncertainty_data:
            continue

        scenario_data = uncertainty_data[fut_type]
        row = [_make_scenario_name(fut_type, 'redd' in uncertainty_data)]
        row += scenario_data['sequest']
        row += scenario_data['value']
        table.add_row(row)

    return table


def _make_valuation_intro(args):
    intro = [
        '<strong>Positive values</strong> in this table indicate that '
        'carbon storage increased. In this case, the positive Net Present '
        'Value represents the value of the sequestered carbon.',
        '<strong>Negative values</strong> indicate that carbon storage '
        'decreased. In this case, the negative Net Present Value represents '
        'the cost of carbon emission.'
        ]

    if args['do_uncertainty']:
        intro.append(
            'Entries in the table with the label "confident cells only" '
            'represent results for sequestration and value if we consider '
            'sequestration that occurs only in those cells where we are '
            'confident that carbon storage will either increase or decrease.')

    return intro


def _make_outfile_table(args, biophysical_outputs, valuation_outputs, html_uri):
    table = html.Table(id='outfile_table')
    table.add_row(['Filename', 'Description'], is_header=True)

    descriptions = collections.OrderedDict()

    if biophysical_outputs:
        descriptions.update(_make_biophysical_outfile_descriptions(
                biophysical_outputs, args))

    if valuation_outputs:
        descriptions.update(_make_valuation_outfile_descriptions(
                valuation_outputs))

    html_filename = os.path.basename(html_uri)
    descriptions[html_filename] = 'This summary file.' # dude, that's so meta

    for filename, description in sorted(descriptions.items()):
        table.add_row([filename, description])

    return table


def _make_biophysical_outfile_descriptions(outfile_uris, args):
    '''Return a dict with descriptions of biophysical outfiles.'''

    def name(scenario_type):
        return _make_scenario_name(scenario_type,
                                  do_redd=('tot_C_redd' in outfile_uris),
                                  capitalize=False)

    def total_carbon_description(scenario_type):
        return ('Maps the total carbon stored in the %s scenario, in '
                'Mg per grid cell.') % name(scenario_type)

    def sequest_description(scenario_type):
        return ('Maps the sequestered carbon in the %s scenario, relative to '
                'the %s scenario, in Mg per grid cell.') % (
            name(scenario_type), name('cur'))

    def conf_description(scenario_type):
        return ('Maps confident areas for carbon sequestration and emissions '
                'between the current scenario and the %s scenario. '
                'Grid cells where we are at least %.2f%% confident that '
                'carbon storage will increase have a value of 1. Grid cells '
                'where we are at least %.2f%% confident that carbon storage will '
                'decrease have a value of -1. Grid cells with a value of 0 '
                'indicate regions where we are not %.2f%% confident that carbon '
                'storage will either increase or decrease.') % (
            tuple([name(scenario_type)] + [args['confidence_threshold']] * 3))

    file_key_to_func = {
        'tot_C_%s': total_carbon_description,
        'sequest_%s': sequest_description,
        'conf_%s': conf_description
        }

    return _make_outfile_descriptions(outfile_uris, ['cur', 'fut', 'redd'],
                                     file_key_to_func)

def _make_valuation_outfile_descriptions(outfile_uris):
    '''Return a dict with descriptions of valuation outfiles.'''

    def name(scenario_type):
        return _make_scenario_name(scenario_type,
                                  do_redd=('sequest_redd' in outfile_uris),
                                  capitalize=False)

    def value_file_description(scenario_type):
        return ('Maps the economic value of carbon sequestered between the '
                'current and %s scenarios, with values in dollars per grid '
                'cell.') % name(scenario_type)

    def value_mask_file_description(scenario_type):
        return ('Maps the economic value of carbon sequestered between the '
                'current and %s scenarios, but only for cells where we are '
                'confident that carbon storage will either increase or '
                'decrease.') % name(scenario_type)

    def carbon_mask_file_description(scenario_type):
        return ('Maps the increase in carbon stored between the current and '
                '%s scenarios, in Mg per grid cell, but only for cells where '
                ' we are confident that carbon storage will either increase or '
                'decrease.') % name(scenario_type)

    file_key_to_func = {
        '%s_val': value_file_description,
        '%s_seq_mask': carbon_mask_file_description,
        '%s_val_mask': value_mask_file_description
        }

    return _make_outfile_descriptions(outfile_uris, ['base', 'redd'],
                                     file_key_to_func)


def _make_outfile_descriptions(outfile_uris, scenarios, file_key_to_func):
    descriptions = collections.OrderedDict()
    for scenario_type in scenarios:
        for file_key, description_func in file_key_to_func.items():
            try:
                uri = outfile_uris[file_key % scenario_type]
            except KeyError:
                continue

            filename = os.path.basename(uri)
            descriptions[filename] = description_func(scenario_type)

    return descriptions


def _make_scenario_name(scenario, do_redd=True, capitalize=True):
    names = {
        'cur': 'current',
        'fut': 'baseline' if do_redd else 'future',
        'redd': 'REDD policy'
        }
    names['base'] = names['fut']
    name = names[scenario]
    if capitalize:
        return name[0].upper() + name[1:]
    return name


def make_suffix(model_args):
    '''Return the suffix from the args (prepending '_' if necessary).'''
    try:
        file_suffix = model_args['suffix']
        if not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''
    return file_suffix


def setup_dirs(workspace_dir, *dirnames):
    '''Create the requested directories, and return the pathnames.'''
    dirs = {name: os.path.join(workspace_dir, name) for name in dirnames}
    for new_dir in dirs.values():
        if not os.path.exists(new_dir):
            LOGGER.debug('Creating directory %s', new_dir)
            os.makedirs(new_dir)
    if len(dirs) == 1:
        return dirs.values()[0]
    return dirs


def sum_pixel_values_from_uri(uri):
    '''Return the sum of the values of all pixels in the given file.'''
    dataset = gdal.Open(uri)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    total_sum = 0.0
    # Loop over each row in out_band
    for row_index in range(band.YSize):
        row_array = band.ReadAsArray(0, row_index, band.XSize, 1)
        total_sum += numpy.sum(row_array[row_array != nodata])
    return total_sum


def execute_valuation(args):
    """This function calculates carbon sequestration valuation.

        args - a python dictionary with at the following *required* entries:

        args['workspace_dir'] - a uri to the directory that will write output
            and other temporary files during calculation. (required)
        args['suffix'] - a string to append to any output file name (optional)
        args['sequest_uri'] - is a uri to a GDAL raster dataset describing the
            amount of carbon sequestered (baseline scenario, if this is REDD)
        args['sequest_redd_uri'] (optional) - uri to the raster dataset for
            sequestration under the REDD policy scenario
        args['conf_uri'] (optional) - uri to the raster dataset indicating
            confident pixels for sequestration or emission
        args['conf_redd_uri'] (optional) - as above, but for the REDD scenario
        args['carbon_price_units'] - a string indicating whether the price is
            in terms of carbon or carbon dioxide. Can value either as
            'Carbon (C)' or 'Carbon Dioxide (CO2)'.
        args['V'] - value of a sequestered ton of carbon or carbon dioxide in
            dollars per metric ton
        args['r'] - the market discount rate in terms of a percentage
        args['c'] - the annual rate of change in the price of carbon
        args['yr_cur'] - the year at which the sequestration measurement
            started
        args['yr_fut'] - the year at which the sequestration measurement ended

        returns a dict with output file URIs."""

    output_directory = carbon_utils.setup_dirs(args['workspace_dir'], 'output')

    if args['carbon_price_units'] == 'Carbon Dioxide (CO2)':
        #Convert to price per unit of Carbon do this by dividing
        #the atomic mass of CO2 (15.9994*2+12.0107) by the atomic
        #mass of 12.0107.  Values gotten from the periodic table of
        #elements.
        args['V'] *= (15.9994*2+12.0107)/12.0107

    LOGGER.info('Constructing valuation formula.')
    n = args['yr_fut'] - args['yr_cur'] - 1
    ratio = 1.0 / ((1 + args['r'] / 100.0) * (1 + args['c'] / 100.0))
    valuation_constant = args['V'] / (args['yr_fut'] - args['yr_cur']) * \
        (1.0 - ratio ** (n + 1)) / (1.0 - ratio)

    nodata_out = -1.0e10

    outputs = _make_outfile_uris(output_directory, args)

    conf_uris = {}
    if args.get('conf_uri'):
        conf_uris['base'] = args['conf_uri']
    if args.get('conf_redd_uri'):
        conf_uris['redd'] = args['conf_redd_uri']

    for scenario_type in ['base', 'redd']:
        try:
            sequest_uri = outputs['sequest_%s' % scenario_type]
        except KeyError:
            # REDD analysis might not be enabled, so just keep going.
            continue

        LOGGER.info('Beginning valuation of %s scenario.', scenario_type)

        sequest_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(sequest_uri)

        def value_op(sequest):
            if sequest == sequest_nodata:
                return nodata_out
            return sequest * valuation_constant

        pixel_size_out = pygeoprocessing.geoprocessing.get_cell_size_from_uri(sequest_uri)
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [sequest_uri], value_op, outputs['%s_val' % scenario_type],
            gdal.GDT_Float32, nodata_out, pixel_size_out, "intersection")


        if scenario_type in conf_uris:
            LOGGER.info('Creating masked rasters for %s scenario.', scenario_type)
            # Produce a raster for sequestration, masking out uncertain areas.
            _create_masked_raster(sequest_uri, conf_uris[scenario_type],
                                  outputs['%s_seq_mask' % scenario_type])

            # Produce a raster for value sequestration,
            # again masking out uncertain areas.
            _create_masked_raster(
                outputs['%s_val' % scenario_type],
                conf_uris[scenario_type],
                outputs['%s_val_mask' % scenario_type])

    if 'uncertainty_data' in args:
        uncertainty_data = _compute_uncertainty_data(
            args['uncertainty_data'], valuation_constant)
        if uncertainty_data:
            outputs['uncertainty_data'] = uncertainty_data

    return outputs


def _make_outfile_uris(output_directory, args):
    '''Return a dict with uris for outfiles.'''
    file_suffix = carbon_utils.make_suffix(args)

    def outfile_uri(prefix, scenario_type='', filetype='tif'):
        '''Create the URI for the appropriate output file.'''
        if not args.get('sequest_redd_uri'):
            # We're not doing REDD analysis, so don't append anything,
            # since there's only one scenario.
            scenario_type = ''
        elif scenario_type:
            scenario_type = '_' + scenario_type
        filename = '%s%s%s.%s' % (prefix, scenario_type, file_suffix, filetype)
        return os.path.join(output_directory, filename)

    outfile_uris = collections.OrderedDict()

    # Value sequestration for base scenario.
    outfile_uris['base_val'] = outfile_uri('value_seq', 'base')

    # Confidence-masked rasters for base scenario.
    if args.get('conf_uri'):
        outfile_uris['base_seq_mask'] = outfile_uri('seq_mask', 'base')
        outfile_uris['base_val_mask'] = outfile_uri('val_mask', 'base')

    # Outputs for REDD scenario.
    if args.get('sequest_redd_uri'):
        # Value sequestration.
        outfile_uris['redd_val'] = outfile_uri('value_seq', 'redd')

        # Confidence-masked rasters for REDD scenario.
        if args.get('conf_redd_uri'):
            outfile_uris['redd_seq_mask'] = outfile_uri('seq_mask', 'redd')
            outfile_uris['redd_val_mask'] = outfile_uri('val_mask', 'redd')

    # These sequestration rasters are actually input files (not output files),
    # but it's convenient to have them in this dictionary.
    outfile_uris['sequest_base'] = args['sequest_uri']
    if args.get('sequest_redd_uri'):
        outfile_uris['sequest_redd'] = args['sequest_redd_uri']

    return outfile_uris


def _create_masked_raster(orig_uri, mask_uri, result_uri):
    '''Creates a raster at result_uri with some areas masked out.

    orig_uri -- uri of the original raster
    mask_uri -- uri of the raster to use as a mask
    result_uri -- uri at which the new raster should be created

    Masked data in the result file is denoted as no data (not necessarily zero).

    Data is masked out at locations where mask_uri is 0 or no data.
    '''
    nodata_orig = pygeoprocessing.geoprocessing.get_nodata_from_uri(orig_uri)
    nodata_mask = pygeoprocessing.geoprocessing.get_nodata_from_uri(mask_uri)
    def mask_op(orig_val, mask_val):
        '''Return orig_val unless mask_val indicates uncertainty.'''
        if mask_val == 0 or mask_val == nodata_mask:
            return nodata_orig
        return orig_val

    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(orig_uri)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [orig_uri, mask_uri], mask_op, result_uri, gdal.GDT_Float32,
        nodata_orig, pixel_size, 'intersection', dataset_to_align_index=0)

def _compute_uncertainty_data(biophysical_uncertainty_data, valuation_const):
    """Computes mean and standard deviation for sequestration value."""
    LOGGER.info('Computing uncertainty data.')
    results = {}
    for fut_type in ['fut', 'redd']:
        try:
            # Get the tuple with mean and standard deviation for sequestration.
            sequest = biophysical_uncertainty_data['sequest_%s' % fut_type]
        except KeyError:
            continue

        results[fut_type] = {}

        # Note sequestered carbon (mean and std dev).
        results[fut_type]['sequest'] = sequest

        # Compute the value of sequestered carbon (mean and std dev).
        results[fut_type]['value'] = tuple(carbon * valuation_const
                                           for carbon in sequest)
    return results
