"""The Crop Production module excutes the Crop Production model."""

import logging
import pprint as pp
import csv
import os
import copy
from collections import Counter
import tempfile
import shutil

import numpy as np
from osgeo import gdal

import pygeoprocessing.geoprocessing as geoprocess
from .. import utils as invest_utils


LOGGER = logging.getLogger('natcap.invest.crop_production.crop_production')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

_OUTPUT = {
    'nutrient_contents_table': 'nutritional_analysis.csv',
    'financial_analysis_table': 'financial_analysis.csv',
    'yield_raster': 'yield.tif'
}

_INTERMEDIATE = {
    'aoi_raster': 'aoi.tif',
    'irrigation_raster': 'irrigation.tif'
}


def execute(args):
    """Crop Production.

    :param str args['workspace_dir']: location into which all intermediate
        and output files should be placed.

    :param str args['results_suffix']: a string to append to output filenames

    :param str args['lookup_table']: filepath to a CSV table used to
        convert the crop code provided in the Crop Map to the crop name that
        can be used for searching through inputs and formatting outputs.

    :param str args['aoi_raster']: a GDAL-supported raster representing a
        crop management scenario.

    :param str args['dataset_dir']: the provided folder should contain
        a set of folders and data specified in the 'Running the Model' section
        of the model's User Guide.

    :param str args['yield_function']: the method used to compute crop yield.
        Can be one of three: 'observed', 'percentile', and 'regression'.

    :param str args['percentile_column']: for percentile yield function, the
        table column name must be provided so that the program can fetch the
        correct yield values for each climate bin.

    :param str args['fertilizer_dir']: path to folder that contains a set of
        GDAL-supported rasters representing the amount of Nitrogen (N),
        Phosphorous (P2O5), and Potash (K2O) applied to each area of land
        (kg/ha).

    :param str args['irrigation_raster']: filepath to a GDAL-supported
        raster representing whether irrigation occurs or not. A zero value
        indicates that no irrigation occurs.  A one value indicates that
        irrigation occurs.  If any other values are provided, irrigation is
        assumed to occur within that cell area.

    :param boolean args['compute_nutritional_contents']: if true, calculates
        nutrition from crop production and creates associated outputs.

    :param str args['nutrient_table']: filepath to a CSV table containing
        information about the nutrient contents of each crop.

    :param boolean args['compute_financial_analysis']: if true, calculates
        economic returns from crop production and creates associated outputs.

    :param str args['economics_table']: filepath to a CSV table containing
        information related to market price of a given crop and the costs
        involved with producing that crop.

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'results_suffix': 'scenario_name',
            'lookup_table': 'path/to/lookup_table',
            'aoi_raster': 'path/to/aoi_raster',
            'dataset_dir': 'path/to/dataset_dir/',
            'yield_function': 'regression',
            'percentile_column': 'yield_95th',
            'fertilizer_dir':'path/to/fertilizer_rasters_dir/',
            'irrigation_raster': 'path/to/is_irrigated_raster',
            'compute_nutritional_contents': True,
            'nutrient_table': 'path/to/nutrition_table',
            'compute_financial_analysis': True,
            'economics_table': 'path/to/economics_table'
        }
    """
    LOGGER.info("Beginning Model Run...")

    check_inputs(args)

    cache_dir = os.path.join(args['workspace_dir'], 'intermediate')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    output_dir = os.path.join(args['workspace_dir'], 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_registry = invest_utils.build_file_registry(
        [(_OUTPUT, output_dir), (_INTERMEDIATE, cache_dir)],
        args['results_suffix'])

    reproject_raster(
        args['aoi_raster'], args['aoi_raster'], file_registry['aoi_raster'])
    global_dataset_dict = get_global_dataset(args['dataset_dir'])
    lookup_dict = get_lookup_dict(
        file_registry['aoi_raster'], args['lookup_table'])

    fertilizer_dict = None
    if args['yield_function'] == 'observed':
        LOGGER.info("Calculating Yield from Observed Regional Data...")
        yield_dict = run_observed_yield(
            global_dataset_dict['observed'],
            cache_dir,
            file_registry['aoi_raster'],
            lookup_dict,
            file_registry['yield_raster'])
    elif args['yield_function'] == 'percentile':
        LOGGER.info("Calculating Yield from Climate-based Distribution of "
                    "Observed Yields...")
        yield_dict = run_percentile_yield(
            global_dataset_dict['climate_bin_maps'],
            global_dataset_dict['percentile'],
            cache_dir,
            file_registry['aoi_raster'],
            lookup_dict,
            file_registry['yield_raster'],
            args['percentile_column'])
    elif args['yield_function'] == 'regression':
        LOGGER.info("Calculating Yield from Regression Model with "
                    "Climate-based Parameters...")
        reproject_raster(
            args['irrigation_raster'],
            file_registry['aoi_raster'],
            file_registry['irrigation_raster'])
        fertilizer_dict = get_fertilizer_rasters(
            args['fertilizer_dir'], cache_dir, file_registry['aoi_raster'])
        yield_dict = run_regression_yield(
            global_dataset_dict['climate_bin_maps'],
            global_dataset_dict['regression'],
            cache_dir,
            file_registry['aoi_raster'],
            fertilizer_dict,
            file_registry['irrigation_raster'],
            lookup_dict,
            file_registry['yield_raster'])

    if args['compute_nutritional_contents'] == True:
        LOGGER.info("Calculating Nutritional Contents...")
        compute_nutritional_contents(
            yield_dict,
            lookup_dict,
            args['nutrient_table'],
            file_registry['nutrient_contents_table'])

    if args['compute_financial_analysis'] == True:
        LOGGER.info("Calculating Financial Analysis...")
        compute_financial_analysis(
            yield_dict,
            args['economics_table'],
            file_registry['aoi_raster'],
            lookup_dict,
            fertilizer_dict,
            file_registry['financial_analysis_table'])

    shutil.rmtree(cache_dir)
    LOGGER.info("...Model Run Complete.")


def check_inputs(args):
    """Check user provides inputs necessary for particular yield functions.

    Args:
        args (dict): user-provided arguments dictionary.
    """
    if args['yield_function'] == 'percentile':
        if 'percentile_column' not in args or \
                args['percentile_column'] in ['', None]:
            LOGGER.error('User must provide a percentile column for the '
                         'percentile yield function.')
            raise ValueError('percentile column must be provided.')
    elif args['yield_function'] == 'regression':
        if 'fertilizer_dir' not in args or not os.path.exists(
                args['fertilizer_dir']):
            LOGGER.error('User must provide a set of fertilizer application '
                         'rate rasters for the regression yield function.')
            raise ValueError('fertilizer directory must be provided.')
        if 'irrigation_raster' not in args or not os.path.exists(
                args['irrigation_raster']):
            LOGGER.error('User must provide an raster indicating whether cell '
                         'is irrigated or not  for the regression yield '
                         'function.')
            raise ValueError('irrigation raster must be provided.')


def get_files_in_dir(path):
    """Fetch mapping of files in directory.

    Each key in the mapping is the first part of the filename split by an
        underscore.
    Each value in the mapping is the filepath.

    Args:
        path (str): path to directory.

    Returns:
        files_dict (dict): dict([(filename, filepath), ...]).
    """
    base_dir = os.path.dirname(path)
    files = list(filter(
        lambda x: x.endswith('tif') or x.endswith('csv'), os.listdir(path)))
    return dict([(f.split('_')[0], os.path.join(path, f)) for f in files])


def get_global_dataset(dataset_dir):
    """Get global dataset.

    Args:
        dataset_dir (str): path to spatial dataset.

    Returns:
        dataset_dict (dict): tree-like structure of spatial dataset filenames
            and filepaths.
    """
    subdirs = {
        'climate_bin_maps': 'climate_bin_maps',
        'observed': 'observed_yield',
        'percentile': 'climate_percentile_yield',
        'regression': 'climate_regression_yield',
    }
    return dict([(k, get_files_in_dir(os.path.join(dataset_dir, v)))
                 for k, v in subdirs.items()])


def get_lookup_dict(aoi_raster, lookup_table):
    """Get lookup information for AOI.

    Args:
        aoi_raster (str): path to aoi raster.
        lookup_table (str): path to lookup table.

    Returns:
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
    """
    iterblocks = geoprocess.iterblocks(aoi_raster)
    lookup_dict = geoprocess.get_lookup_from_table(lookup_table, 'code')

    s = set(geoprocess.unique_raster_values_uri(aoi_raster))
    t = set(lookup_dict.keys())

    if set() != s-t:
        u = s-t
        LOGGER.warn("raster contains values not in lookup table: %s" % u)

    # delete lookup items not in raster
    for i in (t-s):
        del lookup_dict[i]

    for code in lookup_dict.keys():
        lookup_dict[code]['name'] = lookup_dict[code]['name'].lower()
        lookup_dict[code]['is_crop'] = lookup_dict[code]['is_crop'].lower()

        # delete lookup items that are not crops
        if lookup_dict[code]['is_crop'] != 'true':
            del lookup_dict[code]

    return lookup_dict


def reproject_raster(src_path, template_path, dst_path):
    """Reproject raster.

    Block-size set to 256 x 256.

    Args:
        src_path (str): path to source raster.
        template_path (str): path to template raster.
        dst_path (str): path to destination raster.
    """
    # Source
    src = gdal.Open(src_path, 0)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # Match
    match_ds = gdal.Open(template_path, 0)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Destination
    block_size = [256, 256]
    opt = ['TILED=YES', 'BLOCKXSIZE=%d' % block_size[0],
           'BLOCKYSIZE=%d' % block_size[1]]
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(dst_path, wide, high, 1, gdal.GDT_Float32, options=opt)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    # Reproject
    gdal.ReprojectImage(src, dst, src_proj, match_proj, 0)
    dst = None


def reproject_global_rasters(global_dataset_dict, cache_dir, aoi_raster,
                             lookup_dict):
    """Reproject global rasters.

    Args:
        global_dataset_dict (dict): mapping of crops to their respective data
            filepaths.
        cache_dir (str): path to directory in which to store reprojected
            rasters.
        aoi_raster (str): path to aoi raster.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.

    Returns:
        observed_yield_dict (dict): mapping of crops to observed yield rasters.
    """
    crops = [v['name'] for v in lookup_dict.values() if v['is_crop'] == 'true']
    crop_to_code_dict = dict(
        (val['name'], code) for code, val in lookup_dict.items())

    observed_yield_dict = {}
    for crop in crops:
        dst_path = os.path.join(cache_dir, crop + '.tif')
        code = crop_to_code_dict[crop]
        observed_yield_dict[code] = dst_path
        reproject_raster(
            global_dataset_dict[crop],
            aoi_raster,
            dst_path)

    return observed_yield_dict


def write_to_raster(output_raster, array, xoff, yoff):
    """Write numpy array to raster block.

    Args:
        output_raster (str): filepath to output raster.
        array (np.array): block to save to raster.
        xoff (int): offset index for x-dimension.
        yoff (int): offset index for y-dimension.
    """
    ds = gdal.Open(output_raster, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    band.WriteArray(array, xoff, yoff)
    ds = None


def read_from_raster(input_raster, offset_block):
    """Read numpy array from raster block.

    Args:
        input_raster (str): filepath to input raster.
        offset_block (dict): dictionary of offset information.  Keys in the
            dictionary include 'xoff', 'yoff', 'win_xsize', and 'win_ysize'.

    Returns:
        array (np.array): a blocked array of the input raster.
    """
    ds = gdal.Open(input_raster)
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray(**offset_block)
    ds = None
    return array


def compute_observed_yield(aoi_raster, lookup_dict, observed_yield_dict,
                           yield_raster):
    """Compute observed yield.

    Args:
        aoi_raster (str): path to aoi raster.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        observed_yield_dict (dict): mapping of crops to observed yield rasters.
        yield_raster (str): path to output directory.

    Returns:
        yield_dict (collections.Counter): mapping from crop code to total
            yield.
    """
    geoprocess.new_raster_from_base_uri(
        aoi_raster, yield_raster, 'GTiff', -9999, gdal.GDT_Float32)
    cell_size = geoprocess.get_cell_size_from_uri(aoi_raster)
    m2_per_cell = cell_size ** 2
    ha_per_m2 = 0.0001
    ha_per_cell = ha_per_m2 * m2_per_cell

    crop_to_code_dict = dict(
        [(val['name'], code) for code, val in lookup_dict.items()])
    iterate_aoi = geoprocess.iterblocks(aoi_raster)

    yield_dict = Counter()
    for aoi_offset, aoi_block in iterate_aoi:
        accum_block = np.zeros(aoi_block.shape)
        for code, crop_raster in observed_yield_dict.items():
            observed_yield_block = read_from_raster(crop_raster, aoi_offset)
            observed_yield_block[observed_yield_block < 0] = 0.
            aoi_mask = np.where(aoi_block == code, 1., 0.)
            yield_ = aoi_mask * observed_yield_block * ha_per_cell
            yield_dict[code] += yield_.sum()
            accum_block += yield_
        write_to_raster(
            yield_raster,
            accum_block,
            aoi_offset['xoff'],
            aoi_offset['yoff'])

    return yield_dict


def reclass(array, d, nodata=0.):
    """Reclassify values in numpy ndarray.

    Values in array that are not in d are reclassed to np.nan.

    Args:
        array (np.array): input data.
        d (dict): reclassification map.
        nodata (float): reclass value for number not provided in
            reclassification map.

    Returns:
        reclass_array (np.array): reclassified array.
    """
    u = np.unique(array)
    has_map = np.in1d(u, d.keys())

    reclass_array = array.copy()
    for i in u[~has_map]:
        reclass_array = np.where(array == i, nodata, reclass_array)
    d[nodata] = nodata
    a_ravel = reclass_array.ravel()
    k = sorted(d.keys())
    v = np.array([d[key] for key in k])
    index = np.digitize(a_ravel, k, right=True)
    reclass_array = v[index].reshape(array.shape)

    return reclass_array


def get_percentile_yields(percentile_tables, lookup_dict):
    """Get percentile yield information.

    Args:
        percentile_tables (dict): mapping of crops to their respective tables.
            filepaths.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.

    Returns:
        percentile_yields_dict (dict): mapping of crops to their respective
            information.
    """
    crops = [v['name'] for v in lookup_dict.values() if v['is_crop'] == 'true']
    crop_to_code_dict = dict(
        [(val['name'], code) for code, val in lookup_dict.items()])

    percentile_yields_dict = {}
    for crop in crops:
        percentile_yields = geoprocess.get_lookup_from_table(
            percentile_tables[crop], 'climate_bin')
        for k, v in percentile_yields.items():
            del v['climate_bin']
            percentile_yields[k] = v
        percentile_yields_dict[crop_to_code_dict[crop]] = percentile_yields

    return percentile_yields_dict


def compute_percentile_yield(aoi_raster, lookup_dict, climate_bin_dict,
                             percentile_yield_dict, yield_raster,
                             percentile_yield):
    """Compute yield using percentile method.

    Args:
        aoi_raster (str): path to aoi raster.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        climate_bin_dict (dict): mapping of codes to climate bin rasters.
        percentile_yields_dict (dict): mapping of crops to their respective
            information.
        yield_raster (str): path to output raster.
        percentile_yield (str): selected yield percentile.

    Returns:
        yield_dict (collections.Counter): mapping from crop code to total
            yield.
    """
    reclass_dict = {}
    for code, yield_dict in percentile_yield_dict.items():
        reclass_dict[code] = dict(
            [(bin_, v[percentile_yield]) for bin_, v in yield_dict.items()])

    geoprocess.new_raster_from_base_uri(
        aoi_raster, yield_raster, 'GTiff', -9999, gdal.GDT_Float32)
    cell_size = geoprocess.get_cell_size_from_uri(aoi_raster)
    m2_per_cell = cell_size ** 2
    ha_per_m2 = 0.0001
    ha_per_cell = ha_per_m2 * m2_per_cell

    crop_to_code_dict = dict(
        [(val['name'], code) for code, val in lookup_dict.items()])
    iterate_aoi = geoprocess.iterblocks(aoi_raster)

    yield_dict = Counter()
    for aoi_offset, aoi_block in iterate_aoi:
        accum_block = np.zeros(aoi_block.shape)
        for code, climate_bin_raster in climate_bin_dict.items():
            climate_bin_block = read_from_raster(
                climate_bin_raster, aoi_offset)
            yield_block = reclass(climate_bin_block, reclass_dict[code])
            aoi_mask = np.where(aoi_block == code, 1., 0.)
            yield_ = aoi_mask * yield_block * ha_per_cell
            yield_dict[code] += yield_.sum()
            accum_block += yield_
        write_to_raster(
            yield_raster, accum_block, aoi_offset['xoff'], aoi_offset['yoff'])

    return yield_dict


def create_map(d, sub_dict_key):
    """"Shorten nested dictionary into a one-to-one mapping.

    Args:
        d (dict): nested dictionary.
        sub_dict_key (object): key in sub-dictionary whose value becomes value
            in return dictionary.

    Returns:
        one_to_one_dict (dict): dictionary that is a one-to-one mapping.
    """
    return dict([(k, v[sub_dict_key]) for k, v in d.items()])


def get_regression_coefficients(regression_tables, lookup_dict):
    """Get regression coefficients.

    Args:
        regression_tables (dict): mapping of codes to regression coeffeicent
            tables.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.

    Returns:
        regression_coefficients_dict (dict): nested dictionary of regression
            coefficients for each crop code.
    """
    crops = [v['name'] for v in lookup_dict.values() if v['is_crop'] == 'true']
    crop_to_code_dict = dict(
        [(val['name'], code) for code, val in lookup_dict.items()])

    regression_coefficients_dict = {}
    for crop in crops:
        regression_coefficients = geoprocess.get_lookup_from_table(
            regression_tables[crop], 'climate_bin')
        for k, v in regression_coefficients.items():
            del v['climate_bin']
            for k2, v2 in v.items():
                if v2 == '':
                    v[k2] = np.nan
            regression_coefficients[k] = v
        regression_coefficients_dict[crop_to_code_dict[crop]] = \
            regression_coefficients

    return regression_coefficients_dict


def get_fertilizer_rasters(fertilizer_dir, cache_dir, aoi_raster):
    """Get fertilizer rasters.

    Args:
        fertilizer_dir (str): path to directory containing fertilizer rasters.
        cache_dir (str): path to cache directory.
        aoi_raster (str): path to aoi raster.

    Returns:
        fertilizer_dict (dict): mapping of fertilizers to their respective
            raster paths.
    """
    fertilizer_types = set(['potash', 'phosphorus', 'nitrogen'])
    files = list(filter(
        lambda x: x.endswith('tif'), os.listdir(fertilizer_dir)))
    orig_fertilizer_dict = dict(
        [(f.split('.')[0], os.path.join(fertilizer_dir, f)) for f in files])
    fertilizer_dict = {}
    for fertilizer, fertilizer_raster in orig_fertilizer_dict.items():
        if fertilizer in fertilizer_types:
            dst_path = os.path.join(cache_dir, fertilizer)
            reproject_raster(fertilizer_raster, aoi_raster, dst_path)
            fertilizer_dict[fertilizer] = dst_path
    return fertilizer_dict


def compute_regression_yield(aoi_raster, lookup_dict, climate_bin_dict,
                             regression_coefficient_dict, fertilizer_dict,
                             irrigation_raster, yield_raster):
    """Compute regression yield.

    Args:
        aoi_raster (str): path to aoi raster.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        climate_bin_dict (dict): mapping of codes to climate bin rasters.
        fertilizer_dir (str): path to directory containing fertilizer rasters.
        regression_coefficients_dict (dict): nested dictionary of regression
            coefficients for each crop code.
        irrigation_raster (str): path to is_irrigated raster.
        yield_raster (str): path to output raster.

    Returns:
        yield_dict (collections.Counter): mapping from crop code to total
            yield.
    """
    geoprocess.new_raster_from_base_uri(
        aoi_raster, yield_raster, 'GTiff', -9999, gdal.GDT_Float32)
    cell_size = geoprocess.get_cell_size_from_uri(aoi_raster)
    m2_per_cell = cell_size ** 2
    ha_per_m2 = 0.0001
    ha_per_cell = ha_per_m2 * m2_per_cell

    crop_to_code_dict = dict(
        [(val['name'], code) for code, val in lookup_dict.items()])
    iterate_aoi = geoprocess.iterblocks(aoi_raster)

    yield_dict = Counter()
    for aoi_offset, aoi_block in iterate_aoi:
        accum_block = np.zeros(aoi_block.shape)
        for code, climate_bin_raster in climate_bin_dict.items():
            climate_bin_block = read_from_raster(
                climate_bin_raster, aoi_offset)
            yc = reclass(climate_bin_block, create_map(
                regression_coefficient_dict[code], 'yield_ceiling'))
            yc_rf = reclass(climate_bin_block, create_map(
                regression_coefficient_dict[code], 'yield_ceiling_rf'))
            b_nut = reclass(climate_bin_block, create_map(
                regression_coefficient_dict[code], 'b_nut'))
            b_K2O = reclass(climate_bin_block, create_map(
                regression_coefficient_dict[code], 'b_k2o'))
            c_N = reclass(climate_bin_block, create_map(
                regression_coefficient_dict[code], 'c_n'))
            c_P2O5 = reclass(climate_bin_block, create_map(
                regression_coefficient_dict[code], 'c_p2o5'))
            c_K2O = reclass(climate_bin_block, create_map(
                regression_coefficient_dict[code], 'c_k2o'))
            N_block = read_from_raster(fertilizer_dict['nitrogen'], aoi_offset)
            P_block = read_from_raster(
                fertilizer_dict['phosphorus'], aoi_offset)
            K_block = read_from_raster(fertilizer_dict['potash'], aoi_offset)
            Is_Irr_block = read_from_raster(
                irrigation_raster, aoi_offset).astype(int)
            Is_Irr_block[Is_Irr_block != 0] = 1
            PctMaxYield_N = 1 - (b_nut * (np.e ** (-c_N * N_block)))
            PctMaxYield_P = 1 - (b_nut * (np.e ** (-c_P2O5 * P_block)))
            PctMaxYield_K = 1 - (b_K2O * (np.e ** (-c_K2O * K_block)))
            PercentMaxYield = np.fmax(np.fmin(np.fmin(
                PctMaxYield_N, PctMaxYield_P), PctMaxYield_K), 0)
            MaxYield = PercentMaxYield * yc
            MaxYield = np.fmin(MaxYield, yc)
            MaxYieldRainFed = np.fmin(yc_rf, MaxYield)
            Is_RF_block = reclass(Is_Irr_block, {1: 0, 0: 1})
            Yield_block = (MaxYieldRainFed * Is_RF_block) + (
                MaxYield * Is_Irr_block)
            aoi_mask = np.where(aoi_block == code, 1., 0.)
            Yield_ = Yield_block * aoi_mask * ha_per_cell
            yield_dict[code] += Yield_.sum()
            accum_block += Yield_

        write_to_raster(
            yield_raster, accum_block, aoi_offset['xoff'], aoi_offset['yoff'])

    return yield_dict


def run_observed_yield(
        global_dataset_dict, cache_dir, aoi_raster, lookup_dict, yield_raster):
    """Run observed yield model.

    Args:
        global_dataset_dict (dict): mapping of crops to their respective data
            filepaths.
        cache_dir (str): path to cache directory.
        aoi_raster (str): path to aoi raster.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        yield_raster (str): path to output raster.

    Returns:
        yield_dict (collections.Counter): mapping from crop code to total
            yield.
    """
    observed_cache_dir = os.path.join(cache_dir, 'observed')
    os.makedirs(observed_cache_dir)
    observed_yield_dict = reproject_global_rasters(
        global_dataset_dict,
        observed_cache_dir,
        aoi_raster,
        lookup_dict)
    return compute_observed_yield(
        aoi_raster, lookup_dict, observed_yield_dict, yield_raster)


def run_percentile_yield(climate_bin_maps, percentile_tables, cache_dir,
                         aoi_raster, lookup_dict, yield_raster,
                         percentile_yield):
    """Run percentile yield model.

    Args:
        climate_bin_dict (dict): mapping of codes to climate bin rasters.
        percentile_tables (dict): mapping of crops to their respective tables.
            filepaths.
        cache_dir (str): path to cache directory.
        aoi_raster (str): path to aoi raster.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        yield_raster (str): path to output raster.
        percentile_yield (str): selected yield percentile.

    Returns:
        yield_dict (collections.Counter): mapping from crop code to total
            yield.
    """
    percentile_cache_dir = os.path.join(cache_dir, 'percentile')
    os.makedirs(percentile_cache_dir)
    climate_bin_dict = reproject_global_rasters(
        climate_bin_maps,
        percentile_cache_dir,
        aoi_raster,
        lookup_dict)
    percentile_yield_dict = get_percentile_yields(
        percentile_tables, lookup_dict)
    return compute_percentile_yield(
        aoi_raster,
        lookup_dict,
        climate_bin_dict,
        percentile_yield_dict,
        yield_raster,
        percentile_yield)


def run_regression_yield(climate_bin_maps, regression_tables, cache_dir,
                         aoi_raster, fertilizer_dict, irrigation_raster,
                         lookup_dict, yield_raster):
    """Run regression yield model.

    Args:
        climate_bin_maps (dict): mapping of crops to climate bin rasters.
        regression_tables (dict): mapping of codes to regression coeffeicent
            tables.
        cache_dir (str): path to cache directory.
        aoi_raster (str): path to aoi raster.
        fertilizer_dict (dict): mapping of fertilizers to their respective
            raster paths.
        irrigation_raster (str): path to intermediate is_irrigated raster.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        yield_raster (str): path to output raster.

    Returns:
        yield_dict (collections.Counter): mapping from crop code to total
            yield.
    """
    regression_cache_dir = os.path.join(cache_dir, 'regression')
    os.makedirs(regression_cache_dir)
    climate_bin_dict = reproject_global_rasters(
        climate_bin_maps,
        regression_cache_dir,
        aoi_raster,
        lookup_dict)
    regression_coefficient_dict = get_regression_coefficients(
        regression_tables, lookup_dict)
    return compute_regression_yield(
        aoi_raster,
        lookup_dict,
        climate_bin_dict,
        regression_coefficient_dict,
        fertilizer_dict,
        irrigation_raster,
        yield_raster)


def compute_nutritional_contents(yield_dict, lookup_dict, nutrient_table,
                                 nutritional_contents_table):
    """Compute nutritional contents of crop yields.

    Args:
        yield_dict (collections.Counter): mapping from crop code to total
            yield.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        nutrient_table (str): path to table containing information about the
            nutrient contents of each crop.
        nutritional_contents_table (str): path to output table.
    """
    codes = yield_dict.keys()
    crop_to_code_dict = dict(
        [(val['name'], code) for code, val in lookup_dict.items()])
    code_to_crop_dict = dict([(v, k) for k, v in crop_to_code_dict.items()])

    nutrients_dict = geoprocess.get_lookup_from_table(nutrient_table, 'crop')
    for k, v in nutrients_dict.items():
        del v['crop']

    d = {}
    for code, total_yield in yield_dict.items():
        name = code_to_crop_dict[code]
        crop_nutritional_contents_dict = {}
        crop_nutrients_dict = nutrients_dict[name]
        for nutrient, nutrient_amount in crop_nutrients_dict.items():
            contents = total_yield * nutrient_amount
            crop_nutritional_contents_dict[nutrient] = ("%.2f" % contents)
        crop_nutritional_contents_dict['total_yield'] = ("%.2f" % total_yield)
        d[name] = crop_nutritional_contents_dict

    for i in d.keys():
        d[i]['crop'] = i
    csvfile = open(nutritional_contents_table, 'w')
    fieldnames = d[d.keys()[0]].keys()
    fieldnames.remove('crop')
    fieldnames.remove('total_yield')
    fieldnames.sort()
    fieldnames.insert(0, 'total_yield')
    fieldnames.insert(0, 'crop')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in d.values():
        writer.writerow(row)


def calc_fertilizer_costs(code, code_dict, aoi_raster, fertilizer_dict):
    """Calculate fertilizer application rate costs.

    Args:
        code (int): crop code.
        code_dict (dict): economic information of crop.
        aoi_raster (str): path to aoi raster.
        fertilizer_dict (dict): mapping of fertilizers to their respective
            raster paths.

    Returns:
        fertilizer_costs (float): total cost of fertilizer application for
            a given crop.
    """
    fert_lookup = [
        ('nitrogen', 'cost_nitrogen_per_kg'),
        ('potash', 'cost_potash_per_kg'),
        ('phosphorus', 'cost_phosphorus_per_kg')]

    iterblock = geoprocess.iterblocks(aoi_raster)
    fertilizer_costs = 0
    for offset_dict, block in iterblock:
        mask = copy.copy(block)
        mask[block == code] = 1.
        mask[block != code] = 0.
        for fert, column in fert_lookup:
            fert_block = read_from_raster(fertilizer_dict[fert], offset_dict)
            fert_amount = (fert_block * mask).sum()
            fert_cost = code_dict[column]
            fertilizer_costs += fert_cost * fert_amount

    return fertilizer_costs


def calc_area_costs(lookup_dict, economics_dict, aoi_raster):
    """Calculate area-related costs (e.g. labor, seed, machine, irrigation).

    Args:
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        economics_dict (dict): economic information.
        aoi_raster (str): path to aoi raster.

    Returns:
        area_cost_dict (dict): {<code>: <total of area-related costs>}
    """
    # find num cells per code
    cells_per_code = dict((code, 0,) for code in lookup_dict.keys())
    iterblock = geoprocess.iterblocks(aoi_raster)
    for block_dict, block in iterblock:
        cells_per_code = dict((code, cells + len(block[block == code]))
                              for code, cells in cells_per_code.items())

    # find total hectares per code
    cell_size = geoprocess.get_cell_size_from_uri(aoi_raster)
    m2_per_cell = cell_size ** 2
    ha_per_m2 = 0.0001
    ha_per_cell = ha_per_m2 * m2_per_cell
    ha_per_code = dict((code, num_cells * ha_per_cell)
                       for code, num_cells in cells_per_code.items())

    # find area cost per code
    area_costs_dict = dict((code, 0.) for code in lookup_dict.keys())
    for code in lookup_dict.keys():
        sub_dict = economics_dict[code]
        for column, amount in sub_dict.items():
            if column.endswith('per_ha'):
                area_costs_dict[code] = amount * ha_per_code[code]

    return area_costs_dict


def compute_financial_analysis(yield_dict, economics_table, aoi_raster,
                               lookup_dict, fertilizer_dict,
                               financial_analysis_table):
    """Compute financial analysis.

    Args:
        yield_dict (collections.Counter): mapping from crop code to total
            yield.
        economics_table (str): path to table containing economic information
            for each crop.
        aoi_raster (str): path to aoi raster.
        lookup_dict (dict): mapping of codes to lookup info for crops in aoi.
        fertilizer_dict (dict): mapping of fertilizers to their respective
            raster paths.
        financial_analysis_table (str): path to output table.
    """
    codes = yield_dict.keys()
    crop_to_code_dict = dict(
        (val['name'], code) for code, val in lookup_dict.items())
    code_to_crop_dict = dict((v, k) for k, v in crop_to_code_dict.items())

    economics_dict = geoprocess.get_lookup_from_table(economics_table, 'crop')
    economics_dict = dict((crop_to_code_dict[k], v)
                          for k, v in economics_dict.items()
                          if k in crop_to_code_dict)

    area_cost_dict = calc_area_costs(lookup_dict, economics_dict, aoi_raster)

    financial_analysis_dict = {}
    for code, code_dict in economics_dict.items():
        fertilizer_cost = 0
        if fertilizer_dict:
            fertilizer_cost = calc_fertilizer_costs(
                code, code_dict, aoi_raster, fertilizer_dict)
        costs = fertilizer_cost + area_cost_dict[code]
        revenues = code_dict['price_per_ton'] * yield_dict[code]
        returns = revenues - costs
        financial_analysis_dict[code] = {}
        financial_analysis_dict[code]['total_yield'] = (
            "%.2f" % yield_dict[code])
        financial_analysis_dict[code]['costs'] = ("%.2f" % costs)
        financial_analysis_dict[code]['revenues'] = ("%.2f" % revenues)
        financial_analysis_dict[code]['returns'] = ("%.2f" % returns)

    for i in financial_analysis_dict.keys():
        financial_analysis_dict[i]['crop'] = code_to_crop_dict[i]
    csvfile = open(financial_analysis_table, 'w')
    fieldnames = \
        financial_analysis_dict[financial_analysis_dict.keys()[0]].keys()
    fieldnames.remove('crop')
    fieldnames.remove('total_yield')
    fieldnames.sort()
    fieldnames.insert(0, 'total_yield')
    fieldnames.insert(0, 'crop')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in financial_analysis_dict.values():
        writer.writerow(row)
