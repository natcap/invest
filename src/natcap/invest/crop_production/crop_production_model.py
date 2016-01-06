'''
The Crop Production Model module contains functions for running the model
'''

import os
import logging
import pprint

from osgeo import gdal
import numpy as np
import pygeoprocessing

import crop_production_io as io
from raster import Raster
from vector import Vector

LOGGER = logging.getLogger('natcap.invest.crop_production.model')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

pp = pprint.PrettyPrinter(indent=4)

NODATA_INT = -9999
NODATA_FLOAT = -16777216


def calc_observed_yield(vars_dict):
    '''
    Calculates yield using observed yield function

    Args:
        vars_dict (dict): descr

    Example Args::

        vars_dict = {
            # ...

            'lulc_map_uri': '/path/to/lulc_map_uri',
            'crop_lookup_dict': {
                'code': 'crop_name',
                ...
            },
            'observed_yields_maps_dict': {
                'crop': '/path/to/crop_climate_bin_map',
                ...
            },
            'economics_table_dict': {
                'crop': {
                    'price': <float>,
                    ...
                }
                ...
            },
        }
    '''
    vars_dict['crop_production_dict'] = {}
    vars_dict = _create_yield_func_output_folder(
        vars_dict, "observed_yield")

    lulc_raster = Raster.from_file(
        vars_dict['lulc_map_uri']).set_nodata(NODATA_INT)
    aoi_vector = Vector.from_shapely(
        lulc_raster.get_aoi(), lulc_raster.get_projection())

    # setup useful base rasters
    base_raster_float = lulc_raster.set_datatype_and_nodata(
        gdal.GDT_Float64, NODATA_FLOAT)

    if vars_dict['do_economic_returns']:
        returns_raster = base_raster_float.zeros()

    crops = vars_dict['crops_in_aoi_list']
    for crop in crops:
        LOGGER.info('Calculating observed yield for %s' % crop)
        # Wrangle Data...
        observed_yield_over_aoi_raster = _get_observed_yield_from_dataset(
            vars_dict,
            crop,
            aoi_vector,
            base_raster_float)

        ObservedLocalYield_raster = _get_yield_given_lulc(
            vars_dict,
            crop,
            lulc_raster,
            observed_yield_over_aoi_raster)

        # Operations as Noted in User's Guide...
        Production_raster = _calculate_production_for_crop(
            vars_dict,
            crop,
            ObservedLocalYield_raster)

        total_production = float(round(
            Production_raster.sum(), 2))
        vars_dict['crop_production_dict'][crop] = total_production

        if vars_dict['do_economic_returns']:
            returns_raster_crop = _calc_crop_returns(
                vars_dict,
                crop,
                lulc_raster,
                Production_raster,
                returns_raster,
                vars_dict['economics_table_dict'][crop])
            if not np.isnan(returns_raster_crop.sum()):
                returns_raster = returns_raster + returns_raster_crop

        # Clean Up Rasters...
        del observed_yield_over_aoi_raster
        del ObservedLocalYield_raster
        del Production_raster

    if vars_dict['do_nutrition']:
        vars_dict = _calc_nutrition(vars_dict)

    # Results Table
    io.create_results_table(vars_dict)

    if all([vars_dict['do_economic_returns'],
            vars_dict['create_crop_production_maps']]):
        output_observed_yield_dir = vars_dict['output_yield_func_dir']
        returns_uri = os.path.join(
            output_observed_yield_dir, 'economic_returns_map.tif')
        returns_raster.save_raster(returns_uri)

    return vars_dict


def _create_yield_func_output_folder(vars_dict, folder_name):
    '''
    Example Returns::

        vars_dict = {
            # ...

            'output_yield_func_dir': '/path/to/outputs/yield_func/',
            'output_production_maps_dir':
                '/path/to/outputs/yield_func/production/'
        }

    Output:
        .
        |-- [yield_func]_[results_suffix]
            |-- production

    '''
    if vars_dict['results_suffix']:
        folder_name = folder_name + '_' + vars_dict['results_suffix']
    output_yield_func_dir = os.path.join(vars_dict['output_dir'], folder_name)
    if not os.path.exists(output_yield_func_dir):
        os.makedirs(output_yield_func_dir)
    vars_dict['output_yield_func_dir'] = output_yield_func_dir

    if vars_dict['create_crop_production_maps']:
        output_production_maps_dir = os.path.join(
            output_yield_func_dir, 'crop_production_maps')
        if not os.path.exists(output_production_maps_dir):
            os.makedirs(output_production_maps_dir)
        vars_dict['output_production_maps_dir'] = output_production_maps_dir

    return vars_dict


def _get_observed_yield_from_dataset(vars_dict, crop, aoi_vector, base_raster_float):
    '''
    Clips the observed crop yield values in the global dataset, reprojects and
        resamples those values to a new raster aligned to the given LULC raster
        that is then returned to the user.
    '''
    crop_observed_yield_raster = Raster.from_file(
        vars_dict['observed_yields_maps_dict'][crop])

    reproj_aoi_vector = aoi_vector.reproject(
        crop_observed_yield_raster.get_projection())

    clipped_crop_raster = crop_observed_yield_raster.clip(
        reproj_aoi_vector.uri).set_nodata(NODATA_FLOAT)

    if clipped_crop_raster.get_shape() == (1, 1):
        observed_yield_val = float(clipped_crop_raster.get_band(1)[0, 0])
        aligned_crop_raster = observed_yield_val * base_raster_float.ones()
    else:
        # this reprojection could result in very long computation times
        reproj_crop_raster = clipped_crop_raster.reproject(
            base_raster_float.get_projection(),
            'nearest',
            base_raster_float.get_affine().a)

        aligned_crop_raster = reproj_crop_raster.align_to(
            base_raster_float, 'nearest')

    return aligned_crop_raster


def _get_yield_given_lulc(vars_dict, crop, lulc_raster, observed_yield_over_aoi_raster):
    '''
    Maskes out the cells in the observed yield raster who's corrsponding cells
        in the LULC raster are not of the current crop type.
    '''
    masked_lulc_int_raster = _get_masked_lulc_raster(
        vars_dict, crop, lulc_raster)
    masked_lulc_raster = masked_lulc_int_raster.set_datatype_and_nodata(
        gdal.GDT_Float64, NODATA_FLOAT)

    Yield_given_lulc_raster = observed_yield_over_aoi_raster * masked_lulc_raster

    return Yield_given_lulc_raster


def _get_masked_lulc_raster(vars_dict, crop, lulc_raster):
    '''
    Returns a mask raster containing ones in cells that correspond to one
        crop and zeros in cells corresponding to all other crops
    '''
    crop_lookup_dict = vars_dict['crop_lookup_dict']
    inv_crop_lookup_dict = {v: k for k, v in crop_lookup_dict.items()}

    reclass_table = {}
    for key in vars_dict['observed_yields_maps_dict'].keys():
        reclass_table[inv_crop_lookup_dict[key]] = 0
    reclass_table[inv_crop_lookup_dict[crop]] = 1

    masked_lulc_raster = lulc_raster.reclass(reclass_table)

    return masked_lulc_raster


def _calculate_production_for_crop(vars_dict, crop, yield_raster, percentile=None):
    '''
    Converts a yield raster to a production raster and saves the production
        raster if specified by the user.
    '''
    ha_per_m2 = 0.0001
    ha_per_cell = yield_raster.get_cell_area() * ha_per_m2

    Production_raster = yield_raster * ha_per_cell

    if vars_dict['create_crop_production_maps'] and percentile is None:
        filename = crop + '_production_map.tif'
        dst_uri = os.path.join(vars_dict[
            'output_production_maps_dir'], filename)
        Production_raster.save_raster(dst_uri)

    elif vars_dict['create_crop_production_maps']:
        filename = crop + '_production_map_' + percentile + '.tif'
        dst_uri = os.path.join(vars_dict[
            'output_production_maps_dir'], filename)
        Production_raster.save_raster(dst_uri)

    return Production_raster


def _calc_crop_returns(vars_dict, crop, lulc_raster, production_raster, returns_raster, economics_table):
    '''
    Implements the following equations provided in the User Guide:

    Cost_crop = CostPerTonInputTotal_crop + CostPerHectareInputTotal_crop
    Revenue_crop = Production_crop * Price_crop
    Returns_crop = Revenue_crop - Cost_crop
    '''
    cost_per_hectare_input_raster = _calc_cost_of_per_hectare_inputs(
        vars_dict, crop, lulc_raster)

    if vars_dict['fertilizer_maps_dir']:
        cost_per_ton_input_raster = _calc_cost_of_per_ton_inputs(
            vars_dict, crop, lulc_raster)

        cost_raster = cost_per_hectare_input_raster + cost_per_ton_input_raster
    else:
        cost_raster = cost_per_hectare_input_raster

    price = vars_dict['economics_table_dict'][crop]['price_per_tonne']
    revenue_raster = production_raster * price

    returns_raster = revenue_raster - cost_raster

    total_cost = float(round(
        cost_raster.sum(), 2))
    total_revenue = float(round(
        revenue_raster.sum(), 2))
    total_returns = float(round(
        returns_raster.sum(), 2))

    vars_dict['economics_table_dict'][crop]['total_cost'] = total_cost
    vars_dict['economics_table_dict'][crop]['total_revenue'] = total_revenue
    vars_dict['economics_table_dict'][crop]['total_returns'] = total_returns

    return returns_raster


def _calc_cost_of_per_ton_inputs(vars_dict, crop, lulc_raster):
    '''
    Implements the following equations provided in the User Guide:

    sum_across_fert(FertAppRate_fert * LULCCropCellArea * CostPerTon_fert)
    '''

    economics_table_crop = vars_dict['economics_table_dict'][crop]
    fert_maps_dict = vars_dict['fertilizer_maps_dict']

    masked_lulc_raster = _get_masked_lulc_raster(vars_dict, crop, lulc_raster)
    masked_lulc_raster_float = masked_lulc_raster.set_datatype_and_nodata(
        gdal.GDT_Float64, NODATA_FLOAT)

    CostPerTonInputTotal_raster = masked_lulc_raster_float.zeros()

    try:
        cost_nitrogen_per_kg = economics_table_crop['cost_nitrogen_per_kg']
        Nitrogen_raster = Raster.from_file(
            fert_maps_dict['nitrogen']).set_nodata(NODATA_FLOAT)
        NitrogenCost_raster = Nitrogen_raster * cost_nitrogen_per_kg
        CostPerTonInputTotal_raster += NitrogenCost_raster
    except KeyError:
        LOGGER.warning("Skipping nitrogen cost because insufficient amount "
                       "of information provided.")
    try:
        cost_phosphorous_per_kg = economics_table_crop[
            'cost_phosphorous_per_kg']
        Phosphorous_raster = Raster.from_file(
            fert_maps_dict['phosphorous']).set_nodata(NODATA_FLOAT)
        PhosphorousCost_raster = Phosphorous_raster * cost_phosphorous_per_kg
        CostPerTonInputTotal_raster += PhosphorousCost_raster
    except KeyError:
        LOGGER.warning("Skipping phosphorous cost because insufficient amount "
                       "of information provided.")
    try:
        cost_potash_per_kg = economics_table_crop['cost_potash_per_kg']
        Potash_raster = Raster.from_file(
            fert_maps_dict['potash']).set_nodata(NODATA_FLOAT)
        PotashCost_raster = Potash_raster * cost_potash_per_kg
        CostPerTonInputTotal_raster += PotashCost_raster
    except KeyError:
        LOGGER.warning("Skipping potash cost because insufficient amount of "
                       "information provided.")

    CostPerTonInputTotal_masked_raster = CostPerTonInputTotal_raster * masked_lulc_raster_float

    return CostPerTonInputTotal_masked_raster


def _calc_cost_of_per_hectare_inputs(vars_dict, crop, lulc_raster):
    '''
    CostPerHectareInputTotal_crop = Mask_raster * CostPerHectare_input *
        ha_per_cell
    '''

    # Determine the crop lucode based on its name
    crop_lucode = None
    for lucode, luname in vars_dict['crop_lookup_dict'].iteritems():
        if luname == crop:
            crop_lucode = lucode
            continue

    lulc_nodata = pygeoprocessing.get_nodata_from_uri(lulc_raster.uri)
    economics_table_crop = vars_dict['economics_table_dict'][crop]
    datatype_out = gdal.GDT_Float32
    nodata_out = NODATA_FLOAT
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(lulc_raster.uri)
    ha_per_m2 = 0.0001
    cell_area_ha = pixel_size_out**2 * ha_per_m2

    # The scalar cost is identical for all crop pixels of the current class,
    # and is based on the presence of absence of columns in the user-provided
    # economics table.  We only need to calculate this once.
    cost_scalar = 0.0
    for key in ['cost_labor_per_ha', 'cost_machine_per_ha', 'cost_seed_per_ha', 'cost_irrigation_per_ha']:
        try:
            cost_scalar += (economics_table_crop[key] * cell_area_ha)
        except KeyError:
            LOGGER.warning('Key missing from economics table: %s', key)

    def _calculate_cost(lulc_matrix):
        """
        Calculate the total cost on a single pixel.

        <pseudocode>
            If lulc_pixel is nodata:
                return nodata
            else:
                if lulc_pixel is of our crop type:
                    return the cost of this crop (in cost_scalar, above)
                else:
                    return 0.0
        </pseudocode>
        """
        return np.where(lulc_matrix == lulc_nodata, nodata_out,
                        np.where(lulc_matrix == crop_lucode, cost_scalar, 0.0))

    new_raster_uri = pygeoprocessing.geoprocessing.temporary_filename()
    pygeoprocessing.vectorize_datasets(
        [lulc_raster.uri],
        _calculate_cost,
        new_raster_uri,
        datatype_out,
        nodata_out,
        pixel_size_out,
        bounding_box_mode='intersection',
        vectorize_op=False,
        datasets_are_pre_aligned=True
    )

    return Raster.from_file(new_raster_uri, 'GTiff')


def calc_percentile_yield(vars_dict):
    '''
    Calculates yield using the percentile yield function

    Example Args::

        vars_dict = {
            'percentile_yield_dict': {
                ''
            },
            '': ''
        }
    '''
    vars_dict['crop_production_dict'] = {}
    vars_dict = _create_yield_func_output_folder(
        vars_dict, "climate_percentile_yield")

    lulc_raster = Raster.from_file(
        vars_dict['lulc_map_uri']).set_nodata(NODATA_INT)
    aoi_vector = Vector.from_shapely(
        lulc_raster.get_aoi(), lulc_raster.get_projection())
    percentile_yield_dict = vars_dict['percentile_yield_dict']

    # setup useful base rasters
    base_raster_float = lulc_raster.set_datatype_and_nodata(
        gdal.GDT_Float64, NODATA_FLOAT)

    crops = vars_dict['crops_in_aoi_list']
    crop = crops[0]
    climate_bin = percentile_yield_dict[crop].keys()[0]
    percentiles = percentile_yield_dict[crop][climate_bin].keys()

    percentile_count = 1
    for percentile in percentiles:
        vars_dict['crop_production_dict'] = {}
        if vars_dict['do_economic_returns']:
            economics_table = vars_dict['economics_table_dict']
            returns_raster = base_raster_float.zeros()

        for crop in crops:
            LOGGER.info('Calculating percentile yield for %s in %s' % (
                crop, percentile))
            # Wrangle Data...
            climate_bin_raster = _get_climate_bin_over_lulc(
                vars_dict, crop, aoi_vector, base_raster_float)

            reclass_dict = {}
            climate_bins = percentile_yield_dict[crop].keys()
            for climate_bin in climate_bins:
                reclass_dict[climate_bin] = percentile_yield_dict[
                    crop][climate_bin][percentile]

            # Find Yield and Production
            crop_yield_raster = climate_bin_raster.reclass(reclass_dict)

            masked_lulc_raster = _get_masked_lulc_raster(
                vars_dict, crop, lulc_raster).set_datatype_and_nodata(
                gdal.GDT_Float64, NODATA_FLOAT)

            yield_raster = crop_yield_raster.reclass_masked_values(
                masked_lulc_raster, 0)

            Production_raster = _calculate_production_for_crop(
                vars_dict, crop, yield_raster, percentile=percentile)

            total_production = float(round(
                Production_raster.sum(), 2))
            vars_dict['crop_production_dict'][crop] = total_production

            if vars_dict['do_economic_returns']:
                returns_raster_crop = _calc_crop_returns(
                    vars_dict,
                    crop,
                    lulc_raster,
                    Production_raster,
                    returns_raster,
                    economics_table[crop])
                returns_raster = returns_raster + returns_raster_crop

            # Clean Up Rasters...
            del climate_bin_raster
            del crop_yield_raster
            del masked_lulc_raster
            del yield_raster
            del Production_raster

        if vars_dict['do_nutrition']:
            vars_dict = _calc_nutrition(vars_dict)

        # Results Table
        if percentile_count == 1:
            io.create_results_table(vars_dict, percentile=percentile)
        else:
            io.create_results_table(
                vars_dict, percentile=percentile, first=False)
        percentile_count += 1

        if all([vars_dict['do_economic_returns'], vars_dict[
                'create_crop_production_maps']]):
            output_observed_yield_dir = vars_dict['output_yield_func_dir']
            returns_uri = os.path.join(
                output_observed_yield_dir,
                'economic_returns_map_' + percentile + '.tif')
            returns_raster.save_raster(returns_uri)

    return vars_dict


def _get_climate_bin_over_lulc(vars_dict, crop, aoi_vector, base_raster_float):
    '''
    Clips the climate bin values in the global dataset, reprojects and
        resamples those values to a new raster aligned to the given LULC raster
        that is then returned to the user.
    '''
    climate_bin_raster = Raster.from_file(
        vars_dict['climate_bin_maps_dict'][crop])

    reproj_aoi_vector = aoi_vector.reproject(
        climate_bin_raster.get_projection())

    clipped_climate_bin_raster = climate_bin_raster.clip(
        reproj_aoi_vector.uri).set_nodata(NODATA_INT)

    if clipped_climate_bin_raster.get_shape() == (1, 1):
        climate_bin_val = float(clipped_climate_bin_raster.get_band(
            1)[0, 0])
        aligned_climate_bin_raster = base_raster_float.ones() * climate_bin_val
    else:
        # note: this reprojection could result in very long computation times
        reproj_climate_bin_raster = clipped_climate_bin_raster.reproject(
            base_raster_float.get_projection(),
            'nearest',
            base_raster_float.get_affine().a)

        aligned_climate_bin_raster = reproj_climate_bin_raster.align_to(
            base_raster_float, 'nearest')

    return aligned_climate_bin_raster


def calc_regression_yield(vars_dict):
    '''
    Calculates yield using the regression model yield function

    Example Args::

        vars_dict = {
            ...

            'fertilizer_maps_dict': {...},
            'modeled_irrigation_map_uri': '',
            'modeled_yield_dict': {...}
        }
    '''
    vars_dict = _create_yield_func_output_folder(
        vars_dict, "climate_regression_yield")

    lulc_raster = Raster.from_file(
        vars_dict['lulc_map_uri']).set_nodata(NODATA_INT)
    aoi_vector = Vector.from_shapely(
        lulc_raster.get_aoi(), lulc_raster.get_projection())

    # setup useful base rasters
    base_raster_float = lulc_raster.set_datatype_and_nodata(
        gdal.GDT_Float64, NODATA_FLOAT)

    vars_dict['crop_production_dict'] = {}
    if vars_dict['do_economic_returns']:
        economics_table = vars_dict['economics_table_dict']
        returns_raster = base_raster_float.zeros()

    crops = vars_dict['crops_in_aoi_list']
    for crop in crops:
        LOGGER.info('Calculating regression yield for %s' % crop)
        # Wrangle data...
        climate_bin_raster = _get_climate_bin_over_lulc(
            vars_dict, crop, aoi_vector, base_raster_float)

        masked_lulc_raster = _get_masked_lulc_raster(
            vars_dict, crop, lulc_raster).set_datatype_and_nodata(
            gdal.GDT_Float64, NODATA_FLOAT)

        # Operations as Noted in User's Guide...
        Yield_raster = _calc_regression_yield_for_crop(
            vars_dict,
            crop,
            climate_bin_raster)

        Yield_given_lulc_raster = Yield_raster * masked_lulc_raster

        Production_raster = _calculate_production_for_crop(
            vars_dict, crop, Yield_given_lulc_raster)

        total_production = float(round(
            Production_raster.sum(), 2))
        vars_dict['crop_production_dict'][crop] = total_production

        if vars_dict['do_economic_returns']:
            returns_raster_crop = _calc_crop_returns(
                vars_dict,
                crop,
                lulc_raster,
                Production_raster.set_nodata(NODATA_FLOAT),
                returns_raster,
                economics_table[crop])
            returns_raster = returns_raster + returns_raster_crop

        # Clean Up Rasters...
        del climate_bin_raster
        del Yield_raster
        del masked_lulc_raster
        del Yield_given_lulc_raster
        del Production_raster

    if vars_dict['do_nutrition']:
        vars_dict = _calc_nutrition(vars_dict)

    # Results Table
    io.create_results_table(vars_dict)

    if all([vars_dict['do_economic_returns'],
            vars_dict['create_crop_production_maps']]):
        output_observed_yield_dir = vars_dict['output_yield_func_dir']
        returns_uri = os.path.join(
            output_observed_yield_dir, 'economic_returns_map.tif')
        returns_raster.save_raster(returns_uri)

    return vars_dict


def _calc_regression_yield_for_crop(vars_dict, crop, climate_bin_raster):
    '''
    Calculates yield for an individual crop using the percentile yield function
    '''

    # Fetch Fertilizer Maps
    fert_maps_dict = vars_dict['fertilizer_maps_dict']
    NitrogenAppRate_raster = Raster.from_file(
        fert_maps_dict['nitrogen']).set_nodata(NODATA_FLOAT)
    PhosphorousAppRate_raster = Raster.from_file(
        fert_maps_dict['phosphorous']).set_nodata(NODATA_FLOAT)
    PotashAppRate_raster = Raster.from_file(
        fert_maps_dict['potash']).set_nodata(NODATA_FLOAT)
    Irrigation_raster = Raster.from_file(
        vars_dict['modeled_irrigation_map_uri']).set_datatype_and_nodata(
        gdal.GDT_Int16, NODATA_INT)

    irrigated_lulc_mask = (Irrigation_raster).set_datatype_and_nodata(
        gdal.GDT_Float64, NODATA_FLOAT)
    rainfed_lulc_mask = ((Irrigation_raster * -1) + 1).set_datatype_and_nodata(
        gdal.GDT_Float64, NODATA_FLOAT)

    # Create Rasters of Yield Parameters
    yield_params = vars_dict['modeled_yield_dict'][crop]

    nodata = climate_bin_raster.get_nodata(1)

    b_K2O = _create_reg_yield_reclass_dict(
        yield_params, 'b_K2O', nodata)
    b_nut = _create_reg_yield_reclass_dict(
        yield_params, 'b_nut', nodata)
    c_N = _create_reg_yield_reclass_dict(
        yield_params, 'c_N', nodata)
    c_P2O5 = _create_reg_yield_reclass_dict(
        yield_params, 'c_P2O5', nodata)
    c_K2O = _create_reg_yield_reclass_dict(
        yield_params, 'c_K2O', nodata)
    yc = _create_reg_yield_reclass_dict(
        yield_params, 'yield_ceiling', nodata)
    yc_rf = _create_reg_yield_reclass_dict(
        yield_params, 'yield_ceiling_rf', nodata)

    b_K2O_raster = climate_bin_raster.reclass(b_K2O)
    b_nut_raster = climate_bin_raster.reclass(b_nut)
    c_N_raster = climate_bin_raster.reclass(c_N)
    c_P2O5_raster = climate_bin_raster.reclass(c_P2O5)
    c_K2O_raster = climate_bin_raster.reclass(c_K2O)
    YieldCeiling_raster = climate_bin_raster.reclass(yc)
    YieldCeilingRainfed_raster = climate_bin_raster.reclass(yc_rf)

    # Operations as Noted in User's Guide...
    PercentMaxYieldNitrogen_raster = 1 - (
        b_nut_raster * (np.e ** -c_N_raster) * NitrogenAppRate_raster)
    PercentMaxYieldPhosphorous_raster = 1 - (
        b_nut_raster * (np.e ** -c_P2O5_raster) * PhosphorousAppRate_raster)
    PercentMaxYieldPotassium_raster = 1 - (
        b_K2O_raster * (np.e ** -c_K2O_raster) * PotashAppRate_raster)

    PercentMaxYield_raster = (PercentMaxYieldNitrogen_raster.fminimum(
        PercentMaxYieldPhosphorous_raster.fminimum(
            PercentMaxYieldPotassium_raster)))

    MaxYield_raster = PercentMaxYield_raster * YieldCeiling_raster
    Yield_irrigated_raster = MaxYield_raster.reclass_masked_values(
        irrigated_lulc_mask, 0)
    Yield_rainfed_raster = YieldCeilingRainfed_raster.minimum(
        MaxYield_raster).reclass_masked_values(
        rainfed_lulc_mask, 0)

    Yield_raster = Yield_irrigated_raster + Yield_rainfed_raster

    return Yield_raster


def _create_reg_yield_reclass_dict(dictionary, nested_key, nodata):
    '''
    Fetching nested values from a dictionary and returns a new dictionary
    '''
    reclass_dict = {}
    for k in dictionary.keys():
        reclass_dict[k] = dictionary[k][nested_key]

    return reclass_dict


def _calc_nutrition(vars_dict):
    '''
    Calculates the nutritional value of each crop.

    total_nutrient_amount = production_tons * nutrient_unit *
        (1 - fraction_refuse)

    Example Args::

        vars_dict = {
            'crop_production_dict': {
                'corn': 12.3,
                'soy': 13.4,
                ...
            },
            'nutrition_table_dict': {
                'crop': {
                    'percent_refuse': <float>,
                    'protein': <float>,
                    'lipid': <float>,
                    'energy': <float>,
                    'ca': <float>,
                    'fe': <float>,
                    'mg': <float>,
                    'ph': <float>,
                    ...
                },
                ...
            }
        }

    Example Returns::

        vars_dict = {
            ...
            'crop_total_nutrition_dict': {
                'crop': {
                    'percent_refuse': <float>,
                    'protein': <float>,
                    'lipid': <float>,
                    'energy': <float>,
                    'ca': <float>,
                    'fe': <float>,
                    'mg': <float>,
                    'ph': <float>,
                    ...
                }
            }
        }
    '''
    crop_production_dict = vars_dict['crop_production_dict']
    crop_total_nutrition_dict = {}

    for crop in crop_production_dict.keys():
        production = crop_production_dict[crop]
        nutrition_row_per_unit = vars_dict['nutrition_table_dict'][crop]
        fraction_refuse = nutrition_row_per_unit['fraction_refuse']
        nutrition_row_total = {}
        for nutrient in nutrition_row_per_unit.keys():
            if (nutrient == 'fraction_refuse') or (nutrient == 'crop'):
                continue
            nutrient_unit = nutrition_row_per_unit[nutrient]
            if type(nutrient_unit) not in [int, float]:
                nutrient_unit = 0
            total_nutrient = float(round((
                production * nutrient_unit * (1 - fraction_refuse)),
                2))
            nutrition_row_total[nutrient] = total_nutrient
        crop_total_nutrition_dict[crop] = nutrition_row_total

    vars_dict['crop_total_nutrition_dict'] = crop_total_nutrition_dict
    return vars_dict
