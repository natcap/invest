"""Coastal Blue Carbon Model."""

import logging
import os
import collections

import gdal
import pygeoprocessing as pygeo
import numpy as np

from natcap.invest.coastal_blue_carbon.utilities import io
from natcap.invest.coastal_blue_carbon.utilities.raster import Raster
from natcap.invest.coastal_blue_carbon.utilities.cbc_model_classes import \
    DisturbedCarbonStock, AccumulatedCarbonStock

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.coastal_blue_carbon.coastal_blue_carbon')

# Global Variables
NODATA_FLOAT = -16777216
NODATA_INT = -9999
HA_PER_M2 = 0.0001


def execute(args):
    """Entry point for Coastal Blue Carbon model.

    :param str args['workspace']: location into which all intermediate
        and output files should be placed.

    :param str args['results_suffix']: a string to append to output filenames.

    :param str args['lulc_lookup_uri']: filepath to a CSV table used to convert
        the lulc code to a name. Also used to determine if a given lulc type is
        a coastal blue carbon habitat.

    :param str args['lulc_transition_uri']:

    :param str args['lulc_snapshot_list']:

    :param str args['lulc_snapshot_years_list']:

    :param int args['analysis_year']:

    :param str args['carbon_pool_initial_uri']:

    :param str args['carbon_pool_transient_uri']:

    Example Args::

        args = {
            'workspace': 'path/to/workspace',
            'results_suffix': '',
            'lulc_lookup_uri': 'path/to/lulc_lookup_uri',
            'lulc_transition_uri': 'path/to/lulc_transition_uri',
            'lulc_snapshot_list': [raster1_uri, raster2_uri, ...],
            'lulc_snapshot_years_list': [2000, 2005, ...],
            'analysis_year': 2100,
            'carbon_pool_initial_uri': 'path/to/carbon_pool_initial_uri',
            'carbon_pool_transient_uri': 'path/to/carbon_pool_transient_uri'
        }
    """
    # Get Inputs
    vars_dict = io.get_inputs(args)

    # Set Initial Conditions
    vars_dict = _set_initial_stock(vars_dict)

    # Run Transient Analysis
    _run_transient_analysis(vars_dict)


def _set_initial_stock(vars_dict):
    """Create and return an initial carbon stock raster based on user inputs.

    Returns:
        vars_dict['biomass_carbon_stock_raster_list'][0]
        vars_dict['soil_carbon_stock_raster_list'][0]
        vars_dict['total_carbon_stock_raster_list'][0]
    """
    carbon_pool_initial_dict = vars_dict['carbon_pool_initial_dict']
    lulc_to_code_dict = vars_dict['lulc_to_code_dict']

    code_to_biomass_reclass_dict = dict(
        [(lulc_to_code_dict[item[0]],
            item[1]['biomass']) for item in carbon_pool_initial_dict.items()])
    code_to_soil_reclass_dict = dict(
        [(lulc_to_code_dict[item[0]],
            item[1]['soil']) for item in carbon_pool_initial_dict.items()])

    init_lulc_raster = Raster.from_file(vars_dict['lulc_snapshot_list'][0])

    # Create initial carbon stock rasters for biomass and soil
    init_carbon_stock_biomass_raster = init_lulc_raster.reclass(
        code_to_biomass_reclass_dict,
        out_datatype=gdal.GDT_Float32,
        out_nodata=NODATA_FLOAT)
    init_carbon_stock_soil_raster = init_lulc_raster.reclass(
        code_to_soil_reclass_dict,
        out_datatype=gdal.GDT_Float32,
        out_nodata=NODATA_FLOAT)

    # Create total initial carbon stock raster and save rasters to lists
    init_carbon_stock_raster = \
        init_carbon_stock_biomass_raster + init_carbon_stock_soil_raster
    vars_dict['total_carbon_stock_raster_list'][0] = init_carbon_stock_raster
    vars_dict['biomass_carbon_stock_raster_list'][0] = \
        init_carbon_stock_biomass_raster
    vars_dict['soil_carbon_stock_raster_list'][0] = \
        init_carbon_stock_soil_raster

    # Save total initial stock as output
    initial_carbon_stock_filename = 'total_stock_at_%i.tif' % (
        vars_dict['lulc_snapshot_year_list'][0])
    init_carbon_stock_raster.save_raster(
        os.path.join(vars_dict['output_dir'], initial_carbon_stock_filename))

    return vars_dict


def _run_transient_analysis(vars_dict):
    """Run transient analysis of carbon stock over time."""
    for idx in range(len(vars_dict['lulc_snapshot_list'])-1):
        start_year = vars_dict['lulc_snapshot_years_list'][idx]
        end_year = vars_dict['lulc_snapshot_years_list'][idx+1]

        # Create New Accumulated and Disturbed Stock Objects
        vars_dict = _update_transient_carbon_reclass_dicts(vars_dict, idx)
        vars_dict = _update_accumulated_carbon_object_list(vars_dict, idx)
        vars_dict = _update_disturbed_carbon_object_list(vars_dict, idx)

        # Sequestration
        a = vars_dict['accumulated_carbon_stock_object_list
        sequestered_over_time_raster = a.get_total_sequestered_by_year(
            end_year)
        sequestered_over_time_filename = 'sequestration_from_%i_to_%i.tif' % (
            start_year, end_year)
        sequestered_over_time_raster.save_raster(
            os.path.join(
                vars_dict['outputs_dir'], sequestered_over_time_filename))

        # Emissions
        d_list = vars_dict['disturbed_carbon_stock_object_list'][idx]
        emitted_over_time_raster = sequestered_over_time_raster.zeros()
        for i in range(0, idx+1):
            emitted_over_time_raster += d_list.get_total_emissions_between_years(start_year, end_year)
        emitted_over_time_filename = 'emissions_from_%i_to_%i.tif' % (
            start_year, end_year)
        emitted_over_time_raster.save_raster(
            os.path.join(vars_dict['outputs_dir'], emitted_over_time_filename))

        # Net Sequestration
        net_sequestered_over_time_raster = sequestered_over_time_raster - \
            emitted_over_time_raster
        vars_dict['net_sequestration_raster_list'][
            idx] = net_sequestered_over_time_raster
        net_sequestered_over_time_filename = \
            'net_sequestration_from_%i_to_%i.tif' % (start_year, end_year)
        net_sequestered_over_time_raster.save_raster(
            os.path.join(
                vars_dict['outputs_dir'], net_sequestered_over_time_filename))

        # Stock
        prev_carbon_stock_raster = vars_dict['carbon_stock_raster_list'][idx]
        next_carbon_stock_raster = prev_carbon_stock_raster + \
            net_sequestered_over_time_raster
        vars_dict['carbon_stock_raster_list'][idx+1] = next_carbon_stock_raster
        next_carbon_stock_filename = 'carbon_stock_at_%i.tif' % (end_year)
        next_carbon_stock_raster.save_raster(
            os.path.join(vars_dict['outputs_dir'], next_carbon_stock_filename))


def _update_transient_carbon_reclass_dicts(vars_dict, snapshot_idx):
    """Create and return lulc to carbon reclass dictionaries for a given snapshot
    transition.

    Args:

        vars_dict['code_to_lulc_dict']
        vars_dict['lulc_transition_dict']
        vars_dict['carbon_pool_transient_dict']
        vars_dict['lulc_snapshot_list'][idx+1]

    Returns:

        vars_dict['accumulation_biomass_reclass_dict']
        vars_dict['accumulation_soil_reclass_dict']
        vars_dict['disturbance_biomass_reclass_dict']
        vars_dict['disturbance_soil_reclass_dict']
        vars_dict['half_life_biomass_reclass_dict']
        vars_dict['half_life_soil_reclass_dict']

    """
    def _create_accumulation_reclass_dicts(vars_dict, next_lulc_raster, pool):
        """Create accumulation reclass dicts.

        accumulation_biomass_reclass_dict = {}
        accumulation_soil_reclass_dict = {
            next_lulc_code: accumulation_rate,
            lulc_codes_not_in_transition_dict: 0
        }
        """
        code_to_lulc_dict = vars_dict['code_to_lulc_dict']
        lulc_transition_dict = vars_dict['lulc_transition_dict']
        carbon_pool_transient_dict = vars_dict['carbon_pool_transient_dict']
        lulc_vals = set(next_lulc_raster.get_band(1).flatten())
        d = {}
        for i in lulc_vals:
            if (code_to_lulc_dict[i], pool) in carbon_pool_transient_dict:
                d[i] = carbon_pool_transient_dict[
                    (code_to_lulc_dict[i], pool)]['yearly_sequestration_per_ha']
            else:
                d[i] = 0
        return d

    def _create_disturbance_reclass_dicts(vars_dict, pool):
        """Create disturbance reclass dicts.

        disturbance_biomass_reclass_dict = {}
        disturbance_soil_reclass_dict = {
            (prev_lulc_code, next_lulc_code): pct_disturbed
        }
        """
        lulc_to_code_dict = vars_dict['lulc_to_code_dict']
        lulc_transition_dict = vars_dict['lulc_transition_dict']
        carbon_pool_transient_dict = vars_dict['carbon_pool_transient_dict']
        d = {}
        for prev_lulc, val in lulc_transition_dict.items():
            for next_lulc, carbon_mag_and_dir in val.items():
                if carbon_mag_and_dir not in ['', 'accumulation']:
                    disturbance_val = carbon_pool_transient_dict[
                        (next_lulc, pool)][carbon_mag_and_dir]
                    d[(lulc_to_code_dict[prev_lulc], lulc_to_code_dict[
                        next_lulc])] = disturbance_val
                else:
                    d[(lulc_to_code_dict[
                        prev_lulc], lulc_to_code_dict[next_lulc])] = 0
        return d

    def _create_half_life_reclass_dicts(vars_dict, pool):
        """Create half-life reclass dicts.

        half_life_biomass_reclass_dict = {}
        half_life_soil_reclass_dict = {
            (prev_lulc_code, next_lulc_code): pct_disturbed
        }
        """
        lulc_to_code_dict = vars_dict['lulc_to_code_dict']
        lulc_transition_dict = vars_dict['lulc_transition_dict']
        carbon_pool_transient_dict = vars_dict['carbon_pool_transient_dict']
        d = {}
        for prev_lulc, val in lulc_transition_dict.items():
            for next_lulc, carbon_mag_and_dir in val.items():
                if carbon_mag_and_dir not in ['', 'accumulation']:
                    disturbance_val = carbon_pool_transient_dict[
                        (next_lulc, pool)]['half-life']
                    d[(lulc_to_code_dict[prev_lulc], lulc_to_code_dict[
                        next_lulc])] = disturbance_val
                else:
                    d[(lulc_to_code_dict[prev_lulc], lulc_to_code_dict[
                        next_lulc])] = 0
        return d

    next_raster = Raster.from_file(vars_dict['lulc_snapshot_list'][
        snapshot_idx + 1])

    vars_dict['accumulation_biomass_reclass_dict'] = \
        _create_accumulation_reclass_dicts(vars_dict, next_raster, 'biomass')
    vars_dict['accumulation_soil_reclass_dict'] = \
        _create_accumulation_reclass_dicts(vars_dict, next_raster, 'soil')

    vars_dict['disturbance_biomass_reclass_dict'] = \
        _create_disturbance_reclass_dicts(vars_dict, 'biomass')
    vars_dict['disturbance_soil_reclass_dict'] = \
        _create_disturbance_reclass_dicts(vars_dict, 'soil')

    vars_dict['half_life_biomass_reclass_dict'] = \
        _create_half_life_reclass_dicts(vars_dict, 'biomass')
    vars_dict['half_life_soil_reclass_dict'] = \
        _create_half_life_reclass_dicts(vars_dict, 'soil')

    return vars_dict


def _update_accumulated_carbon_object_list(vars_dict, snapshot_idx):
    """Create and return a new AccumulatedCarbonStock object.

    Returns:

        vars_dict['accumulated_carbon_stock_object_list'][snapshot_idx]
    """
    next_lulc_raster = Raster.from_file(
        vars_dict['lulc_snapshot_list'][snapshot_idx+1])
    accumulation_biomass_reclass_dict = \
        vars_dict['accumulation_biomass_reclass_dict']
    accumulation_soil_reclass_dict = \
        vars_dict['accumulation_soil_reclass_dict']

    yearly_sequest_biomass_raster = next_lulc_raster.reclass(
        accumulation_biomass_reclass_dict,
        out_datatype=gdal.GDT_Float32,
        out_nodata=NODATA_FLOAT) * (
            next_lulc_raster.get_cell_area() * HA_PER_M2)
    # multiply by ha_per_cell (should check that units are in meters)
    yearly_sequest_soil_raster = next_lulc_raster.reclass(
        accumulation_soil_reclass_dict,
        out_datatype=gdal.GDT_Float32,
        out_nodata=NODATA_FLOAT) * (
            next_lulc_raster.get_cell_area() * HA_PER_M2)
    # multiply by ha_per_cell (should check that units are in meters)

    accumulated_carbon_stock_object = AccumulatedCarbonStock(
        vars_dict['lulc_snapshot_years_list'][snapshot_idx],
        yearly_sequest_biomass_raster,
        yearly_sequest_soil_raster)

    vars_dict['accumulated_carbon_stock_object_list'][
        snapshot_idx] = accumulated_carbon_stock_object

    return vars_dict


def _update_disturbed_carbon_object_list(vars_dict, snapshot_idx):
    """pct_disturbed_raster * stock_raster --> total_carbon_disturbed_raster.

    Returns:

        vars_dict['disturbed_carbon_stock_object_list'][snapshot_idx]
    """
    def _reclass_lulc_to_pct_disturbed(vars_dict, idx, pool):
        if pool == 'biomass':
            d = vars_dict['disturbance_biomass_reclass_dict']
        else:
            d = vars_dict['disturbance_soil_reclass_dict']
        return _reclass_lulc(vars_dict, idx, pool, d)

    def _reclass_lulc_to_half_life(vars_dict, idx, pool):
        if pool == 'biomass':
            d = vars_dict['half_life_biomass_reclass_dict']
        else:
            d = vars_dict['half_life_soil_reclass_dict']
        return _reclass_lulc(vars_dict, idx, pool, d)

    def _reclass_lulc(vars_dict, idx, pool, reclass_dict):
        prev_raster = Raster.from_file(vars_dict['lulc_snapshot_list'][idx])
        next_raster = Raster.from_file(vars_dict['lulc_snapshot_list'][idx+1])

        prev_lulc = prev_raster.get_band(1)
        next_lulc = next_raster.get_band(1)

        lookup = dict([((i, j), reclass_dict[
                        (i, j)]) for i, j in set(
                            zip(prev_lulc.flatten(), next_lulc.flatten()))])

        flat_lookup = collections.defaultdict(dict)
        for (i, j), val in lookup.iteritems():
            flat_lookup[i][j] = val

        next_lulc_keys = {}
        next_lulc_values = {}

        for i in flat_lookup:
            next_lulc_keys[i] = sorted(flat_lookup[i].keys())
            next_lulc_values[i] = \
                np.array([flat_lookup[i][j] for j in next_lulc_keys[i]])

        def op(prev_lulc, next_lulc):
            result = np.empty(prev_lulc.shape)
            result[:] = NODATA_FLOAT
            for prev_lulc_value in np.unique(prev_lulc):
                prev_lulc_value_mask = prev_lulc == prev_lulc_value
                index = np.digitize(
                    next_lulc[prev_lulc_value_mask].ravel(),
                    next_lulc_keys[prev_lulc_value],
                    right=True)
                result[prev_lulc_value_mask] = \
                    next_lulc_values[prev_lulc_value][index]
            return result

        bounding_box_mode = "dataset"
        resample_method = "nearest"
        dataset_uri_list = [prev_raster.uri, next_raster.uri]
        resample_list = [resample_method] * 2
        dataset_out_uri = pygeo.geoprocessing.temporary_filename()
        datatype_out = gdal.GDT_Float32
        nodata_out = NODATA_FLOAT
        pixel_size_out = pygeo.geoprocessing.get_cell_size_from_uri(
            prev_raster.uri)

        pygeo.geoprocessing.vectorize_datasets(
            dataset_uri_list,
            op,
            dataset_out_uri,
            datatype_out,
            nodata_out,
            pixel_size_out,
            bounding_box_mode,
            resample_method_list=resample_list,
            dataset_to_align_index=0,
            dataset_to_bound_index=0,
            assert_datasets_projected=False,
            vectorize_op=False)

        return dataset_out_uri

    # Find percent distrubed from transitions
    pct_biomass_stock_disturbed_raster = Raster.from_file(
        _reclass_lulc_to_pct_disturbed(vars_dict, snapshot_idx, 'biomass'))
    pct_soil_stock_disturbed_raster = Raster.from_file(
        _reclass_lulc_to_pct_disturbed(vars_dict, snapshot_idx, 'soil'))

    # Get pre-transition carbon stock
    prev_biomass_carbon_stock_biomass_raster = \
        vars_dict['biomass_carbon_stock_raster_list'][snapshot_idx]
    prev_soil_carbon_stock_soil_raster = \
        vars_dict['soil_carbon_stock_raster_list'][snapshot_idx]

    # Calculate total amount of carbon stock disturbed
    final_biomass_stock_disturbed_raster = \
        pct_biomass_stock_disturbed_raster * prev_biomass_carbon_stock_biomass_raster
    final_soil_stock_disturbed_raster = \
        pct_soil_stock_disturbed_raster * prev_soil_carbon_stock_soil_raster

    # Find half-lives
    biomass_half_life_raster = Raster.from_file(
        _reclass_lulc_to_half_life(vars_dict, snapshot_idx, 'biomass'))
    soil_half_life_raster = Raster.from_file(
        _reclass_lulc_to_half_life(vars_dict, snapshot_idx, 'soil'))

    # Create DisturbedCarbonStock object
    disturbed_carbon_stock_object = DisturbedCarbonStock(
        vars_dict['lulc_snapshot_years_list'][snapshot_idx],
        final_biomass_stock_disturbed_raster,
        final_soil_stock_disturbed_raster,
        biomass_half_life_raster,
        soil_half_life_raster)

    # Add object to list
    vars_dict['disturbed_carbon_stock_object_list'][snapshot_idx] = \
        disturbed_carbon_stock_object

    return vars_dict
