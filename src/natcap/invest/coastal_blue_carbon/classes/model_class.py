"""Classes specific to the Coastal Blue Carbon model."""

import logging
import os
import collections
import pprint as pp

import gdal
import pygeoprocessing as pygeo
import numpy as np

from natcap.invest.coastal_blue_carbon.global_variables import *
from natcap.invest.coastal_blue_carbon.classes.model_carbon_classes import \
    DisturbedCarbonStock, AccumulatedCarbonStock
from natcap.invest.coastal_blue_carbon.classes.raster import Raster

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.coastal_blue_carbon.coastal_blue_carbon')


class CBCModel(object):

    """Coastal Blue Carbon Model Run class."""

    def __init__(self, vars_dict):
        self.vars_dict = vars_dict

        # useful variables
        self.num_lulc_maps = len(vars_dict['lulc_snapshot_list'])
        self.num_snapshots = self.num_lulc_maps
        if (vars_dict['analysis_year'] != '' and vars_dict['analysis_year']
                > vars_dict['lulc_snapshot_years_list'][-1]):
            self.num_snapshots += 1
            last_lulc_snapshot_url = vars_dict['lulc_snapshot_list'][-1]
            vars_dict['lulc_snapshot_list'].append(last_lulc_snapshot_url)
        self.num_transitions = self.num_snapshots - 1
        self.do_economic_analysis = vars_dict['do_economic_analysis']
        self.yearly_carbon_price_dict = vars_dict['yearly_carbon_price_dict']

        self.lulc_snapshot_list = vars_dict['lulc_snapshot_list']
        self.lulc_snapshot_years_list = vars_dict['lulc_snapshot_years_list']

        # Intermediate Maps
        self.accumulated_carbon_stock_object_list = range(
            0, self.num_transitions)
        self.disturbed_carbon_stock_object_list = range(
            0, self.num_transitions)
        self.biomass_carbon_stock_raster_list = range(0, self.num_snapshots)
        self.soil_carbon_stock_raster_list = range(0, self.num_snapshots)
        self.litter_carbon_stock_raster_list = range(0, self.num_snapshots)

        # Key Maps
        self.total_carbon_stock_raster_list = range(0, self.num_snapshots)
        self.sequestration_raster_list = range(0, self.num_transitions)
        self.emissions_raster_list = range(0, self.num_transitions)
        self.net_sequestration_raster_list = range(0, self.num_transitions)

        self.npv_raster = Raster.from_file(self.lulc_snapshot_list[
            0]).zeros().set_datatype_and_nodata(gdal.GDT_Float32, NODATA_FLOAT)

    def __str__(self):
        string =  '\nCBCModelRun Class -----'
        string += '\n   lulc_snapshot_list:' + str(self.lulc_snapshot_list)
        string += '\n   lulc_snapshot_years_list:' + str(self.lulc_snapshot_years_list)
        string += '\n   total_carbon_stock_list:' + str([type(i) for i in self.total_carbon_stock_raster_list])
        string += '\n   biomass_carbon_stock_list:' + str([type(i) for i in self.biomass_carbon_stock_raster_list])
        string += '\n   soil_carbon_stock_list:' + str([type(i) for i in self.soil_carbon_stock_raster_list])
        string += '\n   accum_object_list:' + str([type(i) for i in self.accumulated_carbon_stock_object_list])
        string += '\n   dist_object_list:' + str([type(i) for i in self.disturbed_carbon_stock_object_list])
        string += '\n   sequest_list:' + str([type(i) for i in self.sequestration_raster_list])
        string += '\n   emissions_list:' + str([type(i) for i in self.emissions_raster_list])
        string += '\n   net_sequest_list:' + str([type(i) for i in self.net_sequestration_raster_list])
        string += '\n---------\n'
        return string

    def run(self):
        LOGGER.info("Running model...")
        self.initialize_stock()
        self.run_transient_analysis()
        self.save_rasters()
        LOGGER.info("...model run finished.")

    def initialize_stock(self):
        """Set inital stock for biomass, soil, total (plus litter).

        Changes:

            vars_dict['biomass_carbon_stock_raster_list'][0]
            vars_dict['soil_carbon_stock_raster_list'][0]
            vars_dict['total_carbon_stock_raster_list'][0]
        """
        LOGGER.info("Initializing stock...")
        carbon_pool_initial_dict = self.vars_dict['carbon_pool_initial_dict']
        lulc_to_code_dict = self.vars_dict['lulc_to_code_dict']

        code_to_biomass_reclass_dict = dict(
            [(lulc_to_code_dict[item[0]],
                item[1]['biomass']) for item in carbon_pool_initial_dict.items()])
        code_to_soil_reclass_dict = dict(
            [(lulc_to_code_dict[item[0]],
                item[1]['soil']) for item in carbon_pool_initial_dict.items()])

        init_lulc_raster = Raster.from_file(self.lulc_snapshot_list[0])

        # process litter rasters
        for i in range(0, self.num_snapshots):
            self._compute_litter_raster(i)

        # Create initial carbon stock rasters
        init_carbon_stock_biomass_raster = init_lulc_raster.reclass(
            code_to_biomass_reclass_dict,
            out_datatype=gdal.GDT_Float32,
            out_nodata=NODATA_FLOAT) * init_lulc_raster.get_cell_area() * HA_PER_M2
        init_carbon_stock_soil_raster = init_lulc_raster.reclass(
            code_to_soil_reclass_dict,
            out_datatype=gdal.GDT_Float32,
            out_nodata=NODATA_FLOAT) * init_lulc_raster.get_cell_area() * HA_PER_M2
        init_carbon_stock_total_raster = \
            init_carbon_stock_biomass_raster + init_carbon_stock_soil_raster \
            + self.litter_carbon_stock_raster_list[0]

        # Add rasters to lists
        self.total_carbon_stock_raster_list[0] = init_carbon_stock_total_raster
        self.biomass_carbon_stock_raster_list[0] = \
            init_carbon_stock_biomass_raster
        self.soil_carbon_stock_raster_list[0] = init_carbon_stock_soil_raster
        LOGGER.info("...stock initialized.")

    def _compute_litter_raster(self, idx):
        LOGGER.info("Computing litter pool stock...")
        lulc_to_code_dict = self.vars_dict['lulc_to_code_dict']
        carbon_pool_initial_dict = self.vars_dict['carbon_pool_initial_dict']
        reclass_dict = dict([(lulc_to_code_dict[k], v['litter'])
                            for k, v in carbon_pool_initial_dict.items()])
        lulc_float_raster = Raster.from_file(
            self.lulc_snapshot_list[idx]).set_datatype_and_nodata(
                gdal.GDT_Float32, NODATA_FLOAT)
        init_litter_raster = lulc_float_raster.reclass(
            reclass_dict) * lulc_float_raster.get_cell_area() * HA_PER_M2
        self.litter_carbon_stock_raster_list[idx] = init_litter_raster

    def run_transient_analysis(self):
        LOGGER.info("Running transient analysis...")
        for idx in range(0, self.num_transitions):
            self._compute_transient_step(idx)
            if self.do_economic_analysis:
                self._update_npv_raster(idx)
        LOGGER.info("...transient analysis complete.")

    def _get_end_year(self, idx):
        """Get end year of next transition."""
        if (idx == self.num_lulc_maps - 1):
            if self.vars_dict['analysis_year'] != '':
                end_year = self.vars_dict['analysis_year']
            else:
                end_year = self.lulc_snapshot_years_list[idx] + 1
        else:
            end_year = self.lulc_snapshot_years_list[idx+1]

        return end_year

    def _compute_transient_step(self, idx):
        LOGGER.info("Computing transient step %i..." % idx)
        start_year = self.lulc_snapshot_years_list[idx]
        end_year = self._get_end_year(idx)

        self._update_transient_carbon_reclass_dicts(idx)
        self._update_accumulated_carbon_object_list(idx)
        self._update_disturbed_carbon_object_list(idx)

        # Sequestration between Start_Year and End_Year
        a = self.accumulated_carbon_stock_object_list[idx]
        sequestered_over_time_raster = a.get_total_sequestered_by_year(end_year)
        self.sequestration_raster_list[idx] = sequestered_over_time_raster

        # Emissions between Start_Year and End_Year
        d_list = self.disturbed_carbon_stock_object_list
        emitted_over_time_raster = d_list[0].final_biomass_stock_disturbed_raster.zeros()
        for i in range(0, idx+1):
            emitted_over_time_raster += d_list[i].get_total_emissions_between_years(start_year, end_year)
        self.emissions_raster_list[idx] = emitted_over_time_raster

        # Net Sequestration between Start_Year and End_Year
        net_sequestered_over_time_raster = sequestered_over_time_raster - \
            emitted_over_time_raster
        self.net_sequestration_raster_list[idx] = net_sequestered_over_time_raster

        # Stock at End_Year
        prev_carbon_stock_raster = self.total_carbon_stock_raster_list[idx]
        litter_change_raster = self.litter_carbon_stock_raster_list[idx+1] \
            - self.litter_carbon_stock_raster_list[idx]
        next_carbon_stock_raster = prev_carbon_stock_raster + \
            net_sequestered_over_time_raster + litter_change_raster
        self.total_carbon_stock_raster_list[idx+1] = next_carbon_stock_raster
        LOGGER.info("...transient step %i complete." % idx)

    def _update_transient_carbon_reclass_dicts(self, idx):
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
            carbon_pool_transient_dict = vars_dict['carbon_pool_transient_dict']
            lulc_vals = set(next_lulc_raster.get_band(1).data.flatten())
            d = {}
            for i in lulc_vals:
                try:
                    if (code_to_lulc_dict[i], pool) in carbon_pool_transient_dict:
                        d[i] = carbon_pool_transient_dict[
                            (code_to_lulc_dict[i], pool)]['yearly_accumulation']
                    else:
                        d[i] = 0
                except:
                    d[i] = 0
            return d

        def _create_disturbance_reclass_dicts(vars_dict, pool):
            """Create disturbance reclass dicts.

            Note: pull this out into a one-time call at some point

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
                    if carbon_mag_and_dir not in [
                            '', 'accum', 'NCC']:
                        disturbance_val = carbon_pool_transient_dict[
                            (prev_lulc, pool)][carbon_mag_and_dir]
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
            h = {}
            undisturbed_set = set(['', 'accum', 'NCC'])
            for prev_lulc, val in lulc_transition_dict.items():
                for next_lulc, carbon_mag_and_dir in val.items():
                    if carbon_mag_and_dir not in undisturbed_set:
                        half_life_val = carbon_pool_transient_dict[
                            (next_lulc, pool)]['half-life']
                        h[(lulc_to_code_dict[prev_lulc], lulc_to_code_dict[
                            next_lulc])] = half_life_val
                    else:
                        h[(lulc_to_code_dict[prev_lulc], lulc_to_code_dict[
                            next_lulc])] = 1
            return h

        LOGGER.info("Updating carbon reclass dictionaries...")
        next_raster = Raster.from_file(
            self.vars_dict['lulc_snapshot_list'][idx+1])

        self.vars_dict['accumulation_biomass_reclass_dict'] = \
            _create_accumulation_reclass_dicts(
                self.vars_dict, next_raster, 'biomass')
        self.vars_dict['accumulation_soil_reclass_dict'] = \
            _create_accumulation_reclass_dicts(
                self.vars_dict, next_raster, 'soil')

        self.vars_dict['disturbance_biomass_reclass_dict'] = \
            _create_disturbance_reclass_dicts(
                self.vars_dict, 'biomass')
        self.vars_dict['disturbance_soil_reclass_dict'] = \
            _create_disturbance_reclass_dicts(
                self.vars_dict, 'soil')

        self.vars_dict['half_life_biomass_reclass_dict'] = \
            _create_half_life_reclass_dicts(
                self.vars_dict, 'biomass')
        self.vars_dict['half_life_soil_reclass_dict'] = \
            _create_half_life_reclass_dicts(
                self.vars_dict, 'soil')
        LOGGER.info("...carbon reclass dictionaries complete.")

    def _update_accumulated_carbon_object_list(self, idx):
        LOGGER.info("Updating accumulated carbon stock...")
        next_lulc_raster = Raster.from_file(
            self.vars_dict['lulc_snapshot_list'][idx+1])
        accumulation_biomass_reclass_dict = \
            self.vars_dict['accumulation_biomass_reclass_dict']
        accumulation_soil_reclass_dict = \
            self.vars_dict['accumulation_soil_reclass_dict']

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
            self.lulc_snapshot_years_list[idx],
            yearly_sequest_biomass_raster,
            yearly_sequest_soil_raster)

        self.accumulated_carbon_stock_object_list[idx] = \
            accumulated_carbon_stock_object
        LOGGER.info("...accumulated carbon stock update complete.")

    def _update_disturbed_carbon_object_list(self, idx):
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

            prev_lulc = prev_raster.get_band(1).data
            next_lulc = next_raster.get_band(1).data

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

        LOGGER.info("Updating disturbed carbon stock...")
        # Find percent distrubed from transitions
        pct_biomass_stock_disturbed_raster = Raster.from_file(
            _reclass_lulc_to_pct_disturbed(self.vars_dict, idx, 'biomass'))
        pct_soil_stock_disturbed_raster = Raster.from_file(
            _reclass_lulc_to_pct_disturbed(self.vars_dict, idx, 'soil'))

        # Get pre-transition carbon stock
        prev_biomass_carbon_stock_biomass_raster = \
            self.biomass_carbon_stock_raster_list[idx]
        prev_soil_carbon_stock_soil_raster = \
            self.soil_carbon_stock_raster_list[idx]

        # Calculate total amount of carbon stock disturbed
        final_biomass_stock_disturbed_raster = \
            pct_biomass_stock_disturbed_raster * prev_biomass_carbon_stock_biomass_raster
        final_soil_stock_disturbed_raster = \
            pct_soil_stock_disturbed_raster * prev_soil_carbon_stock_soil_raster

        # Find half-lives
        biomass_half_life_raster = Raster.from_file(
            _reclass_lulc_to_half_life(self.vars_dict, idx, 'biomass'))
        soil_half_life_raster = Raster.from_file(
            _reclass_lulc_to_half_life(self.vars_dict, idx, 'soil'))

        # Create DisturbedCarbonStock object
        disturbed_carbon_stock_object = DisturbedCarbonStock(
            self.lulc_snapshot_years_list[idx],
            final_biomass_stock_disturbed_raster,
            final_soil_stock_disturbed_raster,
            biomass_half_life_raster,
            soil_half_life_raster)

        # Add object to list
        self.disturbed_carbon_stock_object_list[idx] = \
            disturbed_carbon_stock_object
        LOGGER.info("...disturbed carbon stock update complete.")

    def _update_npv_raster(self, idx):
        LOGGER.info("Updating NPV raster...")
        start_year = self.lulc_snapshot_years_list[idx]
        end_year = self._get_end_year(idx)

        for year in range(start_year, end_year):
            LOGGER.info("Computing net present value for year %s" % year)
            net_sequestered_raster = self._compute_net_sequestration(idx, year)
            npv_raster = net_sequestered_raster * \
                self.yearly_carbon_price_dict[year]
            self.npv_raster += npv_raster
        LOGGER.info("...NPV raster updated.")

    def _compute_net_sequestration(self, idx, year):
        # Sequestration for a given year
        a = self.accumulated_carbon_stock_object_list[idx]
        sequestered_raster = a.get_total_sequestered_between_years(
            year, year+1)

        # Emissions for a given year
        d_list = self.disturbed_carbon_stock_object_list
        emitted_raster = d_list[0].final_biomass_stock_disturbed_raster.zeros()
        for i in range(0, idx+1):
            emitted_raster += d_list[i].get_total_emissions_between_years(
                year, year+1)

        # Net Sequestration for a given year
        net_sequestered_raster = sequestered_raster - emitted_raster

        return net_sequestered_raster

    def save_rasters(self):
        LOGGER.info("Saving rasters...")
        years_list = self.lulc_snapshot_years_list
        if self.vars_dict['analysis_year'] != '':
            years_list.append(self.vars_dict['analysis_year'])
        # Total Carbon Stock
        for i in range(0, len(self.total_carbon_stock_raster_list)):
            r = self.total_carbon_stock_raster_list[i]
            filename = 'carbon_stock_at_%s.tif' % years_list[i]
            r.save_raster(
                os.path.join(self.vars_dict['outputs_dir'], filename))
        # Total Emissions
        for i in range(0, len(self.emissions_raster_list)):
            r = self.emissions_raster_list[i]
            filename = 'carbon_emissions_between_%s_and_%s.tif' % (
                years_list[i], years_list[i+1])
            r.save_raster(os.path.join(
                self.vars_dict['outputs_dir'], filename))
        # Total Sequestration
        for i in range(0, len(self.sequestration_raster_list)):
            r = self.sequestration_raster_list[i]
            filename = 'carbon_sequestration_between_%s_and_%s.tif' % (
                years_list[i], years_list[i+1])
            r.save_raster(os.path.join(
                self.vars_dict['outputs_dir'], filename))
        # Net Sequestration
        for i in range(0, len(self.net_sequestration_raster_list)):
            r = self.net_sequestration_raster_list[i]
            filename = 'net_carbon_sequestration_between_%s_and_%s.tif' % (
                years_list[i], years_list[i+1])
            r.save_raster(os.path.join(
                self.vars_dict['outputs_dir'], filename))
        # Net Present Value
        if self.do_economic_analysis:
            r = self.npv_raster
            filename = 'net_present_value.tif'
            r.save_raster(os.path.join(
                self.vars_dict['outputs_dir'], filename))
        LOGGER.info("...rasters saved.")
