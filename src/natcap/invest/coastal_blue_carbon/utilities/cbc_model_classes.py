"""Classes specific to the Coastal Blue Carbon model."""

import logging
import os
import collections

import gdal
import pygeoprocessing as pygeo
import numpy as np

from natcap.invest.coastal_blue_carbon.utilities.raster import Raster

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.coastal_blue_carbon.coastal_blue_carbon')

# Global Variables
NODATA_FLOAT = -16777216
NODATA_INT = -9999
HA_PER_M2 = 0.0001


class DisturbedCarbonStock(object):

    """Disturbed Carbon Stock class."""

    def __init__(self,
                 start_year,
                 final_biomass_stock_disturbed_raster,
                 final_soil_stock_disturbed_raster,
                 biomass_half_life_raster,
                 soil_half_life_raster):
        self.start_year = start_year
        self.final_biomass_stock_disturbed_raster = final_biomass_stock_disturbed_raster
        self.final_soil_stock_disturbed_raster = final_soil_stock_disturbed_raster
        self.biomass_half_life_raster = biomass_half_life_raster
        self.soil_half_life_raster = soil_half_life_raster

    def __str__(self):
        string =  '\n=== DisturbedCarbonStock Object ==='
        string += '\nStart Year: %s' % self.start_year
        string += '\nFINAL AMT BIOMASS DISTURBED' + self.final_biomass_stock_disturbed_raster.__str__()
        string += '\nBIOMASS HALF-LIFE' + self.biomass_half_life_raster.__str__()
        string += '\nFINAL AMT SOIL DISTURBED' + self.final_soil_stock_disturbed_raster.__str__()
        string += '\nSOIL HALF-LIFE' + self.soil_half_life_raster.__str__()
        string += '==================================='
        return string

    def get_biomass_emissions_between_years(self, year_1, year_2):
        emissions_by_year_1_raster = self.final_biomass_stock_disturbed_raster * (
            1 - (0.5 ** ((year_1 - self.start_year) / self.biomass_half_life_raster)))
        emissions_by_year_2_raster = self.final_biomass_stock_disturbed_raster * (
            1 - (0.5 ** ((year_2 - self.start_year) / self.biomass_half_life_raster)))
        return emissions_by_year_2_raster - emissions_by_year_1_raster

    def get_soil_emissions_between_years(self, year_1, year_2):
        emissions_by_year_1_raster = self.final_soil_stock_disturbed_raster * (
            1 - (0.5 ** ((year_1 - self.start_year) / self.soil_half_life_raster)))
        emissions_by_year_2_raster = self.final_soil_stock_disturbed_raster * (
            1 - (0.5 ** ((year_2 - self.start_year) / self.soil_half_life_raster)))
        return emissions_by_year_2_raster - emissions_by_year_1_raster

    def get_total_emissions_between_years(self, year_1, year_2):
        biomass_raster = self.get_biomass_emissions_between_years(year_1, year_2)
        soil_raster = self.get_soil_emissions_between_years(year_1, year_2)
        return biomass_raster + soil_raster


class AccumulatedCarbonStock(object):

    """Accumulated Carbon Stock class."""

    def __init__(self, start_year, yearly_sequest_biomass_raster, yearly_sequest_soil_raster):
        self.start_year = start_year

        # 0's where no sequestration, else yearly_sequstration_per_ha
        self.yearly_sequest_biomass_raster = yearly_sequest_biomass_raster
        self.yearly_sequest_soil_raster = yearly_sequest_soil_raster

    def __str__(self):
        string =  '\nAccumulatedCarbonStock Object'
        string += '\nStart Year: %i' % self.start_year
        string += '\nBIOMASS' + self.yearly_sequest_biomass_raster.__str__()
        string += '\nSOIL' + self.yearly_sequest_soil_raster.__str__()
        return string

    def get_biomass_sequestered_by_year(self, year):
        years = year - self.start_year
        if years >= 0:
            return self.yearly_sequest_biomass_raster * years
        else:
            raise ValueError

    def get_soil_sequestered_by_year(self, year):
        years = year - self.start_year
        if years >= 0:
            return self.yearly_sequest_soil_raster * years
        else:
            raise ValueError

    def get_total_sequestered_by_year(self, year):
        years = year - self.start_year
        if years >= 0:
            return (self.yearly_sequest_biomass_raster + self.yearly_sequest_soil_raster) * years
        else:
            raise ValueError

class CBCModelRun(object):

    """Accumulated Carbon Stock class."""

    def __init__(self, vars_dict):
        self.vars_dict = vars_dict

        self.num_snapshots = len(vars_dict['lulc_snapshot_list'])
        if (vars_dict['analysis_year'] != '' and vars_dict['analysis_year']
                > vars_dict['lulc_snapshot_years_list'][-1]):
            self.num_snapshots += 1
        self.num_transitions = self.num_snapshots - 1

        self.lulc_snapshot_list = vars_dict['lulc_snapshot_list']
        self.lulc_snapshot_years_list = vars_dict['lulc_snapshot_years_list']

        # Stock
        self.total_carbon_stock_raster_list = range(0, self.num_snapshots)
        self.biomass_carbon_stock_raster_list = range(0, self.num_snapshots)
        self.soil_carbon_stock_raster_list = range(0, self.num_snapshots)

        # AccumulatedCarbonStock and DisturbedCarbonStock Objects
        self.accumulated_carbon_stock_object_list = range(
            0, self.num_transitions)
        self.disturbed_carbon_stock_object_list = range(
            0, self.num_transitions)

        # Sequestration
        self.sequestration_raster_list = range(0, self.num_transitions)

        # Emissions
        self.emissions_raster_list = range(0, self.num_transitions)

        # Net Sequstration
        self.net_sequestration_raster_list = range(0, self.num_transitions)

    def run(self):
        self.initialize_stock()
        self.run_transient_analysis()
        self.save_rasters()

    def initialize_stock(self):
        """Set inital stock for biomass, soil, total (plus litter).

        Changes:

            vars_dict['biomass_carbon_stock_raster_list'][0]
            vars_dict['soil_carbon_stock_raster_list'][0]
            vars_dict['total_carbon_stock_raster_list'][0]
        """
        carbon_pool_initial_dict = self.vars_dict['carbon_pool_initial_dict']
        lulc_to_code_dict = self.vars_dict['lulc_to_code_dict']

        code_to_biomass_reclass_dict = dict(
            [(lulc_to_code_dict[item[0]],
                item[1]['biomass']) for item in carbon_pool_initial_dict.items()])
        code_to_soil_reclass_dict = dict(
            [(lulc_to_code_dict[item[0]],
                item[1]['soil']) for item in carbon_pool_initial_dict.items()])

        init_lulc_raster = Raster.from_file(self.lulc_snapshot_list[0])

        # Create initial carbon stock rasters
        init_carbon_stock_biomass_raster = init_lulc_raster.reclass(
            code_to_biomass_reclass_dict,
            out_datatype=gdal.GDT_Float32,
            out_nodata=NODATA_FLOAT)
        init_carbon_stock_soil_raster = init_lulc_raster.reclass(
            code_to_soil_reclass_dict,
            out_datatype=gdal.GDT_Float32,
            out_nodata=NODATA_FLOAT)
        init_carbon_stock_total_raster = \
            init_carbon_stock_biomass_raster + init_carbon_stock_soil_raster

        # Add rasters to lists
        self.total_carbon_stock_raster_list = init_carbon_stock_total_raster
        self.biomass_carbon_stock_raster_list = \
            init_carbon_stock_biomass_raster
        self.soil_carbon_stock_raster_list = init_carbon_stock_soil_raster

    def run_transient_analysis(self):
        for idx in range(0, self.num_transitions):
            self._compute_transient_step(idx)

    def _compute_transient_step(self, idx):
        start_year = self.lulc_snapshot_years_list[idx]
        end_year = self.lulc_snapshot_years_list[idx+1]

        self._update_transient_carbon_reclass_dicts(idx)
        self._update_accumulated_carbon_object_list(idx)
        self._update_disturbed_carbon_object_list(idx)

        # Sequestration between Start_Year and End_Year
        a = self.accumulated_carbon_stock_object_list[idx]
        sequestered_over_time_raster = a.get_total_sequestered_by_year(end_year)
        self.sequestration_raster_list[idx] = sequestered_over_time_raster

        # Emissions between Start_Year and End_Year
        d_list = self.disturbed_carbon_stock_object_list[idx]
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
        next_carbon_stock_raster = prev_carbon_stock_raster + \
            net_sequestered_over_time_raster
        self.total_carbon_stock_raster_list[idx+1] = next_carbon_stock_raster

    def _update_transient_carbon_reclass_dicts(self):
        pass

    def _update_accumulated_carbon_object_list(self):
        pass

    def _update_disturbed_carbon_object_list(self):
        pass

    def save_rasters(self):
        pass
