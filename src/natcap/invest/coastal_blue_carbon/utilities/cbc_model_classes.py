"""Classes specific to the Coastal Blue Carbon model."""

HA_PER_M2 = 0.0001

class DisturbedCarbonStock(object):
    """Disturbed Carbon Stock class"""

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
    """Accumulated Carbon Stock class"""
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
