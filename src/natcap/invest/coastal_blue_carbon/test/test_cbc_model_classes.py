"""Test Cases for CBC Model Classes.

python -m unittest test_cbc_model_classes.TestDisturbedCarbonStock
"""

import unittest
import os
import pprint

import numpy
from numpy import testing
import gdal

from natcap.invest.coastal_blue_carbon.global_variables import *
from natcap.invest.coastal_blue_carbon.classes.affine import Affine
from natcap.invest.coastal_blue_carbon.classes.raster import Raster
from natcap.invest.coastal_blue_carbon.classes.raster_factory import RasterFactory
from natcap.invest.coastal_blue_carbon.classes.model_carbon_classes import \
    DisturbedCarbonStock, AccumulatedCarbonStock

pp = pprint.PrettyPrinter(indent=4)


class TestDisturbedCarbonStock(unittest.TestCase):
    def setUp(self):
        # AOI Rasters
        # set arguments
        shape = (2, 2)  # (2, 2)  #(1889, 1325)
        affine = Affine(30.0, 0.0, 443723.127328, 0.0, -30.0, 4956546.905980)
        proj = 26910
        datatype = gdal.GDT_Float32
        nodata_val = NODATA_FLOAT

        # initialize factory
        self.aoi_float_factory = RasterFactory(
            proj, datatype, nodata_val, shape[0], shape[1], affine=affine)

        self.start_year = 2000
        self.final_biomass_stock_disturbed_raster = self.aoi_float_factory.uniform(0.5)
        self.final_soil_stock_disturbed_raster = self.aoi_float_factory.uniform(0.4)
        self.biomass_half_life_raster = self.aoi_float_factory.alternating(1, 0.5)
        self.soil_half_life_raster = self.aoi_float_factory.alternating(1, 0.5)

    # def test_disturbed_carbon_stock_object(self):
    #     final_biomass_stock_disturbed_raster = self.aoi_float_factory.uniform(0.5)
    #     final_soil_stock_disturbed_raster = self.aoi_float_factory.uniform(0.4)
    #     biomass_half_life_raster = self.aoi_float_factory.alternating(1, 0.5)
    #     soil_half_life_raster = self.aoi_float_factory.alternating(1, 0.5)
    #
    #     d = DisturbedCarbonStock(
    #         self.start_year,
    #         final_biomass_stock_disturbed_raster,
    #         final_soil_stock_disturbed_raster,
    #         biomass_half_life_raster,
    #         soil_half_life_raster)
    #
    #     # emissions between 2001 and 2002
    #     assert(d.get_biomass_emissions_between_years(
    #         2001, 2002).get_band(1)[0, 0] == 0.125)
    #     testing.assert_array_almost_equal(
    #         d.get_soil_emissions_between_years(
    #           2001, 2002).get_band(1)[0, 0], numpy.array(0.1), decimal=5)
    #     testing.assert_array_almost_equal(
    #         d.get_total_emissions_between_years(
    #           2001, 2002).get_band(1)[0, 0], numpy.array(0.225), decimal=5)

    def test_disturbed_carbon_stock_object_2(self):
        final_biomass_stock_disturbed_raster = self.aoi_float_factory.alternating(float('nan'), 0.5)
        final_soil_stock_disturbed_raster = self.aoi_float_factory.alternating(float('nan'), 0.4)
        biomass_half_life_raster = self.aoi_float_factory.alternating(0, 0.5)
        soil_half_life_raster = self.aoi_float_factory.alternating(0, 0.3)

        d = DisturbedCarbonStock(
            self.start_year,
            final_biomass_stock_disturbed_raster,
            final_soil_stock_disturbed_raster,
            biomass_half_life_raster,
            soil_half_life_raster)
        self.assertEqual(
            d.get_total_emissions_between_years(2000, 2005)[0, 0], 0)


class TestAccumulatedCarbonStock(unittest.TestCase):
    def setUp(self):
        # AOI Rasters
        # set arguments
        shape = (2, 2)  # (2, 2)  #(1889, 1325)
        affine = Affine(30.0, 0.0, 443723.127328, 0.0, -30.0, 4956546.905980)
        proj = 26910
        datatype = gdal.GDT_Float32
        nodata_val = NODATA_FLOAT

        # initialize factory
        aoi_float_factory = RasterFactory(
            proj, datatype, nodata_val, shape[0], shape[1], affine=affine)

        self.start_year = 2000
        self.yearly_sequest_biomass_raster = aoi_float_factory.uniform(2)
        self.yearly_sequest_soil_raster = aoi_float_factory.uniform(1)

    def test_disturbed_carbon_stock_object(self):
        a = AccumulatedCarbonStock(
            self.start_year,
            self.yearly_sequest_biomass_raster,
            self.yearly_sequest_soil_raster)
        end_year = 2001

        # sequestration between 2001 and 2002
        self.assertEqual(
            a.get_biomass_sequestered_by_year(end_year).get_band(1)[0, 0], 2)
        self.assertEqual(
            a.get_soil_sequestered_by_year(end_year).get_band(1)[0, 0], 1)
        self.assertEqual(
            a.get_total_sequestered_by_year(end_year).get_band(1)[0, 0], 3)


if __name__ == '__main__':
    unittest.main()
