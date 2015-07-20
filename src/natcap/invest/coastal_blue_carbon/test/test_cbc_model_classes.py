'''
python -m unittest test_cbc_model_classes.TestDisturbedCarbonStock
'''

import unittest
import os
import pprint

import gdal

import natcap.invest.coastal_blue_carbon as cbc
from cbc.utilities.affine import Affine
from cbc.utilities.raster import Raster
from cbc.utilities.raster_factory import RasterFactory
from cbc.utilities.cbc_model_classes import DisturbedCarbonStock, AccumulatedCarbonStock

pp = pprint.PrettyPrinter(indent=4)

NODATA_FLOAT = -16777216
NODATA_INT = -9999


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
        aoi_float_factory = RasterFactory(
            proj, datatype, nodata_val, shape[0], shape[1], affine=affine)

        self.start_year = 2000
        self.final_biomass_stock_disturbed_raster = aoi_float_factory.uniform(0.5)
        self.final_soil_stock_disturbed_raster = aoi_float_factory.uniform(0.4)
        self.biomass_half_life_raster = aoi_float_factory.uniform(1)
        self.soil_half_life_raster = aoi_float_factory.uniform(1)

    def test_disturbed_carbon_stock_object(self):
        d = DisturbedCarbonStock(
            self.start_year,
            self.final_biomass_stock_disturbed_raster,
            self.final_soil_stock_disturbed_raster,
            self.biomass_half_life_raster,
            self.soil_half_life_raster)

        print "BIOMASS EMISSIONS BETWEEN 2000 and 2001"
        print d.get_biomass_emissions_between_years(2000, 2001)
        # d.get_soil_emissions_between_years(2000, 2001)
        # d.get_total_emissions_between_years(2000, 2001)

    def tearDown(self):
        pass




class TestCBCPreprocessor(unittest.TestCase):
    def setUp(self):
        # create lookup.csv
        cwd = os.path.dirname(os.path.realpath(__file__))
        table = [
            ['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
            ['seagrass', '1', 'true'],
            ['man-made', '2', 'false'],
            ['marsh', '3', 'true'],
            ['mangrove', '4', 'true']]
        self.lookup_table_uri = os.path.join(cwd, 'lookup.csv')
        write_csv(self.lookup_table_uri, table)

        # set arguments
        shape = (2, 2)  # (2, 2)  #(1889, 1325)
        affine = Affine(30.0, 0.0, 443723.127328, 0.0, -30.0, 4956546.905980)
        proj = 26910
        datatype = gdal.GDT_Int32
        nodata_val = 255

        # initialize factory
        aoi_int_factory = RasterFactory(
            proj, datatype, nodata_val, shape[0], shape[1], affine=affine)

        # LULC Map
        self.year1_raster = aoi_int_factory.alternating(1, 2)
        self.year2_raster = aoi_int_factory.alternating(2, 1)
        self.year3_raster = aoi_int_factory.alternating(3, 1)
        self.year4_raster = aoi_int_factory.alternating(4, 1)

        self.workspace_dir = os.path.join(cwd, 'workspace')

        self.args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'lulc_lookup_uri': self.lookup_table_uri,
            'lulc_snapshot_list': [
                self.year1_raster.uri,
                self.year2_raster.uri,
                self.year3_raster.uri,
                self.year4_raster.uri]
        }

    def test_cbc_preprocessor(self):
        cbc_preprocessor.execute(self.args)
        transition_dict = get_lookup_from_csv(
            os.path.join(self.workspace_dir, 'outputs', 'transitions.csv'), 'lulc-class')
        assert(transition_dict['seagrass']['seagrass'] == 'accumulation')

    def tearDown(self):
        # remove lookup.csv
        if os.path.isfile(self.lookup_table_uri):
            os.remove(self.lookup_table_uri)

        # remove transition.csv
        transition_table_uri = os.path.join(
            self.workspace_dir, 'outputs', 'transitions.csv')
        if os.path.isfile(transition_table_uri):
            os.remove(transition_table_uri)

        # remove outputs and workspace
        output_dir = os.path.join(self.workspace_dir, 'outputs')
        if os.path.isdir(output_dir):
            os.removedirs(output_dir)
        if os.path.isdir(self.workspace_dir):
            os.removedirs(self.workspace_dir)


if __name__ == '__main__':
    unittest.main()
