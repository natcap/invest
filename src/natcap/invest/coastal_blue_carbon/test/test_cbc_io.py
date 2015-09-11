"""Test Cases for CBC IO Functions.

python -m unittest test_cbc_io
"""

import unittest
import os
import pprint
import shutil

import gdal

import natcap.invest.coastal_blue_carbon.utilities.io as io
from natcap.invest.coastal_blue_carbon.global_variables import *
from natcap.invest.coastal_blue_carbon.classes.raster_factory import \
    RasterFactory
from natcap.invest.coastal_blue_carbon.classes.affine import Affine

pp = pprint.PrettyPrinter(indent=4)


class TestGetInputs(unittest.TestCase):

    """Test io.get_inputs."""

    def setUp(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        workspace = os.path.join(cwd, 'workspace')
        if not os.path.exists(workspace):
            os.mkdir(workspace)
        self.workspace = workspace
        self.results_suffix = ""

        table = [
            ['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
            ['seagrass', '1', 'true'],
            ['man-made', '2', 'false'],
            ['marsh', '3', 'true'],
            ['mangrove', '4', 'true']]
        self.lulc_lookup_uri = os.path.join(self.workspace, 'lookup.csv')
        io.write_csv(self.lulc_lookup_uri, table)

        table = [
            ['lulc-class', 'seagrass', 'man-made', 'marsh', 'mangrove'],
            ['seagrass', 'accum', 'disturb', '', ''],
            ['man-made', 'accum', '', 'accum', ''],
            ['marsh', '', '', '', 'accum'],
            ['mangrove', '', '', '', '']]
        self.lulc_transition_uri = os.path.join(
            self.workspace, 'transition.csv')
        io.write_csv(self.lulc_transition_uri, table)

        shape = (2, 2)  # (2, 2)  #(1889, 1325)
        affine = Affine(30.0, 0.0, 443723.127328, 0.0, -30.0, 4956546.905980)
        proj = 26910
        datatype = gdal.GDT_Int32
        nodata_val = 255
        aoi_int_factory = RasterFactory(
            proj, datatype, nodata_val, shape[0], shape[1], affine=affine)
        year1_raster = aoi_int_factory.alternating(1, 2)
        year2_raster = aoi_int_factory.alternating(2, 1)
        year3_raster = aoi_int_factory.alternating(3, 1)
        year4_raster = aoi_int_factory.alternating(4, 1)
        self.lulc_snapshot_list = [
            year1_raster.uri,
            year2_raster.uri,
            year3_raster.uri,
            year4_raster.uri]

        self.lulc_snapshot_years_list = [2000, 2005, 2020, 2050]
        self.analysis_year = 2100

        table = [
            ['lulc-class', 'biomass', 'soil', 'litter'],
            ['seagrass', '1.0', '1.0', '0.5'],
            ['man-made', '0.0', '0.0', '0'],
            ['marsh', '2.0', '2.0', '1.0'],
            ['mangrove', '3.0', '3.0', '1.5']]
        self.carbon_pool_initial_uri = os.path.join(
            self.workspace, 'initial.csv')
        io.write_csv(self.carbon_pool_initial_uri, table)

        table = [
            ['lulc-class', 'pool', 'half-life', 'yearly_accumulation', 'low-impact-disturb', 'med-impact-disturb', 'high-impact-disturb'],
            ['seagrass', 'biomass', '1', '10', '0.1', '0.3', '0.7'],
            ['seagrass', 'soil', '2', '10', '0.1', '0.3', '0.7'],
            ['man-made', 'biomass', '0', '0', '0', '0', '0'],
            ['man-made', 'soil', '0', '0', '0', '0', '0'],
            ['marsh', 'biomass', '1', '20', '0.2', '0.4', '0.8'],
            ['marsh', 'soil', '2', '20', '0.2', '0.4', '0.8'],
            ['mangrove', 'biomass', '1', '30', '0.3', '0.5', '0.7'],
            ['mangrove', 'soil', '2', '30', '0.3', '0.5', '0.7']]
        self.carbon_pool_transient_uri = os.path.join(
            self.workspace, 'transient.csv')
        io.write_csv(self.carbon_pool_transient_uri, table)

        table = [['year', 'price']]
        for year in range(2000, 2101):
            table.append([str(year), '10.0'])
        self.price_table_uri = os.path.join(self.workspace, 'price_table.csv')
        io.write_csv(self.price_table_uri, table)

        self.args = {
            'workspace_dir': self.workspace,
            'results_suffix': self.results_suffix,
            'lulc_lookup_uri': self.lulc_lookup_uri,
            'lulc_transition_uri': self.lulc_transition_uri,
            'lulc_snapshot_list': self.lulc_snapshot_list,
            'lulc_snapshot_years_list': self.lulc_snapshot_years_list,
            'analysis_year': self.analysis_year,
            'carbon_pool_initial_uri': self.carbon_pool_initial_uri,
            'carbon_pool_transient_uri': self.carbon_pool_transient_uri,
            'do_economic_analysis': True,
            'do_price_table': True,
            'price': '10.0',
            'interest_rate': '5.0',
            'price_table_uri': self.price_table_uri,
            'discount_rate': '6.0'
        }

    def test_get_inputs(self):
        vars_dict = io.get_inputs(self.args)
        # pp.pprint(vars_dict)
        self.assertDictContainsSubset({'discount_rate': '6.0'}, vars_dict)
        self.assertTrue('lulc_lookup_dict' in vars_dict)

    # def test_get_discounted_price_dictionary(self):
    #     discounted_price_dict = io._get_discounted_price_dict(self.args)
    #     pp.pprint(discounted_price_dict)
    #     assert(discounted_price_dict[2100] == 0.029472262287399215)

    def tearDown(self):
        shutil.rmtree(self.workspace)


if __name__ == '__main__':
    unittest.main()
