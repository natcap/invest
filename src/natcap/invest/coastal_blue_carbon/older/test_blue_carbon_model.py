import unittest
import os
import pprint
import tempfile

from numpy import testing
import numpy as np
import rasterio as rio

import blue_carbon

input_dir = '../../test/invest-data/BlueCarbon/input'
pp = pprint.PrettyPrinter(indent=4)


class TestModel1(unittest.TestCase):
    def setUp(self):
        workspace_dir = tempfile.mkdtemp("subdir")
        # filepath = os.path.join(subdir, 'test.tif')

        self.args = {
            'workspace_dir': workspace_dir,
            'lulc_uri_1': os.path.join(
                input_dir, 'GBJC_2004_mean_Resample.tif'),
            'year_1': 2004,
            'lulc_uri_2': os.path.join(
                input_dir, 'GBJC_2050_mean_Resample.tif'),
            'year_2': 2050,
            'lulc_uri_3': os.path.join(
                input_dir, 'GBJC_2100_mean_Resample.tif'),
            'year_3': 2100,
            'analysis_year': 2150,
            'soil_disturbance_csv_uri': os.path.join(
                input_dir, 'soil_disturbance.csv'),
            'biomass_disturbance_csv_uri': os.path.join(
                input_dir, 'biomass_disturbance.csv'),
            'carbon_pools_uri': os.path.join(input_dir, 'carbon.csv'),
            'half_life_csv_uri': os.path.join(input_dir, 'half_life.csv'),
            'transition_matrix_uri': os.path.join(input_dir, 'transition.csv'),
            'do_private_valuation': True,
            'discount_rate': 5,
            'do_price_table': True,
            'carbon_schedule': os.path.join(input_dir, 'SCC5.csv')
            #'carbon_value': None,
            #'rate_change': None,
        }

    def test_run(self):
        blue_carbon.execute(self.args)

        print os.listdir(self.args['workspace_dir'])

        with rio.open(os.path.join(
                self.args['workspace_dir'], 'stock_2050.tif')) as src:
            print src.width
            print src.height
            print src.count
            print src.read_band(1)[0:100:5, 0:100:5]




if __name__ == '__main__':
    unittest.main()
