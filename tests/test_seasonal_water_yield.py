"""InVEST Seasonal water yield model tests that use the InVEST sample data"""

import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'seasonal_water_yield')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), 'data', 'seasonal_water_yield')


class ExampleTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_base_regression(self):
        """
        SWY base regression test
        """
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        args = {
            u'alpha_m': u'1/12',
            u'aoi_path': os.path.join(SAMPLE_DATA, 'watershed.shp'),
            u'beta_i': u'1.0',
            u'biophysical_table_path': os.path.join(SAMPLE_DATA, 'biophysical_table.csv'),
            u'climate_zone_raster_path': os.path.join(SAMPLE_DATA, 'climate_zones.tif'),
            u'climate_zone_table_path': os.path.join(SAMPLE_DATA, 'climate_zone_events.csv'),
            u'dem_raster_path': os.path.join(SAMPLE_DATA, 'dem.tif'),
            u'et0_dir': os.path.join(SAMPLE_DATA, 'eto_dir'),
            u'gamma': u'1.0',
            u'lulc_raster_path': os.path.join(SAMPLE_DATA, 'lulc.tif'),
            u'precip_dir': os.path.join(SAMPLE_DATA, 'precip_dir'),
            u'rain_events_table_path': os.path.join(SAMPLE_DATA, 'rain_events_table.csv'),
            u'results_suffix': '',
            u'soil_group_path': os.path.join(SAMPLE_DATA, 'soil_group.tif'),
            u'threshold_flow_accumulation': u'1000',
            u'user_defined_climate_zones': False,
            u'user_defined_local_recharge': False,
            u'workspace_dir': tempfile.mkdtemp(),
        }

        seasonal_water_yield.execute(args)

        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(args['workspace_dir'], 'L.tif'),
            os.path.join(REGRESSION_DATA, 'base_swy_output', 'L.tif'))

        shutil.rmtree(args['workspace_dir'])
