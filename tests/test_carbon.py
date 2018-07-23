"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
from osgeo import osr
import numpy
import pygeoprocessing.testing


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'carbon')


def make_lulc_rasters(args, raster_keys, start_val):
    """Create LULC rasters with specified raster names and starting value.

    Parameters:
        args (dictionary): the arguments used in the testing function.
        raster_keys (list): a list of raster name(s) that are either 
            'lulc_cur_path', 'lulc_fut_path' or 'lulc_redd_path'.
        start_val (int): the starting value used for filling the rasters(s).
    Returns:
        None.
    """

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    for val, key in enumerate(raster_keys, start=start_val):
        lulc_array = numpy.empty((10, 10))
        lulc_array.fill(val)
        lulc_path = os.path.join('.', key+'.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [lulc_array], (461261, 4923265), projection_wkt, -1, (1, -1), 
            filename=lulc_path)
        args[key] = lulc_path


def assert_npv(args, actual_npv, out_npv_filename):
    """Assert that the output npv array is the same as the npv array 
    computed manually based on the synthetic data.
    
    Parameters:
        args (dictionary): the arguments used in the testing function.
        actual_npv (float): the actual npv to be filled in the array.
        out_npv_filename (string): the filename of the output npv TIFF file.
    Returns:
        None.
    """
    actual_npv_arr = numpy.empty((10, 10))
    actual_npv_arr.fill(actual_npv)

    out_npv_raster = gdal.OpenEx(os.path.join(args['workspace_dir'], out_npv_filename))
    out_npv_raster_band = out_npv_raster.GetRasterBand(1)
    out_npv_arr = out_npv_raster_band.ReadAsArray()

    numpy.testing.assert_almost_equal(actual_npv_arr, out_npv_arr)


class CarbonTests(unittest.TestCase):
    """Tests for the Carbon Model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)


    def test_carbon_full_fast(self):
        """Carbon: full model run with synthetic data."""
        from natcap.invest import carbon

        args = {
            u'workspace_dir': self.workspace_dir,
            u'do_valuation': True,
            u'price_per_metric_ton_of_c': 43.0,
            u'rate_change': 2.8,
            u'lulc_cur_year': 2016,
            u'lulc_fut_year': 2030,
            u'discount_rate': -7.1,
        }

        make_lulc_rasters(args, ['lulc_cur_path', 'lulc_fut_path', 'lulc_redd_path'], 1)

        csv_file = os.path.join(self.workspace_dir, 'pools.csv')
        with open(csv_file, 'w') as open_table:
            open_table.write('C_above,C_below,C_soil,C_dead,lucode,LULC_Name\n')
            open_table.write('15,10,60,1,1,"lulc code 1"\n')
            open_table.write('5,3,20,0,2,"lulc code 2"\n')
            open_table.write('2,1,5,0,3,"lulc code 3"\n')
        args['carbon_pools_path'] = csv_file

        carbon.execute(args)

        #Add assertions for npv for future and REDD scenarios
        assert_npv(args, -0.34220789207450352, 'npv_fut.tif')
        assert_npv(args, -0.4602106134795047, 'npv_redd.tif')


    def test_carbon_future_fast(self):
        """Carbon: regression testing future scenario using synthetic data."""
        from natcap.invest import carbon
        args = {
            u'carbon_pools_path': os.path.join(
                SAMPLE_DATA, 'carbon/carbon_pools_samp.csv'),
            u'workspace_dir': self.workspace_dir,
            u'do_valuation': True,
            u'price_per_metric_ton_of_c': 43.0,
            u'rate_change': 2.8,
            u'lulc_cur_year': 2016,
            u'lulc_fut_year': 2030,
            u'discount_rate': -7.1,
        }

        make_lulc_rasters(args, ['lulc_cur_path', 'lulc_fut_path'], 1)
        carbon.execute(args)
        #Add assertions for npv for the future scenario
        assert_npv(args, -0.34220789207450352, 'npv_fut.tif')


    def test_carbon_missing_landcover_values_fast(self):
        """Carbon: testing expected exception on incomplete with synthetic data."""
        from natcap.invest import carbon
        args = {
            u'carbon_pools_path': os.path.join(
                REGRESSION_DATA, 'carbon_pools_missing_coverage.csv'),
            u'lulc_cur_path': os.path.join(
                SAMPLE_DATA, 'Base_Data/Terrestrial/lulc_samp_cur'),
            u'lulc_fut_path': os.path.join(
                SAMPLE_DATA, 'Base_Data/Terrestrial/lulc_samp_fut'),
            u'workspace_dir': self.workspace_dir,
            u'do_valuation': False,
        }

        make_lulc_rasters(args, ['lulc_cur_path', 'lulc_fut_path'], 200)
        with self.assertRaises(ValueError):
            carbon.execute(args)
            