"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os


import natcap.invest.pygeoprocessing_0_3_3.testing
from natcap.invest.pygeoprocessing_0_3_3.testing import scm
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import pygeoprocessing.testing


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'carbon')


class CarbonTests(unittest.TestCase):
    """Tests for the Timber Model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)


    def make_lulc_rasters(self, args, raster_keys, start_num):
        """Create LULC rasters"""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26910)
        projection_wkt = srs.ExportToWkt()

        for val, key in enumerate(raster_keys, start=start_num):
            lulc_array = numpy.empty((10,10))
            lulc_array.fill(val)
            lulc_path = os.path.join('.', key+'.tif')
            pygeoprocessing.testing.create_raster_on_disk(
                [lulc_array], (461261,4923265), projection_wkt, -1, (1, -1), 
                filename=lulc_path)
            args[key] = lulc_path


    def assert_npv(self, args, fill_val, npv_filename):
        """Assert that the output npv is the same as the real npv."""
        real_npv = numpy.empty((10,10))
        real_npv.fill(fill_val)

        npv_raster = gdal.OpenEx(os.path.join(args['workspace_dir'], npv_filename))
        npv_raster_band = npv_raster.GetRasterBand(1)
        out_npv = npv_raster_band.ReadAsArray()

        numpy.testing.assert_almost_equal(real_npv, out_npv)


    def test_carbon_full_fast(self):
        """Carbon: full model run with synthetic data."""
        from natcap.invest import carbon

        args = {
            u'carbon_pools_path': os.path.join(
                SAMPLE_DATA, 'carbon/carbon_pools_samp.csv'),
            u'lulc_cur_path': os.path.join(
                SAMPLE_DATA, 'Base_Data/Terrestrial/lulc_samp_cur'),
            u'lulc_fut_path': os.path.join(
                SAMPLE_DATA, 'Base_Data/Terrestrial/lulc_samp_fut'),
            u'lulc_redd_path': os.path.join(
                SAMPLE_DATA, 'carbon/lulc_samp_redd.tif'),
            u'workspace_dir': 'carbon_test',
            u'do_valuation': True,
            u'price_per_metric_ton_of_c': 43.0,
            u'rate_change': 2.8,
            u'lulc_cur_year': 2016,
            u'lulc_fut_year': 2030,
            u'discount_rate': -7.1,
        }

        self.make_lulc_rasters(args, ['lulc_cur_path', 'lulc_fut_path', 'lulc_redd_path'], 1)

        csv_file = os.path.join(self.workspace_dir, 'pools.csv')
        with open(csv_file, 'w') as open_table:
            open_table.write('C_above,C_below,C_soil,C_dead,lucode,LULC_Name\n')
            open_table.write('15,10,60,1,1,"lulc code 1"\n')
            open_table.write('5,3,20,0,2,"lulc code 2"\n')
            open_table.write('2,1,5,0,3,"lulc code 3"\n')
        args['carbon_pools_path'] = csv_file

        carbon.execute(args)

        #Add assertions for npv for future and REDD scenarios
        self.assert_npv(args, -0.34220789207450352, 'npv_fut.tif')
        self.assert_npv(args, -0.4602106134795047, 'npv_redd.tif')


    def test_carbon_future_fast(self):
        """Carbon: regression testing future scenario using synthetic data."""
        from natcap.invest import carbon
        args = {
            u'carbon_pools_path': os.path.join(
                SAMPLE_DATA, 'carbon/carbon_pools_samp.csv'),
            u'lulc_cur_path': os.path.join(
                SAMPLE_DATA, 'Base_Data/Terrestrial/lulc_samp_cur'),
            u'lulc_fut_path': os.path.join(
                SAMPLE_DATA, 'Base_Data/Terrestrial/lulc_samp_fut'),
            u'workspace_dir': self.workspace_dir,
            u'do_valuation': True,
            u'price_per_metric_ton_of_c': 43.0,
            u'rate_change': 2.8,
            u'lulc_cur_year': 2016,
            u'lulc_fut_year': 2030,
            u'discount_rate': -7.1,
        }

        self.make_lulc_rasters(args, ['lulc_cur_path', 'lulc_fut_path'], 1)
        carbon.execute(args)
        #Add assertions for npv for the future scenario
        self.assert_npv(args, -0.34220789207450352, 'npv_fut.tif')


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

        self.make_lulc_rasters(args, ['lulc_cur_path', 'lulc_fut_path'], 200)
        with self.assertRaises(ValueError):
            carbon.execute(args)


    @staticmethod
    def _test_same_files(base_list_path, directory_path):
        """Assert files in `base_list_path` are in `directory_path`.

        Parameters:
            base_list_path (string): a path to a file that has one relative
                file path per line.
            directory_path (string): a path to a directory whose contents will
                be checked against the files listed in `base_list_file`

        Returns:
            None

        Raises:
            AssertionError when there are files listed in `base_list_file`
                that don't exist in the directory indicated by `path`
        """
        missing_files = []
        with open(base_list_path, 'r') as file_list:
            for file_path in file_list:
                full_path = os.path.join(directory_path, file_path.rstrip())
                if full_path == '':
                    continue
                if not os.path.isfile(full_path):
                    missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '\n'.join(missing_files))
