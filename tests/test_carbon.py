"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os


import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from osgeo import gdal
from osgeo import ogr
import numpy


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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_carbon(self):
        """Carbon: regression testing the entire carbon model."""
        from natcap.invest import carbon
        args = {
            u'price_per_metric_ton_of_c': 43.0,
            u'rate_change': 2.8,
            u'carbon_pools_path': os.path.join(
                SAMPLE_DATA, 'carbon/carbon_pools_samp.csv'),
            u'do_valuation': True,
            u'lulc_cur_path': os.path.join(
                SAMPLE_DATA, 'Base_Data/Terrestrial/lulc_samp_cur'),
            u'lulc_cur_year': 2016,
            u'lulc_fut_path': os.path.join(
                SAMPLE_DATA, 'Base_Data/Terrestrial/lulc_samp_fut'),
            u'lulc_fut_year': 2030,
            u'lulc_redd_path': os.path.join(
                SAMPLE_DATA, 'carbon/lulc_samp_redd.tif'),
            u'discount_rate': -7.1,
            u'workspace_dir': self.workspace_dir,
        }
        carbon.execute(args)
        CarbonTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list.txt'),
            args['workspace_dir'])
        for npv_filename in ['npv_fut.tif', 'npv_redd.tif']:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(REGRESSION_DATA, npv_filename),
                os.path.join(self.workspace_dir, npv_filename), 1e-6)

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
