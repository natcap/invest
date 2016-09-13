"""Module for Regression Testing the InVEST Pollination model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'pollination')


class PollinationTests(unittest.TestCase):
    """Tests for the Pollination model."""

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
    def test_pollination_regression(self):
        """Pollination: regression testing sample data."""
        from natcap.invest.pollination import pollination

        args = {
            'ag_classes': (
                '67 68 71 72 73 74 75 76 78 79 80 81 82 83 84 85 88 90 91 92'),
            'do_valuation': True,
            'guilds_uri': os.path.join(
                SAMPLE_DATA, 'Pollination', 'Input', 'Guild.csv'),
            'half_saturation': 0.125,
            'landuse_attributes_uri': os.path.join(
                SAMPLE_DATA, 'Pollination', 'Input', 'LU.csv'),
            'landuse_cur_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Terrestrial', 'lulc_samp_cur'),
            'landuse_fut_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Terrestrial', 'lulc_samp_fut'),
            'results_suffix': u'',
            'wild_pollination_proportion': 1.0,
            'workspace_dir': self.workspace_dir,
        }

        pollination.execute(args)

        PollinationTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'expected_file_list_regression.txt'),
            args['workspace_dir'])

        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(self.workspace_dir, 'output', 'frm_avg_cur.tif'),
            os.path.join(REGRESSION_DATA, 'frm_avg_cur_regression.tif'), 1e-6)

        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(self.workspace_dir, 'output', 'frm_avg_fut.tif'),
            os.path.join(REGRESSION_DATA, 'frm_avg_fut_regression.tif'), 1e-6)

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
