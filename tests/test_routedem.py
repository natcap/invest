"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'Base_Data',
    'Freshwater')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'routedem')


class RouteDEMTests(unittest.TestCase):
    """Tests for RouteDEM."""

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
    def test_routedem_single_threshold(self):
        """RouteDem: regression testing single stream threshold."""
        from natcap.invest.routing import routedem
        args = {
            'calculate_stream_threshold': True,
            'results_suffix': 'test',
            'calculate_downstream_distance': True,
            'calculate_slope': True,
            'dem_path': os.path.join(SAMPLE_DATA, 'dem'),
            'calculate_stream_threshold': True,
            'calculate_flow_accumulation': True,
            'threshold_flow_accumulation': '1000',
            'workspace_dir': self.workspace_dir,
        }
        routedem.execute(args)
        RouteDEMTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_single.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'v_stream_1000.tif'),
            os.path.join(self.workspace_dir, 'stream_mask_test.tif'), 1e-6)

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
