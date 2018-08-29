"""Module for Testing DelineateIt."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing


REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'delineateit')
REGRESSION_DATA = r"C:\Users\Joanna Lin\Desktop\test_folder\delineateIT\invest-test-data"


class DelineateItTests(unittest.TestCase):
    """Tests for RouteDEM."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_routedem_multi_threshold(self):
        """DelineateIt: regression testing full run."""
        import natcap.invest.routing.delineateit

        args = {
            'dem_uri': os.path.join(REGRESSION_DATA, 'input', 'dem.tif'),
            'flow_threshold': '500',
            'outlet_shapefile_uri': os.path.join(
                REGRESSION_DATA, 'input', 'outlets.shp'),
            'snap_distance': '20',
            'workspace_dir': self.workspace_dir,
        }
        natcap.invest.routing.delineateit.execute(args)

        DelineateItTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_vectors_equal(
            os.path.join(REGRESSION_DATA, 'watersheds.shp'),
            os.path.join(self.workspace_dir, 'watersheds.shp'), 1e-6)

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
