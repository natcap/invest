"""Module for Regression Testing Scenario Proximity Generator."""
import unittest
import tempfile
import shutil
import os

import pandas
import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'scenario_gen_proximity', 'input')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'scenario_gen_proximity')


class ScenarioProximityTests(unittest.TestCase):
    """Tests for the Scenario Proximity Generator."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across all three regression
        tests"""
        args = {
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'scenario_proximity_aoi.shp'),
            'area_to_convert': '20000.0',
            'base_lulc_path': os.path.join(
                SAMPLE_DATA, 'clipped_lulc.tif'),
            'workspace_dir': workspace_dir,
            'convertible_landcover_codes': '1 2 3 4 5',
            'focal_landcover_codes': '1 2 3 4 5',
            'n_fragmentation_steps': '1',
            'replacment_lucode': '12'
        }
        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_scenario_gen_regression(self):
        """Scenario Gen Proximity: regression testing all functionality."""
        from natcap.invest import scenario_gen_proximity

        args = ScenarioProximityTests.generate_base_args(self.workspace_dir)
        args['convert_farthest_from_edge'] = True
        args['convert_nearest_to_edge'] = True

        scenario_gen_proximity.execute(args)
        ScenarioProximityTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'expected_file_list_regression.txt'),
            args['workspace_dir'])

        base_table = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'farthest_from_edge.csv'))
        expected_table = pandas.read_csv(
            os.path.join(
                REGRESSION_DATA, 'farthest_from_edge_regression.csv'))
        pandas.testing.assert_frame_equal(base_table, expected_table)

        base_table = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'nearest_to_edge.csv'))
        expected_table = pandas.read_csv(
            os.path.join(
                REGRESSION_DATA, 'nearest_to_edge_regression.csv'))
        pandas.testing.assert_frame_equal(base_table, expected_table)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_scenario_gen_far_scenario(self):
        """Scenario Gen Proximity: testing small far functionality."""
        from natcap.invest import scenario_gen_proximity

        args = ScenarioProximityTests.generate_base_args(self.workspace_dir)
        args['convert_farthest_from_edge'] = True
        args['convert_nearest_to_edge'] = False

        scenario_gen_proximity.execute(args)
        ScenarioProximityTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'expected_file_list_small_farthest.txt'),
            args['workspace_dir'])

        pygeoprocessing.testing.assertions.assert_csv_equal(
            os.path.join(args['workspace_dir'], 'farthest_from_edge.csv'),
            os.path.join(
                REGRESSION_DATA, 'small_farthest_from_edge_regression.csv'),
            rel_tol=1e-6)

    def test_scenario_gen_no_scenario(self):
        """Scenario Gen Proximity: no scenario should raise an exception."""
        from natcap.invest import scenario_gen_proximity

        args = ScenarioProximityTests.generate_base_args(self.workspace_dir)
        args['convert_farthest_from_edge'] = False
        args['convert_nearest_to_edge'] = False

        # both scenarios false should raise a value error
        with self.assertRaises(ValueError):
            scenario_gen_proximity.execute(args)

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
