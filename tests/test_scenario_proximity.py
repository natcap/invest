"""Module for Regression Testing Scenario Proximity Generator."""
import unittest
import tempfile
import shutil
import os

import pandas

TEST_DATA_DIR = os.path.join(
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
        """Generate an args list consistent across all regression tests."""
        args = {
            'aoi_path': os.path.join(
                TEST_DATA_DIR, 'input', 'scenario_proximity_aoi.gpkg'),
            'area_to_convert': '3218.0',
            'base_lulc_path': os.path.join(
                TEST_DATA_DIR, 'input', 'clipped_lulc.tif'),
            'workspace_dir': workspace_dir,
            'convertible_landcover_codes': '1 2 3 4 5',
            'focal_landcover_codes': '1 2 3 4 5',
            'n_fragmentation_steps': '1',
            'replacement_lucode': '12',
            'n_workers': '-1',
        }
        return args

    def test_scenario_gen_regression(self):
        """Scenario Gen Proximity: regression testing all functionality."""
        from natcap.invest import scenario_gen_proximity

        args = ScenarioProximityTests.generate_base_args(self.workspace_dir)
        args['convert_farthest_from_edge'] = True
        args['convert_nearest_to_edge'] = True

        scenario_gen_proximity.execute(args)
        ScenarioProximityTests._test_same_files(
            os.path.join(
                TEST_DATA_DIR, 'expected_file_list_regression.txt'),
            args['workspace_dir'])

        base_table = pandas.read_csv(
            os.path.join(self.workspace_dir, 'farthest_from_edge.csv'))
        expected_table = pandas.read_csv(
            os.path.join(
                TEST_DATA_DIR, 'farthest_from_edge_regression.csv'))
        pandas.testing.assert_frame_equal(base_table, expected_table)

        base_table = pandas.read_csv(
            os.path.join(self.workspace_dir, 'nearest_to_edge.csv'))
        expected_table = pandas.read_csv(
            os.path.join(
                TEST_DATA_DIR, 'nearest_to_edge_regression.csv'))
        pandas.testing.assert_frame_equal(base_table, expected_table)

    def test_scenario_gen_farthest(self):
        """Scenario Gen Proximity: testing small far functionality."""
        from natcap.invest import scenario_gen_proximity

        args = ScenarioProximityTests.generate_base_args(self.workspace_dir)
        args['convert_farthest_from_edge'] = True
        args['convert_nearest_to_edge'] = False
        # running without an AOI
        del args['aoi_path']
        scenario_gen_proximity.execute(args)
        ScenarioProximityTests._test_same_files(
            os.path.join(
                TEST_DATA_DIR, 'expected_file_list_farthest.txt'),
            args['workspace_dir'])

        model_df = pandas.read_csv(
            os.path.join(self.workspace_dir, 'farthest_from_edge.csv'))
        reg_df = pandas.read_csv(
            os.path.join(TEST_DATA_DIR, 'farthest_from_edge_farthest.csv'))
        pandas.testing.assert_frame_equal(model_df, reg_df)

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

        Args:
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


class ScenarioGenValidationTests(unittest.TestCase):
    """Tests for the Scenario Generator ARGS_SPEC and validation."""

    def setUp(self):
        """Initiate list of required keys."""
        self.base_required_keys = [
            'focal_landcover_codes',
            'replacement_lucode',
            'workspace_dir',
            'n_fragmentation_steps',
            'convertible_landcover_codes',
            'area_to_convert',
            'base_lulc_path',
            'convert_nearest_to_edge',
            'convert_farthest_from_edge'
        ]

    def test_missing_keys(self):
        """SG Validate: assert missing required keys."""
        from natcap.invest import scenario_gen_proximity
        from natcap.invest import validation

        # empty args dict.
        validation_errors = scenario_gen_proximity.validate({})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_invalid_conversion_methods(self):
        """SG Validate: assert message if both conversion methods false."""
        from natcap.invest import scenario_gen_proximity

        validation_errors = scenario_gen_proximity.validate(
            {'convert_nearest_to_edge': False,
             'convert_farthest_from_edge': False})
        actual_messages = set()
        for keys, error_strings in validation_errors:
            actual_messages.add(error_strings)
        self.assertTrue(scenario_gen_proximity.MISSING_CONVERT_OPTION_MSG
            in actual_messages)
