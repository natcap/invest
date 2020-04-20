"""Module for Regression Testing the InVEST Finfish Aquaculture model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'aquaculture',
    'Input')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'aquaculture')

EXPECTED_FILE_LIST = [
    'output/Finfish_Harvest.dbf',
    'output/Finfish_Harvest.prj',
    'output/Finfish_Harvest.shp',
    'output/Finfish_Harvest.shx',
    'output/Harvest_Results.html']


def _make_harvest_shp(workspace_dir):
    """Within workspace, make an output folder with dummy Finfish_Harvest.shp.

    Parameters:
        workspace_dir: path to workspace for creating the output folder.
    """
    output_path = os.path.join(workspace_dir, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, 'Finfish_Harvest.shp'), 'wb') as shp:
        shp.write(b'')


class FinfishTests(unittest.TestCase):
    """Tests for Finfish Aquaculture."""

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
        """Generate an args list that is consistent for both regression tests"""
        args = {
            'farm_ID': 'FarmID',
            'farm_op_tbl': os.path.join(SAMPLE_DATA, 'Farm_Operations.csv'),
            'ff_farm_loc': os.path.join(SAMPLE_DATA, 'Finfish_Netpens.shp'),
            'g_param_a': 0.038,
            'g_param_a_sd': 0.005,
            'g_param_b': 0.6667,
            'g_param_b_sd': 0.05,
            'g_param_tau': 0.08,
            'outplant_buffer': 3,
            'use_uncertainty': True,
            'water_temp_tbl': os.path.join(SAMPLE_DATA, 'Temp_Daily.csv'),
            'workspace_dir': workspace_dir,
        }
        return args

    def test_finfish_full_run(self):
        """Finfish: regression test to run model with all options on."""
        import natcap.invest.finfish_aquaculture.finfish_aquaculture

        args = FinfishTests.generate_base_args(self.workspace_dir)
        args['discount'] = 0.000192
        args['do_valuation'] = True
        args['frac_p'] = 0.3
        args['num_monte_carlo_runs'] = 10
        args['p_per_kg'] = 2.25

        _make_harvest_shp(self.workspace_dir)  # to test if it's recreated

        natcap.invest.finfish_aquaculture.finfish_aquaculture.execute(args)
        FinfishTests._test_same_files(
            EXPECTED_FILE_LIST, args['workspace_dir'])
        pygeoprocessing.testing.assert_vectors_equal(
            os.path.join(REGRESSION_DATA, 'Finfish_Harvest.shp'),
            os.path.join(self.workspace_dir, 'output', 'Finfish_Harvest.shp'), 1E-6)

    def test_finfish_mc_no_valuation(self):
        """Finfish: run model with MC analysis and no valuation."""
        import natcap.invest.finfish_aquaculture.finfish_aquaculture

        args = FinfishTests.generate_base_args(self.workspace_dir)
        args['do_valuation'] = False
        args['num_monte_carlo_runs'] = 101

        natcap.invest.finfish_aquaculture.finfish_aquaculture.execute(args)

        FinfishTests._test_same_files(
            EXPECTED_FILE_LIST, args['workspace_dir'])
        pygeoprocessing.testing.assert_vectors_equal(
            os.path.join(REGRESSION_DATA, 'Finfish_Harvest_no_valuation.shp'),
            os.path.join(self.workspace_dir, 'output', 'Finfish_Harvest.shp'), 1E-6)

    @staticmethod
    def _test_same_files(base_path_list, directory_path):
        """Assert files in `base_path_list` are in `directory_path`.

        Parameters:
            base_path_list (list): list of strings which are relative
                file paths.
            directory_path (string): a path to a directory whose contents will
                be checked against the files listed in `base_list_file`

        Returns:
            None

        Raises:
            AssertionError when there are files listed in `base_list_file`
                that don't exist in the directory indicated by `path`
        """
        missing_files = []
        for file_path in base_path_list:
            full_path = os.path.join(
                directory_path,
                file_path.rstrip().replace('//', os.path.sep))
            if full_path == '':
                continue
            if not os.path.isfile(full_path):
                missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '/n'.join(missing_files))


class FinfishValidationTests(unittest.TestCase):
    """Tests for the Finfish Model ARGS_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'ff_farm_loc',
            'farm_ID',
            'g_param_a',
            'g_param_b',
            'g_param_tau',
            'water_temp_tbl',
            'farm_op_tbl',
            'outplant_buffer',
        ]

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys(self):
        """Finfish Validate: assert missing required keys."""
        from natcap.invest.finfish_aquaculture import finfish_aquaculture
        from natcap.invest import validation

        validation_errors = finfish_aquaculture.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['do_valuation',
             'use_uncertainty'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_use_uncertainty(self):
        """Finfish Validate: assert missing required keys for uncertainty."""
        from natcap.invest.finfish_aquaculture import finfish_aquaculture
        from natcap.invest import validation

        validation_errors = finfish_aquaculture.validate(
            {'use_uncertainty': True})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['do_valuation',
             'g_param_a_sd',
             'g_param_b_sd',
             'num_monte_carlo_runs'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_do_valuation(self):
        """Finfish Validate: assert missing required keys for valuation."""
        from natcap.invest.finfish_aquaculture import finfish_aquaculture
        from natcap.invest import validation

        validation_errors = finfish_aquaculture.validate(
            {'do_valuation': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['use_uncertainty',
             'p_per_kg',
             'frac_p',
             'discount'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_field_in_farm_vector(self):
        """Finfish Validate: warning message on invalid fieldname."""
        from natcap.invest.finfish_aquaculture import finfish_aquaculture

        farm_vector_path = os.path.join(SAMPLE_DATA, 'Finfish_Netpens.shp')
        validation_warnings = finfish_aquaculture.validate(
            {'ff_farm_loc': farm_vector_path,
             'farm_ID': 'foo'})
        expected_message = "Value must be one of: ['FarmID']"
        actual_messages = set()
        for keys, error_strings in validation_warnings:
            actual_messages.add(error_strings)
        self.assertTrue(expected_message in actual_messages)
