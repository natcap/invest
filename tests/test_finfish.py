"""Module for Regression Testing the InVEST Finfish Aquaculture model."""
import unittest
import tempfile
import shutil
import os

from natcap.invest.pygeoprocessing_0_3_3.testing import scm


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'aquaculture',
    'Input')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'aquaculture')


class FinfishTests(unittest.TestCase):
    """Tests for Finfish Aquaculture."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        # self.workspace_dir = tempfile.mkdtemp()
        self.workspace_dir = r"C:\Users\Joanna Lin\Desktop\test_folder\finfish"

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        # shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_finfish_full_run(self):
        """Finfish: regression test to run model with all options on."""
        import natcap.invest.finfish_aquaculture.finfish_aquaculture

        args = {
            'discount': 0.000192,
            'do_valuation': True,
            'farm_ID': 'FarmID',
            'farm_op_tbl': os.path.join(SAMPLE_DATA, 'Farm_Operations.csv'),
            'ff_farm_loc': os.path.join(SAMPLE_DATA, 'Finfish_Netpens.shp'),
            'frac_p': 0.3,
            'g_param_a': 0.038,
            'g_param_a_sd': 0.005,
            'g_param_b': 0.6667,
            'g_param_b_sd': 0.05,
            'g_param_tau': 0.08,
            'num_monte_carlo_runs': 10,
            'outplant_buffer': 3,
            'p_per_kg': 2.25,
            'use_uncertainty': True,
            'water_temp_tbl': os.path.join(SAMPLE_DATA, 'Temp_Daily.csv'),
            'workspace_dir': self.workspace_dir,
        }

        natcap.invest.finfish_aquaculture.finfish_aquaculture.execute(args)

        FinfishTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list.txt'),
            args['workspace_dir'])
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_vectors_equal(
            os.path.join(REGRESSION_DATA, 'Finfish_Harvest.shp'),
            os.path.join(self.workspace_dir, 'output', 'Finfish_Harvest.shp'))



    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_finfish_simple_run(self):
        """Finfish: run model with no valuation or stats sampling."""
        import natcap.invest.finfish_aquaculture.finfish_aquaculture

        args = {
            'do_valuation': False,
            'farm_ID': 'FarmID',
            'farm_op_tbl': os.path.join(SAMPLE_DATA, 'Farm_Operations.csv'),
            'ff_farm_loc': os.path.join(SAMPLE_DATA, 'Finfish_Netpens.shp'),
            'g_param_a': 0.038,
            'g_param_b': 0.6667,
            'g_param_tau': 0.08,
            'outplant_buffer': 3,
            'use_uncertainty': False,
            'water_temp_tbl': os.path.join(SAMPLE_DATA, 'Temp_Daily.csv'),
            'workspace_dir': self.workspace_dir,
        }

        natcap.invest.finfish_aquaculture.finfish_aquaculture.execute(args)

        FinfishTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_simple.txt'),
            args['workspace_dir'])
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_vectors_equal(
            os.path.join(REGRESSION_DATA, 'Finfish_Harvest_no_valuation.shp'),
            os.path.join(self.workspace_dir, 'output', 'Finfish_Harvest.shp'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_finfish_mc_no_valuation(self):
        """Finfish: run model with MC analysis and no valuation."""
        import natcap.invest.finfish_aquaculture.finfish_aquaculture

        args = {
            'do_valuation': False,
            'farm_ID': 'FarmID',
            'farm_op_tbl': os.path.join(SAMPLE_DATA, 'Farm_Operations.csv'),
            'ff_farm_loc': os.path.join(SAMPLE_DATA, 'Finfish_Netpens.shp'),
            'g_param_a': 0.038,
            'g_param_a_sd': 0.005,
            'g_param_b': 0.6667,
            'g_param_b_sd': 0.05,
            'g_param_tau': 0.08,
            'num_monte_carlo_runs': 101,
            'outplant_buffer': 3,
            'use_uncertainty': True,
            'water_temp_tbl': os.path.join(SAMPLE_DATA, 'Temp_Daily.csv'),
            'workspace_dir': self.workspace_dir,
        }

        natcap.invest.finfish_aquaculture.finfish_aquaculture.execute(args)

        FinfishTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'expected_file_list_no_valuation.txt'),
            args['workspace_dir'])
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_vectors_equal(
            os.path.join(REGRESSION_DATA, 'Finfish_Harvest_no_valuation.shp'),
            os.path.join(self.workspace_dir, 'output', 'Finfish_Harvest.shp'))

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
                full_path = os.path.join(
                    directory_path,
                    file_path.rstrip().replace('\\', os.path.sep))
                if full_path == '':
                    continue
                if not os.path.isfile(full_path):
                    missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '\n'.join(missing_files))
