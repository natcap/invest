"""Module for Regression Testing the InVEST Habitat Risk Assessment model."""
import unittest
import tempfile
import shutil
import os

import natcap.invest.pygeoprocessing_0_3_3.testing
from natcap.invest.pygeoprocessing_0_3_3.testing import scm

# remove later
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'habitat_risk_assessment', 'synthetic_data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'habitat_risk_assessment')

# SAMPLE_DATA = r"C:\Users\Joanna Lin\Desktop\test_folder\HRA\invest-data"
# REGRESSION_DATA = r"C:\Users\Joanna Lin\Desktop\test_folder\HRA\invest-test-data"
# tempdir = r"C:\Users\Joanna Lin\Desktop\test_folder\HRA\tempdir"

class HRATests(unittest.TestCase):
    """Tests for Habitat Risk Assessment."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        # self.workspace_dir = tempdir
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_hra_euc_none(self):
        """HRA: euclidean and no decay."""
        import natcap.invest.habitat_risk_assessment.hra

        args = {
            'aoi_tables': os.path.join(
                SAMPLE_DATA, 'Input', 'subregions.shp'),
            'csv_uri': os.path.join(
                SAMPLE_DATA, 'habitat_stressor_ratings_sample'),
            'decay_eq': 'None',
            'grid_size': 30,
            'max_rating': 3,
            'max_stress': 2,
            'risk_eq': 'Euclidean',
            'workspace_dir': self.workspace_dir,
        }
        natcap.invest.habitat_risk_assessment.hra.execute(args)

        HRATests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_euc_none.txt'),
            args['workspace_dir'])
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'ecosys_risk_euc_none.tif'),
            os.path.join(
                self.workspace_dir, 'output', 'Maps', 'ecosys_risk.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_hra_mult_none(self):
        """HRA: multiplicative and no decay."""
        import natcap.invest.habitat_risk_assessment.hra

        args = {
            'aoi_tables': os.path.join(
                SAMPLE_DATA, 'Input', 'subregions.shp'),
            'csv_uri': os.path.join(
                SAMPLE_DATA, 'habitat_stressor_ratings_sample'),
            'decay_eq': 'None',
            'grid_size': 30,
            'max_rating': 3,
            'max_stress': 2,
            'risk_eq': 'Multiplicative',
            'workspace_dir': self.workspace_dir,
        }
        natcap.invest.habitat_risk_assessment.hra.execute(args)

        HRATests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_mult_none.txt'),
            args['workspace_dir'])
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'ecosys_risk_mult_none.tif'),
            os.path.join(
                self.workspace_dir, 'output', 'Maps', 'ecosys_risk.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_hra_euc_lin(self):
        """HRA: euclidean and linear."""
        import natcap.invest.habitat_risk_assessment.hra

        args = {
            'aoi_tables': os.path.join(
                SAMPLE_DATA, 'Input', 'subregions.shp'),
            'csv_uri': os.path.join(
                SAMPLE_DATA, 'habitat_stressor_ratings_sample'),
            'decay_eq': 'Linear',
            'grid_size': 30,
            'max_rating': 3,
            'max_stress': 2,
            'risk_eq': 'Euclidean',
            'workspace_dir': self.workspace_dir,
        }
        natcap.invest.habitat_risk_assessment.hra.execute(args)

        HRATests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_euc_lin.txt'),
            args['workspace_dir'])
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'ecosys_risk_euc_lin.tif'),
            os.path.join(
                self.workspace_dir, 'output', 'Maps', 'ecosys_risk.tif'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_hra_euc_exp(self):
        """HRA: euclidean and exponential."""
        import natcap.invest.habitat_risk_assessment.hra

        args = {
            'aoi_tables': os.path.join(
                SAMPLE_DATA, 'Input', 'subregions.shp'),
            'csv_uri': os.path.join(
                SAMPLE_DATA, 'habitat_stressor_ratings_sample'),
            'decay_eq': 'Exponential',
            'grid_size': 30,
            'max_rating': 3,
            'max_stress': 2,
            'risk_eq': 'Euclidean',
            'workspace_dir': self.workspace_dir,
        }
        natcap.invest.habitat_risk_assessment.hra.execute(args)

        HRATests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_euc_exp.txt'),
            args['workspace_dir'])
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'ecosys_risk_euc_exp.tif'),
            os.path.join(
                self.workspace_dir, 'output', 'Maps', 'ecosys_risk.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_hra_preprocessor(self):
        """HRA: preprocessor coverage."""
        import natcap.invest.habitat_risk_assessment.hra_preprocessor
        args = {
            'criteria_dir': os.path.join(
                SAMPLE_DATA, 'Input', 'Spatially_Explicit_Criteria'),
            'exposure_crits': [
                'Intensity Rating',
                'Management Effectiveness',
                'Temporal Overlap Rating',
            ],
            'habitats_dir': os.path.join(
                SAMPLE_DATA, 'Input', 'HabitatLayers'),
            'resilience_crits': [
                'Connectivity Rate',
                'Natural Mortality Rate',
                'Recovery Time',
                'Recruitment Rate',
            ],
            'sensitivity_crits': [
                'Change in Area Rating',
                'Change in Structure Rating',
                'Frequency of Disturbance',
            ],
            'stressors_dir': os.path.join(
                SAMPLE_DATA, 'Input', 'StressorLayers'),
            'workspace_dir': self.workspace_dir,
        }
        natcap.invest.habitat_risk_assessment.hra_preprocessor.execute(args)
        HRATests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'expected_file_list_hra_preprocessor.txt'),
            args['workspace_dir'])

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
