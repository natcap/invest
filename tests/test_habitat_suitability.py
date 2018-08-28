"""InVEST Habitat Suitability model tests."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing.assertions

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'habitat_suitability')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'habitat_suitability')


class HabitatSuitabilityTests(unittest.TestCase):
    """Regression tests for InVEST Habitat Suitability model."""

    def setUp(self):
        """Initialize SDRRegression tests."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_base_regression(self):
        """Habitat suitability base regression on sample data.

        Execute Habitat suitability with sample data and checks that the
        output files are generated and that the aggregate shapefile fields are
        the same as the regression case.
        """
        from natcap.invest import habitat_suitability

        # use predefined directory so test can clean up files during teardown
        args = HabitatSuitabilityTests._generate_base_args(
            self.workspace_dir)
        # make args explicit that this is a base run of SWY
        habitat_suitability.execute(args)

        HabitatSuitabilityTests._assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list.txt'),
            [os.path.join(args['workspace_dir'], 'hsi.tif'),
             os.path.join(args['workspace_dir'], 'hsi_threshold.tif'),
             os.path.join(args['workspace_dir'],
                          'hsi_threshold_screened.tif')],
            [os.path.join(REGRESSION_DATA, 'hsi.tif'),
             os.path.join(REGRESSION_DATA, 'hsi_threshold.tif'),
             os.path.join(REGRESSION_DATA, 'hsi_threshold_screened.tif')])

    def test_categorical_regression(self):
        """Habitat suitability regression with categorical sample data.

        Execute Habitat suitability with sample data that includes categorical
        vector inputs and checks that the output files are generated and that
        the aggregate shapefile fields are the same as the regression case.
        """
        from natcap.invest import habitat_suitability

        # use predefined directory so test can clean up files during teardown
        args = HabitatSuitabilityTests._generate_base_args(
            self.workspace_dir)
        # make args explicit that this is a base run of SWY
        args['categorical_geometry'] = {
            'substrate': {
                'vector_path': os.path.join(
                    SAMPLE_DATA, 'substrate_3005.shp'),
                'fieldname': 'Suitabilit',
            }
        }
        args['output_cell_size'] = 500
        args['results_suffix'] = 'categorical'
        habitat_suitability.execute(args)

        HabitatSuitabilityTests._assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_categorical.txt'),
            [os.path.join(args['workspace_dir'], 'hsi_categorical.tif'),
             os.path.join(args['workspace_dir'], 'hsi_threshold_categorical.tif'),
             os.path.join(args['workspace_dir'],
                          'hsi_threshold_screened_categorical.tif')],
            [os.path.join(REGRESSION_DATA, 'hsi_categorical.tif'),
             os.path.join(REGRESSION_DATA, 'hsi_threshold_categorical.tif'),
             os.path.join(REGRESSION_DATA, 'hsi_threshold_screened_categorical.tif')])

    def test_regression_cell_size(self):
        """Habitat suitability regression test on cell size set to 500m."""
        from natcap.invest import habitat_suitability

        # use predefined directory so test can clean up files during teardown
        args = HabitatSuitabilityTests._generate_base_args(
            self.workspace_dir)
        args['output_cell_size'] = 500
        args['results_suffix'] = '500'
        # make args explicit that this is a base run of SWY
        habitat_suitability.execute(args)

        HabitatSuitabilityTests._assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_500.txt'),
            [os.path.join(args['workspace_dir'], 'hsi_500.tif'),
             os.path.join(args['workspace_dir'], 'hsi_threshold_500.tif'),
             os.path.join(args['workspace_dir'],
                          'hsi_threshold_screened_500.tif')],
            [os.path.join(REGRESSION_DATA, 'hsi_500.tif'),
             os.path.join(REGRESSION_DATA, 'hsi_threshold_500.tif'),
             os.path.join(REGRESSION_DATA, 'hsi_threshold_screened_500.tif')])

    @staticmethod
    def _generate_base_args(workspace_dir):
        """Generate a base sample args dict for Habitat Suitability."""
        args = {
            'workspace_dir': workspace_dir,
            'results_suffix': '',
            'aoi_path': os.path.join(SAMPLE_DATA, 'AOI_ocean.shp'),
            'exclusion_path_list': [
                os.path.join(SAMPLE_DATA, 'example_masks', "mask_1.shp"),
                os.path.join(SAMPLE_DATA, 'example_masks', "mask_2.shp")],
            'habitat_threshold': 0.2,
            'hsi_ranges': {
                'depth': {
                    'raster_path': os.path.join(
                        SAMPLE_DATA, "bathyclip"),
                    'suitability_range': (-50, -30, -10, -10),
                },
                'temperature': {
                    'raster_path': os.path.join(SAMPLE_DATA, "sst"),
                    'suitability_range': (5, 7, 12.5, 16),
                },
                'salinity': {
                    'raster_path': os.path.join(SAMPLE_DATA, "sss"),
                    'suitability_range': (28, 30, 31, 32),
                },
                'tidal_speed': {
                    'raster_path': os.path.join(
                        SAMPLE_DATA, "tidalspeedcms.tif"),
                    'suitability_range': (1.5, 5, 15, 30),
                },
            },
            'categorical_geometry': {},
        }
        return args

    @staticmethod
    def _assert_regression_results_eq(
            workspace_dir, file_list_path, result_path_list,
            expected_result_path_list):
        """Test workspace state against expected aggregate results.

        Parameters:
            workspace_dir (string): path to the completed model workspace
            file_list_path (string): path to a file that has a list of all
                the expected files relative to the workspace base
            result_path_list (list): list of raster paths.
            expected_result_path_list (list): list of raster paths
                with values expected.

        Returns:
            None

        Raises:
            AssertionError if any files are missing or results are out of
            range by `tolerance_places`
        """
        # test that the workspace has the same files as we expect
        HabitatSuitabilityTests._test_same_files(
            file_list_path, workspace_dir)

        # The tolerance of 1e-6 was determined by experimentation
        tolerance = 1e-6

        for result_path, expected_path in zip(
                result_path_list, expected_result_path_list):
            pygeoprocessing.testing.assertions.assert_rasters_equal(
                result_path, expected_path, tolerance)

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
