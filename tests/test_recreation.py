"""InVEST Recreation model tests."""

import unittest
import tempfile
import shutil
import os

import numpy
from osgeo import ogr
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'Recreation')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'recreation_model')


class RecreationRegressionTests(unittest.TestCase):
    """Regression tests for InVEST Seasonal Water Yield model."""

    def setUp(self):
        """Setup workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_base_regression(self):
        """Recreation base regression test on sample data.

        Executes Recreation model with default data and default arguments.
        """
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'BC_AOI.shp'),
            'cell_size': 5000.0,
            'compute_regression': True,
            'start_year': '2004',
            'end_year': '2014',
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors.csv'),
            'results_suffix': u'',
            'scenario_predictor_table_path': os.path.join(
                SAMPLE_DATA, 'scenario_predictors.csv'),
            'workspace_dir': self.workspace_dir,
        }

        recmodel_client.execute(args)

        RecreationRegressionTests._assert_regression_results_equal(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_base.txt'),
            os.path.join(args['workspace_dir'], 'scenario_results.shp'),
            os.path.join(REGRESSION_DATA, 'scenario_results.csv'))

    @staticmethod
    def _assert_regression_results_equal(
            workspace_dir, file_list_path, result_vector_path,
            agg_results_path):
        """Test that workspace against the expected list of files
        and aggregated results.

        Parameters:
            workspace_dir (string): path to the completed model workspace
            file_list_path (string): path to a file that has a list of all
                the expected files relative to the workspace base
            result_vector_path (string): path to the summary shapefile
                produced by the SWY model.
            agg_results_path (string): path to a csv file that has the
                expected aggregated_results.shp table in the form of
                fid,vri_sum,qb_val per line

        Returns:
            None

        Raises:
            AssertionError if any files are missing or results are out of
            range by `tolerance_places`
        """
         # Test that the workspace has the same files as we expect
        RecreationRegressionTests._test_same_files(
            file_list_path, workspace_dir)

        # we expect a file called 'aggregated_results.shp'
        result_vector = ogr.Open(result_vector_path)
        result_layer = result_vector.GetLayer()

        # The tolerance of 3 digits after the decimal was determined by
        # experimentation on the application with the given range of numbers.
        # This is an apparently reasonable approach as described by ChrisF:
        # http://stackoverflow.com/a/3281371/42897
        # and even more reading about picking numerical tolerance (it's hard):
        # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        tolerance_places = 3

        headers = [
            'FID', 'PUD_YR_AVG', 'PUD_JAN', 'PUD_FEB', 'PUD_MAR', 'PUD_APR',
            'PUD_MAY', 'PUD_JUN', 'PUD_JUL', 'PUD_AUG', 'PUD_SEP', 'PUD_OCT',
            'PUD_NOV', 'PUD_DEC', 'bc_parks_a', 'bc_r_mean', 'bc_points',
            'bc_parks_p', 'bc_r_sum', 'bc_paths', 'PUD_EST']

        with open(agg_results_path, 'rb') as agg_result_file:
            for line in agg_result_file:
                expected_result_lookup = dict(
                    zip(headers, [float(x) for x in line.split(',')]))
                feature = result_layer.GetFeature(
                    int(expected_result_lookup['FID']))
                for field, value in expected_result_lookup.iteritems():
                    numpy.testing.assert_almost_equal(
                        feature.GetField(field), value,
                        decimal=tolerance_places)
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None

    @staticmethod
    def _test_same_files(base_list_path, directory_path):
        """Assert expected files are in the `directory_path`.

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
                    #skip blank lines
                    continue
                if not os.path.isfile(full_path):
                    missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '\n'.join(missing_files))
