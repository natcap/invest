"""InVEST Recreation model tests."""

import multiprocessing
import multiprocessing.pool
import unittest
import tempfile
import shutil
import os
import functools
import logging

import numpy
from osgeo import ogr
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'recreation')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'recreation_model')

LOGGER = logging.getLogger('test_recreation')

def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(self, *args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, (self,) + args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator


class TestLocalRecServer(unittest.TestCase):
    """Tests using a local rec server."""

    def setUp(self):
        """Setup workspace and server."""
        multiprocessing.freeze_support()

        from natcap.invest.recreation import recmodel_server
        self.workspace_dir = 'local_rec_workspace' #tempfile.mkdtemp()
        self.recreation_server = recmodel_server.RecModel(
            os.path.join(REGRESSION_DATA, 'sample_data.csv'),
            os.path.join(self.workspace_dir, 'server_cache'))

    def test_local_aoi(self):
        """Recreation test local AOI with local server."""
        aoi_path = os.path.join(REGRESSION_DATA, 'test_aoi_for_subset.shp')
        date_range = (
            numpy.datetime64('2005-01-01'),
            numpy.datetime64('2014-12-31'))
        out_vector_filename = 'pud.shp'
        LOGGER.debug(out_vector_filename)
        self.recreation_server._calc_aggregated_points_in_aoi(
            aoi_path, self.workspace_dir, date_range, out_vector_filename)

        output_lines = open(os.path.join(
            self.workspace_dir, 'monthly_table.csv'), 'rb').readlines()
        expected_lines = open(os.path.join(
            REGRESSION_DATA, 'expected_monthly_table_for_subset.csv'),
                              'rb').readlines()

        if output_lines != expected_lines:
            raise ValueError(
                "Output table not the same as input. "
                "Expected:\n%s\nGot:\n%s" % (expected_lines, output_lines))

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)


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

    def test_local_server(self):
        """Launch a local server with a reduced set of point data."""
        pass

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @timeout(1.0)
    def test_base_regression(self):
        """Recreation base regression test on sample data.

        Executes Recreation model with default data and default arguments.
        """
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'compute_regression': True,
            'start_year': '2004',
            'end_year': '2015',
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors.csv'),
            'results_suffix': u'',
            'scenario_predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors_scenario.csv'),
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
        print result_vector_path
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
            'PUD_NOV', 'PUD_DEC', 'bonefish', 'airdist', 'ports', 'bathy',
            'PUD_EST']

        with open(agg_results_path, 'rb') as agg_result_file:
            header_line = agg_result_file.readline().strip()
            error_in_header = False
            for expected, actual in zip(headers, header_line.split(',')):
                if actual != expected:
                    error_in_header = True
            if error_in_header:
                raise ValueError(
                    "Header not as expected, got\n%s\nexpected:\n%s" % (
                        str(header_line.split(',')), headers))
            for line in agg_result_file:
                try:
                    expected_result_lookup = dict(
                        zip(headers, [float(x) for x in line.split(',')]))
                except ValueError:
                    print line
                    raise
                feature = result_layer.GetFeature(
                    int(expected_result_lookup['FID']))
                for field, value in expected_result_lookup.iteritems():
                    print (
                        "field, value, feature.GetField(field), FID %s %s %s %s" % (
                            field, value, feature.GetField(field),
                            int(expected_result_lookup['FID'])))
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
                    # skip blank lines
                    continue
                if not os.path.isfile(full_path):
                    missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '\n'.join(missing_files))
