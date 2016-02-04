"""InVEST Recreation model tests."""

import socket
import threading
import multiprocessing
import multiprocessing.pool
import unittest
import tempfile
import shutil
import os
import functools
import logging

import pygeoprocessing
from pygeoprocessing.testing import scm
import numpy
from osgeo import ogr

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


class TestLocalPyroRecServer(unittest.TestCase):
    """Tests that set up local rec server on a port and call through."""

    def setUp(self):
        """Setup Pyro port."""
        multiprocessing.freeze_support()
        self.workspace_dir = tempfile.mkdtemp()

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @timeout(10.0)
    def test_empty_server(self):
        """Recreation test a client call to custom server."""
        from natcap.invest.recreation import recmodel_server
        from natcap.invest.recreation import recmodel_client

        pygeoprocessing.create_directories([self.workspace_dir])
        empty_point_data_path = os.path.join(
            self.workspace_dir, 'empty_table.csv')
        open(empty_point_data_path, 'w').close()  # touch the file

        # attempt to get an open port; could result in race condition but
        # will be okay for a test. if this test ever fails because of port
        # in use, that's probably why
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        sock = None

        server_args = {
            'hostname': 'localhost',
            'port': port,
            'raw_csv_point_data_path': empty_point_data_path,
            'cache_workspace': self.workspace_dir,
            'min_year': 2004,
            'max_year': 2015,
        }

        server_thread = threading.Thread(
            target=recmodel_server.execute, args=(server_args,))
        server_thread.start()

        client_args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'hostname': 'localhost',
            'port': port,
            'compute_regression': False,
            'start_year': '2005',
            'end_year': '2014',
            'grid_aoi': False,
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
        }
        recmodel_client.execute(client_args)

        # testing for file existence seems reasonable since mostly we are
        # testing that a local server starts and a client connects to it
        _test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list_empty_local_server.txt'),
            self.workspace_dir)

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)


class TestLocalRecServer(unittest.TestCase):
    """Tests using a local rec server."""

    def setUp(self):
        """Setup workspace and server."""
        multiprocessing.freeze_support()

        from natcap.invest.recreation import recmodel_server
        self.workspace_dir = tempfile.mkdtemp()
        self.recreation_server = recmodel_server.RecModel(
            os.path.join(REGRESSION_DATA, 'sample_data.csv'),
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_raster_sum_mean_no_nodata(self):
        """Recreation test sum/mean if raster doesn't have nodata defined."""
        from natcap.invest.recreation import recmodel_client

        # The following raster has no nodata value
        raster_path = os.path.join(REGRESSION_DATA, 'no_nodata_raster.tif')

        response_vector_path = os.path.join(SAMPLE_DATA, 'andros_aoi.shp')
        tmp_indexed_vector_path = os.path.join(
            self.workspace_dir, 'tmp_indexed_vector.shp')
        tmp_fid_raster_path = os.path.join(
            self.workspace_dir, 'tmp_fid_raster.tif')
        fid_values = recmodel_client._raster_sum_mean(
            response_vector_path, raster_path, tmp_indexed_vector_path,
            tmp_fid_raster_path)

        # These constants were calculated by hand by Rich.
        numpy.testing.assert_equal(fid_values['count'][0], 5178)
        numpy.testing.assert_equal(fid_values['sum'][0], 67314)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_raster_sum_mean_nodata(self):
        """Recreation test sum/mean if raster is all nodata."""
        from natcap.invest.recreation import recmodel_client

        # The following raster has no nodata value
        raster_path = os.path.join(REGRESSION_DATA, 'nodata_raster.tif')

        response_vector_path = os.path.join(SAMPLE_DATA, 'andros_aoi.shp')
        tmp_indexed_vector_path = os.path.join(
            self.workspace_dir, 'tmp_indexed_vector.shp')
        tmp_fid_raster_path = os.path.join(
            self.workspace_dir, 'tmp_fid_raster.tif')
        fid_values = recmodel_client._raster_sum_mean(
            response_vector_path, raster_path, tmp_indexed_vector_path,
            tmp_fid_raster_path)

        # These constants were calculated by hand by Rich.
        numpy.testing.assert_equal(fid_values['count'][0], 0)
        numpy.testing.assert_equal(fid_values['sum'][0], 0)
        numpy.testing.assert_equal(fid_values['mean'][0], 0)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @timeout(100.0)
    def test_base_regression(self):
        """Recreation base regression test on sample data.

        Executes Recreation model with default data and default arguments.
        """
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'compute_regression': True,
            'start_year': '2005',
            'end_year': '2014',
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

        RecreationRegressionTests._assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_base.txt'),
            os.path.join(args['workspace_dir'], 'scenario_results.shp'),
            os.path.join(REGRESSION_DATA, 'scenario_results.csv'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_square_grid_regression(self):
        """Recreation square grid regression test."""
        from natcap.invest.recreation import recmodel_client

        out_grid_vector_path = os.path.join(
            self.workspace_dir, 'square_grid_vector_path.shp')

        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'square', 20000.0,
            out_grid_vector_path)

        expected_grid_vector_path = os.path.join(
            REGRESSION_DATA, 'square_grid_vector_path.shp')

        pygeoprocessing.testing.assert_vectors_equal(
            out_grid_vector_path, expected_grid_vector_path, 0.0)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_all_metrics(self):
        """Recreation test with all but trivial predictor metrics."""
        from natcap.invest.recreation import recmodel_client
        args = {
            'aoi_path': os.path.join(
                REGRESSION_DATA, 'andros_aoi_with_extra_fields.shp'),
            'cell_size': 20000.0,
            'compute_regression': True,
            'start_year': '2005',
            'end_year': '2014',
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors_all.csv'),
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
        }
        recmodel_client.execute(args)

        out_grid_vector_path = os.path.join(
            self.workspace_dir, 'regression_coefficients.shp')
        expected_grid_vector_path = os.path.join(
            REGRESSION_DATA, 'trivial_regression_coefficients.shp')
        pygeoprocessing.testing.assert_vectors_equal(
            out_grid_vector_path, expected_grid_vector_path, 1e-5)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_hex_grid_regression(self):
        """Recreation hex grid regression test."""
        from natcap.invest.recreation import recmodel_client

        out_grid_vector_path = os.path.join(
            self.workspace_dir, 'hex_grid_vector_path.shp')

        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'hexagon', 20000.0,
            out_grid_vector_path)

        expected_grid_vector_path = os.path.join(
            REGRESSION_DATA, 'hex_grid_vector_path.shp')

        pygeoprocessing.testing.assert_vectors_equal(
            out_grid_vector_path, expected_grid_vector_path, 0.0)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_no_grid_regression(self):
        """Recreation base regression on ungridded AOI."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': False,
            'start_year': '2005',
            'end_year': '2014',
            'grid_aoi': False,
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
        }

        recmodel_client.execute(args)

        output_lines = open(os.path.join(
            self.workspace_dir, 'monthly_table.csv'), 'rb').readlines()
        expected_lines = open(os.path.join(
            REGRESSION_DATA, 'expected_monthly_table_for_no_grid.csv'),
                              'rb').readlines()

        if output_lines != expected_lines:
            raise ValueError(
                "Output table not the same as input. "
                "Expected:\n%s\nGot:\n%s" % (expected_lines, output_lines))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_predictor_id_too_long(self):
        """Recreation test ID too long raises ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': True,
            'start_year': '2005',
            'end_year': '2014',
            'grid_aoi': True,
            'grid_type': 'square',
            'cell_size': 20000,
            'predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors_id_too_long.csv'),
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_existing_output_shapefiles(self):
        """Recreation grid test when output files need to be overwritten."""
        from natcap.invest.recreation import recmodel_client

        out_grid_vector_path = os.path.join(
            self.workspace_dir, 'hex_grid_vector_path.shp')

        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'hexagon', 20000.0,
            out_grid_vector_path)
        # overwrite output
        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'hexagon', 20000.0,
            out_grid_vector_path)

        expected_grid_vector_path = os.path.join(
            REGRESSION_DATA, 'hex_grid_vector_path.shp')

        pygeoprocessing.testing.assert_vectors_equal(
            out_grid_vector_path, expected_grid_vector_path, 0.0)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_existing_regression_coef(self):
        """Recreation test regression coefficients handle existing output."""
        from natcap.invest.recreation import recmodel_client

        response_vector_path = os.path.join(
            self.workspace_dir, 'hex_grid_vector_path.shp')

        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'hexagon', 20000.0,
            response_vector_path)

        predictor_table_path = os.path.join(REGRESSION_DATA, 'predictors.csv')

        tmp_indexed_vector_path = os.path.join(
            self.workspace_dir, 'tmp_indexed_vector.shp')
        tmp_fid_raster_path = os.path.join(
            self.workspace_dir, 'tmp_fid_raster_path.shp')
        out_coefficient_vector_path = os.path.join(
            self.workspace_dir, 'out_coefficient_vector.shp')
        out_predictor_id_list = []

        recmodel_client._build_regression_coefficients(
            response_vector_path, predictor_table_path,
            tmp_indexed_vector_path, tmp_fid_raster_path,
            out_coefficient_vector_path, out_predictor_id_list)

        # build again to test against overwritting output
        recmodel_client._build_regression_coefficients(
            response_vector_path, predictor_table_path,
            tmp_indexed_vector_path, tmp_fid_raster_path,
            out_coefficient_vector_path, out_predictor_id_list)

        expected_coeff_vector_path = os.path.join(
            REGRESSION_DATA, 'test_regression_coefficients.shp')

        pygeoprocessing.testing.assert_vectors_equal(
            out_coefficient_vector_path, expected_coeff_vector_path, 1e-4)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_absolute_regression_coef(self):
        """Recreation test validation from full path."""
        from natcap.invest.recreation import recmodel_client

        response_vector_path = os.path.join(
            self.workspace_dir, 'hex_grid_vector_path.shp')

        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'hexagon', 20000.0,
            response_vector_path)

        predictor_table_path = 'predictors.csv' # os.path.join(self.workspace_dir, 'predictors.csv')

        # these are absolute paths for predictor data
        predictor_list = [
            ('ports', os.path.join(SAMPLE_DATA, 'dredged_ports.shp'),
             'point_count'),
            ('airdist', os.path.join(SAMPLE_DATA, 'airport.shp'),
             'point_nearest_distance'),
            ('bonefish', os.path.join(SAMPLE_DATA, 'bonefish.shp'),
             'polygon_percent_coverage'),
            ('bathy', os.path.join(SAMPLE_DATA, 'dem90m.tif'),
             'raster_mean'),
            ]

        with open(predictor_table_path, 'wb') as table_file:
            table_file.write('id,path,type\n')
            for predictor_id, path, predictor_type in predictor_list:
                table_file.write(
                    '%s,%s,%s\n' % (predictor_id, path, predictor_type))

        recmodel_client._validate_same_projection(
            response_vector_path, predictor_table_path)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_bad_grid_type(self):
        """Recreation ensure that bad grid type raises ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'compute_regression': False,
            'start_year': '2005',
            'end_year': '2014',
            'grid_aoi': True,
            'grid_type': 'circle',  # intentionally bad gridtype
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_year_order(self):
        """Recreation ensure that end year < start year raise ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'compute_regression': True,
            'start_year': '2014',  # note start_year > end_year
            'end_year': '2005',
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors.csv'),
            'results_suffix': u'',
            'scenario_predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors_scenario.csv'),
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_start_year_out_of_range(self):
        """Recreation that start_year out of range raise ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'compute_regression': True,
            'start_year': '2219',  # start year ridiculously out of range
            'end_year': '2250',
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors.csv'),
            'results_suffix': u'',
            'scenario_predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors_scenario.csv'),
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_end_year_out_of_range(self):
        """Recreation that end_year out of range raise ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'compute_regression': True,
            'start_year': '2005',
            'end_year': '2219',  # end year ridiculously out of range
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors.csv'),
            'results_suffix': u'',
            'scenario_predictor_table_path': os.path.join(
                REGRESSION_DATA, 'predictors_scenario.csv'),
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

    @staticmethod
    def _assert_regression_results_eq(
            workspace_dir, file_list_path, result_vector_path,
            agg_results_path):
        """Test workspace against the expected list of files and results.

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
        _test_same_files(file_list_path, workspace_dir)

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
                    numpy.testing.assert_almost_equal(
                        feature.GetField(field), value,
                        decimal=tolerance_places)
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None


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
