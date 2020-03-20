"""InVEST Recreation model tests."""
import datetime
import glob
import zipfile
import socket
import threading
import unittest
import tempfile
import shutil
import os
import functools
import logging
import json
import queue

import Pyro4
import pygeoprocessing
import pygeoprocessing.testing
import numpy
import pandas
from osgeo import gdal
import taskgraph

from natcap.invest import utils

Pyro4.config.SERIALIZER = 'marshal'  # allow null bytes in strings

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'recreation')
SAMPLE_DATA = os.path.join(REGRESSION_DATA, 'input')

LOGGER = logging.getLogger('test_recreation')


def _timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(target):
        """Wrap the original function."""
        work_queue = queue.Queue()
        result_queue = queue.Queue()

        def worker():
            """Read one func,args,kwargs tuple and execute."""
            try:
                func, args, kwargs = work_queue.get()
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(e)
                raise

        work_thread = threading.Thread(target=worker)
        work_thread.daemon = True
        work_thread.start()

        @functools.wraps(target)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            try:
                work_queue.put((target, args, kwargs))
                result = result_queue.get(timeout=max_timeout)
                if isinstance(result, Exception):
                    raise result
                return result
            except queue.Empty:
                raise RuntimeError("Timeout of %f exceeded" % max_timeout)
        return func_wrapper
    return timeout_decorator


def _make_empty_files(base_file_list):
    """Create a list of empty files.

    Parameters:
        base_file_list: a list of paths to empty files to be created.

    Returns:
        None.

    """
    for file_path in base_file_list:
        with open(file_path, 'w') as open_file:
            open_file.write('')


def _resample_csv(base_csv_path, base_dst_path, resample_factor):
    """Resample (downsize) a csv file by a certain resample factor.

    Parameters:
        base_csv_path (str): path to the source csv file to be resampled.
        base_dst_path (str): path to the destination csv file.
        resample_factor (int): the factor used to determined how many rows
            should be skipped before writing a row to the destination file.

    Returns:
        None

    """
    with open(base_csv_path, 'r') as read_table:
        with open(base_dst_path, 'w') as write_table:
            for i, line in enumerate(read_table):
                if i % resample_factor == 0:
                    write_table.write(line)


class TestBufferedNumpyDiskMap(unittest.TestCase):
    """Tests for BufferedNumpyDiskMap."""

    def setUp(self):
        """Setup workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_basic_operation(self):
        """Recreation test buffered file manager basic ops w/ no buffer."""
        from natcap.invest.recreation import buffered_numpy_disk_map
        file_manager = buffered_numpy_disk_map.BufferedNumpyDiskMap(
            os.path.join(self.workspace_dir, 'test'), 0)

        file_manager.append(1234, numpy.array([1, 2, 3, 4]))
        file_manager.append(1234, numpy.array([1, 2, 3, 4]))
        file_manager.append(4321, numpy.array([-4, -1, -2, 4]))

        numpy.testing.assert_equal(
            file_manager.read(1234), numpy.array([1, 2, 3, 4, 1, 2, 3, 4]))

        numpy.testing.assert_equal(
            file_manager.read(4321), numpy.array([-4, -1, -2, 4]))

        file_manager.delete(1234)
        with self.assertRaises(IOError):
            file_manager.read(1234)


class TestRecServer(unittest.TestCase):
    """Tests that set up local rec server on a port and call through."""

    def setUp(self):
        """Setup workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.resampled_data_path = os.path.join(
            self.workspace_dir, 'resampled_data.csv')
        _resample_csv(
            os.path.join(SAMPLE_DATA, 'sample_data.csv'),
            self.resampled_data_path, resample_factor=10)

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def test_hashfile(self):
        """Recreation test for hash of file."""
        from natcap.invest.recreation import recmodel_server
        file_hash = recmodel_server._hashfile(
            self.resampled_data_path, blocksize=2**20, fast_hash=False)
        # The exact encoded string that is hashed is dependent on python version,
        # with Python 3 including b prefix and \n suffix.
        # these hashes are for [py2.7, py3.6]
        self.assertIn(file_hash, ['c052e7a0a4c5e528', 'c8054b109d7a9d2a'])

    def test_hashfile_fast(self):
        """Recreation test for hash and fast hash of file."""
        from natcap.invest.recreation import recmodel_server
        file_hash = recmodel_server._hashfile(
            self.resampled_data_path, blocksize=2**20, fast_hash=True)
        # we can't assert the full hash since it is dependant on the file
        # last access time and we can't reliably set that in Python.
        # instead we just check that at the very least it ends with _fast_hash
        self.assertTrue(file_hash.endswith('_fast_hash'))

    def test_year_order(self):
        """Recreation ensure that end year < start year raise ValueError."""
        from natcap.invest.recreation import recmodel_server

        with self.assertRaises(ValueError):
            # intentionally construct start year > end year
            recmodel_server.RecModel(
                self.resampled_data_path,
                2014, 2005, os.path.join(self.workspace_dir, 'server_cache'))

    @_timeout(30.0)
    def test_workspace_fetcher(self):
        """Recreation test workspace fetcher on a local Pyro4 empty server."""
        from natcap.invest.recreation import recmodel_server
        from natcap.invest.recreation import recmodel_workspace_fetcher

        # Attempt a few connections, we've had this test be flaky on the
        # entire suite run which we suspect is because of a race condition
        server_launched = False
        for _ in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', 0))
                port = sock.getsockname()[1]
                sock.close()
                sock = None

                server_args = {
                    'hostname': 'localhost',
                    'port': port,
                    'raw_csv_point_data_path': self.resampled_data_path,
                    'cache_workspace': self.workspace_dir,
                    'min_year': 2010,
                    'max_year': 2015,
                }

                server_thread = threading.Thread(
                    target=recmodel_server.execute, args=(server_args,))
                server_thread.daemon = True
                server_thread.start()
                server_launched = True
                break
            except:
                LOGGER.warn("Can't start server process on port %d", port)
        if not server_launched:
            self.fail("Server didn't start")

        path = "PYRO:natcap.invest.recreation@localhost:%s" % port
        LOGGER.info("Local server path %s", path)
        recreation_server = Pyro4.Proxy(path)
        aoi_path = os.path.join(
            SAMPLE_DATA, 'test_aoi_for_subset.shp')
        basename = os.path.splitext(aoi_path)[0]
        aoi_archive_path = os.path.join(
            self.workspace_dir, 'aoi_zipped.zip')
        with zipfile.ZipFile(aoi_archive_path, 'w') as myzip:
            for filename in glob.glob(basename + '.*'):
                myzip.write(filename, os.path.basename(filename))

        # convert shapefile to binary string for serialization
        zip_file_binary = open(aoi_archive_path, 'rb').read()
        date_range = (('2005-01-01'), ('2014-12-31'))
        out_vector_filename = 'test_aoi_for_subset_pud.shp'

        _, workspace_id = (
            recreation_server.calc_photo_user_days_in_aoi(
                zip_file_binary, date_range, out_vector_filename))
        fetcher_args = {
            'workspace_dir': self.workspace_dir,
            'hostname': 'localhost',
            'port': port,
            'workspace_id': workspace_id,
        }
        try:
            recmodel_workspace_fetcher.execute(fetcher_args)
        except:
            LOGGER.error(
                "Server process failed (%s) is_alive=%s",
                str(server_thread), server_thread.is_alive())
            raise

        out_workspace_dir = os.path.join(
            self.workspace_dir, 'workspace_zip')
        os.makedirs(out_workspace_dir)
        workspace_zip_path = os.path.join(
            self.workspace_dir, workspace_id + '.zip')
        zipfile.ZipFile(workspace_zip_path, 'r').extractall(
            out_workspace_dir)
        pygeoprocessing.testing.assert_vectors_equal(
            aoi_path,
            os.path.join(out_workspace_dir, 'test_aoi_for_subset.shp'), 1E-6)

    @_timeout(30.0)
    def test_empty_server(self):
        """Recreation test a client call to simple server."""
        from natcap.invest.recreation import recmodel_server
        from natcap.invest.recreation import recmodel_client

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
        server_thread.daemon = True
        server_thread.start()

        client_args = {
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'test_aoi_for_subset.shp'),
            'cell_size': 7000.0,
            'hostname': 'localhost',
            'port': port,
            'compute_regression': False,
            'start_year': '2005',
            'end_year': '2014',
            'grid_aoi': False,
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
        }
        recmodel_client.execute(client_args)

        # testing for file existence seems reasonable since mostly we are
        # testing that a local server starts and a client connects to it
        _test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list_empty_local_server.txt'),
            self.workspace_dir)

    def test_local_aggregate_points(self):
        """Recreation test single threaded local AOI aggregate calculation."""
        from natcap.invest.recreation import recmodel_server

        recreation_server = recmodel_server.RecModel(
            self.resampled_data_path, 2005, 2014,
            os.path.join(self.workspace_dir, 'server_cache'))

        aoi_path = os.path.join(SAMPLE_DATA, 'test_aoi_for_subset.shp')

        basename = os.path.splitext(aoi_path)[0]
        aoi_archive_path = os.path.join(
            self.workspace_dir, 'aoi_zipped.zip')
        with zipfile.ZipFile(aoi_archive_path, 'w') as myzip:
            for filename in glob.glob(basename + '.*'):
                myzip.write(filename, os.path.basename(filename))

        # convert shapefile to binary string for serialization
        zip_file_binary = open(aoi_archive_path, 'rb').read()

        # transfer zipped file to server
        date_range = (('2005-01-01'), ('2014-12-31'))
        out_vector_filename = 'test_aoi_for_subset_pud.shp'
        zip_result, workspace_id = (
            recreation_server.calc_photo_user_days_in_aoi(
                zip_file_binary, date_range, out_vector_filename))

        # unpack result
        result_zip_path = os.path.join(self.workspace_dir, 'pud_result.zip')
        open(result_zip_path, 'wb').write(zip_result)
        zipfile.ZipFile(result_zip_path, 'r').extractall(self.workspace_dir)

        result_vector_path = os.path.join(
            self.workspace_dir, out_vector_filename)
        expected_vector_path = os.path.join(
            REGRESSION_DATA, 'test_aoi_for_subset_pud.shp')
        pygeoprocessing.testing.assert_vectors_equal(
            expected_vector_path, result_vector_path, 1E-6)

        # ensure the remote workspace is as expected
        workspace_zip_binary = recreation_server.fetch_workspace_aoi(
            workspace_id)
        out_workspace_dir = os.path.join(self.workspace_dir, 'workspace_zip')
        os.makedirs(out_workspace_dir)
        workspace_zip_path = os.path.join(out_workspace_dir, 'workspace.zip')
        open(workspace_zip_path, 'wb').write(workspace_zip_binary)
        zipfile.ZipFile(workspace_zip_path, 'r').extractall(out_workspace_dir)
        pygeoprocessing.testing.assert_vectors_equal(
            aoi_path,
            os.path.join(out_workspace_dir, 'test_aoi_for_subset.shp'), 1E-6)

    def test_local_calc_poly_pud(self):
        """Recreation test single threaded local PUD calculation."""
        from natcap.invest.recreation import recmodel_server

        recreation_server = recmodel_server.RecModel(
            self.resampled_data_path,
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'))

        date_range = (
            numpy.datetime64('2005-01-01'),
            numpy.datetime64('2014-12-31'))

        poly_test_queue = queue.Queue()
        poly_test_queue.put(0)
        poly_test_queue.put('STOP')
        pud_poly_feature_queue = queue.Queue()
        recmodel_server._calc_poly_pud(
            recreation_server.qt_pickle_filename,
            os.path.join(SAMPLE_DATA, 'test_aoi_for_subset.shp'),
            date_range, poly_test_queue, pud_poly_feature_queue)

        # assert annual average PUD is the same as regression
        self.assertEqual(
            83.2, pud_poly_feature_queue.get()[1][0])

    def test_local_calc_existing_cached(self):
        """Recreation local PUD calculation on existing quadtree."""
        from natcap.invest.recreation import recmodel_server

        recreation_server = recmodel_server.RecModel(
            self.resampled_data_path,
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'))
        recreation_server = None
        # This will not generate a new quadtree but instead load existing one
        recreation_server = recmodel_server.RecModel(
            self.resampled_data_path,
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'))

        date_range = (
            numpy.datetime64('2005-01-01'),
            numpy.datetime64('2014-12-31'))

        poly_test_queue = queue.Queue()
        poly_test_queue.put(0)
        poly_test_queue.put('STOP')
        pud_poly_feature_queue = queue.Queue()
        recmodel_server._calc_poly_pud(
            recreation_server.qt_pickle_filename,
            os.path.join(SAMPLE_DATA, 'test_aoi_for_subset.shp'),
            date_range, poly_test_queue, pud_poly_feature_queue)

        # assert annual average PUD is the same as regression
        self.assertEqual(
            83.2, pud_poly_feature_queue.get()[1][0])

    def test_parse_input_csv(self):
        """Recreation test parsing raw CSV."""
        from natcap.invest.recreation import recmodel_server

        block_offset_size_queue = queue.Queue()
        block_offset_size_queue.put((0, 2**10))
        block_offset_size_queue.put('STOP')
        numpy_array_queue = queue.Queue()
        recmodel_server._parse_input_csv(
            block_offset_size_queue, self.resampled_data_path,
            numpy_array_queue)
        val = numpy_array_queue.get()
        # we know what the first date is
        self.assertEqual(val[0][0], datetime.date(2013, 3, 16))

    @_timeout(30.0)
    def test_regression_local_server(self):
        """Recreation base regression test on sample data on local server.

        Executes Recreation model all the way through scenario prediction.
        With this florida AOI, raster and vector predictors do not
        intersect the AOI. This makes for a fast test and incidentally
        covers an edge case.
        """
        from natcap.invest.recreation import recmodel_client
        from natcap.invest.recreation import recmodel_server

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
            'raw_csv_point_data_path': self.resampled_data_path,
            'cache_workspace': self.workspace_dir,
            'min_year': 2004,
            'max_year': 2015,
            'max_points_per_node': 200,
        }

        server_thread = threading.Thread(
            target=recmodel_server.execute, args=(server_args,))
        server_thread.daemon = True
        server_thread.start()

        args = {
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'local_recreation_aoi_florida_utm18n.shp'),
            'cell_size': 40000.0,
            'compute_regression': True,
            'start_year': '2005',
            'end_year': '2014',
            'hostname': 'localhost',
            'port': port,
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors.csv'),
            'results_suffix': '',
            'scenario_predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_scenario.csv'),
            'workspace_dir': self.workspace_dir,
        }

        recmodel_client.execute(args)

        _assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_base_florida_aoi.txt'),
            os.path.join(args['workspace_dir'], 'scenario_results.shp'),
            os.path.join(REGRESSION_DATA, 'local_server_scenario_results.csv'))

    def test_all_metrics_local_server(self):
        """Recreation test with all but trivial predictor metrics.

        Executes Recreation model all the way through scenario prediction.
        With this 'extra_fields_features' AOI, we also cover two edge cases:
        1) the AOI has a pre-existing field that the model wishes to create.
        2) the AOI has features only covering nodata raster predictor values."""
        from natcap.invest.recreation import recmodel_client
        from natcap.invest.recreation import recmodel_server

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
            'raw_csv_point_data_path': self.resampled_data_path,
            'cache_workspace': self.workspace_dir,
            'min_year': 2008,
            'max_year': 2015,
            'max_points_per_node': 200,
        }

        server_thread = threading.Thread(
            target=recmodel_server.execute, args=(server_args,))
        server_thread.daemon = True
        server_thread.start()

        args = {
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'andros_aoi_with_extra_fields_features.shp'),
            'compute_regression': True,
            'start_year': '2008',
            'end_year': '2014',
            'grid_aoi': False,
            'predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_all.csv'),
            'scenario_predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_all.csv'),
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'hostname': server_args['hostname'],
            'port': server_args['port'],
        }
        recmodel_client.execute(args)

        out_grid_vector_path = os.path.join(
            args['workspace_dir'], 'predictor_data.shp')
        expected_grid_vector_path = os.path.join(
            REGRESSION_DATA, 'predictor_data_all_metrics.shp')
        _assert_vector_attributes_eq(
            out_grid_vector_path, expected_grid_vector_path, 3)

        out_scenario_path = os.path.join(
            args['workspace_dir'], 'scenario_results.shp')
        expected_scenario_path = os.path.join(
            REGRESSION_DATA, 'scenario_results_all_metrics.shp')
        _assert_vector_attributes_eq(
            out_scenario_path, expected_scenario_path, 3)

    def test_results_suffix_on_serverside_files(self):
        """Recreation test suffix gets added to files created on server."""
        from natcap.invest.recreation import recmodel_client
        from natcap.invest.recreation import recmodel_server

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
            'raw_csv_point_data_path': self.resampled_data_path,
            'cache_workspace': self.workspace_dir,
            'min_year': 2014,
            'max_year': 2015,
            'max_points_per_node': 200,
        }

        server_thread = threading.Thread(
            target=recmodel_server.execute, args=(server_args,))
        server_thread.daemon = True
        server_thread.start()

        args = {
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'andros_aoi_with_extra_fields_features.shp'),
            'compute_regression': False,
            'start_year': '2014',
            'end_year': '2015',
            'grid_aoi': False,
            'results_suffix': 'hello',
            'workspace_dir': self.workspace_dir,
            'hostname': server_args['hostname'],
            'port': server_args['port'],
        }
        recmodel_client.execute(args)

        self.assertTrue(os.path.exists(
            os.path.join(args['workspace_dir'], 'monthly_table_hello.csv')))
        self.assertTrue(os.path.exists(
            os.path.join(args['workspace_dir'], 'pud_results_hello.shp')))


class TestLocalRecServer(unittest.TestCase):
    """Tests using a local rec server."""

    def setUp(self):
        """Setup workspace and server."""
        from natcap.invest.recreation import recmodel_server
        self.workspace_dir = tempfile.mkdtemp()
        self.recreation_server = recmodel_server.RecModel(
            os.path.join(SAMPLE_DATA, 'sample_data.csv'),
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'))

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_local_aoi(self):
        """Recreation test local AOI with local server."""
        aoi_path = os.path.join(SAMPLE_DATA, 'test_local_aoi_for_subset.shp')
        date_range = (
            numpy.datetime64('2010-01-01'),
            numpy.datetime64('2014-12-31'))
        out_vector_filename = os.path.join(self.workspace_dir, 'pud.shp')
        self.recreation_server._calc_aggregated_points_in_aoi(
            aoi_path, self.workspace_dir, date_range, out_vector_filename)

        output_lines = open(os.path.join(
            self.workspace_dir, 'monthly_table.csv'), 'r').readlines()
        expected_lines = open(os.path.join(
            REGRESSION_DATA, 'expected_monthly_table_for_subset.csv'),
                              'r').readlines()

        if output_lines != expected_lines:
            raise ValueError(
                "Output table not the same as input.\n"
                "Expected:\n%s\nGot:\n%s" % (expected_lines, output_lines))


class RecreationRegressionTests(unittest.TestCase):
    """Regression tests for InVEST Recreation model."""

    def setUp(self):
        """Setup workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_data_missing_in_predictors(self):
        """Recreation raise exception if predictor data missing."""
        from natcap.invest.recreation import recmodel_client

        response_vector_path = os.path.join(SAMPLE_DATA, 'andros_aoi.shp')
        table_path = os.path.join(
            SAMPLE_DATA, 'predictors_data_missing.csv')

        with self.assertRaises(ValueError):
            recmodel_client._validate_same_projection(
                response_vector_path, table_path)

    def test_data_different_projection(self):
        """Recreation raise exception if data in different projection."""
        from natcap.invest.recreation import recmodel_client

        response_vector_path = os.path.join(SAMPLE_DATA, 'andros_aoi.shp')
        table_path = os.path.join(
            SAMPLE_DATA, 'predictors_wrong_projection.csv')

        with self.assertRaises(ValueError):
            recmodel_client._validate_same_projection(
                response_vector_path, table_path)

    def test_different_tables(self):
        """Recreation exception if scenario ids different than predictor."""
        from natcap.invest.recreation import recmodel_client

        base_table_path = os.path.join(
            SAMPLE_DATA, 'predictors_data_missing.csv')
        scenario_table_path = os.path.join(
            SAMPLE_DATA, 'predictors_wrong_projection.csv')

        with self.assertRaises(ValueError):
            recmodel_client._validate_same_ids_and_types(
                base_table_path, scenario_table_path)

    def test_delay_op(self):
        """Recreation coverage of delay op function."""
        from natcap.invest.recreation import recmodel_client

        # not much to test here but that the function is invoked
        # guarantee the time has exceeded since we can't have negative time
        last_time = -1.0
        time_delay = 1.0
        called = [False]

        def func():
            """Set `called` to True."""
            called[0] = True
        recmodel_client.delay_op(last_time, time_delay, func)
        self.assertTrue(called[0])

    def test_raster_sum_mean_no_nodata(self):
        """Recreation test sum/mean if raster doesn't have nodata defined."""
        from natcap.invest.recreation import recmodel_client

        # The following raster has no nodata value
        raster_path = os.path.join(SAMPLE_DATA, 'no_nodata_raster.tif')

        response_vector_path = os.path.join(SAMPLE_DATA, 'andros_aoi.shp')
        target_path = os.path.join(self.workspace_dir, "predictor.json")
        recmodel_client._raster_sum_mean(
            raster_path, "mean", response_vector_path, target_path)

        with open(target_path, 'r') as file:
            predictor_results = json.load(file)
        # These constants were calculated by hand by Dave.
        numpy.testing.assert_almost_equal(predictor_results['0'], 13.0)

    def test_raster_sum_mean_nodata(self):
        """Recreation test sum/mean if raster has no valid pixels.

        This may be a raster that does not intersect with the AOI, or
        one that does intersect, but is entirely nodata within the AOI.
        Such a raster is not usable as a predictor variable.
        """
        from natcap.invest.recreation import recmodel_client

        # The following raster has only nodata pixels.
        raster_path = os.path.join(SAMPLE_DATA, 'nodata_raster.tif')
        response_vector_path = os.path.join(SAMPLE_DATA, 'andros_aoi.shp')
        target_path = os.path.join(self.workspace_dir, "predictor.json")

        recmodel_client._raster_sum_mean(
            raster_path, "sum", response_vector_path, target_path)

        with open(target_path, 'r') as file:
            predictor_results = json.load(file)
        # Assert that target file was written and it is an empty dictionary
        assert(len(predictor_results) == 0)

    def test_least_squares_regression(self):
        """Recreation regression test for the least-squares linear model."""
        from natcap.invest.recreation import recmodel_client

        coefficient_vector_path = os.path.join(
            REGRESSION_DATA, 'predictor_data.shp')
        response_vector_path = os.path.join(
            REGRESSION_DATA, 'predictor_data_pud.shp')
        response_id = 'PUD_YR_AVG'

        _, coefficients, ssres, r_sq, r_sq_adj, std_err, dof, se_est = (
            recmodel_client._build_regression(
                response_vector_path, coefficient_vector_path, response_id))

        results = {}
        results['coefficients'] = coefficients
        results['ssres'] = ssres
        results['r_sq'] = r_sq
        results['r_sq_adj'] = r_sq_adj
        results['std_err'] = std_err
        results['dof'] = dof
        results['se_est'] = se_est

        # Dave created these numbers using Recreation model release/3.5.0
        expected_results = {}
        expected_results['coefficients'] = [
            -3.67484238e-03, -8.76864968e-06, 1.75244536e-01, 2.07040116e-01,
            6.59076098e-01]
        expected_results['ssres'] = 11.03734250869611
        expected_results['r_sq'] = 0.5768926587089602
        expected_results['r_sq_adj'] = 0.5256069203706524
        expected_results['std_err'] = 0.5783294255923199
        expected_results['dof'] = 33
        expected_results['se_est'] = [
            5.93275522e-03, 8.49251058e-06, 1.72921342e-01, 6.39079593e-02,
            3.98165865e-01]

        for key in expected_results:
            numpy.testing.assert_allclose(results[key], expected_results[key])

    @unittest.skip("skipping to avoid remote server call (issue #3753)")
    def test_base_regression(self):
        """Recreation base regression test on fast sample data.

        Executes Recreation model with default data and default arguments.
        """
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 40000.0,
            'compute_regression': True,
            'start_year': '2005',
            'end_year': '2014',
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors.csv'),
            'results_suffix': '',
            'scenario_predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_scenario.csv'),
            'workspace_dir': self.workspace_dir,
        }

        recmodel_client.execute(args)
        _assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_base.txt'),
            os.path.join(args['workspace_dir'], 'scenario_results.shp'),
            os.path.join(REGRESSION_DATA, 'scenario_results_40000.csv'))

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
            out_grid_vector_path, expected_grid_vector_path, 1E-6)

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
            out_grid_vector_path, expected_grid_vector_path, 1E-6)

    @unittest.skip("skipping to avoid remote server call (issue #3753)")
    def test_no_grid_regression(self):
        """Recreation base regression on ungridded AOI."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': False,
            'start_year': '2005',
            'end_year': '2014',
            'grid_aoi': False,
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
        }

        recmodel_client.execute(args)

        expected_result_table = pandas.read_csv(os.path.join(
            REGRESSION_DATA, 'expected_monthly_table_for_no_grid.csv'))
        result_table = pandas.read_csv(
            os.path.join(self.workspace_dir, 'monthly_table.csv'))
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

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
                SAMPLE_DATA, 'predictors_id_too_long.csv'),
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

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
            out_grid_vector_path, expected_grid_vector_path, 1E-6)

    def test_existing_regression_coef(self):
        """Recreation test regression coefficients handle existing output."""
        from natcap.invest.recreation import recmodel_client

        # Initialize a TaskGraph
        taskgraph_db_dir = os.path.join(
            self.workspace_dir, '_taskgraph_working_dir')
        n_workers = -1  # single process mode.
        task_graph = taskgraph.TaskGraph(taskgraph_db_dir, n_workers)

        response_vector_path = os.path.join(
            self.workspace_dir, 'no_grid_vector_path.shp')
        response_polygons_lookup_path = os.path.join(
            self.workspace_dir, 'response_polygons_lookup.pickle')
        recmodel_client._copy_aoi_no_grid(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), response_vector_path)

        predictor_table_path = os.path.join(SAMPLE_DATA, 'predictors.csv')

        # make outputs to be overwritten
        predictor_dict = utils.build_lookup_from_csv(predictor_table_path, 'id')
        predictor_list = predictor_dict.keys()
        tmp_working_dir = tempfile.mkdtemp(dir=self.workspace_dir)
        empty_json_list = [
            os.path.join(tmp_working_dir, x + '.json') for x in predictor_list]
        out_coefficient_vector_path = os.path.join(
            self.workspace_dir, 'out_coefficient_vector.shp')
        _make_empty_files(
            [out_coefficient_vector_path] + empty_json_list)

        prepare_response_polygons_task = task_graph.add_task(
            func=recmodel_client._prepare_response_polygons_lookup,
            args=(response_vector_path,
                  response_polygons_lookup_path),
            target_path_list=[response_polygons_lookup_path],
            task_name='prepare response polygons for geoprocessing')
        # build again to test against overwriting output
        recmodel_client._schedule_predictor_data_processing(
            response_vector_path, response_polygons_lookup_path,
            prepare_response_polygons_task, predictor_table_path,
            out_coefficient_vector_path, tmp_working_dir, task_graph)

        expected_coeff_vector_path = os.path.join(
            REGRESSION_DATA, 'test_regression_coefficients.shp')

        _assert_vector_attributes_eq(
            out_coefficient_vector_path, expected_coeff_vector_path, 6)

    def test_predictor_table_absolute_paths(self):
        """Recreation test validation from full path."""
        from natcap.invest.recreation import recmodel_client

        response_vector_path = os.path.join(
            self.workspace_dir, 'no_grid_vector_path.shp')
        recmodel_client._copy_aoi_no_grid(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), response_vector_path)

        predictor_table_path = os.path.join(
            self.workspace_dir, 'predictors.csv')

        # these are absolute paths for predictor data
        predictor_list = [
            ('ports',
             os.path.join(SAMPLE_DATA, 'predictors', 'dredged_ports.shp'),
             'point_count'),
            ('airdist',
             os.path.join(SAMPLE_DATA, 'predictors', 'airport.shp'),
             'point_nearest_distance'),
            ('bonefish',
             os.path.join(SAMPLE_DATA, 'predictors', 'bonefish_simp.shp'),
             'polygon_percent_coverage'),
            ('bathy',
             os.path.join(SAMPLE_DATA, 'predictors', 'dem90m_coarse.tif'),
             'raster_mean'),
            ]

        with open(predictor_table_path, 'w') as table_file:
            table_file.write('id,path,type\n')
            for predictor_id, path, predictor_type in predictor_list:
                table_file.write(
                    '%s,%s,%s\n' % (predictor_id, path, predictor_type))

        # The expected behavior here is that _validate_same_projection does
        # not raise a ValueError.  The try/except block makes that explicit
        # and also explicitly fails the test if it does. Note if a different
        # exception is raised the test will raise an error, thus differentiating
        # between a failed test and an error.
        try:
            recmodel_client._validate_same_projection(
                response_vector_path, predictor_table_path)
        except ValueError:
            self.fail(
                "_validate_same_projection raised ValueError unexpectedly!")

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
                SAMPLE_DATA, 'predictors.csv'),
            'results_suffix': '',
            'scenario_predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_scenario.csv'),
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

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
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

    def test_start_year_out_of_range(self):
        """Recreation that start_year out of range raise ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'compute_regression': True,
            'start_year': '1219',  # start year ridiculously out of range
            'end_year': '2014',
            'grid_aoi': True,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors.csv'),
            'results_suffix': '',
            'scenario_predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_scenario.csv'),
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

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
                SAMPLE_DATA, 'predictors.csv'),
            'results_suffix': '',
            'scenario_predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_scenario.csv'),
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)


class RecreationValidationTests(unittest.TestCase):
    """Tests for the Recreation Model ARGS_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'aoi_path',
            'start_year',
            'end_year'
        ]

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys(self):
        """Recreation Validate: assert missing required keys."""
        from natcap.invest.recreation import recmodel_client
        from natcap.invest import validation

        validation_errors = recmodel_client.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_grid_aoi(self):
        """Recreation Validate: assert missing keys for grid option."""
        from natcap.invest.recreation import recmodel_client
        from natcap.invest import validation

        validation_errors = recmodel_client.validate({'grid_aoi': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys + ['grid_type', 'cell_size'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_compute_regression(self):
        """Recreation Validate: assert missing keys for regression option."""
        from natcap.invest.recreation import recmodel_client
        from natcap.invest import validation

        validation_errors = recmodel_client.validate({'compute_regression': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys + ['predictor_table_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_bad_predictor_table_header(self):
        """Recreation Validate: assert messages for bad table headers."""
        from natcap.invest.recreation import recmodel_client

        table_path = os.path.join(self.workspace_dir, 'table.csv')
        with open(table_path, 'w') as file:
            file.write('foo,bar,baz\n')
            file.write('a,b,c\n')

        expected_message = "Fields are missing from this table: ['ID', 'PATH', 'TYPE']"
        validation_warnings = recmodel_client.validate(
            {'predictor_table_path': table_path})
        actual_messages = set()
        for keys, error_strings in validation_warnings:
            actual_messages.add(error_strings)
        self.assertTrue(expected_message in actual_messages)

        validation_warnings = recmodel_client.validate(
            {'scenario_predictor_table_path': table_path})
        actual_messages = set()
        for keys, error_strings in validation_warnings:
            actual_messages.add(error_strings)
        self.assertTrue(expected_message in actual_messages)


def _assert_vector_attributes_eq(
        actual_vector_path, expected_vector_path, tolerance_places=3):
    """Assert fieldnames and values are equal with no respect to order."""
    try:
        actual_vector = gdal.OpenEx(actual_vector_path, gdal.OF_VECTOR)
        actual_layer = actual_vector.GetLayer()
        expected_vector = gdal.OpenEx(expected_vector_path, gdal.OF_VECTOR)
        expected_layer = expected_vector.GetLayer()

        assert(
            actual_layer.GetFeatureCount() == expected_layer.GetFeatureCount())

        field_names = [field.name for field in expected_layer.schema]
        for feature in expected_layer:
            fid = feature.GetFID()
            expected_values = [
                feature.GetField(field) for field in field_names]

            actual_feature = actual_layer.GetFeature(fid)
            actual_values = [
                actual_feature.GetField(field) for field in field_names]

            for av, ev in zip(actual_values, expected_values):
                if av is not None:
                    numpy.testing.assert_almost_equal(
                        av, ev, decimal=tolerance_places)
                else:
                    # Could happen when a raster predictor is only nodata
                    assert(ev is None)
            feature = None
            actual_feature = None
    finally:
        actual_layer = None
        actual_vector = None
        expected_layer = None
        expected_vector = None


def _assert_regression_results_eq(
        workspace_dir, file_list_path, result_vector_path,
        expected_results_path):
    """Test workspace against the expected list of files and results.

    Parameters:
        workspace_dir (string): path to the completed model workspace
        file_list_path (string): path to a file that has a list of all
            the expected files relative to the workspace base
        result_vector_path (string): path to shapefile
            produced by the Recreation model.
        expected_results_path (string): path to a csv file that has the
            expected results of a scenario prediction model run.

    Returns:
        None

    Raises:
        AssertionError if any files are missing or results are out of
        range by `tolerance_places`
    """
    try:
        # Test that the workspace has the same files as we expect
        _test_same_files(file_list_path, workspace_dir)

        # The tolerance of 3 digits after the decimal was determined by
        # experimentation on the application with the given range of
        # numbers.  This is an apparently reasonable approach as described
        # by ChrisF: http://stackoverflow.com/a/3281371/42897
        # and even more reading about picking numerical tolerance
        # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        tolerance_places = 3

        result_vector = gdal.OpenEx(result_vector_path, gdal.OF_VECTOR)
        result_layer = result_vector.GetLayer()
        expected_results = pandas.read_csv(expected_results_path, dtype=float)
        field_names = list(expected_results)
        for feature in result_layer:
            values = [feature.GetField(field) for field in field_names]
            fid = feature.GetFID()
            expected_values = list(expected_results.iloc[fid])
            for v, ev in zip(values, expected_values):
                if v is not None:
                    numpy.testing.assert_almost_equal(
                        v, ev, decimal=tolerance_places)
                else:
                    # Could happen when a raster predictor is only nodata
                    assert(numpy.isnan(ev))
            feature = None

    finally:
        result_layer = None
        gdal.Dataset.__swig_destroy__(result_vector)
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
