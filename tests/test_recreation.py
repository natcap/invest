"""InVEST Recreation model tests."""
import datetime
import functools
import logging
import json
import math
import multiprocessing
import os
import pickle
import queue
import random
import shutil
import socket
import string
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pandas
import pygeoprocessing
import Pyro5
import shapely
import taskgraph
import warnings

from natcap.invest import utils

gdal.UseExceptions()
Pyro5.config.SERIALIZER = 'marshal'  # allow null bytes in strings

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

    Args:
        base_file_list: a list of paths to empty files to be created.

    Returns:
        None.

    """
    for file_path in base_file_list:
        with open(file_path, 'w') as open_file:
            open_file.write('')


def _resample_csv(base_csv_path, base_dst_path, resample_factor):
    """Resample (downsize) a csv file by a certain resample factor.

    Args:
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


def _make_simple_lat_lon_aoi(geom_list, aoi_path, fields=None, attribute_list=None):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    wkt = srs.ExportToWkt()
    pygeoprocessing.shapely_geometry_to_vector(
        geom_list, aoi_path, wkt, 'GeoJSON', fields, attribute_list)


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

    def test_buffered_multiprocess_operation(self):
        """Recreation test buffered file manager parallel flushes."""
        from natcap.invest.recreation import buffered_numpy_disk_map

        array1 = numpy.array([1, 2, 3, 4])
        array2 = numpy.array([-4, -1, -2, 4])
        arraysize = array1.size * buffered_numpy_disk_map.BufferedNumpyDiskMap._ARRAY_TUPLE_TYPE.itemsize
        buffer_size = arraysize * 2  # will flush every 3rd append

        file_manager = buffered_numpy_disk_map.BufferedNumpyDiskMap(
            os.path.join(self.workspace_dir, 'test'), buffer_size, n_workers=2)

        # Make sure multiple array IDs are in present in
        # the cache to trigger multiprocess flush
        file_manager.append(1234, array1)
        file_manager.append(4321, array2)
        file_manager.append(1234, array1)
        file_manager.append(4321, array2)
        file_manager.append(1234, array1)
        file_manager.append(4321, array2)

        numpy.testing.assert_equal(
            file_manager.read(1234), numpy.tile(array1, 3))

        numpy.testing.assert_equal(
            file_manager.read(4321), numpy.tile(array2, 3))

        file_manager.delete(1234)
        with self.assertRaises(IOError):
            file_manager.read(1234)


class UnitTestRecServer(unittest.TestCase):
    """Tests for recmodel_server functions and the RecModel object."""

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
        # The exact encoded string that is hashed is dependent on python
        # version, with Python 3 including b prefix and \n suffix.
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
                2014, 2005, os.path.join(self.workspace_dir, 'server_cache'),
                raw_csv_filename=self.resampled_data_path)

    def test_local_aggregate_points(self):
        """Recreation test single threaded local AOI aggregate calculation."""
        from natcap.invest.recreation import recmodel_server

        recreation_server = recmodel_server.RecModel(
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'),
            raw_csv_filename=self.resampled_data_path)

        aoi_filename = 'test_aoi.geojson'
        aoi_path = os.path.join(
            self.workspace_dir, aoi_filename)
        # This polygon matches the test data shapefile we used formerly.
        geomstring = """
            POLYGON ((-5.54101768507434 56.1006500736864,
                      1.2562729659521 56.007023480697,
                      1.01284382417981 50.2396253525534,
                      -5.2039619503127 49.9961962107811,
                      -5.54101768507434 56.1006500736864))"""
        polygon = shapely.wkt.loads(geomstring)
        fields = {'poly_id': ogr.OFTInteger}
        attribute_list = [{'poly_id': 0}]
        _make_simple_lat_lon_aoi([polygon], aoi_path, fields, attribute_list)
        aoi_vector = gdal.OpenEx(aoi_path)

        aoi_archive_path = os.path.join(
            self.workspace_dir, 'aoi_zipped.zip')
        with zipfile.ZipFile(aoi_archive_path, 'w') as myzip:
            for filename in aoi_vector.GetFileList():
                myzip.write(filename, os.path.basename(filename))

        # convert shapefile to binary string for serialization
        with open(aoi_archive_path, 'rb') as file:
            zip_file_binary = file.read()

        # transfer zipped file to server
        start_year = '2005'
        end_year = '2014'
        out_vector_filename = 'results_pud.gpkg'

        zip_result, workspace_id, version_str = (
            recreation_server.calc_user_days_in_aoi(
                zip_file_binary, aoi_filename, start_year, end_year,
                out_vector_filename))

        # unpack result
        result_zip_path = os.path.join(self.workspace_dir, 'pud_result.zip')
        with open(result_zip_path, 'wb') as file:
            file.write(zip_result)
        zipfile.ZipFile(result_zip_path, 'r').extractall(self.workspace_dir)
        result_vector_path = os.path.join(
            self.workspace_dir, out_vector_filename)

        expected_attributes = [
           {'poly_id': 0,
            'PUD_YR_AVG': 83.2,
            'PUD_JAN': 2.5,
            'PUD_FEB': 2.4,
            'PUD_MAR': 33.3,
            'PUD_APR': 24.9,
            'PUD_MAY': 6.5,
            'PUD_JUN': 4.6,
            'PUD_JUL': 1.6,
            'PUD_AUG': 1.9,
            'PUD_SEP': 1.3,
            'PUD_OCT': 2.0,
            'PUD_NOV': 1.0,
            'PUD_DEC': 1.2}
        ]
        fields = {field: ogr.OFTReal for field in expected_attributes[0]}

        expected_vector_path = os.path.join(
            self.workspace_dir, 'regression_pud.geojson')
        _make_simple_lat_lon_aoi(
            [polygon], expected_vector_path, fields, expected_attributes)
        utils._assert_vectors_equal(expected_vector_path, result_vector_path)

    def test_local_calc_poly_ud(self):
        """Recreation test single threaded local PUD calculation."""
        from natcap.invest.recreation import recmodel_server

        recreation_server = recmodel_server.RecModel(
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'),
            raw_csv_filename=self.resampled_data_path)

        date_range = (
            numpy.datetime64('2005-01-01'),
            numpy.datetime64('2014-12-31'))

        aoi_path = os.path.join('aoi.geojson')
        # This polygon matches the test data shapefile we used formerly.
        geomstring = """
            POLYGON ((-5.54101768507434 56.1006500736864,
                      1.2562729659521 56.007023480697,
                      1.01284382417981 50.2396253525534,
                      -5.2039619503127 49.9961962107811,
                      -5.54101768507434 56.1006500736864))"""

        polygon = shapely.wkt.loads(geomstring)
        _make_simple_lat_lon_aoi([polygon], aoi_path)

        poly_test_queue = queue.Queue()
        poly_test_queue.put(0)
        poly_test_queue.put('STOP')
        pud_poly_feature_queue = queue.Queue()
        recmodel_server._calc_poly_ud(
            recreation_server.qt_pickle_filename, aoi_path,
            date_range, poly_test_queue, pud_poly_feature_queue)

        # assert annual average PUD is the same as regression
        self.assertEqual(
            83.2, pud_poly_feature_queue.get()[1][0])

    def test_local_calc_poly_ud_bad_aoi(self):
        """Recreation test PUD calculation with missing AOI features."""
        from natcap.invest.recreation import recmodel_server

        recreation_server = recmodel_server.RecModel(
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'),
            raw_csv_filename=self.resampled_data_path)

        date_range = (
            numpy.datetime64('2005-01-01'),
            numpy.datetime64('2014-12-31'))

        aoi_vector_path = os.path.join(self.workspace_dir, 'aoi.gpkg')
        gpkg_driver = gdal.GetDriverByName('GPKG')
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84/UTM zone 31s
        target_vector = gpkg_driver.Create(
            aoi_vector_path, 0, 0, 0, gdal.GDT_Unknown)
        target_layer = target_vector.CreateLayer(
            'target_layer', srs, ogr.wkbUnknown)

        # Testing with an AOI of 2 features, one is missing Geometry.
        input_geom_list = [
            None,
            ogr.CreateGeometryFromWkt(
                'POLYGON ((1 1, 1 0, 0 0, 0 1, 1 1))')]
        poly_test_queue = queue.Queue()
        poly_test_queue.put(1)  # gpkg FIDs start at 1
        poly_test_queue.put(2)
        target_layer.StartTransaction()
        for geometry in input_geom_list:
            feature = ogr.Feature(target_layer.GetLayerDefn())
            feature.SetGeometry(geometry)
            target_layer.CreateFeature(feature)
        target_layer.CommitTransaction()
        poly_test_queue.put('STOP')
        target_layer = None
        target_vector = None

        pud_poly_feature_queue = queue.Queue()
        recmodel_server._calc_poly_ud(
            recreation_server.qt_pickle_filename,
            aoi_vector_path, date_range, poly_test_queue,
            pud_poly_feature_queue)

        # assert PUD was calculated for the one good AOI feature.
        self.assertEqual(
            0.0, pud_poly_feature_queue.get()[1][0])

    def test_reuse_of_existing_quadtree(self):
        """Test init RecModel can reuse an existing quadtree on disk."""
        from natcap.invest.recreation import recmodel_server

        _ = recmodel_server.RecModel(
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'),
            raw_csv_filename=self.resampled_data_path)

        # This will not generate a new quadtree but instead load existing one
        with patch.object(recmodel_server, 'construct_userday_quadtree') as mock:
            _ = recmodel_server.RecModel(
                2005, 2014, os.path.join(self.workspace_dir, 'server_cache'),
                raw_csv_filename=self.resampled_data_path)
            self.assertFalse(mock.called)

    def test_parse_big_input_csv(self):
        """Recreation test parsing raw CSV."""
        from natcap.invest.recreation import recmodel_server

        block_offset_size_queue = queue.Queue()
        block_offset_size_queue.put((0, 2**10))
        block_offset_size_queue.put('STOP')
        numpy_array_queue = queue.Queue()
        recmodel_server._parse_big_input_csv(
            block_offset_size_queue, numpy_array_queue,
            self.resampled_data_path, 'flickr')
        val = recmodel_server._numpy_loads(numpy_array_queue.get())
        # we know what the first date is
        self.assertEqual(val[0][0], datetime.date(2013, 3, 16))

    def test_numpy_pickling_queue(self):
        """Recreation test _numpy_dumps and _numpy_loads"""
        from natcap.invest.recreation import recmodel_server

        numpy_array_queue = multiprocessing.Queue()
        array = numpy.empty(1, dtype='datetime64,f4')
        numpy_array_queue.put(recmodel_server._numpy_dumps(array))

        out_array = recmodel_server._numpy_loads(numpy_array_queue.get())
        numpy.testing.assert_equal(out_array, array)
        # without _numpy_loads, the queue pickles the array imperfectly,
        # adding a metadata value to the `datetime64` dtype.
        # assert that this doesn't happen. 'f0' is the first subdtype.
        self.assertEqual(out_array.dtype['f0'].metadata, None)

        # assert that saving the array does not raise a warning
        with warnings.catch_warnings(record=True) as ws:
            # cause all warnings to always be triggered
            warnings.simplefilter("always")
            numpy.save(os.path.join(self.workspace_dir, 'out'), out_array)
            # assert that no warning was raised
            self.assertTrue(len(ws) == 0)

    def test_construct_query_twitter_qt(self):
        """Recreation test constructing and querying twitter quadtree."""
        from natcap.invest.recreation import recmodel_server

        # user,date,lat,lon
        # 1117195232,2023-01-01,-22.908,-43.1975
        # 54900515,2023-01-01,44.62804,10.60603

        def make_twitter_csv(target_filename):
            dates = numpy.arange(
                numpy.datetime64('2017-01-01'), numpy.datetime64('2017-12-31'))
            lats = numpy.arange(-90.0, 90.0, 180/len(dates))
            lons = numpy.arange(-180.0, 180.0, 360/len(dates))
            users = [
                ''.join(random.choices(string.digits, k=n))
                for n in random.choices(range(7, 18), k=len(dates))
            ]
            pandas.DataFrame({
                'user': users,
                'date': dates,
                'lat': lats,
                'lon': lons
            }, index=None).to_csv(target_filename, index=False)

        raw_csv_file_list = [
            os.path.join(self.workspace_dir, 'a.csv'),
            os.path.join(self.workspace_dir, 'b.csv'),
        ]
        for filename in raw_csv_file_list:
            make_twitter_csv(filename)

        cache_dir = os.path.join(self.workspace_dir, 'cache')
        ooc_qt_picklefilename = os.path.join(cache_dir, 'qt.pickle')

        recmodel_server.construct_userday_quadtree(
            recmodel_server.INITIAL_BOUNDING_BOX,
            raw_csv_file_list, 'twitter',
            cache_dir, ooc_qt_picklefilename,
            recmodel_server.GLOBAL_MAX_POINTS_PER_NODE,
            recmodel_server.GLOBAL_DEPTH)

        min_year = 2016
        max_year = 2018
        server = recmodel_server.RecModel(
            min_year, max_year, cache_dir,
            quadtree_pickle_filename=ooc_qt_picklefilename,
            dataset_name='twitter')

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        target_filename = 'results.shp'  # model uses ESRI Shapefile driver
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        wkt = srs.ExportToWkt()
        polygon = shapely.geometry.box(-120, -60, 120, 60)
        poly_id = 999
        pygeoprocessing.shapely_geometry_to_vector(
            [polygon], aoi_path, wkt, 'GeoJSON',
            fields={'poly_id': ogr.OFTInteger},
            attribute_list=[{'poly_id': poly_id}])

        start_year = '2017'
        end_year = '2017'
        _, monthly_table_path = server._calc_aggregated_points_in_aoi(
            aoi_path, self.workspace_dir, start_year, end_year, target_filename)

        expected_result_table = pandas.DataFrame({
           'poly_id': [999],
           '2017-1': [0],
           '2017-2': [0],
           '2017-3': [58],
           '2017-4': [60],
           '2017-5': [62],
           '2017-6': [60],
           '2017-7': [62],
           '2017-8': [62],
           '2017-9': [60],
           '2017-10': [62],
           '2017-11': [0],
           '2017-12': [0]
        }, index=None)
        result_table = pandas.read_csv(
            monthly_table_path)
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

    def test_local_query_bad_dates(self):
        """Recreation test local AOI aggregation with invalid date range."""
        from natcap.invest.recreation import recmodel_server

        recreation_server = recmodel_server.RecModel(
            2005, 2014, os.path.join(self.workspace_dir, 'server_cache'),
            raw_csv_filename=self.resampled_data_path)

        start_year = 2005
        end_year = 2099
        with self.assertRaises(ValueError) as error:
            recreation_server._calc_aggregated_points_in_aoi(
                'aoi.gpkg', self.workspace_dir, start_year, end_year,
                'results.gpkg')
            self.assertIn('End year must be between', str(error.exception))


def _synthesize_points_in_aoi(aoi_path, target_flickr_path, target_twitter_path, n_points):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    target_wkt = srs.ExportToWkt()

    aoi_info = pygeoprocessing.get_vector_info(aoi_path)
    lonlat_bbox = pygeoprocessing.transform_bounding_box(
        aoi_info['bounding_box'], aoi_info['projection_wkt'], target_wkt)
    origin_x, origin_y = lonlat_bbox[:2]

    x_size = int(math.sqrt(n_points))
    lon = numpy.linspace(lonlat_bbox[0], lonlat_bbox[2], num=x_size)
    lat = numpy.linspace(lonlat_bbox[1], lonlat_bbox[3], num=x_size)
    offsets = numpy.geomspace(0.01, 1, num=x_size)
    lon = lon + offsets
    lat = lat + offsets

    flickr_dates = pandas.date_range(
        numpy.datetime64('2017-01-01 12:12:12'),
        numpy.datetime64('2017-12-31 12:12:12'), freq='D')
    twitter_dates = pandas.date_range(
        numpy.datetime64('2017-01-01'),
        numpy.datetime64('2017-12-31'), freq='D')

    users = numpy.array([
        ''.join(random.choices(string.digits, k=n))
        for n in random.choices(range(7, 18), k=50)  # 50 unique people
    ])

    x, y = numpy.meshgrid(lon, lat)
    user_array = users[numpy.arange(n_points) % len(users)]

    flickr_df = pandas.DataFrame({
        'id': range(n_points),
        'user': user_array,
        'date': flickr_dates[numpy.arange(n_points) % len(flickr_dates)],
        'lat': y.flatten(),
        'lon': x.flatten(),
        'accuracy': [16] * n_points
    }, index=None)
    flickr_df.to_csv(target_flickr_path, index=False)

    twitter_df = flickr_df.drop(columns=['id', 'accuracy'])
    twitter_df['date'] = twitter_dates[numpy.arange(n_points) % len(twitter_dates)]
    twitter_df.to_csv(target_twitter_path, index=False)


class TestRecClientServer(unittest.TestCase):
    """Client regression tests using a server executing in a local process."""

    @classmethod
    def setUpClass(cls):
        """Setup Rec model server."""
        from natcap.invest.recreation import recmodel_server

        cls.server_workspace_dir = tempfile.mkdtemp()
        flickr_csv_path = os.path.join(
            cls.server_workspace_dir, 'flickr_sample_data.csv')
        twitter_csv_path = os.path.join(
            cls.server_workspace_dir, 'twitter_sample_data.csv')
        n_points = 50**2  # a perfect square for convenience
        _synthesize_points_in_aoi(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            flickr_csv_path, twitter_csv_path,
            n_points=n_points)

        # attempt to get an open port; could result in race condition but
        # will be okay for a test. if this test ever fails because of port
        # in use, that's probably why
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        cls.port = sock.getsockname()[1]
        sock.close()
        sock = None
        cls.hostname = 'localhost'

        server_args = {
            'hostname': cls.hostname,
            'port': cls.port,
            'cache_workspace': cls.server_workspace_dir,
            'max_points_per_node': 200,
            'max_allowable_query': n_points - 100,
            'datasets': {
                'flickr': {
                    'raw_csv_point_data_path': flickr_csv_path,
                    'min_year': 2005,
                    'max_year': 2017
                },
                'twitter': {
                    'raw_csv_point_data_path': twitter_csv_path,
                    'min_year': 2012,
                    'max_year': 2022
                }
            }
        }

        cls.server_process = multiprocessing.Process(
            target=recmodel_server.execute, args=(server_args,), daemon=False)
        cls.server_process.start()

        # The recmodel_server.execute takes ~10-20 seconds to build quadtrees
        # before the remote Pyro object is ready.
        proxy = Pyro5.api.Proxy(
            f'PYRO:natcap.invest.recreation@{cls.hostname}:{cls.port}')
        ready = False
        while not ready and cls.server_process.is_alive():
            try:
                # _pyroBind() forces the client-server handshake and
                # seems like a good way to check if the remote object is ready
                proxy._pyroBind()
            except Pyro5.errors.CommunicationError:
                time.sleep(2)
                continue
            ready = True
        proxy._pyroRelease()

    @classmethod
    def tearDownClass(cls):
        """Delete workspace and terminate child process."""
        cls.server_process.terminate()
        shutil.rmtree(cls.server_workspace_dir, ignore_errors=True)

    def setUp(self):
        """Create workspace"""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace"""
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def test_execute_no_regression(self):
        """Recreation test userday metrics exist if not computing regression."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': False,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'grid_aoi': False,
            'workspace_dir': self.workspace_dir,
            'hostname': self.hostname,
            'port': self.port,
        }
        recmodel_client.execute(args)

        out_regression_vector_path = os.path.join(
            args['workspace_dir'], 'regression_data.gpkg')
        # These fields should exist even if `compute_regression` is False
        expected_fields = ['pr_TUD', 'pr_PUD', 'avg_pr_UD']
        # For convenience, assert the sums of the columns instead of all
        # the individual values.
        actual_sums = sum_vector_columns(
            out_regression_vector_path, expected_fields)
        expected_sums = {
            'pr_TUD': 1.0,
            'pr_PUD': 1.0,
            'avg_pr_UD': 1.0
        }
        for key in expected_sums:
            numpy.testing.assert_almost_equal(
                actual_sums[key], expected_sums[key], decimal=3)

    def test_all_metrics_local_server(self):
        """Recreation test with all but trivial predictor metrics."""
        from natcap.invest.recreation import recmodel_client
        from natcap.invest import validation

        suffix = 'foo'
        args = {
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': True,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'grid_aoi': True,
            'cell_size': 30000,
            'grid_type': 'hexagon',
            'predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_all.csv'),
            'scenario_predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_all.csv'),
            'results_suffix': suffix,
            'workspace_dir': self.workspace_dir,
            'hostname': self.hostname,
            'port': self.port,
        }
        recmodel_client.execute(args)

        out_regression_vector_path = os.path.join(
            args['workspace_dir'], f'regression_data_{suffix}.gpkg')

        predictor_df = recmodel_client.MODEL_SPEC.get_input(
            'predictor_table_path').get_validated_dataframe(
            os.path.join(SAMPLE_DATA, 'predictors_all.csv'))
        field_list = list(predictor_df.index) + ['pr_TUD', 'pr_PUD', 'avg_pr_UD']

        # For convenience, assert the sums of the columns instead of all
        # the individual values.
        actual_sums = sum_vector_columns(out_regression_vector_path, field_list)
        expected_sums = {
            'ports': 11.0,
            'airdist': 875291.8190812231,
            'bonefish_a': 4630187907.293639,
            'bathy': 47.16540460441528,
            'roads': 5072.707571235277,
            'bonefish_p': 792.0711806443292,
            'bathy_sum': 348.04177433624864,
            'pr_TUD': 1.0,
            'pr_PUD': 1.0,
            'avg_pr_UD': 1.0
        }
        for key in expected_sums:
            numpy.testing.assert_almost_equal(
                actual_sums[key], expected_sums[key], decimal=3)

        out_scenario_path = os.path.join(
            args['workspace_dir'], f'scenario_results_{suffix}.gpkg')
        field_list = list(predictor_df.index) + ['pr_UD_EST']
        actual_scenario_sums = sum_vector_columns(out_scenario_path, field_list)
        expected_scenario_sums = {
            'ports': 11.0,
            'airdist': 875291.8190812231,
            'bonefish_a': 4630187907.293639,
            'bathy': 47.16540460441528,
            'roads': 5072.707571235277,
            'bonefish_p': 792.0711806443292,
            'bathy_sum': 348.04177433624864,
            'pr_UD_EST': 0.996366808597374
        }
        for key in expected_scenario_sums:
            numpy.testing.assert_almost_equal(
                actual_scenario_sums[key], expected_scenario_sums[key], decimal=3)

        # assert that all tabular outputs are indexed by the same poly_id
        output_aoi_path = os.path.join(
            args['workspace_dir'], 'intermediate', f'aoi_{suffix}.gpkg')
        aoi_vector = gdal.OpenEx(output_aoi_path, gdal.OF_VECTOR)
        aoi_layer = aoi_vector.GetLayer()
        aoi_poly_id_list = [feature.GetField(
            recmodel_client.POLYGON_ID_FIELD) for feature in aoi_layer]
        aoi_layer = aoi_vector = None

        output_vector_list = [
            out_scenario_path,
            out_regression_vector_path,
            os.path.join(args['workspace_dir'], f'PUD_results_{suffix}.gpkg'),
            os.path.join(args['workspace_dir'], f'TUD_results_{suffix}.gpkg')]
        for vector_filepath in output_vector_list:
            vector = gdal.OpenEx(vector_filepath, gdal.OF_VECTOR)
            layer = vector.GetLayer()
            id_list = [feature.GetField(
                recmodel_client.POLYGON_ID_FIELD) for feature in layer]
            self.assertEqual(id_list, aoi_poly_id_list)
            vector = layer = None

    def test_workspace_fetcher(self):
        """Recreation test workspace fetcher on a local Pyro5 server."""
        from natcap.invest.recreation import recmodel_workspace_fetcher
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': False,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'grid_aoi': False,
            'workspace_dir': self.workspace_dir,
            'hostname': self.hostname,
            'port': self.port,
        }
        recmodel_client.execute(args)

        # workspace IDs are stored in this file
        server_version_path = os.path.join(
            args['workspace_dir'], 'intermediate', 'server_version.pickle')
        with open(server_version_path, 'rb') as f:
            server_data = pickle.load(f)

        server = 'flickr'
        workspace_id = server_data[server]['workspace_id']
        fetcher_args = {
            'workspace_dir': os.path.join(self.workspace_dir, server),
            'hostname': 'localhost',
            'port': self.port,
            'workspace_id': workspace_id,
            'server_id': server,
        }

        recmodel_workspace_fetcher.execute(fetcher_args)
        zip_path = os.path.join(
            fetcher_args['workspace_dir'], f'{server}_{workspace_id}.zip')
        zipfile.ZipFile(zip_path, 'r').extractall(
            fetcher_args['workspace_dir'])

        utils._assert_vectors_equal(
            os.path.join(
                fetcher_args['workspace_dir'],
                'aoi.gpkg'),
            os.path.join(
                args['workspace_dir'], 'intermediate',
                'aoi.gpkg'))

    def test_start_year_out_of_range(self):
        """Test server sends valid date-ranges; client raises ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': False,
            'start_year': '1219',  # start year ridiculously out of range
            'end_year': recmodel_client.MAX_YEAR,
            'grid_aoi': False,
            'workspace_dir': self.workspace_dir,
            'hostname': self.hostname,
            'port': self.port
        }

        with self.assertRaises(ValueError) as cm:
            recmodel_client.execute(args)
        actual_message = str(cm.exception)
        expected_message = 'Start year must be between'
        self.assertIn(expected_message, actual_message)

    def test_end_year_out_of_range(self):
        """Test server sends valid date-ranges; client raises ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': False,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': '2219',  # end year ridiculously out of range
            'grid_aoi': False,
            'workspace_dir': self.workspace_dir,
            'hostname': self.hostname,
            'port': self.port
        }

        with self.assertRaises(ValueError) as cm:
            recmodel_client.execute(args)
        actual_message = str(cm.exception)
        expected_message = 'End year must be between'
        self.assertIn(expected_message, actual_message)

    def test_aoi_too_large(self):
        """Test server checks aoi size; client raises exception."""
        from natcap.invest.recreation import recmodel_client

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        wkt = srs.ExportToWkt()

        # This AOI should capture all the points in the quadtree
        # and that should exceed the max_allowable limit set
        # in the server args.
        aoi_geometries = [shapely.geometry.Polygon([
            (-180, -90), (-180, 90), (180, 90), (180, -90), (-180, -90)])]
        pygeoprocessing.shapely_geometry_to_vector(
            aoi_geometries, aoi_path, wkt, 'GeoJSON')
        args = {
            'aoi_path': aoi_path,
            'compute_regression': False,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'grid_aoi': False,
            'workspace_dir': self.workspace_dir,
            'hostname': self.hostname,
            'port': self.port
        }

        with self.assertRaises(ValueError) as cm:
            recmodel_client.execute(args)
        actual_message = str(cm.exception)
        expected_message = 'The AOI extent is too large'
        self.assertIn(expected_message, actual_message)


class RecreationClientRegressionTests(unittest.TestCase):
    """Regression & Unit tests for recmodel_client."""

    def setUp(self):
        """Setup workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_data_different_projection(self):
        """Recreation can validate if data in different projection."""
        from natcap.invest.recreation import recmodel_client

        response_vector_path = os.path.join(SAMPLE_DATA, 'andros_aoi.shp')
        table_path = os.path.join(
            SAMPLE_DATA, 'predictors_wrong_projection.csv')
        msg = recmodel_client._validate_same_projection(
                response_vector_path, table_path)
        self.assertIn('did not match the projection', msg)

    def test_different_tables(self):
        """Recreation can validate if scenario ids different than predictor."""
        from natcap.invest.recreation import recmodel_client

        base_table_path = os.path.join(
            SAMPLE_DATA, 'predictors_all.csv')
        scenario_table_path = os.path.join(
            SAMPLE_DATA, 'predictors.csv')
        msg = recmodel_client._validate_same_ids_and_types(
                base_table_path, scenario_table_path)
        self.assertIn('table pairs unequal', msg)

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
        numpy.testing.assert_allclose(
            predictor_results['0'], 13.0, rtol=0, atol=1e-6)

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
        self.assertEqual(len(predictor_results), 0)

    def test_overlapping_features_in_polygon_predictor(self):
        """Test overlapping predictor features are not double-counted.

        If a polygon predictor contains features that overlap, the overlapping
        area should only be counted once when calculating
        `polygon_area_coverage` or `polygon_percent_coverage`.
        """
        from natcap.invest.recreation import recmodel_client

        response_vector_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        response_polygons_pickle_path = os.path.join(
            self.workspace_dir, 'response.pickle')
        predictor_vector_path = os.path.join(
            self.workspace_dir, 'predictor.geojson')
        predictor_target_path = os.path.join(
            self.workspace_dir, 'predictor.json')

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32610)  # a UTM system

        # A unit square
        response_geom = shapely.geometry.Polygon(
            ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
        pygeoprocessing.shapely_geometry_to_vector(
            [response_geom],
            response_vector_path,
            srs.ExportToWkt(),
            'GEOJSON')

        # Two overlapping polygons, including a unit square
        predictor_geom_list = [
            shapely.geometry.Polygon(
                ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))),
            shapely.geometry.Polygon(
                ((0., 0.), (0., 0.5), (0.5, 0.5), (0.5, 0.), (0., 0.)))]
        pygeoprocessing.shapely_geometry_to_vector(
            predictor_geom_list,
            predictor_vector_path,
            srs.ExportToWkt(),
            'GEOJSON')

        recmodel_client._prepare_response_polygons_lookup(
            response_vector_path, response_polygons_pickle_path)
        recmodel_client._polygon_area(
            'polygon_area_coverage',
            response_polygons_pickle_path,
            predictor_vector_path,
            predictor_target_path)

        with open(predictor_target_path, 'r') as file:
            data = json.load(file)
        actual_value = list(data.values())[0]
        expected_value = 1
        self.assertEqual(actual_value, expected_value)

    def test_compute_and_summarize_regression(self):
        """Recreation regression test for the least-squares linear model."""
        from natcap.invest.recreation import recmodel_client

        data_vector_path = os.path.join(
            self.workspace_dir, 'regression_data.gpkg')
        driver = 'GPKG'
        n_features = 10
        userdays = numpy.linspace(0, 1, n_features)
        avg_pr_UD = userdays / userdays.sum()
        roads = numpy.linspace(0, 100, n_features)
        parks = numpy.linspace(0, 100, n_features)**2
        response_id = 'avg_pr_UD'

        attribute_list = []
        for i in range(n_features):
            attribute_list.append({
                response_id: avg_pr_UD[i],
                'roads': roads[i],
                'parks': parks[i]
            })
        field_map = {
            response_id: ogr.OFTReal,
            'roads': ogr.OFTReal,
            'parks': ogr.OFTReal,
        }

        # The geometries don't matter.
        geom_list = [shapely.geometry.Point(1, -1)] * n_features
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        pygeoprocessing.shapely_geometry_to_vector(
            geom_list,
            data_vector_path,
            srs.ExportToWkt(),
            driver,
            fields=field_map,
            attribute_list=attribute_list,
            ogr_geom_type=ogr.wkbPoint)
        predictor_list = ['roads', 'parks']
        server_version_path = 'server_version.pickle'
        with open(server_version_path, 'ab') as f:
            pickle.dump('version: foo', f)
        target_coefficient_json_path = os.path.join(self.workspace_dir, 'estimates.json')
        target_coefficient_csv_path = os.path.join(self.workspace_dir, 'estimates.csv')
        target_regression_summary_path = os.path.join(self.workspace_dir, 'summary.txt')
        recmodel_client._compute_and_summarize_regression(
            data_vector_path, response_id, predictor_list, server_version_path,
            target_coefficient_json_path, target_coefficient_csv_path,
            target_regression_summary_path)

        coefficient_results = {}
        coefficient_results['estimate'] = [5.953980e-02, -3.056440e-04, -4.388366e+00]
        coefficient_results['stderr'] = [3.769794e-03, 3.629146e-05, 8.094584e-02]
        coefficient_results['t-value'] = [15.793912,  -8.421927, -54.213612]
        summary_results = {
            'SSres': '0.0742',
            'Multiple R-squared': '0.9921',
            'Adjusted R-squared': '0.9898',
            'Residual standard error': '0.1030 on 7 degrees of freedom'
        }

        results = pandas.read_csv(
            target_coefficient_csv_path).to_dict(orient='list')
        for key in coefficient_results:
            numpy.testing.assert_allclose(
                results[key], coefficient_results[key], rtol=1e-05)

        with open(target_regression_summary_path, 'r') as file:
            for line in file.readlines():
                for k, v in summary_results.items():
                    if line.startswith(k):
                        self.assertEqual(line.split(': ')[1].rstrip(), v)

    def test_regression_edge_case(self):
        """Recreation unit test only one non-zero userday observation."""
        from natcap.invest.recreation import recmodel_client

        data_vector_path = os.path.join(
            self.workspace_dir, 'regression_data.gpkg')
        driver = 'GPKG'
        n_features = 10
        userdays = numpy.array([0] * n_features)
        userdays[0] = 1  # one non-zero value
        avg_pr_UD = userdays / userdays.sum()
        roads = numpy.linspace(0, 100, n_features)
        response_id = 'avg_pr_UD'

        attribute_list = []
        for i in range(n_features):
            attribute_list.append({
                response_id: avg_pr_UD[i],
                'roads': roads[i],
            })
        field_map = {
            response_id: ogr.OFTReal,
            'roads': ogr.OFTReal,
        }

        # The geometries don't matter.
        geom_list = [shapely.geometry.Point(1, -1)] * n_features
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        pygeoprocessing.shapely_geometry_to_vector(
            geom_list,
            data_vector_path,
            srs.ExportToWkt(),
            driver,
            fields=field_map,
            attribute_list=attribute_list,
            ogr_geom_type=ogr.wkbPoint)
        predictor_list = ['roads']

        with self.assertRaises(ValueError):
            recmodel_client._build_regression(
                data_vector_path, predictor_list,
                response_id)

    def test_copy_aoi_no_grid(self):
        """Recreation test AOI copy adds poly_id field."""
        from natcap.invest.recreation import recmodel_client

        out_grid_vector_path = os.path.join(
            self.workspace_dir, 'aoi.gpkg')

        recmodel_client._copy_aoi_no_grid(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            out_grid_vector_path)

        vector = gdal.OpenEx(out_grid_vector_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()
        n_features = layer.GetFeatureCount()
        poly_id_list = [feature.GetField(
            recmodel_client.POLYGON_ID_FIELD) for feature in layer]
        self.assertEqual(poly_id_list, list(range(n_features)))
        layer = vector = None

    def test_square_grid(self):
        """Recreation square grid regression test."""
        from natcap.invest.recreation import recmodel_client

        out_grid_vector_path = os.path.join(
            self.workspace_dir, 'square_grid_vector_path.gpkg')

        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'square', 20000.0,
            out_grid_vector_path)

        vector = gdal.OpenEx(out_grid_vector_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()
        n_features = layer.GetFeatureCount()
        poly_id_list = [feature.GetField(
            recmodel_client.POLYGON_ID_FIELD) for feature in layer]
        self.assertEqual(poly_id_list, list(range(n_features)))
        layer = vector = None
        # andros_aoi.shp fits 38 squares at 20000 meters cell size
        self.assertEqual(n_features, 38)

    def test_hex_grid(self):
        """Recreation hex grid regression test."""
        from natcap.invest.recreation import recmodel_client

        out_grid_vector_path = os.path.join(
            self.workspace_dir, 'hex_grid_vector_path.gpkg')

        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'hexagon', 20000.0,
            out_grid_vector_path)

        vector = gdal.OpenEx(out_grid_vector_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()
        n_features = layer.GetFeatureCount()
        layer = vector = None
        # andros_aoi.shp fits 71 hexes at 20000 meters cell size
        self.assertEqual(n_features, 71)

    def test_predictor_id_too_long(self):
        """Recreation can validate predictor ID length."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': True,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'grid_aoi': True,
            'grid_type': 'square',
            'cell_size': 20000,
            'predictor_table_path': os.path.join(
                SAMPLE_DATA, 'predictors_id_too_long.csv'),
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
        }
        msgs = recmodel_client.validate(args)
        self.assertIn('more than 10 characters long', msgs[0][1])

    def test_existing_gridded_aoi_shapefiles(self):
        """Recreation grid test when output files need to be overwritten."""
        from natcap.invest.recreation import recmodel_client

        out_grid_vector_path = os.path.join(
            self.workspace_dir, 'hex_grid_vector_path.gpkg')

        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'hexagon', 20000.0,
            out_grid_vector_path)
        # overwrite output
        recmodel_client._grid_vector(
            os.path.join(SAMPLE_DATA, 'andros_aoi.shp'), 'hexagon', 20000.0,
            out_grid_vector_path)

        vector = gdal.OpenEx(out_grid_vector_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()
        n_features = layer.GetFeatureCount()
        layer = vector = None
        # andros_aoi.shp fits 71 hexes at 20000 meters cell size
        self.assertEqual(n_features, 71)

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
        # exception is raised the test will raise an error, thus
        # differentiating between a failed test and an error.
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
            'compute_regression': False,
            'start_year': recmodel_client.MAX_YEAR,  # note start_year > end_year
            'end_year': recmodel_client.MIN_YEAR,
            'grid_aoi': False,
            'workspace_dir': self.workspace_dir,
        }
        msgs = recmodel_client.validate(args)
        self.assertEqual(
            'Start year must be less than or equal to end year.', msgs[0][1])
        with self.assertRaises(ValueError):
            recmodel_client.execute(args)

    def test_bad_grid_type(self):
        """Recreation ensure that bad grid type raises ValueError."""
        from natcap.invest.recreation import recmodel_client

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'cell_size': 7000.0,
            'compute_regression': False,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'grid_aoi': True,
            'grid_type': 'circle',  # intentionally bad gridtype
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            recmodel_client.execute(args)


class RecreationValidationTests(unittest.TestCase):
    """Tests for the Recreation Model MODEL_SPEC and validation."""

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

        validation_errors = recmodel_client.validate(
            {'compute_regression': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys + ['predictor_table_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_bad_predictor_table_header(self):
        """Recreation Validate: assert messages for bad table headers."""
        from natcap.invest.recreation import recmodel_client
        from natcap.invest import validation

        table_path = os.path.join(self.workspace_dir, 'table.csv')
        with open(table_path, 'w') as file:
            file.write('foo,bar,baz\n')
            file.write('a,b,c\n')

        expected_message = [(
            ['predictor_table_path'],
            validation.MESSAGES['MATCHED_NO_HEADERS'].format(
                header='column', header_name='id'))]
        validation_warnings = recmodel_client.validate({
            'compute_regression': True,
            'predictor_table_path': table_path,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'workspace_dir': self.workspace_dir,
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp')})

        self.assertEqual(validation_warnings, expected_message)

        validation_warnings = recmodel_client.validate({
            'compute_regression': True,
            'predictor_table_path': table_path,
            'scenario_predictor_table_path': table_path,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'workspace_dir': self.workspace_dir,
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp')})
        expected_messages = [
            (['predictor_table_path'],
             validation.MESSAGES['MATCHED_NO_HEADERS'].format(
                header='column', header_name='id')),
            (['scenario_predictor_table_path'],
             validation.MESSAGES['MATCHED_NO_HEADERS'] .format(
                header='column', header_name='id'))]
        self.assertEqual(len(validation_warnings), 2)
        for message in expected_messages:
            self.assertTrue(message in validation_warnings)

    def test_validate_predictor_types_incorrect(self):
        """Recreation Validate: assert error on incorrect type value"""
        from natcap.invest.recreation import recmodel_client

        predictor_id = 'dem90m'
        raster_path = os.path.join(SAMPLE_DATA, 'predictors/dem90m_coarse.tif')
        # include a typo in the type, this should fail
        bad_table_path = os.path.join(self.workspace_dir, 'bad_table.csv')
        with open(bad_table_path, 'w') as file:
            file.write('id,path,type\n')
            file.write(f'{predictor_id},{raster_path},raster?mean\n')

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'andros_aoi.shp'),
            'compute_regression': True,
            'start_year': recmodel_client.MIN_YEAR,
            'end_year': recmodel_client.MAX_YEAR,
            'grid_aoi': False,
            'predictor_table_path': bad_table_path,
            'workspace_dir': self.workspace_dir,
        }
        msgs = recmodel_client.validate(args)
        self.assertIn('The table contains invalid type value(s)', msgs[0][1])


class RecreationProductionServerHealth(unittest.TestCase):
    """Health check for the production server."""

    def test_production_server(self):
        from natcap.invest.recreation import recmodel_client
        import requests

        server_url = requests.get(recmodel_client.SERVER_URL).text.rstrip()
        try:
            proxy = Pyro5.api.Proxy(server_url)
            # _pyroBind() forces the client-server handshake and
            # seems like a good way to check if the remote object is ready
            proxy._pyroBind()
        except Exception as exc:
            self.fail(exc)
        finally:
            proxy._pyroRelease()


def _assert_regression_results_eq(
        workspace_dir, file_list_path, result_vector_path,
        expected_results_path):
    """Test workspace against the expected list of files and results.

    Args:
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
                    numpy.testing.assert_allclose(
                        v, ev, rtol=0, atol=10**-tolerance_places)
                else:
                    # Could happen when a raster predictor is only nodata
                    assert(numpy.isnan(ev))
            feature = None

    finally:
        result_layer = None
        result_vector = None


def _test_same_files(base_list_path, directory_path):
    """Assert expected files are in the `directory_path`.

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
                # skip blank lines
                continue
            if not os.path.isfile(full_path):
                missing_files.append(full_path)
    if len(missing_files) > 0:
        raise AssertionError(
            "The following files were expected but not found: " +
            '\n'.join(missing_files))


def sum_vector_columns(vector_path, field_list):
    """Calculate the sum of values in each field of a gdal vector.

    Args:
        vector_path (str): path to a gdal vector.
        field_list (list): list of field names to sum.

    Returns:
        dict: mapping field name to sum of values.

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    result = {}
    for field in field_list:
        result[field] = sum([
            feature.GetField(field)
            for feature in layer])
    layer = vector = None
    return result
