"""Module for testing the natcap.invest.utils module.."""
import codecs
import glob
import logging
import logging.handlers
import os
import platform
import queue
import re
import shutil
import tempfile
import textwrap
import threading
import unittest
import unittest.mock
import warnings

import numpy
import numpy.testing
import pandas as pd
import pygeoprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from shapely.geometry import Point
from shapely.geometry import Polygon

gdal.UseExceptions()


class TimeFormattingTests(unittest.TestCase):
    """Test Time Formatting."""

    def test_format_time_hours(self):
        """Test format time hours."""
        from natcap.invest.utils import _format_time

        seconds = 3667
        self.assertEqual(_format_time(seconds), '1h 1m 7s')

    def test_format_time_minutes(self):
        """Test format time minutes."""
        from natcap.invest.utils import _format_time

        seconds = 67
        self.assertEqual(_format_time(seconds), '1m 7s')

    def test_format_time_seconds(self):
        """Test format time seconds."""
        from natcap.invest.utils import _format_time

        seconds = 7
        self.assertEqual(_format_time(seconds), '7s')


class LogToFileTests(unittest.TestCase):
    """Test Log To File."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary workspace."""
        shutil.rmtree(self.workspace)

    def test_log_to_file_all_threads(self):
        """Utils: Verify that we can capture messages from all threads."""
        from natcap.invest.utils import log_to_file

        logfile = os.path.join(self.workspace, 'logfile.txt')

        def _log_from_other_thread():
            """Log from other thead."""
            thread_logger = logging.getLogger()
            thread_logger.info('this is from a thread')

        local_logger = logging.getLogger()

        # create the file before we log to it, so we know a warning should
        # be logged.
        with open(logfile, 'w') as new_file:
            new_file.write(' ')

        with log_to_file(logfile) as handler:
            thread = threading.Thread(target=_log_from_other_thread)
            thread.start()
            local_logger.info('this should be logged')
            local_logger.info('this should also be logged')

            thread.join()
            handler.flush()

        with open(logfile) as opened_logfile:
            messages = [msg for msg in opened_logfile.read().split('\n')
                        if msg if msg]
        self.assertEqual(len(messages), 3)

    def test_log_to_file_from_thread(self):
        """Utils: Verify that we can filter from a threading.Thread."""
        from natcap.invest.utils import log_to_file

        logfile = os.path.join(self.workspace, 'logfile.txt')

        def _log_from_other_thread():
            """Log from other thread."""
            thread_logger = logging.getLogger()
            thread_logger.info('this should not be logged')
            thread_logger.info('neither should this message')

        local_logger = logging.getLogger()

        thread = threading.Thread(target=_log_from_other_thread)

        with log_to_file(logfile, exclude_threads=[thread.name]) as handler:
            thread.start()
            local_logger.info('this should be logged')

            thread.join()
            handler.flush()

        with open(logfile) as opened_logfile:
            messages = [msg for msg in opened_logfile.read().split('\n')
                        if msg if msg]
        self.assertEqual(len(messages), 1)


class ThreadFilterTests(unittest.TestCase):
    """Test Thread Filter."""

    def test_thread_filter_same_thread(self):
        """Test threat filter same thread."""
        from natcap.invest.utils import ThreadFilter

        # name, level, pathname, lineno, msg, args, exc_info, func=None
        record = logging.LogRecord(
            name='foo',
            level=logging.INFO,
            pathname=__file__,
            lineno=500,
            msg='some logging message',
            args=(),
            exc_info=None,
            func='test_thread_filter_same_thread')
        filterer = ThreadFilter(threading.current_thread().name)

        # The record comes from the same thread.
        self.assertEqual(filterer.filter(record), False)

    def test_thread_filter_different_thread(self):
        """Test thread filter different thread."""
        from natcap.invest.utils import ThreadFilter

        # name, level, pathname, lineno, msg, args, exc_info, func=None
        record = logging.LogRecord(
            name='foo',
            level=logging.INFO,
            pathname=__file__,
            lineno=500,
            msg='some logging message',
            args=(),
            exc_info=None,
            func='test_thread_filter_same_thread')
        filterer = ThreadFilter('Thread-nonexistent')

        # The record comes from the same thread.
        self.assertEqual(filterer.filter(record), True)


class GDALWarningsLoggingTests(unittest.TestCase):
    """Test GDAL Warnings Logging."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary workspace."""
        shutil.rmtree(self.workspace)

    def test_log_warnings(self):
        """utils: test that we can capture GDAL warnings to logging."""
        from natcap.invest import utils

        logfile = os.path.join(self.workspace, 'logfile.txt')

        invalid_polygon = ogr.CreateGeometryFromWkt(
            'POLYGON ((-20 -20, -16 -20, -20 -16, -16 -16, -20 -20))')

        # This produces a GDAL warning that does not raise an
        # exception with UseExceptions(). Without capture_gdal_logging,
        # it will be printed directly to stderr
        invalid_polygon.IsValid()

        with utils.log_to_file(logfile) as handler:
            with utils.capture_gdal_logging():
                # warning should be captured.
                invalid_polygon.IsValid()
            handler.flush()

        # warning should go to stderr
        invalid_polygon.IsValid()

        with open(logfile) as opened_logfile:
            messages = [msg for msg in opened_logfile.read().split('\n')
                        if msg if msg]

        self.assertEqual(len(messages), 1)

    def test_log_gdal_errors_bad_n_args(self):
        """utils: test error capture when number of args != 3."""
        from natcap.invest import utils

        log_queue = queue.Queue()
        log_queue_handler = logging.handlers.QueueHandler(log_queue)
        utils.LOGGER.addHandler(log_queue_handler)

        try:
            # 1 parameter, expected 3
            utils._log_gdal_errors('foo')
        finally:
            utils.LOGGER.removeHandler(log_queue_handler)

        record = log_queue.get()
        self.assertEqual(record.name, 'natcap.invest.utils')
        self.assertEqual(record.levelno, logging.ERROR)
        self.assertIn(
            '_log_gdal_errors was called with an incorrect number',
            record.msg)

    def test_log_gdal_errors_missing_param(self):
        """utils: test error when specific parameters missing."""
        from natcap.invest import utils

        log_queue = queue.Queue()
        log_queue_handler = logging.handlers.QueueHandler(log_queue)
        utils.LOGGER.addHandler(log_queue_handler)

        try:
            # Missing third parameter, "err_msg"
            utils._log_gdal_errors(
                gdal.CE_Failure, 123,
                bad_param='bad param')  # param obviously bad
        finally:
            utils.LOGGER.removeHandler(log_queue_handler)

        record = log_queue.get()
        self.assertEqual(record.name, 'natcap.invest.utils')
        self.assertEqual(record.levelno, logging.ERROR)
        self.assertIn(
            "_log_gdal_errors called without the argument 'err_msg'",
            record.msg)


class PrepareWorkspaceTests(unittest.TestCase):
    """Test Prepare Workspace."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary workspace."""
        shutil.rmtree(self.workspace)

    def test_prepare_workspace(self):
        """utils: test that prepare_workspace does what is expected."""
        from natcap.invest import utils

        workspace = os.path.join(self.workspace, 'foo')
        with warnings.catch_warnings():
            # restore the warnings filter to default, overriding any
            # global pytest filter. this preserves the warnings so that
            # they may be redirected to the log.
            warnings.simplefilter('default')
            with utils.prepare_workspace(workspace,
                                         'some_model'):
                warnings.warn('deprecated', UserWarning)
                invalid_polygon = ogr.CreateGeometryFromWkt(
                    'POLYGON ((-20 -20, -16 -20, -20 -16, -16 -16, -20 -20))')
                # This produces a GDAL warning that does not raise an
                # exception with UseExceptions()
                invalid_polygon.IsValid()

        self.assertTrue(os.path.exists(workspace))
        logfile_glob = glob.glob(os.path.join(workspace, '*.txt'))
        self.assertEqual(len(logfile_glob), 1)
        self.assertTrue(
            os.path.basename(logfile_glob[0]).startswith('InVEST-some_model'))
        with open(logfile_glob[0]) as logfile:
            logfile_text = logfile.read()
            # all the following strings should be in the logfile.
            self.assertTrue(  # gdal logging captured
                'Self-intersection at or near point -18 -18' in logfile_text)
            self.assertEqual(len(re.findall('WARNING', logfile_text)), 2)
            self.assertTrue('Elapsed time:' in logfile_text)


class ReadCSVToDataframeTests(unittest.TestCase):
    """Tests for natcap.invest.utils.read_csv_to_dataframe."""

    def setUp(self):
        """Make temporary directory for workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_read_csv_to_dataframe(self):
        """utils: read csv with no row or column specs provided"""
        from natcap.invest import utils
        csv_text = ("lucode,desc,val1,val2\n"
                    "1,corn,0.5,2\n"
                    "2,bread,1,4,\n"
                    "3,beans,0.5,4\n"
                    "4,butter,9,1")
        table_path = os.path.join(self.workspace_dir, 'table.csv')
        with open(table_path, 'w') as table_file:
            table_file.write(csv_text)

        df = utils.read_csv_to_dataframe(table_path)
        self.assertEqual(list(df.columns), ['lucode', 'desc', 'val1', 'val2'])
        self.assertEqual(df['lucode'][0], 1)
        self.assertEqual(df['desc'][1], 'bread')
        self.assertEqual(df['val1'][2], 0.5)
        self.assertEqual(df['val2'][3], 1)

    def test_csv_utf8_encoding(self):
        """utils: test that CSV read correctly with UTF-8 encoding."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace_dir, 'csv.csv')
        with open(csv_file, 'w', encoding='utf-8') as file_obj:
            file_obj.write(textwrap.dedent(
                """\
                header1,HEADER2,header3
                1,2,bar
                4,5,FOO
                """
            ))
        lookup_dict = utils.read_csv_to_dataframe(
            csv_file).to_dict(orient='index')
        self.assertEqual(lookup_dict[1]['header2'], 5)
        self.assertEqual(lookup_dict[1]['header3'], 'FOO')

    def test_utf8_bom_encoding(self):
        """utils: test that CSV read correctly with UTF-8 BOM encoding."""
        from natcap.invest import utils
        csv_file = os.path.join(self.workspace_dir, 'csv.csv')
        # writing with utf-8-sig will prepend the BOM
        with open(csv_file, 'w', encoding='utf-8-sig') as file_obj:
            file_obj.write(textwrap.dedent(
                """\
                header1,header2,header3
                1,2,bar
                4,5,FOO
                """
            ))
        # confirm that the file has the BOM prefix
        with open(csv_file, 'rb') as file_obj:
            self.assertTrue(file_obj.read().startswith(codecs.BOM_UTF8))
        df = utils.read_csv_to_dataframe(csv_file)
        # assert the BOM prefix was correctly parsed and skipped
        self.assertEqual(df.columns[0], 'header1')
        self.assertEqual(df['header2'][1], 5)

    def test_csv_latin_1_encoding(self):
        """utils: can read Latin-1 encoded CSV if it uses only ASCII chars."""
        from natcap.invest import utils
        csv_file = os.path.join(self.workspace_dir, 'csv.csv')
        with codecs.open(csv_file, 'w', encoding='iso-8859-1') as file_obj:
            file_obj.write(textwrap.dedent(
                """\
                header 1,HEADER 2,header 3
                1,2,bar1
                4,5,FOO
                """
            ))
        df = utils.read_csv_to_dataframe(csv_file)
        self.assertEqual(df['header 2'][1], 5)
        self.assertEqual(df['header 3'][1], 'FOO')
        self.assertEqual(df['header 1'][0], 1)

    def test_csv_error_non_utf8_character(self):
        """utils: test that error is raised on non-UTF8 character."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace_dir, 'csv.csv')
        with codecs.open(csv_file, 'w', encoding='iso-8859-1') as file_obj:
            file_obj.write(textwrap.dedent(
                """\
                header 1,HEADER 2,header 3
                1,2,bar1
                4,5,FÖÖ
                """
            ))
        with self.assertRaises(ValueError):
            utils.read_csv_to_dataframe(csv_file)

    def test_override_default_encoding(self):
        """utils: test that you can override the default encoding kwarg"""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace_dir, 'csv.csv')

        # encode with ISO Cyrillic, include a non-ASCII character
        with open(csv_file, 'w', encoding='iso8859_5') as file_obj:
            file_obj.write(textwrap.dedent(
                """\
                header,
                fЮЮ,
                bar
                """
            ))
        df = utils.read_csv_to_dataframe(csv_file, encoding='iso8859_5')
        # with the encoding specified, special characters should work
        self.assertEqual(df['header'][0], 'fЮЮ')
        self.assertEqual(df['header'][1], 'bar')

    def test_csv_dialect_detection_semicolon_delimited(self):
        """utils: test that we can parse semicolon-delimited CSVs."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace_dir, 'csv.csv')
        with open(csv_file, 'w') as file_obj:
            file_obj.write(textwrap.dedent(
                """\
                header1;HEADER2;header3;
                1;2;3;
                4;FOO;bar;
                """
            ))

        df = utils.read_csv_to_dataframe(csv_file)
        self.assertEqual(df['header2'][1], 'FOO')
        self.assertEqual(df['header3'][1], 'bar')
        self.assertEqual(df['header1'][0], 1)




class CreateCoordinateTransformationTests(unittest.TestCase):
    """Tests for natcap.invest.utils.create_coordinate_transformer."""

    def test_latlon_to_latlon_transformer(self):
        """Utils: test transformer for lat/lon to lat/lon."""
        from natcap.invest import utils

        # Willamette valley in lat/lon for reference
        lon = -124.525
        lat = 44.525

        base_srs = osr.SpatialReference()
        base_srs.ImportFromEPSG(4326)  # WSG84 EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)

        transformer = utils.create_coordinate_transformer(base_srs, target_srs)
        actual_x, actual_y, _ = transformer.TransformPoint(lon, lat)

        expected_x = -124.525
        expected_y = 44.525

        self.assertAlmostEqual(expected_x, actual_x, 5)
        self.assertAlmostEqual(expected_y, actual_y, 5)

    def test_latlon_to_projected_transformer(self):
        """Utils: test transformer for lat/lon to projected."""
        from natcap.invest import utils

        # Willamette valley in lat/lon for reference
        lon = -124.525
        lat = 44.525

        base_srs = osr.SpatialReference()
        base_srs.ImportFromEPSG(4326)  # WSG84 EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(26910)  # UTM10N EPSG

        transformer = utils.create_coordinate_transformer(base_srs, target_srs)
        actual_x, actual_y, _ = transformer.TransformPoint(lon, lat)

        expected_x = 378816.2531852932
        expected_y = 4931317.807472325

        self.assertAlmostEqual(expected_x, actual_x, 5)
        self.assertAlmostEqual(expected_y, actual_y, 5)

    def test_projected_to_latlon_transformer(self):
        """Utils: test transformer for projected to lat/lon."""
        from natcap.invest import utils

        # Willamette valley in lat/lon for reference
        known_x = 378816.2531852932
        known_y = 4931317.807472325

        base_srs = osr.SpatialReference()
        base_srs.ImportFromEPSG(26910)  # UTM10N EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)  # WSG84 EPSG

        transformer = utils.create_coordinate_transformer(base_srs, target_srs)
        actual_x, actual_y, _ = transformer.TransformPoint(known_x, known_y)

        expected_x = -124.52500000000002
        expected_y = 44.525

        self.assertAlmostEqual(expected_x, actual_x, places=3)
        self.assertAlmostEqual(expected_y, actual_y, places=3)

    def test_projected_to_projected_transformer(self):
        """Utils: test transformer for projected to projected."""
        from natcap.invest import utils

        # Willamette valley in lat/lon for reference
        known_x = 378816.2531852932
        known_y = 4931317.807472325

        base_srs = osr.SpatialReference()
        base_srs.ImportFromEPSG(26910)  # UTM10N EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(26910)  # UTM10N EPSG

        transformer = utils.create_coordinate_transformer(base_srs, target_srs)
        actual_x, actual_y, _ = transformer.TransformPoint(known_x, known_y)

        expected_x = 378816.2531852932
        expected_y = 4931317.807472325

        self.assertAlmostEqual(expected_x, actual_x, 5)
        self.assertAlmostEqual(expected_y, actual_y, 5)


class AssertVectorsEqualTests(unittest.TestCase):
    """Tests for natcap.invest.utils._assert_vectors_equal."""

    def setUp(self):
        """Setup workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_identical_point_vectors(self):
        """Utils: test identical point vectors pass."""
        from natcap.invest import utils

        # Setup parameters to create point shapefile
        fields = {'id': ogr.OFTReal}
        attrs = [{'id': 1}, {'id': 2}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        geometries = [Point(pos_x + 50, pos_y - 50),
                      Point(pos_x + 50, pos_y - 150)]
        shape_path = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPoint)

        shape_copy_path = os.path.join(
            self.workspace_dir, 'point_shape_copy.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_copy_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPoint)

        utils._assert_vectors_equal(shape_path, shape_copy_path)

    def test_identical_polygon_vectors(self):
        """Utils: test identical polygon vectors pass."""
        from natcap.invest import utils

        # Setup parameters to create point shapefile
        fields = {'id': ogr.OFTReal}
        attrs = [{'id': 1}, {'id': 2}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        poly_geoms = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y - 100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x + 100, pos_y - 150),
                       (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200),
                       (pos_x, pos_y - 150)]}

        geometries = [
            Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]

        shape_path = os.path.join(self.workspace_dir, 'poly_shape.shp')
        # Create polygon shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPolygon)

        shape_copy_path = os.path.join(
            self.workspace_dir, 'poly_shape_copy.shp')
        # Create polygon shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_copy_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPolygon)

        utils._assert_vectors_equal(shape_path, shape_copy_path)

    def test_identical_polygon_vectors_unorded_geometry(self):
        """Utils: test identical polygon vectors w/ diff geometry order."""
        from natcap.invest import utils

        # Setup parameters to create point shapefile
        fields = {'id': ogr.OFTReal}
        attrs = [{'id': 1}, {'id': 2}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        poly_geoms = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y - 100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x + 100, pos_y - 150),
                       (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200),
                       (pos_x, pos_y - 150)]}

        poly_geoms_unordered = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y - 100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x, pos_y - 200),
                       (pos_x + 100, pos_y - 200), (pos_x + 100, pos_y - 150),
                       (pos_x, pos_y - 150)]}

        geometries = [
            Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]

        geometries_copy = [
            Polygon(poly_geoms_unordered['poly_1']),
            Polygon(poly_geoms_unordered['poly_2'])]

        shape_path = os.path.join(self.workspace_dir, 'poly_shape.shp')
        # Create polygon shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPolygon)

        shape_copy_path = os.path.join(
            self.workspace_dir, 'poly_shape_copy.shp')
        # Create polygon shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries_copy, shape_copy_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPolygon)

        utils._assert_vectors_equal(shape_path, shape_copy_path)

    def test_different_field_value(self):
        """Utils: test vectors w/ different field value fails."""
        from natcap.invest import utils

        # Setup parameters to create point shapefile
        fields = {'id': ogr.OFTReal, 'foo': ogr.OFTReal}
        attrs = [{'id': 1, 'foo': 2.3456}, {'id': 2, 'foo': 5.6789}]
        attrs_copy = [{'id': 1, 'foo': 2.3467}, {'id': 2, 'foo': 5.6789}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        geometries = [Point(pos_x + 50, pos_y - 50),
                      Point(pos_x + 50, pos_y - 150)]
        shape_path = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPoint)

        shape_copy_path = os.path.join(
            self.workspace_dir, 'point_shape_copy.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_copy_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs_copy,
            ogr_geom_type=ogr.wkbPoint)

        with self.assertRaises(AssertionError) as cm:
            utils._assert_vectors_equal(shape_path, shape_copy_path)

        self.assertTrue(
            "Vector field values are not equal" in str(cm.exception))

    def test_different_field_names(self):
        """Utils: test vectors w/ different field names fails."""
        from natcap.invest import utils

        # Setup parameters to create point shapefile
        fields = {'id': ogr.OFTReal, 'foo': ogr.OFTReal}
        fields_copy = {'id': ogr.OFTReal, 'foobar': ogr.OFTReal}
        attrs = [{'id': 1, 'foo': 2.3456}, {'id': 2, 'foo': 5.6789}]
        attrs_copy = [{'id': 1, 'foobar': 2.3456}, {'id': 2, 'foobar': 5.6789}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        geometries = [Point(pos_x + 50, pos_y - 50),
                      Point(pos_x + 50, pos_y - 150)]
        shape_path = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPoint)

        shape_copy_path = os.path.join(
            self.workspace_dir, 'point_shape_copy.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_copy_path, projection_wkt,
            'ESRI Shapefile', fields=fields_copy, attribute_list=attrs_copy,
            ogr_geom_type=ogr.wkbPoint)

        with self.assertRaises(AssertionError) as cm:
            utils._assert_vectors_equal(shape_path, shape_copy_path)

        self.assertTrue(
            "Vector field names are not the same" in str(cm.exception))

    def test_different_feature_count(self):
        """Utils: test vectors w/ different feature count fails."""
        from natcap.invest import utils

        # Setup parameters to create point shapefile
        fields = {'id': ogr.OFTReal, 'foo': ogr.OFTReal}
        attrs = [{'id': 1, 'foo': 2.3456}, {'id': 2, 'foo': 5.6789}]
        attrs_copy = [
            {'id': 1, 'foo': 2.3456}, {'id': 2, 'foo': 5.6789},
            {'id': 3, 'foo': 5}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        geometries = [Point(pos_x + 50, pos_y - 50),
                      Point(pos_x + 50, pos_y - 150)]

        geometries_copy = [Point(pos_x + 50, pos_y - 50),
                           Point(pos_x + 50, pos_y - 150),
                           Point(pos_x + 55, pos_y - 55)]
        shape_path = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPoint)

        shape_copy_path = os.path.join(
            self.workspace_dir, 'point_shape_copy.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries_copy, shape_copy_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs_copy,
            ogr_geom_type=ogr.wkbPoint)

        with self.assertRaises(AssertionError) as cm:
            utils._assert_vectors_equal(shape_path, shape_copy_path)

        self.assertTrue(
            "Vector feature counts are not the same" in str(cm.exception))

    def test_different_projections(self):
        """Utils: test vectors w/ different projections fails."""
        from natcap.invest import utils

        # Setup parameters to create point shapefile
        fields = {'id': ogr.OFTReal, 'foo': ogr.OFTReal}
        attrs = [{'id': 1, 'foo': 2.3456}, {'id': 2, 'foo': 5.6789}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        geometries = [Point(pos_x + 50, pos_y - 50),
                      Point(pos_x + 50, pos_y - 150)]

        srs_copy = osr.SpatialReference()
        srs_copy.ImportFromEPSG(26910)  # UTM Zone 10N
        projection_wkt_copy = srs_copy.ExportToWkt()

        origin_copy = (1180000, 690000)
        pos_x_copy = origin_copy[0]
        pos_y_copy = origin_copy[1]

        geometries_copy = [Point(pos_x_copy + 50, pos_y_copy - 50),
                           Point(pos_x_copy + 50, pos_y_copy - 150)]

        shape_path = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPoint)

        shape_copy_path = os.path.join(
            self.workspace_dir, 'point_shape_copy.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries_copy, shape_copy_path, projection_wkt_copy,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPoint)

        with self.assertRaises(AssertionError) as cm:
            utils._assert_vectors_equal(shape_path, shape_copy_path)

        self.assertTrue(
            "Vector projections are not the same" in str(cm.exception))

    def test_different_geometry_fails(self):
        """Utils: test vectors w/ diff geometries fail."""
        from natcap.invest import utils

        # Setup parameters to create point shapefile
        fields = {'id': ogr.OFTReal}
        attrs = [{'id': 1}, {'id': 2}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        poly_geoms = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y - 100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x + 100, pos_y - 150),
                       (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200),
                       (pos_x, pos_y - 150)]}

        poly_geoms_diff = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y - 100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x, pos_y - 201),
                       (pos_x + 100, pos_y - 200), (pos_x + 100, pos_y - 150),
                       (pos_x, pos_y - 150)]}

        geometries = [
            Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]

        geometries_diff = [
            Polygon(poly_geoms_diff['poly_1']),
            Polygon(poly_geoms_diff['poly_2'])]

        shape_path = os.path.join(self.workspace_dir, 'poly_shape.shp')
        # Create polygon shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, shape_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPolygon)

        shape_diff_path = os.path.join(
            self.workspace_dir, 'poly_shape_diff.shp')
        # Create polygon shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries_diff, shape_diff_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPolygon)

        with self.assertRaises(AssertionError) as cm:
            utils._assert_vectors_equal(shape_path, shape_diff_path)

        self.assertTrue("Vector geometry assertion fail." in str(cm.exception))


class ReclassifyRasterOpTests(unittest.TestCase):
    """Tests for natcap.invest.utils.reclassify_raster."""

    def setUp(self):
        """Setup workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_exception_raised_with_details(self):
        """Utils: test message w/ details is raised on missing value."""
        from natcap.invest import utils

        srs_copy = osr.SpatialReference()
        srs_copy.ImportFromEPSG(26910)  # UTM Zone 10N
        projection_wkt = srs_copy.ExportToWkt()
        origin = (1180000, 690000)
        raster_path = os.path.join(self.workspace_dir, 'tmp_raster.tif')

        array = numpy.array([[1,1,1], [2,2,2], [3,3,3]], dtype=numpy.int32)

        pygeoprocessing.numpy_array_to_raster(
            array, -1, (1, -1), origin, projection_wkt, raster_path)

        value_map = {1: 10, 2: 20}
        target_raster_path = os.path.join(
            self.workspace_dir, 'tmp_raster_out.tif')

        message_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Biophysical'}

        with self.assertRaises(ValueError) as context:
            utils.reclassify_raster(
                (raster_path, 1), value_map, target_raster_path,
                gdal.GDT_Int32, -1, error_details=message_details)
        expected_message = (
            "Values in the LULC raster were found that are"
            " not represented under the 'lucode' column"
            " of the Biophysical table. The missing values found in"
            " the LULC raster but not the table are: [3].")
        self.assertTrue(
            expected_message in str(context.exception), str(context.exception))


class ExpandPathTests(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    @unittest.skipIf(platform.system() == 'Windows',
                     "Function behavior differs across systems.")
    def test_os_path_normalization_linux(self):
        """Utils: test path separator conversion Win to Linux."""
        from natcap.invest import utils

        # Assumption: a path was created on Windows and is now being loaded on
        # a Mac or Linux computer.
        rel_path = "foo\\bar.shp"
        relative_to = os.path.join(self.workspace_dir, 'test.csv')
        expected_path = os.path.join(self.workspace_dir, "foo/bar.shp")
        path = utils.expand_path(rel_path, relative_to)
        self.assertEqual(path, expected_path)

    @unittest.skipIf(platform.system() != 'Windows',
                     "Function behavior differs across systems.")
    def test_os_path_normalization_windows(self):
        """Utils: test path separator conversion Mac/Linux to Windows."""
        from natcap.invest import utils

        # Assumption: a path was created on mac/linux and is now being loaded
        # on a Windows computer.
        rel_path = "foo/bar.shp"
        relative_to = os.path.join(self.workspace_dir, 'test.csv')
        expected_path = os.path.join(self.workspace_dir, "foo\\bar.shp")
        path = utils.expand_path(rel_path, relative_to)
        self.assertEqual(path, expected_path)

    def test_falsey(self):
        """Utils: test return None when falsey."""
        from natcap.invest import utils

        for value in ('', None, False, 0):
            self.assertEqual(
                None, utils.expand_path(value, self.workspace_dir))

class _GDALPathTests(unittest.TestCase):

    def test_local_path(self):
        from natcap.invest import utils
        gdal_path = utils._GDALPath.from_uri('foo/bar.tif')
        self.assertTrue(gdal_path.is_local)
        self.assertIsNone(gdal_path.scheme)

    @unittest.skipIf(platform.system() != 'Windows',
                     'Drive prefixes only apply on Windows')
    def test_windows_drive_path(self):
        from natcap.invest import utils
        gdal_path = utils._GDALPath.from_uri(r'C:\foo\bar.tif')
        self.assertTrue(gdal_path.is_local)
        self.assertIsNone(gdal_path.scheme)

    def test_https_path(self):
        from natcap.invest import utils
        gdal_path = utils._GDALPath.from_uri('https://example.com/foo/bar.tif')
        self.assertTrue(gdal_path.is_remote)
        self.assertEqual(gdal_path.scheme, 'https')
        self.assertEqual(gdal_path.to_normalized_path(),
                         '/vsicurl/https://example.com/foo/bar.tif')

    def test_https_zip_path(self):
        from natcap.invest import utils
        gdal_path = utils._GDALPath.from_uri('zip+https://example.com/foo.zip/foo/bar.tif')
        self.assertTrue(gdal_path.is_remote)
        self.assertEqual(gdal_path.scheme, 'zip+https')
        self.assertEqual(gdal_path.to_normalized_path(),
                         '/vsizip/vsicurl/https://example.com/foo.zip/foo/bar.tif')


class FormatArgsTest(unittest.TestCase):
    """Args format tests."""
    def test_print_args(self):
        """Datastacks: verify that we format args correctly."""
        from natcap.invest import __version__
        from natcap.invest.utils import format_args_dict

        args_dict = {
            'some_arg': [1, 2, 3, 4],
            'foo': 'bar',
        }

        args_string = format_args_dict(args_dict, 'test_model')
        expected_string = str(
            'Arguments for InVEST test_model %s:\n'
            'foo      bar\n'
            'some_arg [1, 2, 3, 4]\n') % __version__
        self.assertEqual(args_string, expected_string)
