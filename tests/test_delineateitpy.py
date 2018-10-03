"""Module for Testing DelineateIt."""
import unittest
import tempfile
import shutil
import os

from shapely.geometry import Point
import numpy
from osgeo import osr
from osgeo import gdal
import pygeoprocessing.testing


REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'delineateit')


class DelineateItTests(unittest.TestCase):
    """Tests for RouteDEM."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_routedem_multi_threshold(self):
        """DelineateIt: regression testing full run."""
        from natcap.invest import delineateit

        args = {
            'dem_uri': os.path.join(REGRESSION_DATA, 'input', 'dem.tif'),
            'flow_threshold': '500',
            'outlet_shapefile_uri': os.path.join(
                REGRESSION_DATA, 'input', 'outlets.shp'),
            'snap_distance': '20',
            'workspace_dir': self.workspace_dir,
        }
        delineateit.execute(args)

        DelineateItTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_vectors_equal(
            os.path.join(REGRESSION_DATA, 'watersheds.shp'),
            os.path.join(self.workspace_dir, 'watersheds.shp'), 1e-6)

    def test_point_snapping(self):
        from natcap.invest import delineateit

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84/UTM zone 31s
        wkt = srs.ExportToWkt()

        # need stream layer, points
        stream_matrix = numpy.array(
            [[0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]], dtype=numpy.int8)
        stream_raster_path = os.path.join(self.workspace_dir, 'streams.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [stream_matrix],
            origin=(2, -2),
            pixel_size=(2, -2),
            projection_wkt=wkt,
            nodata=255,  # byte datatype
            filename=stream_raster_path)

        source_points_path = os.path.join(self.workspace_dir,
                                          'source_points.shp')
        source_points = [
            Point(-1, -1),  # off the edge of the stream raster.
            Point(3, -5),
            Point(7, -9),
            Point(13, -5)]
        pygeoprocessing.testing.create_vector_on_disk(
            source_points, wkt,
            fields={'foo': 'int',
                    'bar': 'real'},
            attributes=[
                {'foo': 0, 'bar': 0.1},
                {'foo': 1, 'bar': 1.1},
                {'foo': 2, 'bar': 2.1},
                {'foo': 3, 'bar': 3.1}],
        filename=source_points_path)

        snap_distance = 10  # large enough to get multiple streams per point.
        snapped_points_path = os.path.join(self.workspace_dir,
                                           'snapped_points.shp')
        delineateit.snap_points_to_nearest_stream(
            source_points_path, (stream_raster_path, 1),
            snap_distance, snapped_points_path)

        snapped_points_vector = gdal.OpenEx(snapped_points_path,
                                            gdal.OF_VECTOR)
        snapped_points_layer = snapped_points_vector.GetLayer()
        self.assertEqual(3, snapped_points_layer.GetFeatureCount())

        # need to assert the locations of output points, that fields are
        # copied.

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
