"""Module for Testing DelineateIt."""
import unittest
import tempfile
import shutil
import os

import shapely.wkt
import shapely.wkb
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

    def test_delineateit_willamette(self):
        """DelineateIt: regression testing full run."""
        from natcap.invest import delineateit

        args = {
            'dem_path': os.path.join(REGRESSION_DATA, 'input', 'dem.tif'),
            'outlet_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'outlets.shp'),
            'workspace_dir': self.workspace_dir,
            'snap_points': True,
            'snap_distance': '20',
            'flow_threshold': '500',
            'results_suffix': 'w',
            'n_workers': -1,
        }
        delineateit.execute(args)

        vector = gdal.OpenEx(os.path.join(args['workspace_dir'],
                                          'watersheds_w.gpkg'), gdal.OF_VECTOR)
        layer = vector.GetLayer('watersheds')
        self.assertEqual(layer.GetFeatureCount(), 3)

        expected_areas_by_id = {
            1: 143631000.0,
            2: 474300.0,
            3: 3247200.0,
        }
        areas_by_id = {}
        for feature in layer:
            geom = feature.GetGeometryRef()
            areas_by_id[feature.GetField('id')] = geom.Area()

        for id_key, expected_area in expected_areas_by_id.items():
            self.assertAlmostEqual(expected_area, areas_by_id[id_key],
                                   delta=1e-4)

    def test_delineateit_validate(self):
        pass

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
                                          'source_points.geojson')
        source_points = [
            Point(-1, -1),  # off the edge of the stream raster.
            Point(3, -5),
            Point(7, -9),
            Point(13, -5)]
        pygeoprocessing.testing.create_vector_on_disk(
            source_points, wkt,
            fields={'foo': 'int',
                    'bar': 'string'},
            attributes=[
                {'foo': 0, 'bar': 0.1},
                {'foo': 1, 'bar': 1.1},
                {'foo': 2, 'bar': 2.1},
                {'foo': 3, 'bar': 3.1}],
            filename=source_points_path)

        snap_distance = 10  # large enough to get multiple streams per point.
        snapped_points_path = os.path.join(self.workspace_dir,
                                           'snapped_points.gpkg')
        delineateit.snap_points_to_nearest_stream(
            source_points_path, (stream_raster_path, 1),
            snap_distance, snapped_points_path)

        snapped_points_vector = gdal.OpenEx(snapped_points_path,
                                            gdal.OF_VECTOR)
        snapped_points_layer = snapped_points_vector.GetLayer()
        self.assertEqual(3, snapped_points_layer.GetFeatureCount())

        expected_geometries_and_fields = [
            (Point(5, -5), {'foo': 1, 'bar': '1.1'}),
            (Point(5, -9), {'foo': 2, 'bar': '2.1'}),
            (Point(13, -11), {'foo': 3, 'bar': '3.1'}),
        ]
        for feature, (expected_geom, expected_fields) in zip(
                snapped_points_layer, expected_geometries_and_fields):
            shapely_feature = shapely.wkb.loads(
                feature.GetGeometryRef().ExportToWkb())

            self.assertTrue(shapely_feature.equals(expected_geom))
            self.assertEqual(expected_fields, feature.items())

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
