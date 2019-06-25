"""Module for Testing DelineateIt."""
import unittest
import tempfile
import shutil
import os

import shapely.wkt
import shapely.wkb
from shapely.geometry import Point, box
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
            'n_workers': None,  # Trigger error and default to -1
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
        from natcap.invest import delineateit
        missing_keys = {}
        with self.assertRaises(KeyError) as cm:
            delineateit.validate(missing_keys)
        self.assertTrue('keys were expected' in str(cm.exception))

        missing_values_args = {
            'workspace_dir': '',
            'dem_path': None,
            'outlet_vector_path': '',
            'snap_points': False,
        }
        validation_warnings = delineateit.validate(missing_values_args)
        self.assertEqual(len(validation_warnings), 2)
        self.assertEqual(len(validation_warnings[0][0]), 3)
        self.assertTrue('parameter has no value' in validation_warnings[0][1])

        file_not_found_args = {
            'workspace_dir': os.path.join(self.workspace_dir),
            'dem_path': os.path.join(self.workspace_dir, 'dem-not-here.tif'),
            'outlet_vector_path': os.path.join(self.workspace_dir,
                                               'outlets-not-here.shp'),
            'snap_points': False,
        }
        validation_warnings = delineateit.validate(file_not_found_args)
        self.assertEqual(
            validation_warnings,
            [(['dem_path'], 'not found on disk'),
             (['outlet_vector_path'], 'not found on disk')])

        bad_spatial_files_args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'dem-not-here.tif'),
            'outlet_vector_path': os.path.join(self.workspace_dir,
                                               'outlets-not-here.shp'),
            # Also testing point snapping args
            'snap_points': True,
            'flow_threshold': -1,
            'snap_distance': 'fooooo',
        }
        for key in ('dem_path', 'outlet_vector_path'):
            with open(bad_spatial_files_args[key], 'w') as spatial_file:
                spatial_file.write('not a spatial file')

        validation_warnings = delineateit.validate(bad_spatial_files_args)
        self.assertEqual(
            validation_warnings,
            [(['dem_path'], 'not a raster'),
             (['outlet_vector_path'], 'not a vector'),
             (['flow_threshold'], 'must be a positive integer'),
             (['snap_distance'], 'must be an integer')])

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
                                          'source_features.geojson')
        source_features = [
            Point(-1, -1),  # off the edge of the stream raster.
            Point(3, -5),
            Point(7, -9),
            Point(13, -5),
            box(-2, -2, -1, -1),  # Off the edge
        ]
        pygeoprocessing.testing.create_vector_on_disk(
            source_features, wkt,
            fields={'foo': 'int',
                    'bar': 'string'},
            attributes=[
                {'foo': 0, 'bar': 0.1},
                {'foo': 1, 'bar': 1.1},
                {'foo': 2, 'bar': 2.1},
                {'foo': 3, 'bar': 3.1},
                {'foo': 4, 'bar': 4.1}],
            filename=source_points_path)

        snapped_points_path = os.path.join(self.workspace_dir,
                                           'snapped_points.gpkg')

        snap_distance = -1
        with self.assertRaises(ValueError) as cm:
            delineateit.snap_points_to_nearest_stream(
                source_points_path, (stream_raster_path, 1),
                snap_distance, snapped_points_path)
        self.assertTrue('must be >= 0' in str(cm.exception))

        snap_distance = 10  # large enough to get multiple streams per point.
        delineateit.snap_points_to_nearest_stream(
            source_points_path, (stream_raster_path, 1),
            snap_distance, snapped_points_path)

        snapped_points_vector = gdal.OpenEx(snapped_points_path,
                                            gdal.OF_VECTOR)
        snapped_points_layer = snapped_points_vector.GetLayer()

        # snapped layer will include 3 valid points and one polygon.
        self.assertEqual(4, snapped_points_layer.GetFeatureCount())

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
