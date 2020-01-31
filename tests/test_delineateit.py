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
from osgeo import ogr
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
        """DelineateIt: test validation function."""
        from natcap.invest import delineateit
        missing_keys = {}
        validation_warnings = delineateit.validate(missing_keys)
        self.assertEqual(len(validation_warnings), 1)
        self.assertEqual(len(validation_warnings[0][0]), 3)

        missing_values_args = {
            'workspace_dir': '',
            'dem_path': None,
            'outlet_vector_path': '',
            'snap_points': False,
        }
        validation_warnings = delineateit.validate(missing_values_args)
        self.assertEqual(len(validation_warnings), 1)
        self.assertTrue('has no value' in validation_warnings[0][1])

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
            [(['dem_path'], 'File not found'),
             (['outlet_vector_path'], 'File not found')])

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
            [(['dem_path'], 'File could not be opened as a GDAL raster'),
             (['outlet_vector_path'], (
                 'File could not be opened as a GDAL vector')),
             (['flow_threshold'], 'Value does not meet condition value > 0'),
             (['snap_distance'], (
                "Value 'fooooo' could not be interpreted as a number"))])

    def test_point_snapping(self):
        """DelineateIt: test point snapping."""

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

    def test_vector_may_contain_points(self):
        """DelineateIt: Check whether a layer contains points."""
        from natcap.invest import delineateit

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84/UTM zone 31s

        # Verify invalid filepath returns False.
        invalid_filepath = os.path.join(self.workspace_dir, 'nope.foo')
        self.assertFalse(
            delineateit._vector_may_contain_points(invalid_filepath))

        # Verify invalid layer ID returns False
        gpkg_driver = gdal.GetDriverByName('GPKG')
        new_vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')
        new_vector = gpkg_driver.Create(
            new_vector_path, 0, 0, 0, gdal.GDT_Unknown)
        point_layer = new_vector.CreateLayer(
            'point_layer', srs, ogr.wkbPoint)
        polygon_layer = new_vector.CreateLayer(
            'polygon_layer', srs, ogr.wkbPolygon)
        unknown_layer = new_vector.CreateLayer(
            'unknown_type_layer', srs, ogr.wkbUnknown)

        unknown_layer = None
        polygon_layer = None
        point_layer = None
        new_vector = None

        self.assertTrue(delineateit._vector_may_contain_points(
                        new_vector_path, 'point_layer'))
        self.assertFalse(delineateit._vector_may_contain_points(
                         new_vector_path, 'polygon_layer'))
        self.assertTrue(delineateit._vector_may_contain_points(
                        new_vector_path, 'unknown_type_layer'))
        self.assertFalse(delineateit._vector_may_contain_points(
                         new_vector_path, 'NOT A LAYER'))

    def test_check_geometries(self):
        """DelineateIt: Check that we can reasonably repair geometries."""
        from natcap.invest import delineateit
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84/UTM zone 31s

        dem_matrix = numpy.array(
            [[0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]], dtype=numpy.int8)
        dem_raster_path = os.path.join(self.workspace_dir, 'dem.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [dem_matrix],
            origin=(2, -2),
            pixel_size=(2, -2),
            projection_wkt=srs.ExportToWkt(),
            nodata=255,  # byte datatype
            filename=dem_raster_path)

        # empty geometry
        invalid_geometry = ogr.CreateGeometryFromWkt('POLYGON EMPTY')
        self.assertTrue(invalid_geometry.IsEmpty())

        # point outside of the DEM bbox
        invalid_point = ogr.CreateGeometryFromWkt('POINT (-100 -100)')

        # line intersects the DEM but is not contained by it
        valid_line = ogr.CreateGeometryFromWkt(
            'LINESTRING (-100 100, 100 -100)')

        # invalid polygon coult fixed by buffering by 0
        invalid_bowtie_polygon = ogr.CreateGeometryFromWkt(
            'POLYGON ((2 -2, 6 -2, 2 -6, 6 -6, 2 -2))')
        self.assertFalse(invalid_bowtie_polygon.IsValid())

        # Bowtie polygon with vertex in the middle, could be fixed
        # by buffering by 0
        invalid_alt_bowtie_polygon = ogr.CreateGeometryFromWkt(
            'POLYGON ((2 -2, 6 -2, 4 -4, 6 -6, 2 -6, 4 -4, 2 -2))')
        self.assertFalse(invalid_alt_bowtie_polygon.IsValid())

        # invalid polygon could be fixed by closing rings
        invalid_open_ring_polygon = ogr.CreateGeometryFromWkt(
            'POLYGON ((2 -2, 6 -2, 6 -6, 2 -6))')
        self.assertFalse(invalid_open_ring_polygon.IsValid())

        gpkg_driver = gdal.GetDriverByName('GPKG')
        outflow_vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')
        outflow_vector = gpkg_driver.Create(
            outflow_vector_path, 0, 0, 0, gdal.GDT_Unknown)
        outflow_layer = outflow_vector.CreateLayer(
            'outflow_layer', srs, ogr.wkbUnknown)
        outflow_layer.CreateField(ogr.FieldDefn('geom_id', ogr.OFTInteger))

        outflow_layer.StartTransaction()
        for index, geometry in enumerate((invalid_geometry,
                                          invalid_point,
                                          valid_line,
                                          invalid_bowtie_polygon,
                                          invalid_alt_bowtie_polygon,
                                          invalid_open_ring_polygon)):
            if geometry is None:
                self.fail('Geometry could not be created')

            outflow_feature = ogr.Feature(outflow_layer.GetLayerDefn())
            outflow_feature.SetField('geom_id', index)
            outflow_feature.SetGeometry(geometry)
            outflow_layer.CreateFeature(outflow_feature)
        outflow_layer.CommitTransaction()

        self.assertEquals(outflow_layer.GetFeatureCount(), 6)
        outflow_layer = None
        outflow_vector = None

        target_vector_path = os.path.join(self.workspace_dir, 'checked_geometries.gpkg')
        with self.assertRaises(ValueError) as cm:
            delineateit.check_geometries(
                outflow_vector_path, dem_raster_path, target_vector_path,
                skip_invalid_geometry=False
            )
        self.assertTrue('is invalid' in str(cm.exception))

        delineateit.check_geometries(
            outflow_vector_path, dem_raster_path, target_vector_path,
            skip_invalid_geometry=True
        )

        # I only expect to see 1 feature in the output layer, as there's only 1
        # valid geometry.
        expected_geom_areas = {
            2: 0,
        }

        target_vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)
        target_layer = target_vector.GetLayer()
        self.assertEqual(target_layer.GetFeatureCount(), len(expected_geom_areas))

        for feature in target_layer:
            geom = feature.GetGeometryRef()
            self.assertAlmostEqual(
                geom.Area(), expected_geom_areas[feature.GetField('geom_id')])

        target_layer = None
        target_vector = None
