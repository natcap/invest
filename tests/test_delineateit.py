"""Module for Testing DelineateIt."""
import unittest
from unittest import mock
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
import pygeoprocessing


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
        from natcap.invest.delineateit import delineateit

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
        layer = vector.GetLayer('watersheds_w')  # includes suffix
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
        from natcap.invest.delineateit import delineateit
        missing_keys = {}
        validation_warnings = delineateit.validate(missing_keys)
        self.assertEqual(len(validation_warnings), 2)
        self.assertTrue('dem_path' in validation_warnings[0][0])
        self.assertTrue('workspace_dir' in validation_warnings[0][0])
        self.assertEqual(validation_warnings[1][0], ['outlet_vector_path'])

        missing_values_args = {
            'workspace_dir': '',
            'dem_path': None,
            'outlet_vector_path': '',
            'snap_points': False,
            'detect_pour_points': True
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
             (['flow_threshold'], 'Value does not meet condition value > 0'),
             (['outlet_vector_path'], (
                 'File could not be opened as a GDAL vector')),
             (['snap_distance'], (
                "Value 'fooooo' could not be interpreted as a number"))])

    def test_point_snapping(self):
        """DelineateIt: test point snapping."""
        from natcap.invest.delineateit import delineateit

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
        # byte datatype
        pygeoprocessing.numpy_array_to_raster(
            stream_matrix, 255, (2, -2), (2, -2), wkt, stream_raster_path)

        source_points_path = os.path.join(self.workspace_dir,
                                          'source_features.geojson')
        source_features = [
            Point(-1, -1),  # off the edge of the stream raster.
            Point(3, -5),
            Point(7, -9),
            Point(13, -5),
            box(-2, -2, -1, -1),  # Off the edge
        ]
        fields = {'foo': ogr.OFTInteger, 'bar': ogr.OFTString}
        attributes = [
            {'foo': 0, 'bar': 0.1}, {'foo': 1, 'bar': 1.1},
            {'foo': 2, 'bar': 2.1}, {'foo': 3, 'bar': 3.1},
            {'foo': 4, 'bar': 4.1}]
        pygeoprocessing.shapely_geometry_to_vector(
            source_features, source_points_path, wkt, 'GeoJSON',
            fields=fields, attribute_list=attributes,
            ogr_geom_type=ogr.wkbUnknown)

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
        from natcap.invest.delineateit import delineateit

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
        from natcap.invest.delineateit import delineateit
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84/UTM zone 31s
        projection_wkt = srs.ExportToWkt()

        dem_matrix = numpy.array(
            [[0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1],
             [0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]], dtype=numpy.int8)
        dem_raster_path = os.path.join(self.workspace_dir, 'dem.tif')
        # byte datatype
        pygeoprocessing.numpy_array_to_raster(
            dem_matrix, 255, (2, -2), (2, -2), projection_wkt,
            dem_raster_path)

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

        self.assertEqual(outflow_layer.GetFeatureCount(), 6)
        outflow_layer = None
        outflow_vector = None

        target_vector_path = os.path.join(
            self.workspace_dir, 'checked_geometries.gpkg')
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
        self.assertEqual(
            target_layer.GetFeatureCount(), len(expected_geom_areas))

        for feature in target_layer:
            geom = feature.GetGeometryRef()
            self.assertAlmostEqual(
                geom.Area(), expected_geom_areas[feature.GetField('geom_id')])

        target_layer = None
        target_vector = None

    def test_detect_pour_points(self):
        from natcap.invest.delineateit import delineateit
 
        # create a flow direction raster from the sample DEM
        flow_dir_path = os.path.join(REGRESSION_DATA, 'input/flow_dir_gura.tif')
        output_path = os.path.join(self.workspace_dir, 'point_vector.gpkg')

        delineateit.detect_pour_points((flow_dir_path, 1), output_path)

        vector = gdal.OpenEx(output_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()

        points = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            x, y, _ = geom.GetPoint()
            points.append((x, y))
        points.sort()

        expected_points = [
            (277713.1562500205, 9941874.499999935),
            (277713.1562500205, 9941859.499999935)]

        self.assertTrue(numpy.isclose(points[0][0], expected_points[0][0]))
        self.assertTrue(numpy.isclose(points[0][1], expected_points[0][1]))
        self.assertTrue(numpy.isclose(points[1][0], expected_points[1][0]))
        self.assertTrue(numpy.isclose(points[1][1], expected_points[1][1]))

    def test_calculate_pour_point_array(self):
        from natcap.invest.delineateit import delineateit, delineateit_core

        a = 100  # nodata value

        # at this point the flow direction array will already have been
        # buffered with a border of nodata
        flow_dir_array = numpy.array([
            [7, 7, 7, 5],
            [6, 6, 6, 4],
            [0, 1, a, 0],
            [a, 2, 3, 4]
        ], dtype=numpy.intc)

        edges = numpy.array([1, 1, 0, 0], dtype=numpy.intc)  # top, left, bottom, right

        expected_pour_points = {(6.25, 5.25)}

        output = delineateit_core.calculate_pour_point_array(
            flow_dir_array, 
            edges, 
            nodata=a,
            offset=(0, 0),
            origin=(5, 3),
            pixel_size=(0.5, 1.5))

        self.assertTrue(numpy.array_equal(
            output,
            expected_pour_points))

    def test_find_pour_points_by_block(self):
        from natcap.invest.delineateit import delineateit

        a = 100  # nodata value
        flow_dir_array = numpy.array([
            [0, 0, 0, 0, 7, 7, 7, 1, 6, 6],
            [2, 3, 4, 5, 6, 7, 0, 1, 1, 2],
            [2, 2, 2, 2, 0, a, a, 3, 3, a],
            [2, 1, 1, 1, 2, 6, 4, 1, a, a],
            [1, 1, 0, 0, 0, 0, a, a, a, a]
        ], dtype=numpy.int8)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()

        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        pygeoprocessing.numpy_array_to_raster(
            flow_dir_array,
            a,
            (1, 1),
            (0, 0),
            projection_wkt,
            raster_path
        )

        expected_pour_points = {(7.5, 0.5), (5.5, 1.5), (4.5, 2.5), (5.5, 4.5)}

        # Mock iterblocks so that we can test with an array smaller than 128x128
        # to make sure that the algorithm gets pour points on block edges e.g.
        # flow_dir_array[2, 4]
        def mock_iterblocks(*args, **kwargs):
            xoffs = [0, 4, 8]
            win_xsizes = [4, 4, 2]
            for xoff, win_xsize in zip(xoffs, win_xsizes):
                yield {
                    'xoff': xoff, 
                    'yoff': 0,
                    'win_xsize': win_xsize,
                    'win_ysize': 5}

        with mock.patch(
            'natcap.invest.delineateit.delineateit.pygeoprocessing.iterblocks', 
            mock_iterblocks):
            pour_points = delineateit._find_raster_pour_points((raster_path, 1))
            self.assertEqual(pour_points, expected_pour_points)
