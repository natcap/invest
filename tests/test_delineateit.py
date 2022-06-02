"""Module for Testing DelineateIt."""
import contextlib
import logging
import os
import queue
import shutil
import tempfile
import unittest
from unittest import mock

import numpy
import pygeoprocessing
import shapely.wkb
import shapely.wkt
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from shapely.geometry import box
from shapely.geometry import MultiPoint
from shapely.geometry import Point

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'delineateit')


@contextlib.contextmanager
def capture_logging(logger, level=logging.NOTSET):
    """Capture logging within a context manager.

    Args:
        logger (logging.Logger): The logger that should be monitored for
            log records within the scope of the context manager.
        level (int): The log level to set for the new handler.  Defaults to
            ``logging.NOTSET``.

    Yields:
        log_records (list): A list of logging.LogRecord objects.  This list is
            yielded early in the execution, and may have logging progressively
            added to it until the context manager is exited.
    Returns:
        ``None``
    """
    message_queue = queue.Queue()
    queuehandler = logging.handlers.QueueHandler(message_queue)
    queuehandler.setLevel(level)
    logger.addHandler(queuehandler)
    log_records = []
    yield log_records
    logger.removeHandler(queuehandler)

    # Append log records to the existing log_records list.
    while True:
        try:
            log_records.append(message_queue.get_nowait())
        except queue.Empty:
            break


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
        expected_ws_ids_by_id = {
            1: 2,
            2: 1,
            3: 0
        }
        for feature in layer:
            geom = feature.GetGeometryRef()
            id_value = feature.GetField('id')
            self.assertEqual(
                feature.GetField('ws_id'), expected_ws_ids_by_id[id_value])
            self.assertAlmostEqual(
                geom.Area(), expected_areas_by_id[id_value], delta=1e-4)

    def test_delineateit_willamette_detect_pour_points(self):
        """DelineateIt: regression testing full run with pour point detection."""
        from natcap.invest.delineateit import delineateit

        args = {
            'dem_path': os.path.join(REGRESSION_DATA, 'input', 'dem.tif'),
            'outlet_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'outlets.shp'),
            'workspace_dir': self.workspace_dir,
            'detect_pour_points': True,
            'results_suffix': 'w',
            'n_workers': None,  # Trigger error and default to -1
        }
        delineateit.execute(args)

        vector = gdal.OpenEx(os.path.join(args['workspace_dir'],
                                          'watersheds_w.gpkg'), gdal.OF_VECTOR)
        layer = vector.GetLayer('watersheds_w')  # includes suffix
        self.assertEqual(layer.GetFeatureCount(), 102)

        # Assert that every valid pixel is covered by a watershed.
        n_pixels = 0
        raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
        pixel_x, pixel_y = raster_info['pixel_size']
        pixel_area = abs(pixel_x * pixel_y)
        nodata = raster_info['nodata'][0]
        for _, block in pygeoprocessing.iterblocks((args['dem_path'], 1)):
            n_pixels += len(block[~numpy.isclose(block, nodata)])

        valid_pixel_area = n_pixels * pixel_area

        total_area = 0
        for feature in layer:
            geom = feature.GetGeometryRef()
            total_area += geom.Area()

        self.assertAlmostEqual(valid_pixel_area, total_area, 4)

    def test_delineateit_validate(self):
        """DelineateIt: test validation function."""
        from natcap.invest import validation
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
        self.assertEqual(validation_warnings[0][1], validation.MESSAGES['MISSING_VALUE'])

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
            [(['dem_path'], validation.MESSAGES['FILE_NOT_FOUND']),
             (['outlet_vector_path'], validation.MESSAGES['FILE_NOT_FOUND'])])

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
            [(['dem_path'], validation.MESSAGES['NOT_GDAL_RASTER']),
             (['flow_threshold'], validation.MESSAGES['INVALID_VALUE'].format(
                condition='value >= 0')),
             (['outlet_vector_path'], validation.MESSAGES['NOT_GDAL_VECTOR']),
             (['snap_distance'], (
                validation.MESSAGES['NOT_A_NUMBER'].format(
                    value=bad_spatial_files_args['snap_distance'])))])

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
        flow_accum_array = numpy.array(
            [[1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 5, 5, 5, 5],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1]], dtype=numpy.int8)
        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')
        # byte datatype
        pygeoprocessing.numpy_array_to_raster(
            stream_matrix, 255, (2, -2), (2, -2), wkt, stream_raster_path)
        pygeoprocessing.numpy_array_to_raster(
            flow_accum_array, 255, (2, -2), (2, -2), wkt, flow_accum_path)

        source_points_path = os.path.join(self.workspace_dir,
                                          'source_features.geojson')
        source_features = [
            Point(-1, -1),  # off the edge of the stream raster.
            Point(3, -5),
            Point(7, -9),
            Point(13, -5),
            MultiPoint([(13, -5)]),
            box(-2, -2, -1, -1),  # Off the edge
        ]
        fields = {'foo': ogr.OFTInteger, 'bar': ogr.OFTString}
        attributes = [
            {'foo': 0, 'bar': '0.1'},
            {'foo': 1, 'bar': '1.1'},
            {'foo': 2, 'bar': '2.1'},
            {'foo': 3, 'bar': '3.1'},
            {'foo': 3, 'bar': '3.1'},  # intentional duplicate fields
            {'foo': 4, 'bar': '4.1'}]
        pygeoprocessing.shapely_geometry_to_vector(
            source_features, source_points_path, wkt, 'GeoJSON',
            fields=fields, attribute_list=attributes,
            ogr_geom_type=ogr.wkbUnknown)

        snapped_points_path = os.path.join(self.workspace_dir,
                                           'snapped_points.gpkg')

        snap_distance = -1
        with self.assertRaises(ValueError) as cm:
            delineateit.snap_points_to_nearest_stream(
                source_points_path, stream_raster_path, flow_accum_path,
                snap_distance, snapped_points_path)
        self.assertTrue('must be >= 0' in str(cm.exception))

        snap_distance = 10  # large enough to get multiple streams per point.
        delineateit.snap_points_to_nearest_stream(
            source_points_path, stream_raster_path, flow_accum_path,
            snap_distance, snapped_points_path)

        snapped_points_vector = gdal.OpenEx(snapped_points_path,
                                            gdal.OF_VECTOR)
        snapped_points_layer = snapped_points_vector.GetLayer()

        # snapped layer will include 4 valid points and 1 polygon.
        self.assertEqual(5, snapped_points_layer.GetFeatureCount())

        expected_geometries_and_fields = [
            (Point(5, -5), {'foo': 1, 'bar': '1.1'}),
            (Point(5, -9), {'foo': 2, 'bar': '2.1'}),
            (Point(13, -11), {'foo': 3, 'bar': '3.1'}),
            (Point(13, -11), {'foo': 3, 'bar': '3.1'}),  # Multipoint now point
            (box(-2, -2, -1, -1), {'foo': 4, 'bar': '4.1'}),  # unchanged
        ]
        for feature, (expected_geom, expected_fields) in zip(
                snapped_points_layer, expected_geometries_and_fields):
            shapely_feature = shapely.wkb.loads(
                bytes(feature.GetGeometryRef().ExportToWkb()))

            self.assertTrue(shapely_feature.equals(expected_geom))
            self.assertEqual(expected_fields, feature.items())

    def test_point_snapping_multipoint(self):
        """DelineateIt: test multi-point snapping."""
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
        flow_accum_array = numpy.array(
            [[1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 5, 5, 5, 5],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1]], dtype=numpy.int8)
        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')
        # byte datatype
        pygeoprocessing.numpy_array_to_raster(
            stream_matrix, 255, (2, -2), (2, -2), wkt, stream_raster_path)
        pygeoprocessing.numpy_array_to_raster(
            flow_accum_array, 255, (2, -2), (2, -2), wkt, flow_accum_path)

        source_points_path = os.path.join(self.workspace_dir,
                                          'source_features.gpkg')
        gpkg_driver = gdal.GetDriverByName('GPKG')
        points_vector = gpkg_driver.Create(
            source_points_path, 0, 0, 0, gdal.GDT_Unknown)
        layer_name = os.path.splitext(os.path.basename(source_points_path))[0]
        points_layer = points_vector.CreateLayer(layer_name,
                                                 points_vector.GetSpatialRef(),
                                                 ogr.wkbUnknown)
        # Create a bunch of points for the various OGR multipoint types and
        # make sure that they are all snapped to exactly the same place.
        points_layer.StartTransaction()
        for multipoint_type in (ogr.wkbMultiPoint, ogr.wkbMultiPointM,
                                ogr.wkbMultiPointZM, ogr.wkbMultiPoint25D):
            new_feature = ogr.Feature(points_layer.GetLayerDefn())
            new_geom = ogr.Geometry(multipoint_type)
            component_point = ogr.Geometry(ogr.wkbPoint)
            component_point.AddPoint(3, -5)
            new_geom.AddGeometry(component_point)
            new_feature.SetGeometry(new_geom)
            points_layer.CreateFeature(new_feature)

        # Verify point snapping will run if we give it empty multipoints.
        for point_type in (ogr.wkbPoint, ogr.wkbMultiPoint):
            new_feature = ogr.Feature(points_layer.GetLayerDefn())
            new_geom = ogr.Geometry(point_type)
            new_feature.SetGeometry(new_geom)
            points_layer.CreateFeature(new_feature)

        points_layer.CommitTransaction()

        snapped_points_path = os.path.join(self.workspace_dir,
                                           'snapped_points.gpkg')
        snap_distance = 10  # large enough to get multiple streams per point.
        delineateit.snap_points_to_nearest_stream(
            source_points_path, stream_raster_path, flow_accum_path,
            snap_distance, snapped_points_path)

        try:
            snapped_points_vector = gdal.OpenEx(snapped_points_path,
                                                gdal.OF_VECTOR)
            snapped_points_layer = snapped_points_vector.GetLayer()

            # All 4 multipoints should have been snapped to the same place and
            # should all be Point geometries.
            self.assertEqual(4, snapped_points_layer.GetFeatureCount())
            expected_feature = shapely.geometry.Point(5, -5)
            for feature in snapped_points_layer:
                shapely_feature = shapely.wkb.loads(
                    bytes(feature.GetGeometryRef().ExportToWkb()))
                self.assertTrue(shapely_feature.equals(expected_feature))
        finally:
            snapped_points_layer = None
            snapped_points_vector = None

    def test_point_snapping_break_ties(self):
        """DelineateIt: distance ties are broken using flow accumulation."""
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
        flow_accum_array = numpy.array(
            [[1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 1, 1, 1, 1],
             [1, 5, 9, 9, 9, 9],
             [1, 4, 1, 1, 1, 1],
             [1, 4, 1, 1, 1, 1]], dtype=numpy.int8)
        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')
        pygeoprocessing.numpy_array_to_raster(
            stream_matrix, 255, (2, -2), (2, -2), wkt, stream_raster_path)
        pygeoprocessing.numpy_array_to_raster(
            flow_accum_array, -1, (2, -2), (2, -2), wkt, flow_accum_path)

        source_points_path = os.path.join(self.workspace_dir,
                                          'source_features.geojson')
        source_features = [Point(9, -7)]  # equidistant from two streams
        pygeoprocessing.shapely_geometry_to_vector(
            source_features, source_points_path, wkt, 'GeoJSON',
            ogr_geom_type=ogr.wkbUnknown)

        snapped_points_path = os.path.join(self.workspace_dir,
                                           'snapped_points.gpkg')

        snap_distance = 10  # large enough to get multiple streams per point.
        delineateit.snap_points_to_nearest_stream(
            source_points_path, stream_raster_path, flow_accum_path,
            snap_distance, snapped_points_path)

        snapped_points_vector = gdal.OpenEx(snapped_points_path,
                                            gdal.OF_VECTOR)
        snapped_points_layer = snapped_points_vector.GetLayer()

        # should snap to stream point [4, 3] in the array above
        # if not considering flow accumulation, it would snap to the
        # nearest stream point found first in the array, at [2, 1]
        points = [
            shapely.wkb.loads(bytes(feature.GetGeometryRef().ExportToWkb()))
            for feature in snapped_points_layer]
        self.assertEqual(len(points), 1)
        self.assertEqual((points[0].x, points[0].y), (9, -11))

    def test_preprocess_geometries(self):
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

        # invalid polygon could fixed by buffering by 0
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
        outflow_layer.CreateField(ogr.FieldDefn('ws_id', ogr.OFTInteger))

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

            # We'll be overwriting these values with actual WS_IDs
            outflow_feature.SetField('ws_id', index*100)

            outflow_feature.SetGeometry(geometry)
            outflow_layer.CreateFeature(outflow_feature)
        outflow_layer.CommitTransaction()

        self.assertEqual(outflow_layer.GetFeatureCount(), 6)
        outflow_layer = None
        outflow_vector = None

        target_vector_path = os.path.join(
            self.workspace_dir, 'checked_geometries.gpkg')
        with self.assertRaises(ValueError) as cm:
            delineateit.preprocess_geometries(
                outflow_vector_path, dem_raster_path, target_vector_path,
                skip_invalid_geometry=False
            )
        self.assertTrue('is invalid' in str(cm.exception))

        # The only messages we care about for this test are WARNINGs
        logger = logging.getLogger('natcap.invest.delineateit.delineateit')
        with capture_logging(logger, logging.WARNING) as log_records:
            delineateit.preprocess_geometries(
                outflow_vector_path, dem_raster_path, target_vector_path,
                skip_invalid_geometry=True
            )
        self.assertEqual(len(log_records), 6)
        self.assertEqual(
            log_records[0].msg,
            delineateit._WS_ID_OVERWRITE_WARNING.format(
                layer_name='outflow_layer',
                vector_basename=os.path.basename(outflow_vector_path))
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

        expected_ws_ids = [0]  # only 1 valid feature, so we start at 0
        for ws_id, feature in zip(expected_ws_ids, target_layer):
            geom = feature.GetGeometryRef()
            self.assertAlmostEqual(
                geom.Area(), expected_geom_areas[feature.GetField('geom_id')])
            self.assertEqual(feature.GetField('ws_id'), ws_id)

        target_layer = None
        target_vector = None

    def test_preprocess_geometries_added_ws_id(self):
        """DelineateIt: Check that we add field ws_id when preprocessing."""
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

        source_features = [Point(9, -7), Point(10, -3)]
        source_points_path = os.path.join(self.workspace_dir, 'source.geojson')

        pygeoprocessing.shapely_geometry_to_vector(
            source_features, source_points_path, projection_wkt, 'GeoJSON',
            ogr_geom_type=ogr.wkbUnknown)

        target_vector_path = os.path.join(self.workspace_dir, 'preprocessed.gpkg')
        delineateit.preprocess_geometries(
            source_points_path, dem_raster_path, target_vector_path,
            skip_invalid_geometry=False)

        target_vector = gdal.OpenEx(target_vector_path)
        target_layer = target_vector.GetLayer()
        self.assertEqual(target_layer.GetFeatureCount(), 2)

        try:
            for expected_ws_id, feature in zip([0, 1], target_layer):
                self.assertEqual(feature.GetField('ws_id'), expected_ws_id)
        finally:
            target_layer = None
            target_vector = None

    def test_detect_pour_points(self):
        """DelineateIt: low-level test for pour point detection."""
        from natcap.invest.delineateit import delineateit

        # create a flow direction raster from the sample DEM
        flow_dir_path = os.path.join(
            REGRESSION_DATA, 'input/flow_dir_gura.tif')
        output_path = os.path.join(self.workspace_dir, 'point_vector.gpkg')

        delineateit.detect_pour_points((flow_dir_path, 1), output_path)

        vector = gdal.OpenEx(output_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()

        points = []
        ws_ids = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            x, y, _ = geom.GetPoint()
            points.append((x, y))
            ws_ids.append(feature.GetField('ws_id'))
        points.sort()

        expected_points = [
            (277713.1562500205, 9941874.499999935),
            (277713.1562500205, 9941859.499999935)]

        self.assertTrue(numpy.isclose(points[0][0], expected_points[0][0]))
        self.assertTrue(numpy.isclose(points[0][1], expected_points[0][1]))
        self.assertTrue(numpy.isclose(points[1][0], expected_points[1][0]))
        self.assertTrue(numpy.isclose(points[1][1], expected_points[1][1]))
        self.assertEqual(ws_ids, [0, 1])

    def test_calculate_pour_point_array(self):
        """DelineateIt: Extract pour points."""
        from natcap.invest.delineateit import delineateit
        from natcap.invest.delineateit import delineateit_core

        a = 100  # nodata value

        # at this point the flow direction array will already have been
        # buffered with a border of nodata
        flow_dir_array = numpy.array([
            [7, 7, 7, 5],
            [6, 6, 6, 4],
            [0, 1, a, 0],
            [a, 2, 3, 4]
        ], dtype=numpy.intc)

        # top, left, bottom, right
        edges = numpy.array([1, 1, 0, 0], dtype=numpy.intc)

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
        """DelineateIt: test pour point detection against block edges."""
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
            pour_points = delineateit._find_raster_pour_points(
                (raster_path, 1))
            self.assertEqual(pour_points, expected_pour_points)
