"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
import numpy
from osgeo import gdal
from osgeo import osr

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'routedem')


class RouteDEMTests(unittest.TestCase):
    """Tests for RouteDEM with Pygeoprocessing 1.x routing API."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def _make_dem(target_path):
        # makes a 10x10 DEM with a valley in the middle that flows to row 0.
        elevation = numpy.arange(1.1, 2, step=0.1).reshape((9, 1))
        valley = numpy.concatenate((
            numpy.flip(numpy.arange(5)),
            numpy.arange(1, 5)))
        valley_with_sink = numpy.array([5, 4, 3, 2, 1.3, 1.3, 3, 4, 5])

        dem_array = numpy.vstack((
            valley_with_sink,
            numpy.tile(valley, (9, 1)) + elevation))

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)
        srs_wkt = srs.ExportToWkt()

        driver = gdal.GetDriverByName('GTiff')
        dem_raster = driver.Create(
            target_path, dem_array.shape[1], dem_array.shape[0],
            1, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        dem_raster.SetProjection(srs_wkt)
        dem_band = dem_raster.GetRasterBand(1)
        dem_band.SetNoDataValue(-1)
        dem_band.WriteArray(dem_array)
        dem_geotransform = [2, 2, 0, -2, 0, -2]
        dem_raster.SetGeoTransform(dem_geotransform)
        dem_raster = None

    def test_routedem_invalid_algorithm(self):
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'algorithm': 'invalid',
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'results_suffix': 'foo',
            'calculate_flow_direction': True,
        }

        RouteDEMTests._make_dem(args['dem_path'])
        with self.assertRaises(RuntimeError) as cm:
            routedem.execute(args)

        self.assertTrue('Invalid algorithm specified' in str(cm.exception))

    def test_routedem_no_options(self):
        """RouteDEM: assert pitfilling when no other options given."""
        from natcap.invest import routedem

        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'results_suffix': 'foo',
        }
        RouteDEMTests._make_dem(args['dem_path'])
        routedem.execute(args)

        filled_raster_path = os.path.join(args['workspace_dir'], 'filled_foo.tif')
        self.assertTrue(
            os.path.exists(filled_raster_path),
            'Filled DEM not created.')

        # The one sink in the array should have been filled to 1.3.
        expected_filled_array = gdal.OpenEx(args['dem_path']).ReadAsArray()
        expected_filled_array[expected_filled_array < 1.3] = 1.3
        
        filled_array = gdal.OpenEx(filled_raster_path).ReadAsArray()
        numpy.testing.assert_almost_equal(
            expected_filled_array,
            filled_array)

    def test_routedem_slope(self):
        """RouteDEM: assert slope option."""
        from natcap.invest import routedem

        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'results_suffix': 'foo',
            'calculate_slope': True,
        }
        RouteDEMTests._make_dem(args['dem_path'])
        routedem.execute(args)

        for path in ('filled_foo.tif', 'slope_foo.tif'):
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], path)),
                'File not found: %s' % path)

        slope_array = gdal.OpenEx(
            os.path.join(args['workspace_dir'], 'slope_foo.tif')).ReadAsArray()
        # These were determined by inspection of the output array.
        expected_unique_values = numpy.array(
            [4.999998,  4.9999995, 5.000001, 5.0000043, 7.126098,
             13.235317, 45.017357, 48.226353, 48.75, 49.56845,
             50.249374, 50.24938, 50.249382, 55.17727, 63.18101],
            dtype=numpy.float32).reshape((15,))
        numpy.testing.assert_almost_equal(
            expected_unique_values,
            numpy.unique(slope_array))
        numpy.testing.assert_almost_equal(numpy.sum(slope_array), 4088.7358, decimal=4)

    def test_routedem_d8(self):
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'algorithm': 'd8',
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'results_suffix': 'foo',
            'calculate_flow_direction': True,
            'calculate_flow_accumulation': True,
            'calculate_stream_threshold': True,
            'calculate_downstream_distance': True,
            'calculate_slope': True,
            'threshold_flow_accumulation': 4,
        }

        RouteDEMTests._make_dem(args['dem_path'])
        routedem.execute(args)

        for expected_file in (
                'downstream_distance_foo.tif',
                'filled_foo.tif',
                'flow_accumulation_foo.tif',
                'flow_direction_foo.tif',
                'slope_foo.tif',
                'stream_mask_foo.tif'):
            self.assertTrue(
                os.path.exists(
                    os.path.join(args['workspace_dir'], expected_file)),
                'Raster not found: %s' % expected_file)

        expected_stream_mask = numpy.array([
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ])
        numpy.testing.assert_almost_equal(
            expected_stream_mask,
            gdal.OpenEx(os.path.join(
                args['workspace_dir'], 'stream_mask_foo.tif')).ReadAsArray())


        expected_flow_accum = numpy.empty((10, 9), dtype=numpy.float64)
        expected_flow_accum[:, 0:4] = numpy.arange(1, 5)
        expected_flow_accum[:, 5:9] = numpy.flip(numpy.arange(1, 5))
        expected_flow_accum[:, 4] = numpy.array(
            [82, 77, 72, 63, 54, 45, 36, 27, 18, 9])
        expected_flow_accum[1, 5] = 1
        expected_flow_accum[0, 5] = 8

        numpy.testing.assert_almost_equal(
            expected_flow_accum,
            gdal.OpenEx(os.path.join(
                args['workspace_dir'], 'flow_accumulation_foo.tif')).ReadAsArray())

        expected_flow_direction = numpy.empty((10, 9), dtype=numpy.uint8)
        expected_flow_direction[:, 0:4] = 0
        expected_flow_direction[:, 5:9] = 4
        expected_flow_direction[:, 4] = 2
        expected_flow_direction[0:2, 5] = 2
        expected_flow_direction[1, 6] = 3

        numpy.testing.assert_almost_equal(
            expected_flow_direction,
            gdal.OpenEx(os.path.join(
                args['workspace_dir'], 'flow_direction_foo.tif')).ReadAsArray())

        expected_downstream_distance = numpy.empty((10, 9), dtype=numpy.float64)
        expected_downstream_distance[:, 0:5] = numpy.flip(numpy.arange(5))
        expected_downstream_distance[2:, 5:] = numpy.arange(1, 5)
        expected_downstream_distance[0, 5:] = numpy.arange(4)
        expected_downstream_distance[1, 5] = 1
        expected_downstream_distance[1, 6:] = numpy.arange(1, 4) + 0.41421356

        numpy.testing.assert_almost_equal(
            expected_downstream_distance,
            gdal.OpenEx(os.path.join(
                args['workspace_dir'], 'downstream_distance_foo.tif')).ReadAsArray())
            



        

