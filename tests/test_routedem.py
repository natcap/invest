"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os

import numpy
from osgeo import gdal
from osgeo import osr


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
            numpy.flipud(numpy.arange(5)),
            numpy.arange(1, 5)))
        valley_with_sink = numpy.array([5, 4, 3, 2, 1.3, 1.3, 3, 4, 5])

        dem_array = numpy.vstack((
            valley_with_sink,
            numpy.tile(valley, (9, 1)) + elevation))
        nodata_value = -1

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)
        srs_wkt = srs.ExportToWkt()

        driver = gdal.GetDriverByName('GTiff')
        dem_raster = driver.Create(
            target_path, dem_array.shape[1], dem_array.shape[0],
            2, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        dem_raster.SetProjection(srs_wkt)
        ones_band = dem_raster.GetRasterBand(1)
        ones_band.SetNoDataValue(nodata_value)
        ones_band.WriteArray(numpy.ones(dem_array.shape))

        dem_band = dem_raster.GetRasterBand(2)
        dem_band.SetNoDataValue(nodata_value)
        dem_band.WriteArray(dem_array)
        dem_geotransform = [2, 2, 0, -2, 0, -2]
        dem_raster.SetGeoTransform(dem_geotransform)
        dem_raster = None

    def test_routedem_invalid_algorithm(self):
        """RouteDEM: fail when the algorithm isn't recognized."""
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

    def test_routedem_no_options_default_band(self):
        """RouteDEM: default to band 1 when not specified."""
        from natcap.invest import routedem

        # Intentionally leaving out the dem_band_index parameter,
        # should default to band 1.
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

        # The first band has only values of 1, no hydrological pits.
        # So, the filled band should match the source band.
        expected_filled_array = gdal.OpenEx(args['dem_path']).ReadAsArray()[0]
        filled_array = gdal.OpenEx(filled_raster_path).ReadAsArray()
        numpy.testing.assert_almost_equal(
            expected_filled_array,
            filled_array)

    def test_routedem_no_options(self):
        """RouteDEM: assert pitfilling when no other options given."""
        from natcap.invest import routedem

        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'dem_band_index': 2,
            'results_suffix': 'foo',
        }
        RouteDEMTests._make_dem(args['dem_path'])
        routedem.execute(args)

        filled_raster_path = os.path.join(args['workspace_dir'], 'filled_foo.tif')
        self.assertTrue(
            os.path.exists(filled_raster_path),
            'Filled DEM not created.')

        # The one sink in the array should have been filled to 1.3.
        expected_filled_array = gdal.OpenEx(args['dem_path']).ReadAsArray()[1]
        expected_filled_array[expected_filled_array < 1.3] = 1.3

        # Filled rasters currently are copies of the target rasters, though
        # pixels have only been filled from the desired band.
        filled_array = gdal.OpenEx(filled_raster_path).ReadAsArray()[1]
        numpy.testing.assert_almost_equal(
            expected_filled_array,
            filled_array)

    def test_routedem_slope(self):
        """RouteDEM: assert slope option."""
        from natcap.invest import routedem

        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'dem_band_index': 2,
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
        """RouteDEM: test d8 routing."""
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'algorithm': 'd8',
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'dem_band_index': 2,
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
        expected_flow_accum[:, 5:9] = numpy.flipud(numpy.arange(1, 5))
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
        expected_downstream_distance[:, 0:5] = numpy.flipud(numpy.arange(5))
        expected_downstream_distance[2:, 5:] = numpy.arange(1, 5)
        expected_downstream_distance[0, 5:] = numpy.arange(4)
        expected_downstream_distance[1, 5] = 1
        expected_downstream_distance[1, 6:] = numpy.arange(1, 4) + 0.41421356

        numpy.testing.assert_almost_equal(
            expected_downstream_distance,
            gdal.OpenEx(os.path.join(
                args['workspace_dir'], 'downstream_distance_foo.tif')).ReadAsArray())

    def test_routedem_mfd(self):
        """RouteDEM: test mfd routing."""
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'algorithm': 'mfd',
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'dem_band_index': 2,
            'results_suffix': 'foo',
            'calculate_flow_direction': True,
            'calculate_flow_accumulation': True,
            'calculate_stream_threshold': True,
            'calculate_downstream_distance': True,
            'calculate_slope': False,
            'threshold_flow_accumulation': 4,
        }

        RouteDEMTests._make_dem(args['dem_path'])
        routedem.execute(args)

        expected_stream_mask = numpy.array([
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ])
        numpy.testing.assert_almost_equal(
            expected_stream_mask,
            gdal.OpenEx(os.path.join(
                args['workspace_dir'], 'stream_mask_foo.tif')).ReadAsArray())

        # Raster sums are from manually-inspected outputs.
        for filename, expected_sum in (
                ('flow_accumulation_foo.tif', 678.94551294),
                ('flow_direction_foo.tif', 40968303668.0),
                ('downstream_distance_foo.tif', 162.28624753707527)):
            raster_path = os.path.join(args['workspace_dir'], filename)
            raster = gdal.OpenEx(raster_path)
            if raster is None:
                self.fail('Could not open raster %s' % filename)

            self.assertEqual(raster.RasterYSize, expected_stream_mask.shape[0])
            self.assertEqual(raster.RasterXSize, expected_stream_mask.shape[1])

            raster_sum = numpy.sum(raster.ReadAsArray(), dtype=numpy.float64)
            numpy.testing.assert_almost_equal(raster_sum, expected_sum)

    def test_validation_required_args(self):
        """RouteDEM: test required args in validation."""
        from natcap.invest import routedem
        args = {}

        required_keys = ['workspace_dir', 'dem_path']

        with self.assertRaises(KeyError) as cm:
            routedem.validate(args)
        for key in required_keys:
            self.assertTrue(key in repr(cm.exception))

    def test_validation_required_args_threshold(self):
        """RouteDEM: test required args in validation (with threshold)."""
        from natcap.invest import routedem

        args = {'calculate_stream_threshold': True}
        required_keys = [
            'workspace_dir', 'dem_path', 'threshold_flow_accumulation']

        with self.assertRaises(KeyError) as cm:
            routedem.validate(args)
        for key in required_keys:
            self.assertTrue(key in repr(cm.exception))

    def test_validation_required_args_none(self):
        """RouteDEM: test validation of a present but None args."""
        from natcap.invest import routedem
        required_keys = ['workspace_dir', 'dem_path']
        args = dict((k, None) for k in required_keys)

        validation_errors = routedem.validate(args)
        self.assertEqual(len(validation_errors), 1)
        self.assertEqual(
            set(required_keys) - set(validation_errors[0][0]),
            set([]))
        self.assertEqual(
            validation_errors[0][1], 'parameter has no value')

    def test_validation_required_args_empty(self):
        """RouteDEM: test validation of a present but empty args."""
        from natcap.invest import routedem
        required_keys = ['workspace_dir', 'dem_path']
        args = dict((k, '') for k in required_keys)

        validation_errors = routedem.validate(args)
        self.assertEqual(len(validation_errors), 1)
        self.assertEqual(
            set(required_keys) - set(validation_errors[0][0]),
            set([]))
        self.assertEqual(
            validation_errors[0][1], 'parameter has no value')

    def test_validation_dem_not_found(self):
        """RouteDEM: test validation of a missing DEM."""
        from natcap.invest import routedem

        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'notafile.txt')
        }
        validation_errors = routedem.validate(args)
        self.assertEqual(len(validation_errors), 1)
        self.assertEqual(
            validation_errors[0],
            (['dem_path'], 'not found on disk'))

    def test_validation_invalid_raster(self):
        """RouteDEM: test validation of an invalid DEM."""
        from natcap.invest import routedem

        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'badraster.tif'),
        }

        with open(args['dem_path'], 'w') as bad_raster:
            bad_raster.write('This is an invalid raster format.')

        validation_errors = routedem.validate(args)
        self.assertEqual(len(validation_errors), 1)
        self.assertEqual(
            validation_errors[0],
            (['dem_path'], 'not a raster'))

    def test_validation_band_index_type(self):
        """RouteDEM: test validation of an invalid band index."""
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'notafile.txt'),
            'dem_band_index': range(1, 5),
        }

        validation_errors = routedem.validate(args)
        self.assertEqual(len(validation_errors), 2)
        self.assertEqual(
            validation_errors[0],
            (['dem_band_index'], 'must be a positive, nonzero integer'))
        self.assertEqual(
            validation_errors[1],
            (['dem_path'], 'not found on disk'))

    def test_validation_band_index_negative_value(self):
        """RouteDEM: test validation of a negative band index."""
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'notafile.txt'),
            'dem_band_index': -5,
        }

        validation_errors = routedem.validate(args)
        self.assertEqual(len(validation_errors), 2)
        self.assertEqual(
            validation_errors[0],
            (['dem_band_index'], 'must be a positive, nonzero integer'))
        self.assertEqual(
            validation_errors[1],
            (['dem_path'], 'not found on disk'))

    def test_validation_band_index_value_too_large(self):
        """RouteDEM: test validation of a too-large band index."""
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'dem_path': os.path.join(self.workspace_dir, 'raster.tif'),
            'dem_band_index': 5,
        }

        # Has two bands, so band index 5 is too large.
        RouteDEMTests._make_dem(args['dem_path'])

        validation_errors = routedem.validate(args)
        self.assertEqual(len(validation_errors), 1)
        self.assertEqual(
            validation_errors[0],
            (['dem_band_index'], 'Must be between 1 and 2'))
