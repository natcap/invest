"""Module for Regression Testing the InVEST Scenic Quality module."""
import unittest
import tempfile
import shutil
import os
import glob

from osgeo import gdal
from osgeo import osr
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
from shapely.geometry import Polygon, Point
import numpy


class ScenicQualityTests(unittest.TestCase):
    """Tests for the InVEST Scenic Quality model."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def create_dem(dem_path):
        """Create a known DEM at the given path.

        Parameters:
            dem_path (string): Where to store the DEM.

        Returns:
            ``None``

        """
        from pygeoprocessing.testing import create_raster_on_disk
        dem_matrix = numpy.array(
            [[10, 2, 2, 2, 10],
             [2, 10, 2, 10, 2],
             [2, 2, 10, 2, 2],
             [2, 10, 2, 10, 2],
             [10, 2, 2, 2, 10]], dtype=numpy.int8)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32331)  # UTM zone 31s
        wkt = srs.ExportToWkt()
        create_raster_on_disk(
            [dem_matrix],
            origin=(0, 0),
            projection_wkt=wkt,
            nodata=255,  # byte nodata value
            pixel_size=(2, -2),
            filename=dem_path)

    def test_invalid_valuation_function(self):
        """SQ: model raises exception with invalid valuation function."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32331)  # UTM zone 31s
        wkt = srs.ExportToWkt()

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        sampledata.create_vector_on_disk(
            [Point(5.0, -1.0),
             Point(-1.0, -5.0),  # off the edge of DEM, won't be included.
             Point(5.0, -9.0),
             Point(9.0, -5.0)],
            wkt, filename=viewpoints_path)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        sampledata.create_vector_on_disk(
            [Polygon([(0, 0), (0, -9), (9, -9), (9, 0), (0, 0)])],
            wkt, filename=aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'foo',
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'valuation_function': 'INVALID FUNCTION',
            'a_coef': 1,
            'b_coef': 0,
            'max_valuation_radius': 10.0,
        }

        with self.assertRaises(ValueError):
            scenic_quality.execute(args)

    def test_viewshed_field_defaults(self):
        """SQ: run model with default field values."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32331)  # UTM zone 31s
        wkt = srs.ExportToWkt()

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        sampledata.create_vector_on_disk(
            [Point(5.0, -1.0),
             Point(-1.0, -5.0),  # off the edge of DEM, won't be included.
             Point(5.0, -9.0),
             Point(9.0, -5.0)],
            wkt, filename=viewpoints_path)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        sampledata.create_vector_on_disk(
            [Polygon([(0, 0), (0, -9), (9, -9), (9, 0), (0, 0)])],
            wkt, filename=aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'foo',
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'valuation_function': 'linear',
            'a_coef': 1,
            'b_coef': 0,
            'max_valuation_radius': 10.0,
        }

        # Simulate a run where the clipped structures vector already exists.
        # This is needed for coverage in the vector clipping function.
        clipped_structures_path = os.path.join(args['workspace_dir'],
                                               'intermediate',
                                               'structures_clipped_foo.shp')
        os.makedirs(os.path.dirname(clipped_structures_path))
        with open(clipped_structures_path, 'w') as fake_file:
            fake_file.write('this is a vector :)')

        scenic_quality.execute(args)

        # 3 of the 4 viewpoints overlap the DEM, so there should only be files
        # from 3 viewsheds.
        self.assertEqual(len(glob.glob(os.path.join(
            args['workspace_dir'], 'intermediate', 'auxilliary*'))), 3)
        self.assertEqual(len(glob.glob(os.path.join(
            args['workspace_dir'], 'intermediate', 'visibility*'))), 3)
        self.assertEqual(len(glob.glob(os.path.join(
            args['workspace_dir'], 'intermediate', 'value*'))), 3)

        # Verify that the value summation matrix is what we expect it to be.
        expected_value = numpy.array(
            [[1, 1, 1, 1, 2],
             [0, 1, 1, 2, 1],
             [0, 0, 3, 1, 1],
             [0, 1, 1, 2, 1],
             [1, 1, 1, 1, 2]], dtype=numpy.int8)

        value_raster = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'output', 'vshed_value_foo.tif'))
        value_band = value_raster.GetRasterBand(1)
        value_matrix = value_band.ReadAsArray()

        numpy.testing.assert_almost_equal(expected_value, value_matrix)

        # verify that the correct number of viewpoints has been tallied.
        vshed_raster = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'output', 'vshed_foo.tif'))
        vshed_band = vshed_raster.GetRasterBand(1)
        vshed_matrix = vshed_band.ReadAsArray()

        # Because our B coefficient is 0, the vshed matrix should match the
        # value matrix.
        numpy.testing.assert_almost_equal(expected_value, vshed_matrix)

        # Test the visual quality raster.
        expected_visual_quality = numpy.array(
            [[3, 3, 3, 3, 4],
             [0, 3, 3, 4, 3],
             [0, 0, 4, 3, 3],
             [0, 3, 3, 4, 3],
             [3, 3, 3, 3, 4]])
        visual_quality_raster = os.path.join(
            args['workspace_dir'], 'output', 'vshed_qual_foo.tif')
        quality_matrix = gdal.OpenEx(visual_quality_raster).ReadAsArray()
        numpy.testing.assert_almost_equal(expected_visual_quality,
                                          quality_matrix)

    def test_viewshed_with_fields(self):
        """SQ: verify that we can specify viewpoint fields."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32331)  # UTM zone 31s
        wkt = srs.ExportToWkt()

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        sampledata.create_vector_on_disk(
            [Point(5.0, 0.0),
             Point(-1.0, -4.0),  # off the edge of DEM, won't be included.
             Point(5.0, -8.0),
             Point(9.0, -4.0)],
            wkt, filename=viewpoints_path,
            fields={'RADIUS': 'real',
                    'HEIGHT': 'real',
                    'WEIGHT': 'real'},
            attributes=[
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 1.0},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 1.0},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 2.5},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 2.5}])

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        sampledata.create_vector_on_disk(
            [Polygon([(0, 0), (0, -9), (9, -9), (9, 0), (0, 0)])],
            wkt, filename=aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'valuation_function': 'linear',
            'a_coef': 0,
            'b_coef': 1,
            'max_valuation_radius': 10.0,
        }

        scenic_quality.execute(args)

        # Verify that the value summation matrix is what we expect it to be.
        # The weight of two of the points makessome sectors more valuable
        expected_value = numpy.array(
            [[10., 5., 0., 5., 20.],
             [0., 7.07106781, 5., 14.14213562, 5.],
             [0., 0., 24., 5., 0.],
             [0., 2.82842712, 2., 9.89949494, 5.],
             [4., 2., 0., 2., 14.]])

        value_raster = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'output', 'vshed_value.tif'))
        value_band = value_raster.GetRasterBand(1)
        value_matrix = value_band.ReadAsArray()

        numpy.testing.assert_almost_equal(expected_value, value_matrix)

        # Verify that the sum of the viewsheds (which is weighted) is correct.
        expected_weighted_vshed = numpy.array(
            [[2.5, 2.5, 2.5, 2.5, 5.],
             [0., 2.5, 2.5, 5., 2.5],
             [0., 0., 6., 2.5, 2.5],
             [0., 1., 1., 3.5, 2.5],
             [1., 1., 1., 1., 3.5]], dtype=numpy.float32)
        vshed_raster_path = os.path.join(args['workspace_dir'], 'output',
                                         'vshed.tif')
        weighted_vshed_matrix = gdal.OpenEx(vshed_raster_path).ReadAsArray()
        numpy.testing.assert_almost_equal(expected_weighted_vshed,
                                          weighted_vshed_matrix)

        # Test the visual quality raster since this run is weighted.
        expected_visual_quality = numpy.array(
            [[3, 3, 0, 3, 4],
             [0, 3, 3, 4, 3],
             [0, 0, 4, 3, 0],
             [0, 1, 1, 3, 3],
             [1, 1, 0, 1, 4]])
        visual_quality_raster = os.path.join(
            args['workspace_dir'], 'output', 'vshed_qual.tif')
        quality_matrix = gdal.OpenEx(visual_quality_raster).ReadAsArray()
        numpy.testing.assert_almost_equal(expected_visual_quality,
                                          quality_matrix)

    def test_exponential_valuation(self):
        """SQ: verify values on exponential valuation."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32331)  # UTM zone 31s
        wkt = srs.ExportToWkt()

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        sampledata.create_vector_on_disk(
            [Point(5.0, 0.0),
             Point(-1.0, -4.0),  # off the edge of DEM, won't be included.
             Point(5.0, -8.0),
             Point(9.0, -4.0)],
            wkt, filename=viewpoints_path)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        sampledata.create_vector_on_disk(
            [Polygon([(0, 0), (0, -9), (9, -9), (9, 0), (0, 0)])],
            wkt, filename=aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'valuation_function': 'exponential',
            'a_coef': 1,
            'b_coef': 1,
            'max_valuation_radius': 10.0,
        }

        scenic_quality.execute(args)

        # Verify that the value summation matrix is what we expect it to be.
        # The weight of two of the points makessome sectors more valuable
        expected_value = numpy.array(
            [[0.01831564, 0.13533528, 1., 0.13533528, 0.03663128],
             [0., 0.05910575, 0.13533528, 0.11821149, 0.13533528],
             [0., 0., 0.05494692, 0.13533528, 1.],
             [0., 0.05910575, 0.13533528, 0.11821149, 0.13533528],
             [0.01831564, 0.13533528, 1., 0.13533528, 0.03663128]])

        value_raster = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'output', 'vshed_value.tif'))
        value_band = value_raster.GetRasterBand(1)
        value_matrix = value_band.ReadAsArray()

        numpy.testing.assert_almost_equal(expected_value, value_matrix)

    def test_logarithmic_valuation(self):
        """SQ: verify values on logarithmic valuation."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32331)  # UTM zone 31s
        wkt = srs.ExportToWkt()

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        sampledata.create_vector_on_disk(
            [Point(5.0, 0.0),
             Point(-1.0, -4.0),  # off the edge of DEM, won't be included.
             Point(5.0, -8.0),
             Point(9.0, -4.0)],
            wkt, filename=viewpoints_path)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        sampledata.create_vector_on_disk(
            [Polygon([(0, 0), (0, -9), (9, -9), (9, 0), (0, 0)])],
            wkt, filename=aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'valuation_function': 'logarithmic',
            'a_coef': 1,
            'b_coef': 1,
            'max_valuation_radius': 10.0,
        }

        scenic_quality.execute(args)

        # Verify that the value summation matrix is what we expect it to be.
        # The weight of two of the points makessome sectors more valuable
        expected_value = numpy.array(
            [[2.38629436, 1.69314718, 0., 1.69314718, 4.77258872],
             [0., 2.03972077, 1.69314718, 4.07944154, 1.69314718],
             [0., 0., 7.15888308, 1.69314718, 0.],
             [0., 2.03972077, 1.69314718, 4.07944154, 1.69314718],
             [2.38629436, 1.69314718, 0., 1.69314718, 4.77258872]])

        value_raster = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'output', 'vshed_value.tif'))
        value_band = value_raster.GetRasterBand(1)
        value_matrix = value_band.ReadAsArray()

        numpy.testing.assert_almost_equal(expected_value, value_matrix)

    def test_visual_quality(self):
        """SQ: verify visual quality calculations."""
        from natcap.invest.scenic_quality import scenic_quality
        visible_structures = numpy.tile(
            numpy.array([3, 0, 0, 0, 6, 7, 8]), (5, 1))

        n_visible = os.path.join(self.workspace_dir, 'n_visible.tif')
        visual_quality_raster = os.path.join(self.workspace_dir,
                                             'visual_quality.tif')
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create(n_visible, 7, 5, 1, gdal.GDT_Int32)
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(-1)
        band.WriteArray(visible_structures)
        band = None
        raster = None

        scenic_quality._calculate_visual_quality(n_visible,
                                                 self.workspace_dir,
                                                 visual_quality_raster)

        expected_visual_quality = numpy.tile(
            numpy.array([1, 0, 0, 0, 2, 3, 4]), (5, 1))

        visual_quality_matrix = gdal.OpenEx(
            visual_quality_raster).ReadAsArray()
        numpy.testing.assert_almost_equal(expected_visual_quality,
                                          visual_quality_matrix)

    def test_visual_quality_low_count(self):
        """SQ: verify visual quality calculations for low pixel counts."""
        from natcap.invest.scenic_quality import scenic_quality
        visible_structures = numpy.array([[-1, 3, 0, 0, 0, 3, 6, 7]])

        n_visible = os.path.join(self.workspace_dir, 'n_visible.tif')
        visual_quality_raster = os.path.join(self.workspace_dir,
                                             'visual_quality.tif')
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create(n_visible, 8, 1, 1, gdal.GDT_Int32)
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(-1)
        band.WriteArray(visible_structures)
        band = None
        raster = None

        scenic_quality._calculate_visual_quality(n_visible,
                                                 self.workspace_dir,
                                                 visual_quality_raster)

        expected_visual_quality = numpy.array([[255, 2, 0, 0, 0, 2, 3, 4]])

        visual_quality_matrix = gdal.OpenEx(
            visual_quality_raster).ReadAsArray()
        numpy.testing.assert_almost_equal(expected_visual_quality,
                                          visual_quality_matrix)

    def test_visual_quality_floats(self):
        """SQ: verify visual quality calculations for floating-point vshed."""
        from natcap.invest.scenic_quality import scenic_quality
        visible_structures = numpy.array(
            [[-1, 3.33, 0, 0, 0, 3.66, 6.12, 7.8]])

        n_visible = os.path.join(self.workspace_dir, 'n_visible.tif')
        visual_quality_raster = os.path.join(self.workspace_dir,
                                             'visual_quality.tif')
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create(n_visible, 8, 1, 1, gdal.GDT_Float32)
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(-1)
        band.WriteArray(visible_structures)
        band = None
        raster = None

        scenic_quality._calculate_visual_quality(n_visible,
                                                 self.workspace_dir,
                                                 visual_quality_raster)

        expected_visual_quality = numpy.array([[255, 1, 0, 0, 0, 2, 3, 4]])

        visual_quality_matrix = gdal.OpenEx(
            visual_quality_raster).ReadAsArray()
        numpy.testing.assert_almost_equal(expected_visual_quality,
                                          visual_quality_matrix)


class ScenicQualityValidationTests(unittest.TestCase):
    """Tests for Scenic Quality validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys(self):
        """SQ Validate: assert missing keys."""
        from natcap.invest.scenic_quality import scenic_quality
        try:
            scenic_quality.validate({})  # empty args dict.
            self.fail('KeyError expected but not found')
        except KeyError as error_raised:
            missing_keys = sorted(error_raised.args)
            expected_missing_keys = [
                'a_coef',
                'aoi_path',
                'b_coef',
                'dem_path',
                'max_valuation_radius',
                'refraction',
                'structure_path',
                'valuation_function',
                'workspace_dir',
            ]
            self.assertEqual(missing_keys, expected_missing_keys)

    def test_polynomial_required_keys(self):
        """SQ Validate: assert polynomial required keys."""
        from natcap.invest.scenic_quality import scenic_quality
        try:
            args = {'valuation_function': 'polynomial'}
            scenic_quality.validate(args)
            self.fail('KeyError expected but not found')
        except KeyError as error_raised:
            missing_keys = sorted(error_raised.args)
            expected_missing_keys = [
                'a_coef',
                'aoi_path',
                'b_coef',
                'dem_path',
                'max_valuation_radius',
                'refraction',
                'structure_path',
                'workspace_dir',
                # This list doesn't contain key ``valuation_function`` because
                # the key was provided in args.
            ]
            self.assertEqual(missing_keys, expected_missing_keys)

    def test_bad_values(self):
        """SQ Validate: Assert we can catch various validation errors."""
        from natcap.invest.scenic_quality import scenic_quality

        # AOI path is missing
        args = {
            'workspace_dir': '',  # required key, missing value
            'aoi_path': '/bad/vector/path',
            'a_coef': 'foo',  # not a number
            'b_coef': -1,  # valid
            'dem_path': 'not/a/path',  # not a raster
            'refraction': "0.13",
            'max_valuation_radius': None,  # covers missing value.
            'structure_path': 'vector/missing',
            'valuation_function': 'bad function',
        }

        validation_errors = scenic_quality.validate(args)

        self.assertEqual(len(validation_errors), 7)

        # map single-key errors to their errors.
        single_key_errors = {}
        for keys, error in validation_errors:
            if len(keys) == 1:
                single_key_errors[keys[0]] = error

        self.assertTrue('refraction' not in single_key_errors)
        self.assertEqual(single_key_errors['a_coef'], 'Must be a number')
        self.assertEqual(single_key_errors['dem_path'], 'Must be a raster')
        self.assertEqual(single_key_errors['structure_path'],
                         'Must be a vector')
        self.assertEqual(single_key_errors['aoi_path'], 'Must be a vector')
        self.assertEqual(single_key_errors['valuation_function'],
                         'Invalid function')

    def test_dem_projected_in_m(self):
        """SQ Validate: the DEM must be projected in meters."""
        from natcap.invest.scenic_quality import scenic_quality
        from pygeoprocessing.testing import create_raster_on_disk

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84 is not projected.
        filepath = os.path.join(self.workspace_dir, 'dem.tif')
        create_raster_on_disk(
            [numpy.array([[1]])],
            origin=(0, 0),
            projection_wkt=srs.ExportToWkt(),
            nodata=-1,
            pixel_size=(1, -1),
            filename=filepath)

        args = {'dem_path': filepath}

        validation_errors = scenic_quality.validate(args, limit_to='dem_path')
        self.assertEqual(len(validation_errors), 1)
        self.assertTrue('Must be projected in meters' in
                        validation_errors[0][1])


class ViewshedTests(unittest.TestCase):
    """Tests for pygeoprocessing's viewshed."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def create_dem(matrix, filepath, pixel_size=(1, -1), nodata=-1):
        """Create a DEM in WGS84 coordinate system.

        Parameters:
            matrix (numpy.array): A 2D numpy array of pixel values.
            filepath (string): The filepath where the new raster file will be
                written.
            pixel_size=(1, -1): The pixel size to use for the output raster.
            nodata=-1: The nodata value to use for the output raster.

        Returns:
            ``None``.

        """
        from pygeoprocessing.testing import create_raster_on_disk

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        wkt = srs.ExportToWkt()
        create_raster_on_disk(
            [matrix],
            origin=(0, 0),
            projection_wkt=wkt,
            nodata=nodata,
            pixel_size=pixel_size,
            filename=filepath)

    def test_pixels_not_square(self):
        """Viewshed: exception raised when pixels are not square."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.ones((20, 20))
        viewpoint = (10, 10)
        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath,
                                 pixel_size=(1.111111, 1.12))

        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        with self.assertRaises(AssertionError):
            viewshed((dem_filepath, 1), viewpoint, visibility_filepath)

    def test_viewpoint_not_overlapping_dem(self):
        """Viewshed: exception raised when viewpoint is not over the DEM."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.ones((20, 20))
        viewpoint = (-10, -10)
        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)

        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')

        with self.assertRaises(ValueError):
            viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                     aux_filepath=os.path.join(self.workspace_dir,
                                               'auxulliary.tif'))

    def test_max_distance(self):
        """Viewshed: setting a max distance limits visibility distance."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.ones((6, 6))
        viewpoint = (5, 5)
        max_dist = 4

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)

        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')

        viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'),
                 refraction_coeff=1.0, max_distance=max_dist)

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        expected_visibility = numpy.zeros(matrix.shape)

        expected_visibility = numpy.array(
            [[255, 255, 255, 255, 255, 255],
             [255, 255, 255, 255, 255, 0],
             [255, 255, 255, 1, 1, 1],
             [255, 255, 1, 1, 1, 1],
             [255, 255, 1, 1, 1, 1],
             [255, 0, 1, 1, 1, 1]], dtype=numpy.uint8)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_refractivity(self):
        """Viewshed: refractivity partly compensates for earth's curvature."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.array([[2, 1, 1, 2, 1, 1, 1, 1, 1, 50]])
        viewpoint = (0, 0)
        matrix[viewpoint] = 2
        matrix[0, 3] = 2
        pixel_size = (1000, -1000)

        # pixels are 1km.  With the viewpoint at an elevation of 1m,
        # the horizon should be about 3.6km out.  A 50m structure 10km out
        # should be visible above the horizon.

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath,
                                 pixel_size=pixel_size)
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')

        viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'),
                 refraction_coeff=0.1)

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()

        # Because of refractivity calculations (and the size of the pixels),
        # the pixels farther to the right are visible despite being 'hidden'
        # behind the hill at (0,3).  This is due to refractivity.
        expected_visibility = numpy.array(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0, 1]], dtype=numpy.uint8)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_intervening_nodata(self):
        """Viewshed: intervening nodata does not affect visibility."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        nodata = 255
        matrix = numpy.array([[2, 2, nodata, 3]])
        viewpoint = (0, 0)

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath,
                                 nodata=nodata)
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')

        viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'),
                 refraction_coeff=0.0)

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()

        expected_visibility = numpy.array(
            [[1, 1, 0, 1]], dtype=numpy.uint8)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_nodata_undefined(self):
        """Viewshed: assume a reasonable nodata value if none defined."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        nodata = None  # viewshed assumes an unlikely nodata value.
        matrix = numpy.array([[2, 2, 1, 3]])
        viewpoint = (0, 0)

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath,
                                 nodata=nodata)
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')

        viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'),
                 refraction_coeff=0.0)

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()

        expected_visibility = numpy.array(
            [[1, 1, 0, 1]], dtype=numpy.uint8)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_block_size_check(self):
        """Viewshed: exception raised when blocks not equal, power of 2."""
        from natcap.invest.scenic_quality.viewshed import viewshed

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [numpy.ones((10, 10))],
            (0, 0), projection_wkt=srs.ExportToWkt(), nodata=-1,
            pixel_size=(1, -1), dataset_opts=(
                'TILED=NO', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=20', 'BLOCKYSIZE=40'), filename=dem_filepath)

        with self.assertRaises(ValueError):
            viewshed(
                (dem_filepath, 1), (0, 0), visibility_filepath,
                aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
            )

    def test_view_from_valley(self):
        """Viewshed: test visibility from within a pit."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.zeros((9, 9))
        matrix[5:8, 5:8] = 2
        matrix[4:7, 4:7] = 1
        matrix[5, 5] = 0

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed((dem_filepath, 1), (5, 5), visibility_filepath,
                 refraction_coeff=1.0,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'))

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()

        expected_visibility = numpy.zeros(visibility_matrix.shape)
        expected_visibility[matrix != 0] = 1
        expected_visibility[5, 5] = 1
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_tower_view_from_valley(self):
        """Viewshed: test visibility from a 'tower' within a pit."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.zeros((9, 9))
        matrix[5:8, 5:8] = 2
        matrix[4:7, 4:7] = 1
        matrix[5, 5] = 0

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed((dem_filepath, 1), (5, 5), visibility_filepath,
                 viewpoint_height=10,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'))

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()

        expected_visibility = numpy.ones(visibility_matrix.shape)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_primitive_peak(self):
        """Viewshed: looking down from a peak renders everything visible."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.zeros((8, 8))
        matrix[4:7, 4:7] = 1
        matrix[5, 5] = 2

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed((dem_filepath, 1), (5, 5), visibility_filepath,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'),
                 refraction_coeff=1.0)

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        numpy.testing.assert_equal(visibility_matrix, numpy.ones(matrix.shape))

    def test_cliff_bottom_half_visibility(self):
        """Viewshed: visibility for a cliff on bottom half of DEM."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.empty((20, 20))
        matrix.fill(2)
        matrix[7:] = 10  # cliff at row 7
        viewpoint = (5, 10)
        matrix[viewpoint] = 5  # viewpoint

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed(
            dem_raster_path_band=(dem_filepath, 1),
            viewpoint=(viewpoint[1], viewpoint[0]),
            visibility_filepath=visibility_filepath,
            aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
        )

        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[8:] = 0
        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_cliff_top_half_visibility(self):
        """Viewshed: visibility for a cliff on top half of DEM."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.empty((20, 20))
        matrix.fill(2)
        matrix[:8] = 10  # cliff at row 8
        viewpoint = (10, 10)
        matrix[viewpoint] = 5  # viewpoint

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed(
            dem_raster_path_band=(dem_filepath, 1),
            viewpoint=viewpoint,
            visibility_filepath=visibility_filepath,
            aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
        )
        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[:7] = 0
        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_cliff_left_half_visibility(self):
        """Viewshed: visibility for a cliff on left half of DEM."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.empty((20, 20))
        matrix.fill(2)
        matrix[:, :8] = 10  # cliff at column 8
        viewpoint = (10, 10)
        matrix[viewpoint] = 5  # viewpoint

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed(
            dem_raster_path_band=(dem_filepath, 1),
            viewpoint=viewpoint,
            visibility_filepath=visibility_filepath,
            aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
        )
        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[:, :7] = 0
        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_cliff_right_half_visibility(self):
        """Viewshed: visibility for a cliff on right half of DEM."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.empty((20, 20))
        matrix.fill(2)
        matrix[:, 12:] = 10  # cliff at column 8
        viewpoint = (10, 10)
        matrix[viewpoint] = 5  # viewpoint

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed(
            dem_raster_path_band=(dem_filepath, 1),
            viewpoint=viewpoint,
            visibility_filepath=visibility_filepath,
            aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
        )
        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[:, 13:] = 0
        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_pillars(self):
        """Viewshed: put a few pillars in a field, can't see behind them."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.empty((20, 20))
        matrix.fill(2)

        # Put a couple of pillars in there.
        for pillar in (
                (2, 5),
                (18, 5),
                (7, 18)):
            matrix[pillar] = 10

        viewpoint = (10, 10)
        matrix[viewpoint] = 5  # so it stands out in the DEM

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed(
            dem_raster_path_band=(dem_filepath, 1),
            viewpoint=viewpoint,
            visibility_filepath=visibility_filepath,
            aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
        )

        expected_visibility = numpy.array(
            [[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)
