"""Module for Regression Testing the InVEST Scenic Quality module."""
import unittest
import tempfile
import shutil
import os
import glob

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pygeoprocessing
from shapely.geometry import Polygon, Point
import numpy


_SRS = osr.SpatialReference()
_SRS.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
WKT = _SRS.ExportToWkt()


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

        Args:
            dem_path (string): Where to store the DEM.

        Returns:
            ``None``

        """
        dem_matrix = numpy.array(
            [[10, 2, 2, 2, 10],
             [2, 10, 2, 10, 2],
             [2, 2, 10, 2, 2],
             [2, 10, 2, 10, 2],
             [10, 2, 2, 2, 10]], dtype=numpy.int8)

        # byte nodata value
        pygeoprocessing.numpy_array_to_raster(
            dem_matrix, 255, (2, -2), (2, -2), WKT, dem_path,
            raster_driver_creation_tuple=(
                'GTIFF', ['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW']))

    @staticmethod
    def create_aoi(aoi_path):
        """Create a known bounding box that overlaps the DEM.

        The envelope of the AOI perfectly overlaps the outside edge of the DEM.

        Args:
            aoi_path (string): The filepath where the AOI should be written.

        Returns:
            ``None``

        """
        geoms = [Polygon([(2, -2), (2, -12), (12, -12), (12, -2), (2, -2)])]
        pygeoprocessing.shapely_geometry_to_vector(
            geoms, aoi_path, WKT, 'GeoJSON')

    @staticmethod
    def create_viewpoints(viewpoints_path, fields=None, attributes=None):
        """Create a known set of viewpoints for this DEM.

        This vector will contain 4 viewpoints in the WGS84/UTM31S projection.
        The second viewpoint is off the edge of the DEM and will therefore not
        be included in the Scenic Quality analysis.

        Args:
            viewpoints_path (string): The filepath where the viewpoints vector
                should be saved.
            fields=None (dict): If provided, this must be a dict mapping
                fieldnames to datatypes, as expected by
                ``pygeoprocessing.shapely_geometry_to_vector``.
            attributes=None (dict): If provided, this must be a list of dicts
                mapping fieldnames (which match the keys in ``fields``) to
                values that will be used as the column value for each feature
                in sequence.

        Returns:
            ``None``

        """
        geometries = [
            Point(7.0, -3.0),
            Point(1.0, -7.0),  # off the edge of DEM, won't be included.
            Point(7.0, -11.0),
            Point(11.0, -7.0)]

        pygeoprocessing.shapely_geometry_to_vector(
            geometries, viewpoints_path, WKT, 'GeoJSON',
            fields=fields, attribute_list=attributes,
            ogr_geom_type=ogr.wkbPoint)

    def test_exception_when_no_structures_aoi_overlap(self):
        """SQ: model raises exception when AOI does not overlap structures."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        ScenicQualityTests.create_viewpoints(viewpoints_path)

        # AOI DEFINITELY doesn't overlap with the viewpoints.
        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        geometries = [Polygon([(2, 2), (2, 12), (12, 12), (12, 2), (2, 2)])]
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, aoi_path, WKT, 'GeoJSON')

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            # Valuation parameter defaults to False, so leaving it off here.
            'n_workers': -1,
        }

        with self.assertRaises(ValueError) as cm:
            scenic_quality.execute(args)

        self.assertTrue('found no intersection between' in str(cm.exception))

    def test_no_valuation(self):
        """SQ: model works as expected without valuation."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        # Using weighted viewpoints here to make the visual quality output more
        # interesting.
        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        ScenicQualityTests.create_viewpoints(
            viewpoints_path,
            fields={'RADIUS': ogr.OFTReal,
                    'HEIGHT': ogr.OFTReal,
                    'WEIGHT': ogr.OFTReal},
            attributes=[
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 1.0},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 1.0},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 2.5},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 2.5}])

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        ScenicQualityTests.create_aoi(aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            # Valuation parameter defaults to False, so leaving it off here.
            'n_workers': -1,
        }

        scenic_quality.execute(args)

        # vshed.tif and vshed_qual.tif are still created by the model,
        # vshed_value.tif is not when we are not doing valuation.
        for output_filename, should_exist in (
                ('vshed_value.tif', False),
                ('vshed.tif', True),
                ('vshed_qual.tif', True)):
            full_filepath = os.path.join(
                args['workspace_dir'], 'output', output_filename)
            self.assertEqual(os.path.exists(full_filepath), should_exist)

        # In a non-valuation run, vshed_qual.tif is based on the number of
        # visible structures rather than the valuation, so we need to make sure
        # that the raster has the expected values.
        expected_visual_quality = numpy.array(
            [[1, 1, 1, 1, 4],
             [0, 1, 1, 4, 3],
             [0, 0, 4, 3, 3],
             [0, 3, 3, 4, 3],
             [3, 3, 3, 3, 4]])
        visual_quality_raster = os.path.join(
            args['workspace_dir'], 'output', 'vshed_qual.tif')
        quality_matrix = gdal.OpenEx(
            visual_quality_raster, gdal.OF_RASTER).ReadAsArray()
        numpy.testing.assert_allclose(expected_visual_quality,
                                          quality_matrix,
                                          rtol=0, atol=1e-6)

    def test_invalid_valuation_function(self):
        """SQ: model raises exception with invalid valuation function."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        ScenicQualityTests.create_viewpoints(viewpoints_path)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        ScenicQualityTests.create_aoi(aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'foo',
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'do_valuation': True,
            'valuation_function': 'INVALID FUNCTION',
            'a_coef': 1,
            'b_coef': 0,
            'max_valuation_radius': 10.0,
            'n_workers': -1,
        }

        with self.assertRaises(ValueError):
            scenic_quality.execute(args)

    def test_error_invalid_viewpoints(self):
        """SQ: error when no valid viewpoints.

        This also tests for coverage when using logarithmic valuation on pixels
        with size < 1m.
        """
        from natcap.invest.scenic_quality import scenic_quality

        dem_matrix = numpy.array(
            [[-1, -1, 2, -1, -1],
             [-1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1]], dtype=numpy.int32)

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        pygeoprocessing.numpy_array_to_raster(
            dem_matrix, -1, (0.5, -0.5), (0, 0), WKT, dem_path)

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        geometries = [Point(1.25, -0.5),  # Valid in DEM but outside of AOI.
                      Point(-1.0, -5.0),  # off the edge of DEM.
                      Point(1.25, -1.5)]  # Within AOI, over nodata.
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, viewpoints_path, WKT, 'GeoJSON',
            ogr_geom_type=ogr.wkbPoint)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        geometries_aoi = [
            Polygon([(1, -1), (1, -2.5), (2.5, -2.5), (2.5, -1), (1, -1)])]
        pygeoprocessing.shapely_geometry_to_vector(
            geometries_aoi, aoi_path, WKT, 'GeoJSON')

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'foo',
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'valuation_function': 'logarithmic',
            'a_coef': 1,
            'b_coef': 0,
            'max_valuation_radius': 10.0,
            'n_workers': -1,  # use serial mode to ensure correct exception.
        }
        with self.assertRaises(ValueError) as raised_error:
            scenic_quality.execute(args)

        self.assertTrue('No valid viewpoints found.' in
                        str(raised_error.exception))

    def test_viewshed_field_defaults(self):
        """SQ: run model with default field values."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        ScenicQualityTests.create_viewpoints(viewpoints_path)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        ScenicQualityTests.create_aoi(aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'foo',
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'valuation_function': 'linear',
            'do_valuation': True,
            'a_coef': 1,
            'b_coef': 0,
            'max_valuation_radius': 10.0,
            'n_workers': -1,
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

        value_matrix = pygeoprocessing.raster_to_numpy_array(
            os.path.join(
                args['workspace_dir'], 'output', 'vshed_value_foo.tif'))

        numpy.testing.assert_allclose(
            expected_value, value_matrix, rtol=0, atol=1e-6)

        # verify that the correct number of viewpoints has been tallied.
        vshed_matrix = pygeoprocessing.raster_to_numpy_array(
            os.path.join(args['workspace_dir'], 'output', 'vshed_foo.tif'))

        # Because our B coefficient is 0, the vshed matrix should match the
        # value matrix.
        numpy.testing.assert_allclose(
            expected_value, vshed_matrix, rtol=0, atol=1e-6)

        # Test the visual quality raster.
        expected_visual_quality = numpy.array(
            [[3, 3, 3, 3, 4],
             [0, 3, 3, 4, 3],
             [0, 0, 4, 3, 3],
             [0, 3, 3, 4, 3],
             [3, 3, 3, 3, 4]], dtype=numpy.int32)

        quality_matrix = pygeoprocessing.raster_to_numpy_array(
            os.path.join(
                args['workspace_dir'], 'output', 'vshed_qual_foo.tif'))
        numpy.testing.assert_allclose(expected_visual_quality,
                                      quality_matrix,
                                      rtol=0, atol=1e-6)

    def test_viewshed_with_fields(self):
        """SQ: verify that we can specify viewpoint fields."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        ScenicQualityTests.create_viewpoints(
            viewpoints_path,
            fields={'RADIUS': ogr.OFTReal,
                    'HEIGHT': ogr.OFTReal,
                    'WEIGHT': ogr.OFTReal},
            attributes=[
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 1.0},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 1.0},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 2.5},
                {'RADIUS': 6.0, 'HEIGHT': 1.0, 'WEIGHT': 2.5}])

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        ScenicQualityTests.create_aoi(aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'do_valuation': True,
            'valuation_function': 'linear',
            'a_coef': 0,
            'b_coef': 1,
            'max_valuation_radius': 10.0,
            # n_workers is explicitly excluded here to trigger the model
            # default.
        }

        scenic_quality.execute(args)

        # Verify that the value summation matrix is what we expect it to be.
        # The weight of two of the points makes some sectors more valuable
        expected_value = numpy.array(
            [[4., 2., 0., 2., 14.],
             [0., 2.82842712, 2., 9.89949494, 5.],
             [0., 0., 24., 5., 0.],
             [0., 7.07106781, 5., 14.14213562, 5.],
             [10., 5., 0., 5., 20.]], dtype=numpy.float32)

        value_matrix = pygeoprocessing.raster_to_numpy_array(
            os.path.join(args['workspace_dir'], 'output', 'vshed_value.tif'))

        numpy.testing.assert_allclose(
            expected_value, value_matrix, rtol=0, atol=1e-6)

        # Verify that the sum of the viewsheds (which is weighted) is correct.
        expected_weighted_vshed = numpy.array(
            [[1., 1., 1., 1., 3.5],
             [0., 1., 1., 3.5, 2.5],
             [0., 0., 6., 2.5, 2.5],
             [0., 2.5, 2.5, 5., 2.5],
             [2.5, 2.5, 2.5, 2.5, 5.]], dtype=numpy.float32)
        vshed_raster_path = os.path.join(
            args['workspace_dir'], 'output', 'vshed.tif')
        weighted_vshed_matrix = pygeoprocessing.raster_to_numpy_array(
            vshed_raster_path)
        numpy.testing.assert_allclose(expected_weighted_vshed,
                                      weighted_vshed_matrix,
                                      rtol=0, atol=1e-6)

        # Test the visual quality raster since this run is weighted.
        expected_visual_quality = numpy.array(
            [[1, 1, 0, 1, 4],
             [0, 1, 1, 3, 3],
             [0, 0, 4, 3, 0],
             [0, 3, 3, 4, 3],
             [3, 3, 0, 3, 4]], dtype=numpy.int32)
        visual_quality_raster = os.path.join(
            args['workspace_dir'], 'output', 'vshed_qual.tif')
        quality_matrix = pygeoprocessing.raster_to_numpy_array(
            visual_quality_raster)
        numpy.testing.assert_allclose(expected_visual_quality,
                                      quality_matrix,
                                      rtol=0, atol=1e-6)

    def test_exponential_valuation(self):
        """SQ: verify values on exponential valuation."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        ScenicQualityTests.create_viewpoints(viewpoints_path)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        ScenicQualityTests.create_aoi(aoi_path)

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
            'do_valuation': True,
            'n_workers': -1,
        }

        scenic_quality.execute(args)

        # Verify that the value summation matrix is what we expect it to be.
        # The weight of two of the points makes some sectors more valuable
        expected_value = numpy.array(
            [[0.01831564, 0.13533528, 1., 0.13533528, 0.03663128],
             [0., 0.05910575, 0.13533528, 0.11821149, 0.13533528],
             [0., 0., 0.05494692, 0.13533528, 1.],
             [0., 0.05910575, 0.13533528, 0.11821149, 0.13533528],
             [0.01831564, 0.13533528, 1., 0.13533528, 0.03663128]])

        value_matrix = pygeoprocessing.raster_to_numpy_array(
            os.path.join(args['workspace_dir'], 'output', 'vshed_value.tif'))

        numpy.testing.assert_allclose(expected_value, value_matrix, rtol=0, atol=1e-6)

    def test_logarithmic_valuation(self):
        """SQ: verify values on logarithmic valuation."""
        from natcap.invest.scenic_quality import scenic_quality

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        ScenicQualityTests.create_dem(dem_path)

        viewpoints_path = os.path.join(self.workspace_dir,
                                       'viewpoints.geojson')
        ScenicQualityTests.create_viewpoints(viewpoints_path)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.geojson')
        ScenicQualityTests.create_aoi(aoi_path)

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'aoi_path': aoi_path,
            'structure_path': viewpoints_path,
            'dem_path': dem_path,
            'refraction': 0.13,
            'valuation_function': 'logarithmic',
            'do_valuation': True,
            'a_coef': 1,
            'b_coef': 1,
            'max_valuation_radius': 10.0,
            'n_workers': -1,
        }

        scenic_quality.execute(args)

        # Verify that the value summation matrix is what we expect it to be.
        # The weight of two of the points makes some sectors more valuable
        expected_value = numpy.array(
            [[2.60943791, 2.09861229, 1., 2.09861229, 5.21887582],
             [0., 2.34245405, 2.09861229, 4.68490809, 2.09861229],
             [0., 0., 7.82831374, 2.09861229, 1.],
             [0., 2.34245405, 2.09861229, 4.68490809, 2.09861229],
             [2.60943791, 2.09861229, 1., 2.09861229, 5.21887582]])

        value_matrix = pygeoprocessing.raster_to_numpy_array(
            os.path.join(
                args['workspace_dir'], 'output', 'vshed_value.tif'))

        numpy.testing.assert_allclose(
            expected_value, value_matrix, rtol=0, atol=1e-6)

    def test_visual_quality(self):
        """SQ: verify visual quality calculations."""
        from natcap.invest.scenic_quality import scenic_quality
        visible_structures = numpy.tile(
            numpy.array([3, 0, 0, 0, 6, 7, 8], dtype=numpy.int32), (5, 1))

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
            numpy.array([1, 0, 0, 0, 2, 3, 4], dtype=numpy.int32), (5, 1))

        visual_quality_matrix = pygeoprocessing.raster_to_numpy_array(
            visual_quality_raster)
        numpy.testing.assert_allclose(expected_visual_quality,
                                      visual_quality_matrix,
                                      rtol=0, atol=1e-6)

    def test_visual_quality_large_blocks(self):
        """SQ: verify visual quality on large blocks."""
        # This is a regression test for an issue encountered in the
        # percentiles algorithm.  To exercise the fix, we need to
        # calculate percentiles on a raster that does not fit completely into
        # memory in a single percentile buffer.
        from natcap.invest.scenic_quality import scenic_quality
        shape = (512, 512)
        n_blocks = 5
        visible_structures = numpy.concatenate(
            [numpy.full(shape, n*2) for n in range(n_blocks)])

        n_visible = os.path.join(self.workspace_dir, 'n_visible.tif')
        visual_quality_raster = os.path.join(self.workspace_dir,
                                             'visual_quality.tif')
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create(n_visible, shape[0], shape[1]*n_blocks,
                               1, gdal.GDT_Int32)
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(-1)
        band.WriteArray(visible_structures)
        band = None
        raster = None

        scenic_quality._calculate_visual_quality(n_visible,
                                                 self.workspace_dir,
                                                 visual_quality_raster)

        expected_visual_quality = numpy.concatenate(
            [numpy.full(shape, n) for n in range(n_blocks)])

        visual_quality_matrix = pygeoprocessing.raster_to_numpy_array(
            visual_quality_raster)
        numpy.testing.assert_allclose(expected_visual_quality,
                                      visual_quality_matrix,
                                      rtol=0, atol=1e-6)

    def test_visual_quality_low_count(self):
        """SQ: verify visual quality calculations for low pixel counts."""
        from natcap.invest.scenic_quality import scenic_quality
        visible_structures = numpy.array(
            [[-1, 3, 0, 0, 0, 3, 6, 7]], dtype=numpy.int32)

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

        expected_visual_quality = numpy.array(
            [[255, 2, 0, 0, 0, 2, 3, 4]], dtype=numpy.int32)

        visual_quality_matrix = pygeoprocessing.raster_to_numpy_array(
            visual_quality_raster)
        numpy.testing.assert_allclose(expected_visual_quality,
                                      visual_quality_matrix,
                                      rtol=0, atol=1e-6)

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

        expected_visual_quality = numpy.array(
            [[255, 1, 0, 0, 0, 2, 3, 4]], dtype=numpy.int32)

        visual_quality_matrix = pygeoprocessing.raster_to_numpy_array(
            visual_quality_raster)
        numpy.testing.assert_allclose(expected_visual_quality,
                                      visual_quality_matrix,
                                      rtol=0, atol=1e-6)


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
        from natcap.invest import validation

        validation_errors = scenic_quality.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set([
            'aoi_path',
            'dem_path',
            'refraction',
            'structure_path',
            'workspace_dir',
        ])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_polynomial_required_keys(self):
        """SQ Validate: assert polynomial required keys."""
        from natcap.invest.scenic_quality import scenic_quality
        from natcap.invest import validation

        args = {
            'valuation_function': 'polynomial',
            'do_valuation': True,
        }
        validation_errors = scenic_quality.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)

        self.assertEqual(
            invalid_keys,
            set(['a_coef',
                 'aoi_path',
                 'b_coef',
                 'dem_path',
                 'refraction',
                 'structure_path',
                 'workspace_dir',
                 'valuation_function', ])
        )

    def test_novaluation_required_keys(self):
        """SQ Validate: assert required keys without valuation."""
        from natcap.invest.scenic_quality import scenic_quality
        from natcap.invest import validation
        args = {}
        validation_errors = scenic_quality.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set([
            'aoi_path',
            'dem_path',
            'refraction',
            'structure_path',
            'workspace_dir',
        ])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_bad_values(self):
        """SQ Validate: Assert we can catch various validation errors."""
        from natcap.invest.scenic_quality import scenic_quality
        from natcap.invest import validation

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
            'do_valuation': True
        }

        validation_errors = scenic_quality.validate(args)

        self.assertEqual(len(validation_errors), 6)

        # map single-key errors to their errors.
        single_key_errors = {}
        for keys, error in validation_errors:
            if len(keys) == 1:
                single_key_errors[keys[0]] = error

        self.assertTrue('refraction' not in single_key_errors)
        self.assertEqual(
            single_key_errors['a_coef'],
            validation.MESSAGES['NOT_A_NUMBER'].format(value=args['a_coef']))
        self.assertEqual(
            single_key_errors['dem_path'], validation.MESSAGES['FILE_NOT_FOUND'])
        self.assertEqual(
            single_key_errors['structure_path'], validation.MESSAGES['FILE_NOT_FOUND'])
        self.assertEqual(
            single_key_errors['aoi_path'], validation.MESSAGES['FILE_NOT_FOUND'])
        self.assertEqual(
            single_key_errors['valuation_function'],
            validation.MESSAGES['INVALID_OPTION'].format(
                option_list=[
                    'exponential',
                    'linear',
                    'logarithmic']))

    def test_dem_projected_in_m(self):
        """SQ Validate: the DEM must be projected in meters."""
        from natcap.invest.scenic_quality import scenic_quality
        from natcap.invest import validation

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84 is not projected.
        projection_wkt = srs.ExportToWkt()
        filepath = os.path.join(self.workspace_dir, 'dem.tif')

        pygeoprocessing.numpy_array_to_raster(
            numpy.array([[1]], dtype=numpy.int32), -1, (1, -1), (0, 0),
            projection_wkt, filepath)

        args = {'dem_path': filepath}

        validation_errors = scenic_quality.validate(args, limit_to='dem_path')
        self.assertEqual(
            validation_errors,
            [(['dem_path'], validation.MESSAGES['NOT_PROJECTED'])])


class ViewshedTests(unittest.TestCase):
    """Tests for pygeoprocessing's viewshed."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def create_dem(matrix, filepath, pixel_size=(1, 1), nodata=-1):
        """Create a DEM in WGS84 coordinate system.

        Args:
            matrix (numpy.array): A 2D numpy array of pixel values.
            filepath (string): The filepath where the new raster file will be
                written.
            pixel_size=(1, -1): The pixel size to use for the output raster.
            nodata=-1: The nodata value to use for the output raster.

        Returns:
            ``None``.
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        projection_wkt = srs.ExportToWkt()
        pygeoprocessing.numpy_array_to_raster(
            matrix, nodata, pixel_size, (0, 0), projection_wkt, filepath)

    def test_pixels_not_square(self):
        """SQ Viewshed: exception raised when pixels are not square."""
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
        """SQ Viewshed: exception raised when viewpoint is not over the DEM."""
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
                                               'auxiliary.tif'))

    def test_max_distance(self):
        """SQ Viewshed: setting a max distance limits visibility distance."""
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
                                           'auxiliary.tif'),
                 refraction_coeff=1.0, max_distance=max_dist)

        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)
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
        """SQ Vshed: refractivity partly compensates for earth's curvature."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.array(
            [[2, 1, 1, 2, 1, 1, 1, 1, 1, 50]], dtype=numpy.int32)
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
                                           'auxiliary.tif'),
                 refraction_coeff=0.1)

        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)

        # Because of refractivity calculations (and the size of the pixels),
        # the pixels farther to the right are visible despite being 'hidden'
        # behind the hill at (0,3).  This is due to refractivity.
        expected_visibility = numpy.array(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0, 1]], dtype=numpy.uint8)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_intervening_nodata(self):
        """SQ Viewshed: intervening nodata does not affect visibility."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        nodata = 255
        matrix = numpy.array([[2, 2, nodata, 3]], dtype=numpy.int32)
        viewpoint = (0, 0)

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath,
                                 nodata=nodata)
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')

        viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxiliary.tif'),
                 refraction_coeff=0.0)

        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)

        expected_visibility = numpy.array(
            [[1, 1, 0, 1]], dtype=numpy.uint8)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_nodata_undefined(self):
        """SQ Viewshed: assume a reasonable nodata value if none defined."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        nodata = None  # viewshed assumes an unlikely nodata value.
        matrix = numpy.array([[2, 2, 1, 3]], dtype=numpy.int32)
        viewpoint = (0, 0)

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath,
                                 nodata=nodata)
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')

        viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxiliary.tif'),
                 refraction_coeff=0.0)

        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)

        expected_visibility = numpy.array(
            [[1, 1, 0, 1]], dtype=numpy.uint8)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_block_size_check(self):
        """SQ Viewshed: exception raised when blocks not equal, power of 2."""
        from natcap.invest.scenic_quality.viewshed import viewshed

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        projection_wkt = srs.ExportToWkt()

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir,
                                           'visibility.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.ones((10, 10)), -1, (1, -1), (0, 0), projection_wkt,
            dem_filepath, raster_driver_creation_tuple=(
                'GTIFF', ('TILED=NO', 'BIGTIFF=YES', 'COMPRESS=LZW',
                          'BLOCKXSIZE=20', 'BLOCKYSIZE=40')))

        with self.assertRaises(ValueError):
            viewshed(
                (dem_filepath, 1), (0, 0), visibility_filepath,
                aux_filepath=os.path.join(self.workspace_dir, 'auxiliary.tif')
            )

    def test_view_from_valley(self):
        """SQ Viewshed: test visibility from within a pit."""
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
                                           'auxiliary.tif'))

        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)

        expected_visibility = numpy.zeros(visibility_matrix.shape)
        expected_visibility[matrix != 0] = 1
        expected_visibility[5, 5] = 1
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_tower_view_from_valley(self):
        """SQ Viewshed: test visibility from a 'tower' within a pit."""
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
                                           'auxiliary.tif'))

        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)

        expected_visibility = numpy.ones(visibility_matrix.shape)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_primitive_peak(self):
        """SQ Viewshed: looking down from a peak renders everything visible."""
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
                                           'auxiliary.tif'),
                 refraction_coeff=1.0)

        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)
        numpy.testing.assert_equal(visibility_matrix, numpy.ones(matrix.shape))

    def test_cliff_bottom_half_visibility(self):
        """SQ Viewshed: visibility for a cliff on bottom half of DEM."""
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
            aux_filepath=os.path.join(self.workspace_dir, 'auxiliary.tif')
        )

        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[8:] = 0
        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_cliff_top_half_visibility(self):
        """SQ Viewshed: visibility for a cliff on top half of DEM."""
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
            aux_filepath=os.path.join(self.workspace_dir, 'auxiliary.tif')
        )
        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[:7] = 0
        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_cliff_left_half_visibility(self):
        """SQ Viewshed: visibility for a cliff on left half of DEM."""
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
            aux_filepath=os.path.join(self.workspace_dir, 'auxiliary.tif')
        )
        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[:, :7] = 0
        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_cliff_right_half_visibility(self):
        """SQ Viewshed: visibility for a cliff on right half of DEM."""
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
            aux_filepath=os.path.join(self.workspace_dir, 'auxiliary.tif')
        )
        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[:, 13:] = 0
        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_pillars(self):
        """SQ Viewshed: put a few pillars in a field, can't see behind them."""
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
            aux_filepath=os.path.join(self.workspace_dir, 'auxiliary.tif')
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
             [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=numpy.int32)

        visibility_matrix = pygeoprocessing.raster_to_numpy_array(
            visibility_filepath)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)
