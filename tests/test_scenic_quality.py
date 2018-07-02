"""Module for Regression Testing the InVEST Scenic Quality module."""
import unittest
import tempfile
import shutil
import os
import csv

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from pygeoprocessing.testing import sampledata
from shapely.geometry import Polygon
import numpy

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'scenic_quality')


class ScenicQualityTests(unittest.TestCase):
    """Tests for the InVEST Scenic Quality model."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)


class ScenicQualityValidationTests(unittest.TestCase):
    """Tests for Scenic Quality validation."""

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
        except Exception:
            # any other error, fail!
            self.fail('An unexpected error was raised!')

    def test_polynomial_required_keys(self):
        """SQ Validate: assert polynomial required keys"""
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
                'c_coef',
                'd_coef',
                'dem_path',
                'max_valuation_radius',
                'refraction',
                'structure_path',
                'workspace_dir',
                # This list doesn't contain key ``valuation_function`` because
                # the key was provided in args.
            ]
            self.assertEqual(missing_keys, expected_missing_keys)
        except Exception:
            # any other error, fail!
            self.fail('An unexpected error was raised!')

    def test_bad_values(self):
        """SQ Validate: Assert we can catch various validation errors."""
        from natcap.invest.scenic_quality import scenic_quality

        # AOI path is missing
        args = {
            'workspace_dir': 'workspace not validated',
            'aoi_path': '',  # covers required key, missing value.
            'a_coef': 'foo',  # not a number
            'b_coef': 1,  # key still needs to be here
            'dem_path': 'not/a/path',  # not a raster
            'refraction': "0.13",
            'max_valuation_radius': None,  # covers missing value.
            'structure_path': 'vector/missing',
            'valuation_function': 'bad function',
        }

        validation_errors = scenic_quality.validate(args)

        self.assertEqual(len(validation_errors), 5)

        # map single-key errors to their errors.
        single_key_errors = {}
        for keys, error in validation_errors:
            if len(keys) == 1:
                single_key_errors[keys[0]] = error

        self.assertTrue('refraction' not in single_key_errors)
        self.assertEqual(single_key_errors['a_coef'], 'Must be a number')
        self.assertEqual(single_key_errors['dem_path'], 'Must be a raster')
        self.assertEqual(single_key_errors['structure_path'], 'Must be a vector')
        self.assertEqual(single_key_errors['aoi_path'], 'Must be a vector')
        self.assertEqual(single_key_errors['valuation_function'],
                         'Invalid function')


class ScenicQualityUnitTests(unittest.TestCase):
    """Unit tests for Scenic Quality Model."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_add_percent_overlap(self):
        """SQ: test 'add_percent_overlap' function."""
        from natcap.invest.scenic_quality import scenic_quality

        temp_dir = self.workspace_dir
        srs = sampledata.SRS_WILLAMETTE

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]
        fields = {'myid': 'int'}
        attrs = [{'myid': 1}, {'myid': 2}]

        # Create geometry for the polygons
        geom = [
            Polygon(
                [(pos_x, pos_y), (pos_x + 60, pos_y), (pos_x + 60, pos_y - 60),
                 (pos_x, pos_y - 60), (pos_x, pos_y)]),
            Polygon(
                [(pos_x + 80, pos_y - 80), (pos_x + 140, pos_y - 80),
                 (pos_x + 140, pos_y - 140), (pos_x + 80, pos_y - 140),
                 (pos_x + 80, pos_y - 80)])]

        shape_path = os.path.join(temp_dir, 'shape.shp')
        # Create the point shapefile
        shape_path = pygeoprocessing.testing.create_vector_on_disk(
            geom, srs.projection, fields, attrs,
            vector_format='ESRI Shapefile', filename=shape_path)

        pixel_counts = {1: 2, 2: 1}
        pixel_size = 30

        scenic_quality.add_percent_overlap(
            shape_path, 'myid', '%_overlap', pixel_counts, pixel_size)

        exp_result = {1: 50.0, 2: 25.0}

        shape = ogr.Open(shape_path)
        layer = shape.GetLayer()
        for feat in layer:
            myid = feat.GetFieldAsInteger('myid')
            perc_overlap = feat.GetFieldAsDouble('%_overlap')
            self.assertEqual(exp_result[myid], perc_overlap)


class ScenicQualityRegressionTests(unittest.TestCase):
    """Regression Tests for Scenic Quality Model."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across regression tests."""
        args = {
            'workspace_dir': workspace_dir,
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'ScenicQuality', 'Input', 'AOI_WCVI.shp'),
            'structure_path': os.path.join(
                SAMPLE_DATA, 'ScenicQuality', 'Input', 'AquaWEM_points.shp'),
            'keep_feat_viewsheds': 'No',
            'keep_val_viewsheds': 'No',
            'dem_path': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 'claybark_dem'),
            'valuation_function': 0,
            'max_valuation_radius': 8000
        }

        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_scenic_quality_baseline(self):
        """SQ: testing with no optional features and polynomial val function."""
        from natcap.invest.scenic_quality import scenic_quality

        args = ScenicQualityTests.generate_base_args(self.workspace_dir)

        # Check with Rob about good default values here:
        args['poly_a_coef'] = 1.0
        args['poly_b_coef'] = 1.0
        args['poly_c_coef'] = 1.0
        args['poly_d_coef'] = 1.0

        scenic_quality.execute(args)

        raster_results = [
            'vshed.tif', 'viewshed_counts.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'baseline', raster_path),
                1e-9)

    def test_scenic_quality_refraction(self):
        """SQ: testing with refraction coeff and polynomial val function."""
        from natcap.invest.scenic_quality import scenic_quality

        args = ScenicQualityTests.generate_base_args(self.workspace_dir)

        args['refraction'] = 0.13
        # Check with Rob about good default values here:
        args['poly_a_coef'] = 1.0
        args['poly_b_coef'] = 1.0
        args['poly_c_coef'] = 1.0
        args['poly_d_coef'] = 1.0

        scenic_quality.execute(args)

        raster_results = [
            'vshed.tif', 'viewshed_counts.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'refraction', raster_path),
                1e-9)

    def test_scenic_quality_population(self):
        """SQ: testing affected / unaffected population counts."""
        from natcap.invest.scenic_quality import scenic_quality

        args = ScenicQualityTests.generate_base_args(self.workspace_dir)

        args['population_path'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Population', 'global_pop')
        # Check with Rob about good default values here:
        args['poly_a_coef'] = 1.0
        args['poly_b_coef'] = 1.0
        args['poly_c_coef'] = 1.0
        args['poly_d_coef'] = 1.0

        scenic_quality.execute(args)

        raster_results = [
            'vshed.tif', 'viewshed_counts.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'population', raster_path),
                1e-9)

        # Test population HTML output

    def test_scenic_quality_overlap(self):
        """SQ: testing percent overlap of polygons on view pixels."""
        from natcap.invest.scenic_quality import scenic_quality

        args = ScenicQualityTests.generate_base_args(self.workspace_dir)

        args['overlap_path'] = os.path.join(
            SAMPLE_DATA, 'ScenicQuality', 'Input', 'BC_parks.shp')
        # Check with Rob about good default values here:
        args['poly_a_coef'] = 1.0
        args['poly_b_coef'] = 1.0
        args['poly_c_coef'] = 1.0
        args['poly_d_coef'] = 1.0

        scenic_quality.execute(args)

        raster_results = [
            'vshed.tif', 'viewshed_counts.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'overlap', raster_path),
                1e-9)

        vector_results = ['vp_overlap.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'overlap', vector_path),
                1e-9)

    def test_scenic_quality_log_valuation(self):
        """SQ: testing logarithmic valuation function."""
        from natcap.invest.scenic_quality import scenic_quality

        args = ScenicQualityTests.generate_base_args(self.workspace_dir)

        args['valuation_function'] = 1
        # Check with Rob about good default values here:
        args['log_a_coef'] = 1.0
        args['log_b_coef'] = 1.0

        scenic_quality.execute(args)

        raster_results = ['vshed.tif', 'viewshed_counts.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'logarithmic', raster_path),
                1e-9)

    def test_scenic_quality_exponential_valuation(self):
        """SQ: testing exponential decay valuation function."""
        from natcap.invest.scenic_quality import scenic_quality

        args = ScenicQualityTests.generate_base_args(self.workspace_dir)

        args['valuation_function'] = 2
        # Check with Rob about good default values here:
        args['exp_a_coef'] = 1.0
        args['exp_b_coef'] = 1.0

        scenic_quality.execute(args)

        raster_results = [
            'vshed.tif', 'viewshed_counts.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'exponential', raster_path),
                1e-9)


class ViewshedTests(unittest.TestCase):
    """Tests for pygeoprocessing's viewshed."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def create_dem(matrix, filepath, pixel_size=(1, -1)):
        """Create a DEM in WGS84 coordinate system.

        Parameters:
            matrix (numpy.array): A 2D numpy array of pixel values.
            filepath (string): The filepath where the new raster file will be
                written.
            pixel_size=(1, -1): The pixel size to use for the output raster.

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
            nodata=-1,
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

        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
        with self.assertRaises(AssertionError):
            viewshed((dem_filepath, 1), viewpoint, visibility_filepath)

    def test_viewpoint_not_overlapping_dem(self):
        """Viewshed: exception raised when viewpoint is not over the DEM."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.ones((20, 20))
        viewpoint = (-10, -10)
        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)

        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')

        with self.assertRaises(ValueError):
            viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                     aux_filepath=os.path.join(self.workspace_dir,
                                               'auxulliary.tif'))

    def test_max_distance(self):
        """Viewshed: setting a max distance limits visibility distance."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.ones((10, 10))
        viewpoint = (9, 9)
        max_dist = 4

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)

        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')

        viewshed((dem_filepath, 1), viewpoint, visibility_filepath,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'),
                 refraction_coeff=1.0, max_distance=max_dist)

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        expected_visibility = numpy.zeros(matrix.shape)

        expected_visibility = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]],
                                          dtype=numpy.uint8)
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_refractivity(self):
        """Viewshed: refractivity partly compensates for earth's curvature."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        # TODO: verify this height.
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
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')

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

    def test_block_size_check(self):
        """Viewshed: exception raised when blocks not equal, power of 2."""
        from natcap.invest.scenic_quality.viewshed import viewshed

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
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
        matrix[5:8,5:8] = 2
        matrix[4:7,4:7] = 1
        matrix[5,5] = 0

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed((dem_filepath, 1), (5, 5), visibility_filepath,
                 refraction_coeff=1.0,
                 aux_filepath=os.path.join(self.workspace_dir,
                                           'auxulliary.tif'))

        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()

        expected_visibility = numpy.zeros(visibility_matrix.shape)
        expected_visibility[matrix!=0] = 1
        expected_visibility[5,5] = 1
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_tower_view_from_valley(self):
        """Viewshed: test visibility from a 'tower' within a pit."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.zeros((9, 9))
        matrix[5:8,5:8] = 2
        matrix[4:7,4:7] = 1
        matrix[5,5] = 0

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
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
        matrix[4:7,4:7] = 1
        matrix[5,5] = 2

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed((dem_filepath, 1), (5, 5),
                                 visibility_filepath,
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
        matrix = numpy.empty((20,20))
        matrix.fill(2)
        matrix[7:] = 10  # cliff at row 7
        viewpoint = (5, 10)
        matrix[viewpoint] = 5  # viewpoint

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
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
        matrix = numpy.empty((20,20))
        matrix.fill(2)
        matrix[:8] = 10  # cliff at row 8
        viewpoint = (10, 10)
        matrix[viewpoint] = 5  # viewpoint

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
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
        matrix = numpy.empty((20,20))
        matrix.fill(2)
        matrix[:,:8] = 10  # cliff at column 8
        viewpoint = (10, 10)
        matrix[viewpoint] = 5  # viewpoint

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed(
            dem_raster_path_band=(dem_filepath, 1),
            viewpoint=viewpoint,
            visibility_filepath=visibility_filepath,
            aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
        )
        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[:,:7] = 0
        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_cliff_right_half_visibility(self):
        """Viewshed: visibility for a cliff on right half of DEM."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.empty((20,20))
        matrix.fill(2)
        matrix[:,12:] = 10  # cliff at column 8
        viewpoint = (10, 10)
        matrix[viewpoint] = 5  # viewpoint

        dem_filepath = os.path.join(self.workspace_dir, 'dem.tif')
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed(
            dem_raster_path_band=(dem_filepath, 1),
            viewpoint=viewpoint,
            visibility_filepath=visibility_filepath,
            aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
        )
        expected_visibility = numpy.ones(matrix.shape)
        expected_visibility[:,13:] = 0
        visibility_raster = gdal.OpenEx(visibility_filepath)
        visibility_band = visibility_raster.GetRasterBand(1)
        visibility_matrix = visibility_band.ReadAsArray()
        numpy.testing.assert_equal(visibility_matrix, expected_visibility)

    def test_pillars(self):
        """Viewshed: put a few pillars in a field, can't see behind them."""
        from natcap.invest.scenic_quality.viewshed import viewshed
        matrix = numpy.empty((20,20))
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
        visibility_filepath = os.path.join(self.workspace_dir, 'visibility.tif')
        ViewshedTests.create_dem(matrix, dem_filepath)
        viewshed(
            dem_raster_path_band=(dem_filepath, 1),
            viewpoint=viewpoint,
            visibility_filepath=visibility_filepath,
            aux_filepath=os.path.join(self.workspace_dir, 'auxulliary.tif')
        )

        expected_visibility= numpy.array(
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
