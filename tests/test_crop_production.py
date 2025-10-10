"""Module for Regression Testing the InVEST Crop Production models."""
import unittest
import tempfile
import shutil
import os

import numpy
from osgeo import gdal, ogr, osr
import pandas
import pygeoprocessing
from shapely.geometry import Polygon
from natcap.invest.crop_production_regression import CROP_TO_PATH_TABLES

gdal.UseExceptions()
MODEL_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'crop_production_model', 'model_data')
TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'crop_production_model')
TEST_INPUTS_PATH = os.path.join(TEST_DATA_PATH, 'inputs')
TEST_OUTPUTS_PATH = os.path.join(TEST_DATA_PATH, 'outputs')


def _get_default_args() -> dict[str, str]:
    return {
        'results_suffix': '',
        'landcover_raster_path': os.path.join(
            TEST_INPUTS_PATH, 'landcover.tif'),
        'landcover_to_crop_table_path': os.path.join(
            TEST_INPUTS_PATH, 'landcover_to_crop_table.csv'),
        'aggregate_polygon_path': os.path.join(
            TEST_INPUTS_PATH, 'aggregate_shape.shp'),
        CROP_TO_PATH_TABLES.climate_bin: os.path.join(
            TEST_INPUTS_PATH, 'crop_to_climate_bin.csv'),
        CROP_TO_PATH_TABLES.observed_yield: os.path.join(
            TEST_INPUTS_PATH, 'crop_to_observed_yield.csv'),
        'crop_nutrient_table': os.path.join(
            TEST_INPUTS_PATH, 'crop_nutrient.csv'),
        'n_workers': '-1'
    }


def _get_default_args_percentile() -> dict[str, str]:
    args = _get_default_args()
    args[CROP_TO_PATH_TABLES.percentile_yield] = os.path.join(
        TEST_INPUTS_PATH, 'crop_to_percentile_yield.csv')
    return args


def _get_default_args_regression() -> dict[str, str]:
    args = _get_default_args()
    args[CROP_TO_PATH_TABLES.regression_yield] = os.path.join(
        TEST_INPUTS_PATH, 'crop_to_regression_yield.csv')
    args['fertilization_rate_table_path'] = os.path.join(
        TEST_INPUTS_PATH, 'crop_fertilization_rates.csv')
    return args


def _get_pixels_per_hectare(raster_path):
    """Calculate number of pixels per hectare for a given raster.

    Args:
        raster_path (str): full path to the raster.

    Returns:
        A float representing the number of pixels per hectare.

    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pixel_area = abs(numpy.prod(raster_info['pixel_size']))
    return 10000 / pixel_area


def make_aggregate_vector(path_to_shp):
    """Generate shapefile with two overlapping polygons.

    Args:
        path_to_shp (str): path to store results vector

    Returns:
        None

    """
    # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
    shapely_geometry_list = [
        Polygon([(461151, 4923265-50), (461261+50, 4923265-50),
                 (461261+50, 4923265), (461151, 4923265)]),
        Polygon([(461261, 4923265-35), (461261+60, 4923265-35),
                 (461261+60, 4923265+50), (461261, 4923265+50)])
    ]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    vector_format = "ESRI Shapefile"
    fields = {"id": ogr.OFTReal}
    attribute_list = [
        {"id": 0},
        {"id": 1},
        ]

    pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                               path_to_shp, projection_wkt,
                                               vector_format, fields,
                                               attribute_list)


def make_simple_raster(base_raster_path, array):
    """Create a raster on designated path with arbitrary values.

    Args:
        base_raster_path (str): the raster path for making the new raster.

    Returns:
        None

    """
    # UTM Zone 10N
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    origin = (461251, 4923245)
    pixel_size = (30, 30)
    no_data = -1

    pygeoprocessing.numpy_array_to_raster(
        array, no_data, pixel_size, origin, projection_wkt,
        base_raster_path)


def _create_crop_rasters(output_dir, crop_names, file_suffix):
    """Creates raster files for test setup."""
    _OBSERVED_PRODUCTION_FILE_PATTERN = os.path.join(
        '.', '%s_observed_production%s.tif')
    _CROP_PRODUCTION_FILE_PATTERN = os.path.join(
        '.', '%s_regression_production%s.tif')

    for i, crop in enumerate(crop_names):
        observed_yield_path = os.path.join(
            output_dir,
            _OBSERVED_PRODUCTION_FILE_PATTERN % (crop, file_suffix))
        crop_production_raster_path = os.path.join(
            output_dir,
            _CROP_PRODUCTION_FILE_PATTERN % (crop, file_suffix))

        # Create arbitrary raster arrays
        observed_array = numpy.array([[4, i], [i*3, 4]], dtype=numpy.int16)
        crop_array = numpy.array([[i, 1], [i*2, 3]], dtype=numpy.int16)

        make_simple_raster(observed_yield_path, observed_array)
        make_simple_raster(crop_production_raster_path, crop_array)


def _create_crop_pctl_rasters(output_dir, crop_names, file_suffix, pctls):
    """Creates crop percentile raster files for test setup."""
    _OBSERVED_PRODUCTION_FILE_PATTERN = os.path.join(
        '.', '%s_observed_production%s.tif')
    _CROP_PRODUCTION_FILE_PATTERN = os.path.join(
        '.', '%s_%s_production%s.tif')

    for i, crop in enumerate(crop_names):
        observed_yield_path = os.path.join(
                output_dir,
                _OBSERVED_PRODUCTION_FILE_PATTERN % (crop, file_suffix))
        # Create arbitrary raster arrays
        observed_array = numpy.array(
            [[i, 1], [i*2, 3]], dtype=numpy.int16)
        make_simple_raster(observed_yield_path, observed_array)

        for pctl in pctls:
            crop_production_raster_path = os.path.join(
                output_dir,
                _CROP_PRODUCTION_FILE_PATTERN % (crop, pctl, file_suffix))

            crop_array = numpy.array(
                [[i, 1], [i*3, 4]], dtype=numpy.int16) * float(pctl[-2:])/100

            make_simple_raster(crop_production_raster_path, crop_array)


class CropProductionTests(unittest.TestCase):
    """Tests for the Crop Production model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_crop_production_percentile(self):
        """Crop Production Percentile: validate results."""
        from natcap.invest import crop_production_percentile

        args = _get_default_args_percentile()
        args['workspace_dir'] = self.workspace_dir

        crop_production_percentile.execute(args)

        agg_result_table_path = os.path.join(
            args['workspace_dir'], 'aggregate_results.csv')
        expected_agg_result_table_path = os.path.join(
            TEST_OUTPUTS_PATH, 'expected_aggregate_results.csv')
        expected_agg_result_table = pandas.read_csv(
            expected_agg_result_table_path)
        agg_result_table = pandas.read_csv(
            agg_result_table_path)
        pandas.testing.assert_frame_equal(
            expected_agg_result_table, agg_result_table,
            check_dtype=False, check_exact=False)

        expected_result_table = pandas.read_csv(
            os.path.join(TEST_OUTPUTS_PATH, 'expected_result_table.csv')
        )
        result_table = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'result_table.csv'))
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

        # Check raster outputs to make sure values are in Mg/ha.
        # Raster sum is (Mg•px)/(ha•yr).
        # Result table reports totals in Mg/yr.
        # To convert from Mg/yr to (Mg•px)/(ha•yr), multiply by px/ha.
        expected_raster_sums = {}
        for (index, crop) in [(0, 'barley'), (1, 'soybean'), (2, 'wheat')]:
            filename = crop + '_observed_production.tif'
            pixels_per_hectare = _get_pixels_per_hectare(
                os.path.join(args['workspace_dir'], filename))
            expected_raster_sums[filename] = (
                expected_result_table.loc[index]['production_observed']
                * pixels_per_hectare)
            for percentile in ['25', '50', '75', '95']:
                filename = (
                    crop + '_yield_' + percentile + 'th_production.tif')
                col_name = 'production_' + percentile + 'th'
                pixels_per_hectare = _get_pixels_per_hectare(
                    os.path.join(args['workspace_dir'], filename))
                expected_raster_sums[filename] = (
                    expected_result_table.loc[index][col_name]
                    * pixels_per_hectare)

        for filename in expected_raster_sums:
            raster_path = os.path.join(args['workspace_dir'], filename)
            raster_info = pygeoprocessing.get_raster_info(raster_path)
            nodata = raster_info['nodata'][0]
            raster_sum = 0.0
            for _, block in pygeoprocessing.iterblocks((raster_path, 1)):
                raster_sum += numpy.sum(
                    block[~pygeoprocessing.array_equals_nodata(
                            block, nodata)], dtype=numpy.float32)
            expected_sum = expected_raster_sums[filename]
            numpy.testing.assert_allclose(raster_sum, expected_sum,
                                          rtol=0, atol=0.1)

    def test_crop_production_percentile_no_nodata(self):
        """Crop Production Percentile: landcover raster without defined nodata.

        Test with a landcover raster input that has no nodata value
        defined.
        """
        from natcap.invest import crop_production_percentile

        args = _get_default_args_percentile()
        args['workspace_dir'] = self.workspace_dir

        # Create a raster based on the test data geotransform, but smaller and
        # with no nodata value defined.
        base_lulc_info = pygeoprocessing.get_raster_info(
            args['landcover_raster_path'])
        base_geotransform = base_lulc_info['geotransform']
        origin_x = base_geotransform[0]
        origin_y = base_geotransform[3]

        n = 9
        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(base_lulc_info['projection_wkt'])
        new_raster.SetGeoTransform([origin_x, 1.0, 0.0, origin_y, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        array.fill(20)  # 20 is present in the landcover_to_crop_table
        new_band.WriteArray(array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None
        args['landcover_raster_path'] = raster_path

        crop_production_percentile.execute(args)

        result_table_path = os.path.join(
            args['workspace_dir'], 'result_table.csv')
        expected_result_table_path = os.path.join(
            TEST_OUTPUTS_PATH, 'expected_result_table_no_nodata.csv')
        expected_result_table = pandas.read_csv(
            expected_result_table_path)
        result_table = pandas.read_csv(
            result_table_path)
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

    def test_crop_production_percentile_invalid_crop_name(self):
        """Crop Production Percentile: invalid user-specified crop name."""
        from natcap.invest import crop_production_percentile

        args = _get_default_args_percentile()
        args['workspace_dir'] = self.workspace_dir
        args['landcover_to_crop_table_path'] = os.path.join(
                TEST_INPUTS_PATH, 'landcover_to_invalid_crop_percentile.csv')

        with self.assertRaises(ValueError) as context:
            crop_production_percentile.execute(args)
        self.assertTrue("The following crop names were provided in "
                        f"{args['landcover_to_crop_table_path']} but "
                        "are not supported by the model: {'durian'}"
                        in str(context.exception))

    def test_crop_production_percentile_missing_climate_bin(self):
        """Crop Production Percentile: missing climate bin path."""
        from natcap.invest import crop_production_percentile

        args = _get_default_args_percentile()
        args['workspace_dir'] = self.workspace_dir
        args[CROP_TO_PATH_TABLES.climate_bin] = os.path.join(
            TEST_INPUTS_PATH, 'crop_to_missing_climate_bin.csv')

        with self.assertRaises(ValueError) as context:
            crop_production_percentile.execute(args)
        self.assertTrue('No climate bin raster path could be found for wheat'
                        in str(context.exception))

    def test_crop_production_regression_invalid_crop_name(self):
        """Crop Production Regression: invalid user-specified crop name."""
        from natcap.invest import crop_production_regression

        args = _get_default_args_regression()
        args['workspace_dir'] = self.workspace_dir
        args['landcover_to_crop_table_path'] = os.path.join(
                TEST_INPUTS_PATH, 'landcover_to_invalid_crop_regression.csv')

        with self.assertRaises(ValueError) as context:
            crop_production_regression.execute(args)
        self.assertTrue("The following crop names were provided in "
                        f"{args['landcover_to_crop_table_path']} but "
                        "are not supported by the model: {'avocado'}"
                        in str(context.exception))

    def test_crop_production_regression_missing_climate_bin(self):
        """Crop Production Regression: missing climate bin path."""
        from natcap.invest import crop_production_regression

        args = _get_default_args_regression()
        args['workspace_dir'] = self.workspace_dir
        args[CROP_TO_PATH_TABLES.climate_bin] = os.path.join(
            TEST_INPUTS_PATH, 'crop_to_missing_climate_bin.csv')

        with self.assertRaises(ValueError) as context:
            crop_production_regression.execute(args)
        self.assertTrue('No climate bin raster path could be found for wheat'
                        in str(context.exception))

    def test_crop_production_regression(self):
        """Crop Production Regression: validate results."""
        from natcap.invest import crop_production_regression

        args = _get_default_args_regression()
        args['workspace_dir'] = self.workspace_dir

        crop_production_regression.execute(args)

        expected_agg_result_table = pandas.read_csv(
            os.path.join(TEST_OUTPUTS_PATH,
                         'expected_regression_aggregate_results.csv'))
        agg_result_table = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'aggregate_results.csv'))
        pandas.testing.assert_frame_equal(
            expected_agg_result_table, agg_result_table,
            check_dtype=False, check_exact=False)

        result_table_path = os.path.join(
            args['workspace_dir'], 'result_table.csv')
        expected_result_table_path = os.path.join(
            TEST_OUTPUTS_PATH, 'expected_regression_result_table.csv')
        expected_result_table = pandas.read_csv(
            expected_result_table_path)
        result_table = pandas.read_csv(
            result_table_path)
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

        # Check raster outputs to make sure values are in Mg/ha.
        # Raster sum is (Mg•px)/(ha•yr).
        # Result table reports totals in Mg/yr.
        # To convert from Mg/yr to (Mg•px)/(ha•yr), multiply by px/ha.
        expected_raster_sums = {}
        for (index, crop) in [(0, 'barley'), (1, 'soybean'), (2, 'wheat')]:
            filename = crop + '_observed_production.tif'
            pixels_per_hectare = _get_pixels_per_hectare(
                os.path.join(args['workspace_dir'], filename))
            expected_raster_sums[filename] = (
                expected_result_table.loc[index]['production_observed']
                * pixels_per_hectare)
            filename = crop + '_regression_production.tif'
            pixels_per_hectare = _get_pixels_per_hectare(
                os.path.join(args['workspace_dir'], filename))
            expected_raster_sums[filename] = (
                expected_result_table.loc[index]['production_modeled']
                * pixels_per_hectare)

        for filename in expected_raster_sums:
            raster_path = os.path.join(args['workspace_dir'], filename)
            raster_info = pygeoprocessing.get_raster_info(raster_path)
            nodata = raster_info['nodata'][0]
            raster_sum = 0.0
            for _, block in pygeoprocessing.iterblocks((raster_path, 1)):
                raster_sum += numpy.sum(
                    block[~pygeoprocessing.array_equals_nodata(
                            block, nodata)], dtype=numpy.float32)
            expected_sum = expected_raster_sums[filename]
            numpy.testing.assert_allclose(raster_sum, expected_sum,
                                          rtol=0, atol=0.001)

    def test_crop_production_regression_no_nodata(self):
        """Crop Production Regression: landcover raster without defined nodata.

        Test with a landcover raster input that has no nodata value
        defined.
        """
        from natcap.invest import crop_production_regression

        args = _get_default_args_regression()
        args['workspace_dir'] = self.workspace_dir

        # Create a raster based on the test data geotransform, but smaller and
        # with no nodata value defined.
        base_lulc_info = pygeoprocessing.get_raster_info(
            args['landcover_raster_path'])
        base_geotransform = base_lulc_info['geotransform']
        origin_x = base_geotransform[0]
        origin_y = base_geotransform[3]

        n = 9
        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(base_lulc_info['projection_wkt'])
        new_raster.SetGeoTransform([origin_x, 1.0, 0.0, origin_y, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        array.fill(20)  # 20 is present in the landcover_to_crop_table
        new_band.WriteArray(array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None
        args['landcover_raster_path'] = raster_path

        crop_production_regression.execute(args)

        expected_result_table = pandas.read_csv(os.path.join(
            TEST_OUTPUTS_PATH, 'expected_regression_result_table_no_nodata.csv'))
        result_table = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'result_table.csv'))
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

    def test_x_yield_op(self):
        """Test `_x_yield_op"""
        from natcap.invest.crop_production_regression import _x_yield_op

        # make fake data
        y_max = numpy.array([[-1, 3, 2], [4, 5, 3]])
        b_x = numpy.array([[4, 3, 2], [2, 0, 3]])
        c_x = numpy.array([[4, 1, 2], [3, 0, 3]])
        lulc_array = numpy.array([[3, 3, 2], [3, -1, 3]])
        fert_rate = 0.6
        crop_lucode = 3

        actual_result = _x_yield_op(y_max, b_x, c_x, lulc_array, fert_rate,
                                    crop_lucode)
        expected_result = numpy.array([[-1, -1.9393047, -1],
                                       [2.6776089, -1, 1.51231]])

        numpy.testing.assert_allclose(actual_result, expected_result)

    def test_zero_observed_yield_op(self):
        """Test `_zero_observed_yield_op`"""
        from natcap.invest.crop_production_regression import \
            _zero_observed_yield_op

        # make fake data
        observed_yield_array = numpy.array([[0, 1, -1], [5, 6, -1]])
        observed_yield_nodata = -1

        actual_result = _zero_observed_yield_op(observed_yield_array,
                                                observed_yield_nodata)

        expected_result = numpy.array([[0, 1, 0], [5, 6, 0]])

        numpy.testing.assert_allclose(actual_result, expected_result)

    def test_mask_observed_yield_op(self):
        """Test `_mask_observed_yield_op`"""
        from natcap.invest.crop_production_regression import \
            _mask_observed_yield_op

        # make fake data
        lulc_array = numpy.array([[3, 5, -9999], [3, 3, -1]])
        observed_yield_array = numpy.array([[-1, 5, 4], [8, -9999, 91]])
        observed_yield_nodata = -1
        # note: this observed_yield_nodata value becomes the nodata value in
        # the output array but the values in the observed_yield_array with
        # this value are NOT treated as no data within this function

        landcover_nodata = -9999
        crop_lucode = 3

        actual_result = _mask_observed_yield_op(
            lulc_array, observed_yield_array, observed_yield_nodata,
            landcover_nodata, crop_lucode)

        expected_result = numpy.array([[-1, 0, -1], [8, -9999, 0]])

        numpy.testing.assert_allclose(actual_result, expected_result)

    def test_tabulate_regression_results(self):
        """Test `tabulate_regression_results`"""
        from natcap.invest.crop_production_regression import \
            tabulate_regression_results, MODEL_SPEC
        from natcap.invest.file_registry import FileRegistry
        from crop_production.data_helpers import sample_nutrient_df
        from crop_production.data_helpers import tabulate_regr_results_table

        nutrient_df = sample_nutrient_df()

        pixel_area_ha = 10
        workspace_dir = self.workspace_dir
        output_dir = os.path.join(workspace_dir, "OUTPUT")
        os.makedirs(output_dir, exist_ok=True)

        landcover_raster_path = os.path.join(workspace_dir, "landcover.tif")
        landcover_nodata = -1
        make_simple_raster(landcover_raster_path,
                           numpy.array([[2, 1], [2, 3]], dtype=numpy.int16))

        file_suffix = "v1"
        target_table_path = os.path.join(workspace_dir, "output_table.csv")
        crop_names = ["corn", "soybean"]
        file_registry = FileRegistry(MODEL_SPEC.outputs, output_dir, file_suffix)

        _create_crop_rasters(output_dir, crop_names, file_suffix)

        tabulate_regression_results(
            nutrient_df, crop_names, pixel_area_ha,
            landcover_raster_path, landcover_nodata,
            file_registry, target_table_path
        )

        # Read only the first 2 crop's data (skipping total area)
        actual_result_table = pandas.read_csv(target_table_path, nrows=2,
                                              header=0)
        expected_result_table = tabulate_regr_results_table()

        # Compare expected vs actual
        pandas.testing.assert_frame_equal(actual_result_table,
                                          expected_result_table)

    def test_aggregate_regression_results_to_polygons(self):
        """Test `aggregate_regression_results_to_polygons`"""
        from natcap.invest.crop_production_regression import \
            aggregate_regression_results_to_polygons, MODEL_SPEC
        from natcap.invest.file_registry import FileRegistry
        from crop_production.data_helpers import sample_nutrient_df
        from crop_production.data_helpers import \
            aggregate_regr_polygons_table

        workspace = self.workspace_dir

        base_aggregate_vector_path = os.path.join(workspace,
                                                  "agg_vector.shp")
        make_aggregate_vector(base_aggregate_vector_path)

        target_aggregate_vector_path = os.path.join(workspace,
                                                    "agg_vector_prj.shp")

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(26910)  # EPSG:4326 for WGS84
        landcover_raster_projection = spatial_ref.ExportToWkt()

        crop_names = ['corn', 'soybean']
        nutrient_df = sample_nutrient_df()
        output_dir = os.path.join(workspace, "OUTPUT")
        os.makedirs(output_dir, exist_ok=True)
        file_suffix = 'test'
        file_registry = FileRegistry(MODEL_SPEC.outputs, output_dir, file_suffix)

        _AGGREGATE_TABLE_FILE_PATTERN = os.path.join(
            '.', 'aggregate_results%s.csv')

        aggregate_table_path = os.path.join(
            output_dir, _AGGREGATE_TABLE_FILE_PATTERN % file_suffix)

        pixel_area_ha = 10

        _create_crop_rasters(output_dir, crop_names, file_suffix)

        aggregate_regression_results_to_polygons(
            base_aggregate_vector_path, target_aggregate_vector_path,
            aggregate_table_path, landcover_raster_projection, crop_names,
            nutrient_df, pixel_area_ha, file_registry)

        actual_aggregate_table = pandas.read_csv(aggregate_table_path,
                                                 dtype=float)

        expected_aggregate_table = aggregate_regr_polygons_table()

        pandas.testing.assert_frame_equal(
            actual_aggregate_table, expected_aggregate_table)

    def test_aggregate_to_polygons(self):
        """Test `aggregate_to_polygons`"""
        from natcap.invest.crop_production_percentile import \
            aggregate_to_polygons, MODEL_SPEC
        from natcap.invest.file_registry import FileRegistry
        from crop_production.data_helpers import sample_nutrient_df
        from crop_production.data_helpers import aggregate_pctl_polygons_table

        workspace = self.workspace_dir

        base_aggregate_vector_path = os.path.join(workspace,
                                                  "agg_vector.shp")
        make_aggregate_vector(base_aggregate_vector_path)

        target_aggregate_vector_path = os.path.join(workspace,
                                                    "agg_vector_prj.shp")

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(26910)
        landcover_raster_projection = spatial_ref.ExportToWkt()

        crop_names = ['corn', 'soybean']
        nutrient_df = sample_nutrient_df()
        yield_percentile_headers = ['25', '50', '75']
        pixel_area_ha = 1
        output_dir = os.path.join(workspace, "OUTPUT")
        os.makedirs(output_dir, exist_ok=True)
        file_suffix = 'v1'
        target_aggregate_table_path = os.path.join(output_dir,
                                                   "results.csv")
        file_registry = FileRegistry(MODEL_SPEC.outputs, output_dir, file_suffix)

        _create_crop_pctl_rasters(output_dir, crop_names, file_suffix,
                                  yield_percentile_headers)

        aggregate_to_polygons(
            base_aggregate_vector_path, target_aggregate_vector_path,
            landcover_raster_projection, crop_names, nutrient_df,
            yield_percentile_headers, pixel_area_ha, file_registry,
            target_aggregate_table_path)

        actual_aggregate_pctl_table = pandas.read_csv(
            target_aggregate_table_path, dtype=float)
        expected_aggregate_pctl_table = aggregate_pctl_polygons_table()

        pandas.testing.assert_frame_equal(
            actual_aggregate_pctl_table, expected_aggregate_pctl_table)

    def test_tabulate_percentile_results(self):
        """Test `tabulate_results"""
        from natcap.invest.crop_production_percentile import \
            tabulate_results, MODEL_SPEC
        from natcap.invest.file_registry import FileRegistry
        from crop_production.data_helpers import sample_nutrient_df
        from crop_production.data_helpers import tabulate_pctl_results_table

        nutrient_df = sample_nutrient_df()
        output_dir = os.path.join(self.workspace_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        yield_percentile_headers = ['yield_25', 'yield_50', 'yield_75']
        crop_names = ['corn', 'soybean']
        pixel_area_ha = 1
        landcover_raster_path = os.path.join(self.workspace_dir,
                                             "landcover.tif")
        landcover_nodata = -1

        make_simple_raster(landcover_raster_path,
                           numpy.array([[1, 4], [2, 2]], dtype=numpy.int16))
        file_suffix = 'test'
        file_registry = FileRegistry(MODEL_SPEC.outputs, output_dir, file_suffix)
        target_table_path = os.path.join(output_dir, "result_table.csv")
        _create_crop_pctl_rasters(output_dir, crop_names, file_suffix,
                                  yield_percentile_headers)
        tabulate_results(nutrient_df, yield_percentile_headers,
                         crop_names, pixel_area_ha,
                         landcover_raster_path, landcover_nodata,
                         file_registry, target_table_path)

        actual_table = pandas.read_csv(target_table_path, nrows=2)
        expected_table = tabulate_pctl_results_table()

        pandas.testing.assert_frame_equal(actual_table, expected_table,
                                          check_dtype=False)


class CropValidationTests(unittest.TestCase):
    """Tests for the Crop Production models' MODEL_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'landcover_raster_path',
            'landcover_to_crop_table_path',
            CROP_TO_PATH_TABLES.climate_bin,
            CROP_TO_PATH_TABLES.observed_yield,
            'crop_nutrient_table',
        ]

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys_percentile(self):
        """Crop Percentile Validate: assert missing required keys."""
        from natcap.invest import crop_production_percentile
        from natcap.invest import validation

        # empty args dict.
        validation_errors = crop_production_percentile.validate({})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys
            + [CROP_TO_PATH_TABLES.percentile_yield])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_regression(self):
        """Crop Regression Validate: assert missing required keys."""
        from natcap.invest import crop_production_regression
        from natcap.invest import validation

        # empty args dict.
        validation_errors = crop_production_regression.validate({})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys
            + [CROP_TO_PATH_TABLES.regression_yield]
            + ['fertilization_rate_table_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)
