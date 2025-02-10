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

gdal.UseExceptions()
MODEL_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'crop_production_model', 'model_data')
SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'crop_production_model', 'sample_user_data')
TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'crop_production_model')


def make_aggregate_vector(path_to_shp):
    """
    Generate shapefile with two overlapping polygons
    Args:
        path_to_shp (str): path to store watershed results vector
    Outputs:
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
        None.
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


def create_nutrient_df():
    """Creates a nutrient DataFrame for testing."""
    return pandas.DataFrame([
        {'crop': 'corn', 'area (ha)': 21.0, 'production_observed': 0.2,
         'percentrefuse': 7, 'protein': 42., 'lipid': 8, 'energy': 476.,
         'ca': 27.0, 'fe': 15.7, 'mg': 280.0, 'ph': 704.0, 'k': 1727.0,
         'na': 2.0, 'zn': 4.9, 'cu': 1.9, 'fl': 8, 'mn': 2.9, 'se': 0.1,
         'vita': 3.0, 'betac': 16.0, 'alphac': 2.30, 'vite': 0.8,
         'crypto': 1.6, 'lycopene': 0.36, 'lutein': 63.0, 'betat': 0.5,
         'gammat': 2.1, 'deltat': 1.9, 'vitc': 6.8, 'thiamin': 0.4,
         'riboflavin': 1.8, 'niacin': 8.2, 'pantothenic': 0.9,
         'vitb6': 1.4, 'folate': 385.0, 'vitb12': 2.0, 'vitk': 41.0},

        {'crop': 'soybean', 'area (ha)': 5., 'production_observed': 4.,
         'percentrefuse': 9, 'protein': 33., 'lipid': 2., 'energy': 99.,
         'ca': 257., 'fe': 15.7, 'mg': 280., 'ph': 704.0, 'k': 197.0,
         'na': 2., 'zn': 4.9, 'cu': 1.6, 'fl': 3., 'mn': 5.2, 'se': 0.3,
         'vita': 3.0, 'betac': 16.0, 'alphac': 1.0, 'vite': 0.8,
         'crypto': 0.6, 'lycopene': 0.3, 'lutein': 61.0, 'betat': 0.5,
         'gammat': 2.3, 'deltat': 1.2, 'vitc': 3.0, 'thiamin': 0.42,
         'riboflavin': 0.82, 'niacin': 12.2, 'pantothenic': 0.92,
         'vitb6': 5.4, 'folate': 305., 'vitb12': 3., 'vitk': 42.},
         ]).set_index('crop')


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
        """Crop Production: test crop production percentile regression."""
        from natcap.invest import crop_production_percentile

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'aggregate_polygon_path': os.path.join(
                SAMPLE_DATA_PATH, 'aggregate_shape.shp'),
            'model_data_path': MODEL_DATA_PATH,
            'n_workers': '-1'
        }

        crop_production_percentile.execute(args)

        agg_result_table_path = os.path.join(
            args['workspace_dir'], 'aggregate_results.csv')
        expected_agg_result_table_path = os.path.join(
            TEST_DATA_PATH, 'expected_aggregate_results.csv')
        expected_agg_result_table = pandas.read_csv(
            expected_agg_result_table_path)
        agg_result_table = pandas.read_csv(
            agg_result_table_path)
        pandas.testing.assert_frame_equal(
            expected_agg_result_table, agg_result_table,
            check_dtype=False, check_exact=False)

        expected_result_table = pandas.read_csv(
            os.path.join(TEST_DATA_PATH, 'expected_result_table.csv')
        )
        result_table = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'result_table.csv'))
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

    def test_crop_production_percentile_no_nodata(self):
        """Crop Production: test percentile model with undefined nodata raster.

        Test with a landcover raster input that has no nodata value
        defined.
        """
        from natcap.invest import crop_production_percentile

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'model_data_path': MODEL_DATA_PATH,
            'n_workers': '-1'
        }

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
            TEST_DATA_PATH, 'expected_result_table_no_nodata.csv')
        expected_result_table = pandas.read_csv(
            expected_result_table_path)
        result_table = pandas.read_csv(
            result_table_path)
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

    def test_crop_production_percentile_bad_crop(self):
        """Crop Production: test crop production with a bad crop name."""
        from natcap.invest import crop_production_percentile

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                self.workspace_dir, 'landcover_to_badcrop_table.csv'),
            'aggregate_polygon_path': os.path.join(
                SAMPLE_DATA_PATH, 'aggregate_shape.shp'),
            'model_data_path': MODEL_DATA_PATH,
            'n_workers': '-1'
        }

        with open(args['landcover_to_crop_table_path'],
                  'w') as landcover_crop_table:
            landcover_crop_table.write(
                'crop_name,lucode\nfakecrop,20\n')

        with self.assertRaises(ValueError):
            crop_production_percentile.execute(args)

    def test_crop_production_percentile_missing_climate_bin(self):
        """Crop Production: test crop percentile with a missing climate bin."""
        from natcap.invest import crop_production_percentile

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'aggregate_polygon_path': os.path.join(
                SAMPLE_DATA_PATH, 'aggregate_shape.shp'),
            'model_data_path': MODEL_DATA_PATH,
            'n_workers': '-1'
        }

        # copy model data directory to a temp location so that hard coded
        # data paths can be altered for this test.
        tmp_copy_model_data_path = os.path.join(
            self.workspace_dir, 'tmp_model_data')

        shutil.copytree(MODEL_DATA_PATH, tmp_copy_model_data_path)

        # remove a row from the wheat percentile yield table so that a wheat
        # climate bin value is missing
        climate_bin_wheat_table_path = os.path.join(
           MODEL_DATA_PATH, 'climate_percentile_yield_tables',
           'wheat_percentile_yield_table.csv')

        bad_climate_bin_wheat_table_path = os.path.join(
           tmp_copy_model_data_path, 'climate_percentile_yield_tables',
           'wheat_percentile_yield_table.csv')

        os.remove(bad_climate_bin_wheat_table_path)

        table_df = pandas.read_csv(climate_bin_wheat_table_path)
        table_df = table_df[table_df['climate_bin'] != 40]
        table_df.to_csv(bad_climate_bin_wheat_table_path)
        table_df = None

        args['model_data_path'] = tmp_copy_model_data_path
        with self.assertRaises(ValueError) as context:
            crop_production_percentile.execute(args)
        self.assertTrue(
            "The missing values found in the wheat Climate Bin raster but not"
            " the table are: [40]" in str(context.exception))

    def test_crop_production_regression_bad_crop(self):
        """Crop Production: test crop regression with a bad crop name."""
        from natcap.invest import crop_production_regression

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_badcrop_table.csv'),
            'aggregate_polygon_path': os.path.join(
                SAMPLE_DATA_PATH, 'aggregate_shape.shp'),
            'aggregate_polygon_id': 'id',
            'model_data_path': MODEL_DATA_PATH,
            'fertilization_rate_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'crop_fertilization_rates.csv'),
            'nitrogen_fertilization_rate': 29.6,
            'phosphorus_fertilization_rate': 8.4,
            'potassium_fertilization_rate': 14.2,
            'n_workers': '-1'
        }

        with open(args['landcover_to_crop_table_path'],
                  'w') as landcover_crop_table:
            landcover_crop_table.write(
                'crop_name,lucode\nfakecrop,20\n')

        with self.assertRaises(ValueError):
            crop_production_regression.execute(args)

    def test_crop_production_regression_missing_climate_bin(self):
        """Crop Production: test crop regression with a missing climate bin."""
        from natcap.invest import crop_production_regression

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'aggregate_polygon_path': os.path.join(
                SAMPLE_DATA_PATH, 'aggregate_shape.shp'),
            'aggregate_polygon_id': 'id',
            'model_data_path': MODEL_DATA_PATH,
            'fertilization_rate_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'crop_fertilization_rates.csv'),
            'nitrogen_fertilization_rate': 29.6,
            'phosphorus_fertilization_rate': 8.4,
            'potassium_fertilization_rate': 14.2,
            'n_workers': '-1'
        }

        # copy model data directory to a temp location so that hard coded
        # data paths can be altered for this test.
        tmp_copy_model_data_path = os.path.join(
            self.workspace_dir, 'tmp_model_data')

        shutil.copytree(MODEL_DATA_PATH, tmp_copy_model_data_path)

        # remove a row from the wheat regression yield table so that a wheat
        # climate bin value is missing
        climate_bin_wheat_table_path = os.path.join(
           MODEL_DATA_PATH, 'climate_regression_yield_tables',
           'wheat_regression_yield_table.csv')

        bad_climate_bin_wheat_table_path = os.path.join(
           tmp_copy_model_data_path, 'climate_regression_yield_tables',
           'wheat_regression_yield_table.csv')

        os.remove(bad_climate_bin_wheat_table_path)

        table_df = pandas.read_csv(climate_bin_wheat_table_path)
        table_df = table_df[table_df['climate_bin'] != 40]
        table_df.to_csv(bad_climate_bin_wheat_table_path)
        table_df = None

        args['model_data_path'] = tmp_copy_model_data_path
        with self.assertRaises(ValueError) as context:
            crop_production_regression.execute(args)
        self.assertTrue(
            "The missing values found in the wheat Climate Bin raster but not"
            " the table are: [40]" in str(context.exception))

    def test_crop_production_regression(self):
        """Crop Production: test crop production regression model."""
        from natcap.invest import crop_production_regression

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'aggregate_polygon_path': os.path.join(
                SAMPLE_DATA_PATH, 'aggregate_shape.shp'),
            'aggregate_polygon_id': 'id',
            'model_data_path': MODEL_DATA_PATH,
            'fertilization_rate_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'crop_fertilization_rates.csv'),
            'nitrogen_fertilization_rate': 29.6,
            'phosphorus_fertilization_rate': 8.4,
            'potassium_fertilization_rate': 14.2,
        }

        crop_production_regression.execute(args)

        expected_agg_result_table = pandas.read_csv(
            os.path.join(TEST_DATA_PATH, 'expected_regression_aggregate_results.csv'))
        agg_result_table = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'aggregate_results.csv'))
        pandas.testing.assert_frame_equal(
            expected_agg_result_table, agg_result_table,
            check_dtype=False, check_exact=False)

        result_table_path = os.path.join(
            args['workspace_dir'], 'result_table.csv')
        expected_result_table_path = os.path.join(
            TEST_DATA_PATH, 'expected_regression_result_table.csv')
        expected_result_table = pandas.read_csv(
            expected_result_table_path)
        result_table = pandas.read_csv(
            result_table_path)
        pandas.testing.assert_frame_equal(
            expected_result_table, result_table, check_dtype=False)

    def test_crop_production_regression_no_nodata(self):
        """Crop Production: test regression model with undefined nodata raster.

        Test with a landcover raster input that has no nodata value
        defined.
        """
        from natcap.invest import crop_production_regression

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'model_data_path': MODEL_DATA_PATH,
            'fertilization_rate_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'crop_fertilization_rates.csv'),
            'nitrogen_fertilization_rate': 29.6,
            'phosphorus_fertilization_rate': 8.4,
            'potassium_fertilization_rate': 14.2,
        }

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
            TEST_DATA_PATH, 'expected_regression_result_table_no_nodata.csv'))
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
        pixel_area_ha = 10

        actual_result = _x_yield_op(y_max, b_x, c_x, lulc_array, fert_rate,
                                    crop_lucode, pixel_area_ha)
        expected_result = numpy.array([[-1, -19.393047, -1],
                                       [26.776089, -1, 15.1231]])

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
        pixel_area_ha = 10

        actual_result = _mask_observed_yield_op(
            lulc_array, observed_yield_array, observed_yield_nodata,
            landcover_nodata, crop_lucode, pixel_area_ha)

        expected_result = numpy.array([[-10, 0, -1], [80, -99990, 0]])

        numpy.testing.assert_allclose(actual_result, expected_result)

    def test_tabulate_regression_results(self):
        """Test `tabulate_regression_results`"""
        from natcap.invest.crop_production_regression import \
            tabulate_regression_results

        def _create_expected_results():
            """Creates the expected results DataFrame."""
            return pandas.DataFrame([
                {'crop': 'corn', 'area (ha)': 20.0,
                'production_observed': 8.0, 'production_modeled': 4.0,
                'protein_modeled': 1562400.0, 'protein_observed': 3124800.0,
                'lipid_modeled': 297600.0, 'lipid_observed': 595200.0,
                'energy_modeled': 17707200.0, 'energy_observed': 35414400.0,
                'ca_modeled': 1004400.0, 'ca_observed': 2008800.0,
                'fe_modeled': 584040.0, 'fe_observed': 1168080.0,
                'mg_modeled': 10416000.0, 'mg_observed': 20832000.0,
                'ph_modeled': 26188800.0, 'ph_observed': 52377600.0,
                'k_modeled': 64244400.0, 'k_observed': 128488800.0,
                'na_modeled': 74400.0, 'na_observed': 148800.0,
                'zn_modeled': 182280.0, 'zn_observed': 364560.0,
                'cu_modeled': 70680.0, 'cu_observed': 141360.0,
                'fl_modeled': 297600.0, 'fl_observed': 595200.0,
                'mn_modeled': 107880.0, 'mn_observed': 215760.0,
                'se_modeled': 3720.0, 'se_observed': 7440.0,
                'vita_modeled': 111600.0, 'vita_observed': 223200.0,
                'betac_modeled': 595200.0, 'betac_observed': 1190400.0,
                'alphac_modeled': 85560.0, 'alphac_observed': 171120.0,
                'vite_modeled': 29760.0, 'vite_observed': 59520.0,
                'crypto_modeled': 59520.0, 'crypto_observed': 119040.0,
                'lycopene_modeled': 13392.0, 'lycopene_observed': 26784.0,
                'lutein_modeled': 2343600.0, 'lutein_observed': 4687200.0,
                'betat_modeled': 18600.0, 'betat_observed': 37200.0,
                'gammat_modeled': 78120.0, 'gammat_observed': 156240.0,
                'deltat_modeled': 70680.0, 'deltat_observed': 141360.0,
                'vitc_modeled': 252960.0, 'vitc_observed': 505920.0,
                'thiamin_modeled': 14880.0, 'thiamin_observed': 29760.0,
                'riboflavin_modeled': 66960.0, 'riboflavin_observed': 133920.0,
                'niacin_modeled': 305040.0, 'niacin_observed': 610080.0,
                'pantothenic_modeled': 33480.0, 'pantothenic_observed': 66960.0,
                'vitb6_modeled': 52080.0, 'vitb6_observed': 104160.0,
                'folate_modeled': 14322000.0, 'folate_observed': 28644000.0,
                'vitb12_modeled': 74400.0, 'vitb12_observed': 148800.0,
                'vitk_modeled': 1525200.0, 'vitk_observed': 3050400.0},
                {'crop': 'soybean', 'area (ha)': 40.0,
                'production_observed': 12.0, 'production_modeled': 7.0,
                'protein_modeled': 2102100.0, 'protein_observed': 3603600.0,
                'lipid_modeled': 127400.0, 'lipid_observed': 218400.0,
                'energy_modeled': 6306300.0, 'energy_observed': 10810800.0,
                'ca_modeled': 16370900.0, 'ca_observed': 28064400.0,
                'fe_modeled': 1000090.0, 'fe_observed': 1714440.0,
                'mg_modeled': 17836000.0, 'mg_observed': 30576000.0,
                'ph_modeled': 44844800.0, 'ph_observed': 76876800.0,
                'k_modeled': 12548900.0, 'k_observed': 21512400.0,
                'na_modeled': 127400.0, 'na_observed': 218400.0,
                'zn_modeled': 312130.0, 'zn_observed': 535080.0,
                'cu_modeled': 101920.0, 'cu_observed': 174720.0,
                'fl_modeled': 191100.0, 'fl_observed': 327600.0,
                'mn_modeled': 331240.0, 'mn_observed': 567840.0,
                'se_modeled': 19110.0, 'se_observed': 32760.0,
                'vita_modeled': 191100.0, 'vita_observed': 327600.0,
                'betac_modeled': 1019200.0, 'betac_observed': 1747200.0,
                'alphac_modeled': 63700.0, 'alphac_observed': 109200.0,
                'vite_modeled': 50960.0, 'vite_observed': 87360.0,
                'crypto_modeled': 38220.0, 'crypto_observed': 65520.0,
                'lycopene_modeled': 19110.0, 'lycopene_observed': 32760.0,
                'lutein_modeled': 3885700.0, 'lutein_observed': 6661200.0,
                'betat_modeled': 31850.0, 'betat_observed': 54600.0,
                'gammat_modeled': 146510.0, 'gammat_observed': 251160.0,
                'deltat_modeled': 76440.0, 'deltat_observed': 131040.0,
                'vitc_modeled': 191100.0, 'vitc_observed': 327600.0,
                'thiamin_modeled': 26754.0, 'thiamin_observed': 45864.0,
                'riboflavin_modeled': 52234.0, 'riboflavin_observed': 89544.0,
                'niacin_modeled': 777140.0, 'niacin_observed': 1332240.0,
                'pantothenic_modeled': 58604.0, 'pantothenic_observed': 100464.0,
                'vitb6_modeled': 343980.0, 'vitb6_observed': 589680.0,
                'folate_modeled': 19428500.0, 'folate_observed': 33306000.0,
                'vitb12_modeled': 191100.0, 'vitb12_observed': 327600.0,
                'vitk_modeled': 2675400.0, 'vitk_observed': 4586400.0}])

        nutrient_df = create_nutrient_df()

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

        _create_crop_rasters(output_dir, crop_names, file_suffix)

        tabulate_regression_results(
            nutrient_df, crop_names, pixel_area_ha,
            landcover_raster_path, landcover_nodata,
            output_dir, file_suffix, target_table_path
        )

        # Read only the first 2 crop's data (skipping total area)
        actual_result_table = pandas.read_csv(target_table_path, nrows=2,
                                              header=0)
        expected_result_table = _create_expected_results()

        # Compare expected vs actual
        pandas.testing.assert_frame_equal(actual_result_table,
                                          expected_result_table)

    def test_aggregate_regression_results_to_polygons(self):
        """Test `aggregate_regression_results_to_polygons`"""
        from natcap.invest.crop_production_regression import \
            aggregate_regression_results_to_polygons

        def _create_expected_agg_table():
            """Create expected output results"""
            # Define the new values manually
            return pandas.DataFrame([
                {"FID": 0, "corn_modeled": 1, "corn_observed": 4,
                 "soybean_modeled": 2, "soybean_observed": 5,
                 "protein_modeled": 991200, "protein_observed": 3063900,
                 "lipid_modeled": 110800, "lipid_observed": 388600,
                 "energy_modeled": 6228600, "energy_observed": 22211700,
                 "ca_modeled": 4928500, "ca_observed": 12697900,
                 "fe_modeled": 431750, "fe_observed": 1298390,
                 "mg_modeled": 7700000, "mg_observed": 23156000,
                 "ph_modeled": 19360000, "ph_observed": 58220800,
                 "k_modeled": 19646500, "k_observed": 73207900,
                 "na_modeled": 55000, "na_observed": 165400,
                 "zn_modeled": 134750, "zn_observed": 405230,
                 "cu_modeled": 46790, "cu_observed": 143480,
                 "fl_modeled": 129000, "fl_observed": 434100,
                 "mn_modeled": 121610, "mn_observed": 344480,
                 "se_modeled": 6390, "se_observed": 17370,
                 "vita_modeled": 82500, "vita_observed": 248100,
                 "betac_modeled": 440000, "betac_observed": 1323200,
                 "alphac_modeled": 39590, "alphac_observed": 131060,
                 "vite_modeled": 22000, "vite_observed": 66160,
                 "crypto_modeled": 25800, "crypto_observed": 86820,
                 "lycopene_modeled": 8808, "lycopene_observed": 27042,
                 "lutein_modeled": 1696100, "lutein_observed": 5119100,
                 "betat_modeled": 13750, "betat_observed": 41350,
                 "gammat_modeled": 61390, "gammat_observed": 182770,
                 "deltat_modeled": 39510, "deltat_observed": 125280,
                 "vitc_modeled": 117840, "vitc_observed": 389460,
                 "thiamin_modeled": 11364, "thiamin_observed": 33990,
                 "riboflavin_modeled": 31664, "riboflavin_observed": 104270,
                 "niacin_modeled": 298300, "niacin_observed": 860140,
                 "pantothenic_modeled": 25114, "pantothenic_observed": 75340,
                 "vitb6_modeled": 111300, "vitb6_observed": 297780,
                 "folate_modeled": 9131500, "folate_observed": 28199500,
                 "vitb12_modeled": 73200, "vitb12_observed": 210900,
                 "vitk_modeled": 1145700, "vitk_observed": 3436200},
                {"FID": 1, "corn_modeled": 4, "corn_observed": 8,
                 "soybean_modeled": 7, "soybean_observed": 12,
                 "protein_modeled": 3664500, "protein_observed": 6728400,
                 "lipid_modeled": 425000, "lipid_observed": 813600,
                 "energy_modeled": 24013500, "energy_observed": 46225200,
                 "ca_modeled": 17375300, "ca_observed": 30073200,
                 "fe_modeled": 1584130, "fe_observed": 2882520,
                 "mg_modeled": 28252000, "mg_observed": 51408000,
                 "ph_modeled": 71033600, "ph_observed": 129254400,
                 "k_modeled": 76793300, "k_observed": 150001200,
                 "na_modeled": 201800, "na_observed": 367200,
                 "zn_modeled": 494410, "zn_observed": 899640,
                 "cu_modeled": 172600, "cu_observed": 316080,
                 "fl_modeled": 488700, "fl_observed": 922800,
                 "mn_modeled": 439120, "mn_observed": 783600,
                 "se_modeled": 22830, "se_observed": 40200,
                 "vita_modeled": 302700, "vita_observed": 550800,
                 "betac_modeled": 1614400, "betac_observed": 2937600,
                 "alphac_modeled": 149260, "alphac_observed": 280320,
                 "vite_modeled": 80720, "vite_observed": 146880,
                 "crypto_modeled": 97740, "crypto_observed": 184560,
                 "lycopene_modeled": 32502, "lycopene_observed": 59544,
                 "lutein_modeled": 6229300, "lutein_observed": 11348400,
                 "betat_modeled": 50450, "betat_observed": 91800,
                 "gammat_modeled": 224630, "gammat_observed": 407400,
                 "deltat_modeled": 147120, "deltat_observed": 272400,
                 "vitc_modeled": 444060, "vitc_observed": 833520,
                 "thiamin_modeled": 41634, "thiamin_observed": 75624,
                 "riboflavin_modeled": 119194, "riboflavin_observed": 223464,
                 "niacin_modeled": 1082180, "niacin_observed": 1942320,
                 "pantothenic_modeled": 92084, "pantothenic_observed": 167424,
                 "vitb6_modeled": 396060, "vitb6_observed": 693840,
                 "folate_modeled": 33750500, "folate_observed": 61950000,
                 "vitb12_modeled": 265500, "vitb12_observed": 476400,
                 "vitk_modeled": 4200600, "vitk_observed": 7636800}
            ], dtype=float)

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
        nutrient_df = create_nutrient_df()
        output_dir = os.path.join(workspace, "OUTPUT")
        os.makedirs(output_dir, exist_ok=True)
        file_suffix = 'test'
        target_aggregate_table_path = ''  # unused

        _create_crop_rasters(output_dir, crop_names, file_suffix)

        aggregate_regression_results_to_polygons(
            base_aggregate_vector_path, target_aggregate_vector_path,
            landcover_raster_projection, crop_names,
            nutrient_df, output_dir, file_suffix,
            target_aggregate_table_path)

        _AGGREGATE_TABLE_FILE_PATTERN = os.path.join(
            '.','aggregate_results%s.csv')

        aggregate_table_path = os.path.join(
            output_dir, _AGGREGATE_TABLE_FILE_PATTERN % file_suffix)

        actual_aggregate_table = pandas.read_csv(aggregate_table_path,
                                                 dtype=float)
        print(actual_aggregate_table)

        expected_aggregate_table = _create_expected_agg_table()

        pandas.testing.assert_frame_equal(
            actual_aggregate_table, expected_aggregate_table)



class CropValidationTests(unittest.TestCase):
    """Tests for the Crop Productions' MODEL_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'landcover_raster_path',
            'landcover_to_crop_table_path',
            'model_data_path',
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
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_regression(self):
        """Crop Regression Validate: assert missing required keys."""
        from natcap.invest import crop_production_regression
        from natcap.invest import validation

        # empty args dict.
        validation_errors = crop_production_regression.validate({})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['fertilization_rate_table_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)
