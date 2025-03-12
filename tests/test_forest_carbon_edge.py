"""Module for Regression Testing the InVEST Forest Carbon Edge model."""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal, osr, ogr
import numpy
import pygeoprocessing
from shapely.geometry import Polygon

gdal.UseExceptions()
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'forest_carbon_edge_effect')


def make_simple_vector(path_to_shp):
    """
    Generate shapefile with one rectangular polygon
    Args:
        path_to_shp (str): path to target shapefile
    Returns:
        None
    """
    # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)
    shapely_geometry_list = [
        Polygon([(461251, 4923195), (461501, 4923195),
                 (461501, 4923445), (461251, 4923445),
                 (461251, 4923195)])
    ]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    vector_format = "ESRI Shapefile"
    fields = {"id": ogr.OFTReal}
    attribute_list = [{"id": 0}]

    pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                               path_to_shp, projection_wkt,
                                               vector_format, fields,
                                               attribute_list)


def make_simple_raster(base_raster_path, array, nodata_val=-1):
    """Create a raster on designated path.

    Args:
        base_raster_path (str): the raster path for the new raster.
        array (array): numpy array to convert to tif.
        nodata_val (int or None): for defining a raster's nodata value.

    Returns:
        None.

    """

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # UTM Zone 10N
    projection_wkt = srs.ExportToWkt()
    # origin hand-picked for this epsg:
    origin = (461261, 4923265)

    pixel_size = (1, -1)

    pygeoprocessing.numpy_array_to_raster(
        array, nodata_val, pixel_size, origin, projection_wkt,
        base_raster_path)


class ForestCarbonEdgeTests(unittest.TestCase):
    """Tests for the Forest Carbon Edge Model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_carbon_full(self):
        """Forest Carbon Edge: regression testing all functionality."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': True,
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_lulc.tif'),
            'n_nearest_model_points': 10,
            'pools_to_calculate': 'all',
            'tropical_forest_edge_carbon_model_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
            'n_workers': -1
        }
        forest_carbon_edge_effect.execute(args)
        ForestCarbonEdgeTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list.txt'),
            args['workspace_dir'])

        self._assert_vector_results_close(
            args['workspace_dir'], 'id', ['c_sum', 'c_ha_mean'], os.path.join(
                args['workspace_dir'], 'aggregated_carbon_stocks.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_base.shp'))

        expected_carbon_raster = gdal.OpenEx(os.path.join(REGRESSION_DATA, 
                                                          'carbon_map.tif'))
        expected_carbon_band = expected_carbon_raster.GetRasterBand(1)
        expected_carbon_array = expected_carbon_band.ReadAsArray()
        actual_carbon_raster = gdal.OpenEx(os.path.join(REGRESSION_DATA, 
                                                        'carbon_map.tif'))
        actual_carbon_band = actual_carbon_raster.GetRasterBand(1)
        actual_carbon_array = actual_carbon_band.ReadAsArray()
        self.assertTrue(numpy.allclose(expected_carbon_array, 
                                       actual_carbon_array))

    def test_carbon_dup_output(self):
        """Forest Carbon Edge: test for existing output overlap."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': True,
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_lulc.tif'),
            'n_nearest_model_points': 1,
            'pools_to_calculate': 'above_ground',
            'results_suffix': 'small',
            'tropical_forest_edge_carbon_model_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
            'n_workers': -1
        }

        # explicitly testing that invoking twice doesn't cause the model to
        # crash because of existing outputs
        forest_carbon_edge_effect.execute(args)
        forest_carbon_edge_effect.execute(args)
        self.assertTrue(True)  # explicit pass of the model

    def test_carbon_no_forest_edge(self):
        """Forest Carbon Edge: test for no forest edge effects."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input',
                'no_forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': False,
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_lulc.tif'),
            'n_nearest_model_points': 1,
            'pools_to_calculate': 'above_ground',
            'results_suffix': 'small_no_edge_effect',
            'tropical_forest_edge_carbon_model_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
            'n_workers': -1
        }
        forest_carbon_edge_effect.execute(args)

        ForestCarbonEdgeTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'file_list_no_edge_effect.txt'),
            args['workspace_dir'])
        self._assert_vector_results_close(
            args['workspace_dir'], 'id', ['c_sum', 'c_ha_mean'],
            os.path.join(
                args['workspace_dir'],
                'aggregated_carbon_stocks_small_no_edge_effect.shp'),
            os.path.join(
                REGRESSION_DATA, 'agg_results_no_edge_effect.shp'))

    def test_carbon_bad_pool_value(self):
        """Forest Carbon Edge: test with bad carbon pool value."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input',
                'no_forest_edge_carbon_lu_table_bad_pool_value.csv'),
            'compute_forest_edge_effects': False,
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_lulc.tif'),
            'n_nearest_model_points': 1,
            'pools_to_calculate': 'all',
            'results_suffix': 'small_no_edge_effect',
            'tropical_forest_edge_carbon_model_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
            'n_workers': -1
        }

        with self.assertRaises(ValueError) as cm:
            forest_carbon_edge_effect.execute(args)
        expected_message = 'Could not interpret carbon pool value'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)
    
    def test_missing_lulc_value(self):
        """Forest Carbon Edge: test with missing LULC value."""
        from natcap.invest import forest_carbon_edge_effect
        import pandas

        args = {
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': True,
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_lulc.tif'),
            'n_nearest_model_points': 10,
            'pools_to_calculate': 'all',
            'tropical_forest_edge_carbon_model_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
            'n_workers': -1
        }

        bad_biophysical_table_path = os.path.join(
            self.workspace_dir, 'bad_biophysical_table.csv')

        bio_df = pandas.read_csv(args['biophysical_table_path'])
        bio_df = bio_df[bio_df['lucode'] != 4]
        bio_df.to_csv(bad_biophysical_table_path)
        bio_df = None

        args['biophysical_table_path'] = bad_biophysical_table_path
        with self.assertRaises(ValueError) as cm:
            forest_carbon_edge_effect.execute(args)
        expected_message = (
            "The missing values found in the LULC raster but not the table"
            " are: [4.]")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_carbon_nodata_lulc(self):
        """Forest Carbon Edge: ensure nodata lulc raster cause exception."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': True,
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'nodata_lulc.tif'),
            'n_nearest_model_points': 10,
            'pools_to_calculate': 'all',
            'tropical_forest_edge_carbon_model_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
            'n_workers': -1
        }
        with self.assertRaises(ValueError) as cm:
            forest_carbon_edge_effect.execute(args)
        expected_message = 'The landcover raster '
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_combine_carbon_maps(self):
        """Test `combine_carbon_maps`"""
        from natcap.invest.forest_carbon_edge_effect import combine_carbon_maps

        # note that NODATA_VALUE = -1
        carbon_arr1 = numpy.array([[7, 2, -1], [0, -2, -1]])
        carbon_arr2 = numpy.array([[-1, 900, -1], [1, 20, 0]])

        expected_output = numpy.array([[7, 902, -1], [1, 18, 0]])

        actual_output = combine_carbon_maps(carbon_arr1, carbon_arr2)

        numpy.testing.assert_allclose(actual_output, expected_output)

    def test_aggregate_carbon_map(self):
        """Test `_aggregate_carbon_map`"""
        from natcap.invest.forest_carbon_edge_effect import \
            _aggregate_carbon_map

        aoi_vector_path = os.path.join(self.workspace_dir, "aoi.shp")
        carbon_map_path = os.path.join(self.workspace_dir, "carbon.tif")
        target_aggregate_vector_path = os.path.join(self.workspace_dir,
                                                    "agg_carbon.shp")

        # make data
        make_simple_vector(aoi_vector_path)
        carbon_array = numpy.array(([1, 2, 3], [4, 5, 6]))
        make_simple_raster(carbon_map_path, carbon_array)

        _aggregate_carbon_map(aoi_vector_path, carbon_map_path,
                              target_aggregate_vector_path)

        def _validate_fields(vector_path, field_name, expected_values):
            """
            Validate a field in the agg carbon results vector

            Args:
                vector path (str): path to shapefile
                field_name (str): attribute field to check
                expected values (list): list of expected values for field
                error_msg (str): what to print if assertion fails

            Returns:
                None
            """
            with gdal.OpenEx(vector_path, gdal.OF_VECTOR | gdal.GA_Update) as ws_ds:
                ws_layer = ws_ds.GetLayer()
                actual_values = [ws_feat.GetField(field_name)
                                 for ws_feat in ws_layer]
                error_msg = f"Error with {field_name} in {vector_path}"
                self.assertEqual(actual_values, expected_values, msg=error_msg)

        _validate_fields(target_aggregate_vector_path, 'c_sum', [21])
        _validate_fields(target_aggregate_vector_path, 'c_ha_mean', [3.36])


    @staticmethod
    def _test_same_files(base_list_path, directory_path):
        """Assert files in `base_list_path` are in `directory_path`.

        Args:
            base_list_path (string): a path to a file that has one relative
                file path per line.
            directory_path (string): a path to a directory whose contents will
                be checked against the files listed in `base_list_file`

        Returns:
            None

        Raises:
            AssertionError when there are files listed in `base_list_file`
                that don't exist in the directory indicated by `path`
        """
        missing_files = []
        with open(base_list_path, 'r') as file_list:
            for file_path in file_list:
                full_path = os.path.join(directory_path, file_path.rstrip())
                if full_path == '':
                    continue
                if not os.path.isfile(full_path):
                    missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '\n'.join(missing_files))

    def _assert_vector_results_close(
            self, workspace_dir, id_fieldname, field_list, result_vector_path,
            expected_vector_path):
        """Test workspace state against expected aggregate results.

        Args:
            workspace_dir (string): path to the completed model workspace
            id_fieldname (string): fieldname of the unique ID.
            field_list (list of string): list of fields to check
                near-equality.
            result_vector_path (string): path to the summary shapefile
                produced by the Forest Carbon Edge model.
            expected_vector_path (string): path to a vector that has the
                same fields and values as `result_vector_path`.

        Returns:
            None

        Raises:
            AssertionError if results are not nearly equal or missing.

        """
        result_vector = gdal.OpenEx(result_vector_path, gdal.OF_VECTOR)
        try:
            result_layer = result_vector.GetLayer()
            result_lookup = {}
            for feature in result_layer:
                result_lookup[feature.GetField(id_fieldname)] = dict(
                    [(fieldname, feature.GetField(fieldname))
                     for fieldname in field_list])
            expected_vector = gdal.OpenEx(
                expected_vector_path, gdal.OF_VECTOR)
            expected_layer = expected_vector.GetLayer()
            expected_lookup = {}
            for feature in expected_layer:
                expected_lookup[feature.GetField(id_fieldname)] = dict(
                    [(fieldname, feature.GetField(fieldname))
                     for fieldname in field_list])

            self.assertEqual(len(result_lookup), len(expected_lookup))
            not_close_values_list = []
            for feature_id in result_lookup:
                for fieldname in field_list:
                    result = result_lookup[feature_id][fieldname]
                    expected_result = expected_lookup[feature_id][fieldname]
                    if not numpy.isclose(result, expected_result):
                        not_close_values_list.append(
                            'id: %d, %s: %f (actual) vs %f (expected)' % (
                                feature_id, fieldname, result,
                                expected_result))
            if not_close_values_list:
                raise AssertionError(
                    'Values do not match: %s' % not_close_values_list)
        finally:
            result_layer = None
            if result_vector:
                gdal.Dataset.__swig_destroy__(result_vector)
            result_vector = None


class ForestCarbonEdgeValidationTests(unittest.TestCase):
    """Tests for the Forest Carbon Model MODEL_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'biophysical_table_path',
            'lulc_raster_path',
            'pools_to_calculate',
            'compute_forest_edge_effects',
        ]

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys(self):
        """Forest Carbon Validate: assert missing required keys."""
        from natcap.invest import forest_carbon_edge_effect
        from natcap.invest import validation

        # empty args dict.
        validation_errors = forest_carbon_edge_effect.validate({})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_for_edge_effects(self):
        """Forest Carbon Validate: assert missing required for edge effects."""
        from natcap.invest import forest_carbon_edge_effect
        from natcap.invest import validation

        args = {'compute_forest_edge_effects': True}
        validation_errors = forest_carbon_edge_effect.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['n_nearest_model_points',
             'tropical_forest_edge_carbon_model_vector_path',
             'biomass_to_carbon_conversion_factor'])
        expected_missing_keys.difference_update(
            {'compute_forest_edge_effects'})
        self.assertEqual(invalid_keys, expected_missing_keys)
