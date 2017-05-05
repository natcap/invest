"""Module for Regression Testing the InVEST Crop Production models."""
import unittest
import tempfile
import shutil
import os

import numpy
import pygeoprocessing.testing
from pygeoprocessing.testing import scm

MODEL_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'CropProduction', 'model_data')
SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'CropProduction', 'sample_user_data')
TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'crop_production_model')


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

    @scm.skip_if_data_missing(SAMPLE_DATA_PATH)
    @scm.skip_if_data_missing(MODEL_DATA_PATH)
    def test_crop_production_percentile(self):
        """Crop Production: test crop production."""
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
            'aggregate_polygon_id': 'id',
            'model_data_path': MODEL_DATA_PATH
        }
        crop_production_percentile.execute(args)

        result_table_path = os.path.join(
            args['workspace_dir'], 'aggregate_results.csv')
        from natcap.invest import utils
        result_table = utils.build_lookup_from_csv(
            result_table_path, 'id', to_lower=True, numerical_cast=True)

        expected_result_table_path = os.path.join(
            TEST_DATA_PATH, 'expected_aggregate_results.csv')

        expected_result_table = utils.build_lookup_from_csv(
            expected_result_table_path, 'id', to_lower=True,
            numerical_cast=True)

        for id_key in expected_result_table:
            if id_key not in result_table:
                self.fail("Expected ID %s in result table" % id_key)
            for column_key in expected_result_table[id_key]:
                if column_key not in result_table[id_key]:
                    self.fail(
                        "Expected column %s in result table" % column_key)
                # The tolerance of 3 digits after the decimal was determined by
                # experimentation
                tolerance_places = 3
                numpy.testing.assert_almost_equal(
                    expected_result_table[id_key][column_key],
                    result_table[id_key][column_key],
                    decimal=tolerance_places)

    @scm.skip_if_data_missing(SAMPLE_DATA_PATH)
    @scm.skip_if_data_missing(MODEL_DATA_PATH)
    def test_crop_production_percentile_bad_crop(self):
        """Crop Production: test crop production."""
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
            'aggregate_polygon_id': 'id',
            'model_data_path': MODEL_DATA_PATH
        }

        with open(args['landcover_to_crop_table_path'],
                  'wb') as landcover_crop_table:
            landcover_crop_table.write(
                'crop_name,lucode\nfakecrop,20\n')

        with self.assertRaises(ValueError):
            crop_production_percentile.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA_PATH)
    @scm.skip_if_data_missing(MODEL_DATA_PATH)
    def test_crop_production_regression(self):
        """Crop Production: test crop production."""
        from natcap.invest import crop_production_regression

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'k_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'fake_fert_map.tif'),
            'p_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'fake_fert_map.tif'),
            'n_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'fake_fert_map.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'aggregate_polygon_path': os.path.join(
                SAMPLE_DATA_PATH, 'aggregate_shape.shp'),
            'aggregate_polygon_id': 'id',
            'model_data_path': MODEL_DATA_PATH
        }
        crop_production_regression.execute(args)
