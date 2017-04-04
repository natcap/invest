"""Module for Regression Testing the InVEST Crop Production models."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

MODEL_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'CropProduction', 'model_data')
SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'CropProduction', 'sample_user_data')


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
            'workspace_dir': r'C:\Users\rpsharp\Documents\del_test_crop_production_workspace',
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'aggregate_polygon_path': os.path.join(
                SAMPLE_DATA_PATH, 'aggreate_shape.shp'),
             'aggregate_polygon_id': 'id',
            'model_data_path': MODEL_DATA_PATH
        }
        crop_production_percentile.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA_PATH)
    @scm.skip_if_data_missing(MODEL_DATA_PATH)
    def test_crop_production_regression(self):
        """Crop Production: test crop production."""
        from natcap.invest import crop_production_regression

        args = {
            'workspace_dir': r'C:\Users\rpsharp\Documents\del_test_crop_production_regression_workspace',
            'results_suffix': '',
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover.tif'),
            'landcover_to_crop_table_path': os.path.join(
                SAMPLE_DATA_PATH, 'landcover_to_crop_table.csv'),
            'k_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'k.tif'),
            'n_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'n.tif'),
            'pot_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'pot.tif'),
            'irrigation_raster_path': os.path.join(
                SAMPLE_DATA_PATH, 'irrigation.tif'),
            'model_data_path': MODEL_DATA_PATH
            }
        crop_production_regression.execute(args)
