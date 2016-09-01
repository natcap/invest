"""Module for Regression Testing the InVEST Forest Carbon Edge model."""
import unittest
import tempfile
import shutil
import os


import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from osgeo import gdal
from osgeo import ogr
import numpy


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'forest_carbon_edge_effect')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'forest_carbon_edge_effect')


class ForestCarbonEdgeTests(unittest.TestCase):
    """Tests for the Forets Carbon Edge Model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_carbon_full(self):
        """Forest Carbon Edge: regression testing all functionality."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_uri': os.path.join(
                SAMPLE_DATA, 'forest_carbon_edge_demo_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': True,
            'lulc_uri': os.path.join(
                SAMPLE_DATA, 'forest_carbon_edge_lulc_demo.tif'),
            'n_nearest_model_points': 10,
            'pools_to_calculate': 'all',
            'tropical_forest_edge_carbon_model_shape_uri': os.path.join(
                SAMPLE_DATA, 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
        }

        forest_carbon_edge_effect.execute(args)
        ForestCarbonEdgeTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list.txt'),
            args['workspace_dir'])

    @staticmethod
    def _test_same_files(base_list_path, directory_path):
        """Assert files in `base_list_path` are in `directory_path`.

        Parameters:
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
