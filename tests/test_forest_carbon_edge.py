"""Module for Regression Testing the InVEST Forest Carbon Edge model."""
import unittest
import tempfile
import shutil
import os

from pygeoprocessing.testing import scm
from osgeo import ogr
import numpy


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'forest_carbon_edge_effect', 'input')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'forest_carbon_edge_effect')


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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_carbon_full(self):
        """Forest Carbon Edge: regression testing all functionality."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_uri': os.path.join(SAMPLE_DATA, 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': True,
            'lulc_uri': os.path.join(SAMPLE_DATA, 'small_lulc.tif'),
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

        ForestCarbonEdgeTests._assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(
                args['workspace_dir'], 'aggregated_carbon_stocks.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_base.csv'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_carbon_dup_output(self):
        """Forest Carbon Edge: test for existing output overlap."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_uri': os.path.join(SAMPLE_DATA, 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': True,
            'lulc_uri': os.path.join(SAMPLE_DATA, 'small_lulc.tif'),
            'n_nearest_model_points': 1,
            'pools_to_calculate': 'above_ground',
            'results_suffix': 'small',
            'tropical_forest_edge_carbon_model_shape_uri': os.path.join(
                SAMPLE_DATA, 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
        }

        # explicitly testing that invoking twice doesn't cause the model to
        # crash because of existing outputs
        forest_carbon_edge_effect.execute(args)
        forest_carbon_edge_effect.execute(args)
        self.assertTrue(True)  # explicit pass of the model

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_carbon_no_forest_edge(self):
        """Forest Carbon Edge: test for no forest edge effects."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_uri': os.path.join(SAMPLE_DATA, 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'no_forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': False,
            'lulc_uri': os.path.join(SAMPLE_DATA, 'small_lulc.tif'),
            'n_nearest_model_points': 1,
            'pools_to_calculate': 'above_ground',
            'results_suffix': 'small_no_edge_effect',
            'tropical_forest_edge_carbon_model_shape_uri': os.path.join(
                SAMPLE_DATA, 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
        }

        forest_carbon_edge_effect.execute(args)

        ForestCarbonEdgeTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'file_list_no_edge_effect.txt'),
            args['workspace_dir'])
        ForestCarbonEdgeTests._assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(
                args['workspace_dir'],
                'aggregated_carbon_stocks_small_no_edge_effect.shp'),
            os.path.join(
                REGRESSION_DATA, 'agg_results_no_edge_effect.csv'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_carbon_bad_pool_value(self):
        """Forest Carbon Edge: test with bad carbon pool value."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA,
                'no_forest_edge_carbon_lu_table_bad_pool_value.csv'),
            'compute_forest_edge_effects': False,
            'lulc_uri': os.path.join(SAMPLE_DATA, 'small_lulc.tif'),
            'n_nearest_model_points': 1,
            'pools_to_calculate': 'all',
            'results_suffix': 'small_no_edge_effect',
            'tropical_forest_edge_carbon_model_shape_uri': os.path.join(
                SAMPLE_DATA, 'core_data',
                'forest_carbon_edge_regression_model_parameters.shp'),
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            forest_carbon_edge_effect.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_carbon_nodata_lulc(self):
        """Forest Carbon Edge: regression testing all functionality."""
        from natcap.invest import forest_carbon_edge_effect

        args = {
            'aoi_uri': os.path.join(SAMPLE_DATA, 'small_aoi.shp'),
            'biomass_to_carbon_conversion_factor': '0.47',
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'forest_edge_carbon_lu_table.csv'),
            'compute_forest_edge_effects': True,
            'lulc_uri': os.path.join(SAMPLE_DATA, 'nodata_lulc.tif'),
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

        ForestCarbonEdgeTests._assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(
                args['workspace_dir'], 'aggregated_carbon_stocks.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_nodata_lulc.csv'))

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

    @staticmethod
    def _assert_regression_results_eq(
            workspace_dir, result_vector_path, agg_results_path):
        """Test workspace state against expected aggregate results.

        Parameters:
            workspace_dir (string): path to the completed model workspace
            result_vector_path (string): path to the summary shapefile
                produced by the Forest Carbon Edge model.
            agg_results_path (string): path to a csv file that has the
                expected aggregated_results.shp table in the form of
                c_sum,c_ha_mean per line

        Returns:
            None

        Raises:
            AssertionError if any files are missing or results are out of
            range by `tolerance_places`
        """
        result_vector = ogr.Open(result_vector_path)
        result_layer = result_vector.GetLayer()

        with open(agg_results_path, 'rb') as agg_result_file:
            for line in agg_result_file:
                fid, c_sum, c_ha_mean = [float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                for field, value in [
                        ('c_sum', c_sum),
                        ('c_ha_mean', c_ha_mean)]:
                    numpy.testing.assert_allclose(
                        feature.GetField(field), value)
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None
