"""Module for Regression Testing Scenario Proximity Generator."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'scenario_proximity')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'scenario_gen_proximity')


class ScenarioProximityTests(unittest.TestCase):
    """Tests for the Scenario Proximity Generator."""

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
    def test_scenario_gen_regression(self):
        """Scenario Gen Proximity: regression testing all functionality."""
        from natcap.invest import scenario_gen_proximity

        args = {
            'aoi_uri': os.path.join(
                SAMPLE_DATA, 'scenario_proximity_aoi.shp'),
            'area_to_convert': '20000.0',
            'base_lulc_uri': os.path.join(
                SAMPLE_DATA, 'scenario_proximity_lulc.tif'),
            'convert_farthest_from_edge': True,
            'convert_nearest_to_edge': True,
            'convertible_landcover_codes': '1 2 3 4 5',
            'focal_landcover_codes': '1 2 3 4 5',
            'n_fragmentation_steps': '1',
            'replacment_lucode': '12',
            'workspace_dir': self.workspace_dir,
        }

        scenario_gen_proximity.execute(args)
        ScenarioProximityTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'expected_file_list_regression.txt'),
            args['workspace_dir'])

        pygeoprocessing.testing.assertions.assert_csv_equal(
            os.path.join(self.workspace_dir, 'farthest_from_edge.csv'),
            os.path.join(
                REGRESSION_DATA, 'farthest_from_edge_regression.csv'),
            rel_tol=1e-6)

        pygeoprocessing.testing.assertions.assert_csv_equal(
            os.path.join(self.workspace_dir, 'nearest_to_edge.csv'),
            os.path.join(REGRESSION_DATA, 'nearest_to_edge_regression.csv'),
            rel_tol=1e-6)

    def test_scenario_gen_no_scenario(self):
        """Scenario Gen Proximity: no scenario should raise an exception."""
        from natcap.invest import scenario_gen_proximity

        args = {
            'aoi_uri': os.path.join(
                SAMPLE_DATA, 'scenario_proximity_aoi.shp'),
            'area_to_convert': '20000.0',
            'base_lulc_uri': os.path.join(
                SAMPLE_DATA, 'scenario_proximity_lulc.tif'),
            'convert_farthest_from_edge': False,
            'convert_nearest_to_edge': False,
            'convertible_landcover_codes': '1 2 3 4 5',
            'focal_landcover_codes': '1 2 3 4 5',
            'n_fragmentation_steps': '1',
            'replacment_lucode': '12',
            'workspace_dir': self.workspace_dir,
        }

        # both scenarios false should raise a value error
        with self.assertRaises(ValueError):
            scenario_gen_proximity.execute(args)

    @staticmethod
    def _assert_regression_results_equal(
            workspace_dir, file_list_path, result_vector_path,
            agg_results_path):
        """Test workspace state against expected aggregate results.

        Parameters:
            workspace_dir (string): path to the completed model workspace
            file_list_path (string): path to a file that has a list of all
                the expected files relative to the workspace base
            result_vector_path (string): path to the summary shapefile
                produced by the SWY model.
            agg_results_path (string): path to a csv file that has the
                expected aggregated_results.shp table in the form of
                fid,vri_sum,qb_val per line

        Returns:
            None

        Raises:
            AssertionError if any files are missing or results are out of
            range by `tolerance_places`
        """
        # test that the workspace has the same files as we expect
        ScenarioProximityTests._test_same_files(
            file_list_path, workspace_dir)

        # we expect a file called 'aggregated_results.shp'

        # The tolerance of 3 digits after the decimal was determined by
        # experimentation on the application with the given range of numbers.
        # This is an apparently reasonable approach as described by ChrisF:
        # http://stackoverflow.com/a/3281371/42897
        # and even more reading about picking numerical tolerance (it's hard):
        # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        tolerance_places = 3

        with open(agg_results_path, 'rb') as agg_result_file:
            for line in agg_result_file:
                fid, sed_retent, sed_export, usle_tot = [
                    float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                for field, value in [
                        ('sed_retent', sed_retent),
                        ('sed_export', sed_export),
                        ('usle_tot', usle_tot)]:
                    numpy.testing.assert_almost_equal(
                        feature.GetField(field), value,
                        decimal=tolerance_places)
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None

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
