"""Module for Regression Testing the InVEST Pollination model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from osgeo import ogr

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'pollination')


class PollinationTests(unittest.TestCase):
    """Tests for the Pollination model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_pollination_regression(self):
        """Pollination: regression testing sample data."""
        from natcap.invest import pollination
        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA, 'landcover.tif'),
            'guild_table_path': os.path.join(SAMPLE_DATA, 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                SAMPLE_DATA, r'landcover_biophysical_table.csv'),
            'farm_vector_path': os.path.join(SAMPLE_DATA, 'farms.shp'),
        }
        pollination.execute(args)
        expected_farm_yields = {
            'almonds': {
                'p_av_yield': 0.99173113255719,
                't_av_yield': 0.994625236162173
            },
            'blueberries': {
                'p_av_yield': 0.020808936958026,
                't_av_yield': 0.363525809022717
            },
        }
        result_vector = ogr.Open(
            os.path.join(self.workspace_dir, 'farm_yield.shp'))
        result_layer = result_vector.GetLayer()
        self.assertEqual(
            result_layer.GetFeatureCount(), len(expected_farm_yields))
        for feature in result_layer:
            expected_yields = expected_farm_yields[
                feature.GetField('crop_type')]
            for yield_type in expected_yields:
                self.assertAlmostEqual(
                    expected_yields[yield_type],
                    feature.GetField(yield_type))
        result_layer = None
        result_vector = None


    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_pollination_bad_guild_headers(self):
        """Pollination: testing that model detects bad guild headers."""
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)
        bad_guild_table_path = os.path.join(temp_path, 'bad_guild_table.csv')
        with open(bad_guild_table_path, 'wb') as bad_guild_table:
            bad_guild_table.write(
                'species,nesting_suitability_cavity_index,alpha,'
                'relative_abundance\n')
            bad_guild_table.write(
                'apis,0.2,400,1.0\n')
            bad_guild_table.write(
                'bee,0.9,1400,0.1\n')
        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA, 'landcover.tif'),
            'guild_table_path': bad_guild_table_path,
            'landcover_biophysical_table_path': os.path.join(
                SAMPLE_DATA, r'habitat_nesting_suitability.csv'),
            'farm_vector_path': os.path.join(SAMPLE_DATA, 'farms.shp'),
        }
        with self.assertRaises(ValueError):
            pollination.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_pollination_bad_biophysical_headers(self):
        """Pollination: testing that model detects bad biophysical headers."""
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)
        bad_biophysical_table_path = os.path.join(
            temp_path, 'bad_biophysical_table.csv')
        with open(bad_biophysical_table_path, 'wb') as bad_biophysical_table:
            bad_biophysical_table.write(
                'lucode,nesting_cavity_availability_index,nesting_ground_index\n'
                '1,0.3,0.2\n')
        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA, 'landcover.tif'),
            'guild_table_path': os.path.join(SAMPLE_DATA, 'guild_table.csv'),
            'landcover_biophysical_table_path': bad_biophysical_table_path,
            'farm_vector_path': os.path.join(SAMPLE_DATA, 'farms.shp'),
        }
        with self.assertRaises(ValueError):
            pollination.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_pollination_bad_cross_table_headers(self):
        """Pollination: ensure detection of missing headers in one table."""
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)
        bad_biophysical_table_path = os.path.join(
            temp_path, 'bad_biophysical_table.csv')
        # one table has only spring the other has only fall.
        with open(bad_biophysical_table_path, 'wb') as bad_biophysical_table:
            bad_biophysical_table.write(
                'lucode,nesting_cavity_availability_index,nesting_ground_index,floral_resources_spring_index\n'
                '1,0.3,0.2,0.2\n')
        bad_guild_table_path = os.path.join(temp_path, 'bad_guild_table.csv')
        with open(bad_guild_table_path, 'wb') as bad_guild_table:
            bad_guild_table.write(
                'species,nesting_suitability_cavity_index,'
                'foraging_activity_fall_index,alpha,relative_abundance\n')
            bad_guild_table.write(
                'apis,0.2,0.5,400,1.0\n')
            bad_guild_table.write(
                'bee,0.9,0.5,1400,0.5\n')
        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA, 'landcover.tif'),
            'guild_table_path': bad_guild_table_path,
            'landcover_biophysical_table_path': bad_biophysical_table_path,
            'farm_vector_path': os.path.join(SAMPLE_DATA, 'farms.shp'),
        }
        with self.assertRaises(ValueError):
            pollination.execute(args)


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
