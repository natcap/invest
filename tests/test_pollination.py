"""Module for Regression Testing the InVEST Pollination model."""
import numpy
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from pygeoprocessing.testing import sampledata
from osgeo import ogr
import shapely.geometry

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'pollination')
TEST_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'pollination')


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

    @scm.skip_if_data_missing(TEST_DATA)
    def test_pollination_regression(self):
        """Pollination: regression testing sample data."""
        from natcap.invest import pollination

        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                TEST_DATA, 'pollination_example_landcover.tif'),
            'guild_table_path': os.path.join(TEST_DATA, 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                TEST_DATA, r'landcover_biophysical_table.csv'),
            'farm_vector_path': os.path.join(
                TEST_DATA, 'blueberry_ridge_farm.shp'),
        }
        # make an empty farm result to get coverage for removing if necessary
        f = open(os.path.join(self.workspace_dir, 'farm_results.shp'), 'w')
        f.close()
        pollination.execute(args)
        expected_farm_yields = {
            'blueberry': {
                'y_tot': 0.41237348829,
                'y_wild': 0.06237348829
            },
        }
        result_vector = ogr.Open(
            os.path.join(self.workspace_dir, 'farm_results.shp'))
        result_layer = result_vector.GetLayer()
        try:
            self.assertEqual(
                result_layer.GetFeatureCount(), len(expected_farm_yields))
            for feature in result_layer:
                expected_yields = expected_farm_yields[
                    feature.GetField('crop_type')]
                for yield_type in expected_yields:
                    self.assertAlmostEqual(
                        expected_yields[yield_type],
                        feature.GetField(yield_type), places=2)
        finally:
            # make sure vector is closed before removing the workspace
            result_layer = None
            result_vector = None

    @scm.skip_if_data_missing(TEST_DATA)
    def test_pollination_missing_farm_header(self):
        """Pollination: regression testing missing farm headers."""
        from natcap.invest import pollination

        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                TEST_DATA, 'pollination_example_landcover.tif'),
            'guild_table_path': os.path.join(TEST_DATA, 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                TEST_DATA, r'landcover_biophysical_table.csv'),
            'farm_vector_path': os.path.join(
                TEST_DATA, 'missing_headers_farm.shp'),
        }
        # should error when not finding an expected farm header
        with self.assertRaises(ValueError):
            pollination.execute(args)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_pollination_too_many_farm_seasons(self):
        """Pollination: regression testing too many seasons in farm."""
        from natcap.invest import pollination

        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                TEST_DATA, 'pollination_example_landcover.tif'),
            'guild_table_path': os.path.join(TEST_DATA, 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                TEST_DATA, r'landcover_biophysical_table.csv'),
            'farm_vector_path': os.path.join(
                TEST_DATA, 'too_many_seasons_farm.shp'),
        }
        # should error when not finding an expected farm header
        with self.assertRaises(ValueError):
            pollination.execute(args)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_pollination_missing_guild_header(self):
        """Pollination: regression testing extra guild headers."""
        from natcap.invest import pollination

        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                TEST_DATA, 'pollination_example_landcover.tif'),
            'guild_table_path': os.path.join(
                TEST_DATA, 'missing_guild_table_header.csv'),
            'landcover_biophysical_table_path': os.path.join(
                TEST_DATA, r'landcover_biophysical_table.csv'),
        }
        # should error when not finding an expected farm header
        with self.assertRaises(ValueError):
            pollination.execute(args)


    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_pollination_no_farm_regression(self):
        """Pollination: regression testing sample data with no farms."""
        from natcap.invest import pollination

        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                TEST_DATA, 'clipped_landcover.tif'),
            'guild_table_path': os.path.join(SAMPLE_DATA, 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                SAMPLE_DATA, r'landcover_biophysical_table.csv')
        }
        pollination.execute(args)
        result_raster_path = os.path.join(
            self.workspace_dir, 'pollinator_abundance_apis_spring.tif')
        result_sum = numpy.float32(0.0)
        for _, data_block in pygeoprocessing.iterblocks(result_raster_path):
            result_sum += numpy.sum(data_block)
        # the number below is just what the sum rounded to two decimal places
        # when I manually inspected a run that appeared to be correct.
        self.assertAlmostEqual(result_sum, 4790.44, places=2)


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
                SAMPLE_DATA, r'landcover_biophysical_table.csv'),
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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_pollination_bad_farm_type(self):
        """Pollination: ensure detection of bad farm geometry type."""
        from natcap.invest import pollination

        # make some fake farm points
        point_geom = [shapely.geometry.Point(20, - 20)]

        farm_shape_path = os.path.join(self.workspace_dir, 'point_farm.shp')
        # Create the point shapefile
        srs = sampledata.SRS_WILLAMETTE
        fields = {
            'crop_type': 'string',
            'half_sat': 'real',
            'p_managed': 'real'}
        attrs = [
            {'crop_type': 'test', 'half_sat': 0.5, 'p_managed': 0.5}]

        pygeoprocessing.testing.create_vector_on_disk(
            point_geom, srs.projection, fields, attrs,
            vector_format='ESRI Shapefile', filename=farm_shape_path)

        args = {
            'results_suffix': u'',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                SAMPLE_DATA, 'landcover.tif'),
            'guild_table_path': os.path.join(SAMPLE_DATA, 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                SAMPLE_DATA, r'landcover_biophysical_table.csv'),
            'farm_vector_path': farm_shape_path,
        }
        with self.assertRaises(ValueError):
            pollination.execute(args)
