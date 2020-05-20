"""Module for Regression Testing the InVEST Pollination model."""
import numpy
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
from osgeo import ogr
import shapely.geometry

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'pollination')

EXPECTED_FILE_LIST = [
    'farm_pollinators.tif',
    'farm_results.dbf',
    'farm_results.prj',
    'farm_results.shp',
    'farm_results.shx',
    'pollinator_abundance_apis_spring.tif',
    'pollinator_supply_apis.tif',
    'total_pollinator_abundance_spring.tif',
    'total_pollinator_yield.tif',
    'wild_pollinator_yield.tif',
    'intermediate_outputs/blank_raster.tif',
    'intermediate_outputs/convolve_ps_apis.tif',
    'intermediate_outputs/farm_nesting_substrate_index_cavity.tif',
    'intermediate_outputs/farm_pollinator_spring.tif',
    'intermediate_outputs/farm_relative_floral_abundance_index_spring.tif',
    'intermediate_outputs/floral_resources_apis.tif',
    'intermediate_outputs/foraged_flowers_index_apis_spring.tif',
    'intermediate_outputs/habitat_nesting_index_apis.tif',
    'intermediate_outputs/half_saturation_spring.tif',
    'intermediate_outputs/kernel_10.250312.tif',
    'intermediate_outputs/local_foraging_effectiveness_apis.tif',
    'intermediate_outputs/managed_pollinators.tif',
    'intermediate_outputs/nesting_substrate_index_cavity.tif',
    'intermediate_outputs/relative_floral_abundance_index_spring.tif',
    'intermediate_outputs/reprojected_farm_vector.dbf',
    'intermediate_outputs/reprojected_farm_vector.prj',
    'intermediate_outputs/reprojected_farm_vector.shp',
    'intermediate_outputs/reprojected_farm_vector.shx']


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

    def test_pollination_regression(self):
        """Pollination: regression testing sample data."""
        from natcap.invest import pollination

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'pollination_example_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table_simple.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input',
                'landcover_biophysical_table_simple.csv'),
            'farm_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'blueberry_ridge_farm.shp'),
        }
        # make empty result files to get coverage for removing if necessary
        result_files = ['farm_results.shp', 'total_pollinator_yield.tif',
                        'wild_pollinator_yield.tif']
        for file_name in result_files:
            f = open(os.path.join(self.workspace_dir, file_name), 'w')
            f.close()
        pollination.execute(args)
        expected_farm_yields = {
            'blueberry': {
                'y_tot': 0.44934792607,
                'y_wild': 0.09934792607
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

        PollinationTests._test_same_files(
            EXPECTED_FILE_LIST, self.workspace_dir)

    def test_pollination_missing_farm_header(self):
        """Pollination: regression testing missing farm headers."""
        from natcap.invest import pollination

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'pollination_example_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table_simple.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input',
                'landcover_biophysical_table_simple.csv'),
            'farm_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'missing_headers_farm.shp'),
        }
        # should error when not finding an expected farm header
        with self.assertRaises(ValueError):
            pollination.execute(args)

    def test_pollination_too_many_farm_seasons(self):
        """Pollination: regression testing too many seasons in farm."""
        from natcap.invest import pollination

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'pollination_example_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table_simple.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input',
                'landcover_biophysical_table_simple.csv'),
            'farm_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'too_many_seasons_farm.shp'),
        }
        # should error when not finding an expected farm header
        with self.assertRaises(ValueError):
            pollination.execute(args)

    def test_pollination_missing_guild_header(self):
        """Pollination: regression testing missing guild headers."""
        from natcap.invest import pollination

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'pollination_example_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'missing_guild_table_header.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input',
                'landcover_biophysical_table_simple.csv'),
        }
        # should error when not finding an expected farm header
        with self.assertRaises(ValueError):
            pollination.execute(args)

    def test_pollination_no_farm_regression(self):
        """Pollination: regression testing sample data with no farms."""
        from natcap.invest import pollination

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'clipped_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'landcover_biophysical_table.csv')
        }
        pollination.execute(args)
        result_raster_path = os.path.join(
            self.workspace_dir, 'pollinator_abundance_apis_spring.tif')
        result_sum = numpy.float32(0.0)
        for _, data_block in pygeoprocessing.iterblocks(
                (result_raster_path, 1)):
            result_sum += numpy.sum(data_block)
        # the number below is just what the sum rounded to two decimal places
        # when I manually inspected a run that appeared to be correct.
        self.assertAlmostEqual(result_sum, 58.669518, places=2)

    def test_pollination_constant_abundance(self):
        """Pollination: regression testing when abundance is all 1."""
        from natcap.invest import pollination

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'clipped_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table_rel_all_ones.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'landcover_biophysical_table.csv')
        }
        pollination.execute(args)
        result_raster_path = os.path.join(
            self.workspace_dir, 'pollinator_abundance_apis_spring.tif')
        result_sum = numpy.float32(0.0)
        for _, data_block in pygeoprocessing.iterblocks(
                (result_raster_path, 1)):
            result_sum += numpy.sum(data_block)
        # the number below is just what the sum rounded to two decimal places
        # when I manually inspected a run that appeared to be correct.
        self.assertAlmostEqual(result_sum, 68.44777, places=2)

    def test_pollination_bad_guild_headers(self):
        """Pollination: testing that model detects bad guild headers."""
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)
        bad_guild_table_path = os.path.join(temp_path, 'bad_guild_table.csv')
        with open(bad_guild_table_path, 'w') as bad_guild_table:
            bad_guild_table.write(
                'species,nesting_suitability_cavity_index,alpha,'
                'relative_abundance\n')
            bad_guild_table.write(
                'apis,0.2,400,1.0\n')
            bad_guild_table.write(
                'bee,0.9,1400,0.1\n')
        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'clipped_landcover.tif'),
            'guild_table_path': bad_guild_table_path,
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'landcover_biophysical_table.csv'),
            'farm_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'farms.shp'),
        }
        with self.assertRaises(ValueError):
            pollination.execute(args)

    def test_pollination_bad_biophysical_headers(self):
        """Pollination: testing that model detects bad biophysical headers."""
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)
        bad_biophysical_table_path = os.path.join(
            temp_path, 'bad_biophysical_table.csv')
        with open(bad_biophysical_table_path, 'w') as bad_biophysical_table:
            bad_biophysical_table.write(
                'lucode,nesting_cavity_availability_index,nesting_ground_index\n'
                '1,0.3,0.2\n')
        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'clipped_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table.csv'),
            'landcover_biophysical_table_path': bad_biophysical_table_path,
            'farm_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'farms.shp'),
        }
        with self.assertRaises(ValueError):
            pollination.execute(args)

    def test_pollination_bad_cross_table_headers(self):
        """Pollination: ensure detection of missing headers in one table."""
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)
        bad_biophysical_table_path = os.path.join(
            temp_path, 'bad_biophysical_table.csv')
        # one table has only spring the other has only fall.
        with open(bad_biophysical_table_path, 'w') as bad_biophysical_table:
            bad_biophysical_table.write(
                'lucode,nesting_cavity_availability_index,nesting_ground_index,floral_resources_spring_index\n'
                '1,0.3,0.2,0.2\n')
        bad_guild_table_path = os.path.join(temp_path, 'bad_guild_table.csv')
        with open(bad_guild_table_path, 'w') as bad_guild_table:
            bad_guild_table.write(
                'species,nesting_suitability_cavity_index,'
                'foraging_activity_fall_index,alpha,relative_abundance\n')
            bad_guild_table.write(
                'apis,0.2,0.5,400,1.0\n')
            bad_guild_table.write(
                'bee,0.9,0.5,1400,0.5\n')
        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'clipped_landcover.tif'),
            'guild_table_path': bad_guild_table_path,
            'landcover_biophysical_table_path': bad_biophysical_table_path,
            'farm_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'farms.shp'),
        }
        with self.assertRaises(ValueError):
            pollination.execute(args)

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
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'clipped_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'landcover_biophysical_table.csv'),
            'farm_vector_path': farm_shape_path,
        }
        with self.assertRaises(ValueError):
            pollination.execute(args)

    def test_pollination_unequal_raster_pixel_size(self):
        """Pollination: regression testing sample data."""
        from natcap.invest import pollination

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input',
                'pollination_example_landcover_unequalsize.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table_simple.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input',
                'landcover_biophysical_table_simple.csv'),
            'farm_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'blueberry_ridge_farm.shp'),
        }
        # make empty result files to get coverage for removing if necessary
        result_files = ['farm_results.shp', 'total_pollinator_yield.tif',
                        'wild_pollinator_yield.tif']
        for file_name in result_files:
            f = open(os.path.join(self.workspace_dir, file_name), 'w')
            f.close()
        pollination.execute(args)
        expected_farm_yields = {
            'blueberry': {
                'y_tot': 0.48010949294,
                'y_wild': 0.13010949294
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

    @staticmethod
    def _test_same_files(base_file_list, directory_path):
        """Assert files in `base_list_path` are in `directory_path`.

        Parameters:
            base_file_list (list): a list of relative file paths.
            directory_path (string): a path to a directory whose contents will
                be checked against the files listed in `base_list_file`

        Returns:
            None

        Raises:
            AssertionError when there are files listed in `base_file_list`
                that don't exist in the directory indicated by `path`

        """
        missing_files = []
        for file_path in base_file_list:
            full_path = os.path.join(directory_path, file_path)
            if full_path == '':
                continue
            if not os.path.isfile(full_path):
                missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '\n'.join(missing_files))


class PollinationValidationTests(unittest.TestCase):
    """Tests for the Pollination Model ARGS_SPEC and validation."""

    def setUp(self):
        """Create list of always required arguments."""
        self.base_required_keys = [
            'workspace_dir',
            'landcover_raster_path',
            'guild_table_path',
            'landcover_biophysical_table_path',
        ]

    def test_missing_keys(self):
        """Pollination Validate: assert missing required keys."""
        from natcap.invest import pollination
        from natcap.invest import validation

        validation_errors = pollination.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)
