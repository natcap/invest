"""Module for Regression Testing the InVEST Pollination model."""
import os
import shutil
import tempfile
import unittest

import numpy
import pandas
import pygeoprocessing
import shapely.geometry
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

gdal.UseExceptions()
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
    pixel_size = (30, -30)
    no_data = -1

    pygeoprocessing.numpy_array_to_raster(
        array, no_data, pixel_size, origin, projection_wkt,
        base_raster_path)


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

    def test_pollination_regression_multiple_seasons(self):
        """Pollination: regression testing sample data with two seasons."""
        from natcap.invest import pollination

        guild_table_path = os.path.join(self.workspace_dir, 'guild_table.csv')
        biophysical_table_path = os.path.join(
            self.workspace_dir, 'biophysical_table.csv')
        farm_vector_path = os.path.join(self.workspace_dir, 'farms.shp')

        pandas.DataFrame({
            'SPECIES': ['Apis'],
            'nesting_suitability_cavity_index': [1],
            'foraging_activity_spring_index': [1],
            'foraging_activity_summer_index': [0.1],
            'alpha': [500],
            'relative_abundance': [1]
        }).to_csv(guild_table_path)

        pandas.DataFrame({
            'lucode': [1, 2, 3, 4],
            'nesting_cavity_availability_index': [0.05, 0.8, 0.3, 0.05],
            'floral_resources_spring_index': [0.9, 0.3, 0.8, 0],
            'floral_resources_summer_index': [0.2, 0.1, 0.2, 0]
        }).to_csv(biophysical_table_path)

        landcover_raster_path = os.path.join(
            REGRESSION_DATA, 'input', 'pollination_example_landcover.tif')
        lulc_raster_info = pygeoprocessing.get_raster_info(landcover_raster_path)
        # The farm will occupy a space within the landcover raster, which is
        # about 1300x1300 meters in extent
        farm_geom = shapely.box(*lulc_raster_info['bounding_box']).buffer(-400)
        fields = {
            'lucode': ogr.OFTInteger,
            'crop_type': ogr.OFTString,
            'half_sat': ogr.OFTReal,
            'season': ogr.OFTString,
            'fr_spring': ogr.OFTReal,
            'fr_summer': ogr.OFTReal,
            'n_cavity': ogr.OFTReal,
            'p_dep': ogr.OFTReal,
            'p_managed': ogr.OFTReal
        }
        attributes = {
            'lucode': 1,
            'crop_type': 'blueberry',
            'half_sat': 0.5,
            'season': 'spring',
            'fr_spring': 0.9,
            'fr_summer': 0.1,
            'n_cavity': 0.05,
            'p_dep': 0.65,
            'p_managed': 0
        }
        pygeoprocessing.shapely_geometry_to_vector(
            shapely_geometry_list=[farm_geom],
            target_vector_path=farm_vector_path,
            projection_wkt=lulc_raster_info['projection_wkt'],
            vector_format='ESRI Shapefile',
            fields=fields,
            attribute_list=[attributes])

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': landcover_raster_path,
            'guild_table_path': guild_table_path,
            'landcover_biophysical_table_path': biophysical_table_path,
            'farm_vector_path': farm_vector_path,
        }

        pollination.execute(args)
        expected_farm_yields = {
            'blueberry': {
                'y_tot': 0.42998552322,
                'y_wild': 0.07998548448
            },
        }
        result_vector = gdal.OpenEx(
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
            result_layer = None
            result_vector = None

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
        with self.assertRaises(ValueError) as cm:
            pollination.execute(args)
        self.assertIn('Missing expected header', str(cm.exception))

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
        with self.assertRaises(ValueError) as cm:
            pollination.execute(args)
        self.assertIn(
            ('Found seasons in farm polygon that were not specified in the '
             'biophysical table'), str(cm.exception))

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
        with self.assertRaises(ValueError) as cm:
            pollination.execute(args)
        self.assertIn('Expected a biophysical and guild entry for',
                      str(cm.exception))

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
        self.assertAlmostEqual(result_sum, 58.407155, places=2)

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
        self.assertAlmostEqual(result_sum, 68.14167, places=2)

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
        with self.assertRaises(ValueError) as cm:
            pollination.execute(args)
        self.assertIn(
            'Expected a header in guild table that matched the pattern',
            str(cm.exception))

    def test_pollination_bad_biophysical_headers(self):
        """Pollination: testing that model detects bad biophysical headers."""
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)
        bad_biophysical_table_path = os.path.join(
            temp_path, 'bad_biophysical_table.csv')
        with open(bad_biophysical_table_path, 'w') as bad_biophysical_table:
            bad_biophysical_table.write(
                'lucode,nesting_cavity_availability_index,'
                'nesting_ground_index\n'
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
        with self.assertRaises(ValueError) as cm:
            pollination.execute(args)
        self.assertIn(
            'Expected a header in biophysical table that matched the pattern',
            str(cm.exception))

    def test_pollination_missing_lulc_values(self):
        """Pollination: testing that model detects missing lulc values."""
        import pandas
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)

        args = {
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'landcover_raster_path': os.path.join(
                REGRESSION_DATA, 'input', 'clipped_landcover.tif'),
            'guild_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'guild_table.csv'),
            'landcover_biophysical_table_path': os.path.join(
                REGRESSION_DATA, 'input', 'landcover_biophysical_table.csv'),
            'farm_vector_path': os.path.join(
                REGRESSION_DATA, 'input', 'farms.shp'),
        }

        bad_biophysical_table_path = os.path.join(
            temp_path, 'bad_biophysical_table.csv')

        bio_df = pandas.read_csv(args['landcover_biophysical_table_path'])
        bio_df = bio_df[bio_df['lucode'] != 1]
        bio_df.to_csv(bad_biophysical_table_path)
        bio_df = None

        args['landcover_biophysical_table_path'] = bad_biophysical_table_path

        with self.assertRaises(ValueError) as cm:
            pollination.execute(args)

        self.assertIn(
            "The missing values found in the LULC raster but not the table"
            " are: [1]", str(cm.exception))

    def test_pollination_bad_cross_table_headers(self):
        """Pollination: ensure detection of missing headers in one table."""
        from natcap.invest import pollination

        temp_path = tempfile.mkdtemp(dir=self.workspace_dir)
        bad_biophysical_table_path = os.path.join(
            temp_path, 'bad_biophysical_table.csv')
        # one table has only spring the other has only fall.
        with open(bad_biophysical_table_path, 'w') as bad_biophysical_table:
            bad_biophysical_table.write(
                'lucode,nesting_cavity_availability_index,'
                'nesting_ground_index,floral_resources_spring_index\n'
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
        with self.assertRaises(ValueError) as cm:
            pollination.execute(args)
        self.assertIn(
            "Expected a biophysical, guild, and farm entry for 'fall'",
            str(cm.exception))

    def test_pollination_bad_farm_type(self):
        """Pollination: ensure detection of bad farm geometry type."""
        from natcap.invest import pollination

        # make some fake farm points
        point_geom = [shapely.geometry.Point(20, - 20)]

        farm_shape_path = os.path.join(self.workspace_dir, 'point_farm.shp')
        # Create the point shapefile
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()

        fields = {
            'crop_type': ogr.OFTString,
            'half_sat': ogr.OFTReal,
            'p_managed': ogr.OFTReal}
        attrs = [
            {'crop_type': 'test', 'half_sat': 0.5, 'p_managed': 0.5}]

        pygeoprocessing.shapely_geometry_to_vector(
            point_geom, farm_shape_path, projection_wkt, 'ESRI Shapefile',
            fields=fields, attribute_list=attrs, ogr_geom_type=ogr.wkbPoint)

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
        with self.assertRaises(ValueError) as cm:
            pollination.execute(args)
        self.assertIn("Farm layer not a polygon type", str(cm.exception))

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

        Args:
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

    def test_parse_scenario_variables(self):
        """Test `_parse_scenario_variables`"""
        from natcap.invest.pollination import _parse_scenario_variables

        def _create_guild_table_csv(output_path):
            data = {"species": ["Bee_A", "Bee_B", "Butterfly_C"]}
            df = pandas.DataFrame(data).set_index("species")

            df["nesting_suitability_soil_index"] = [0.8, 0.5, 0.2]
            df["nesting_suitability_wood_index"] = [0.3, 0.2, 0.4]
            df["foraging_activity_spring_index"] = [1.0, 0.8, 0.6]
            df["foraging_activity_summer_index"] = [.9, 0.6, 0.3]
            df["alpha"] = [150, 200, 120]
            df["relative_abundance"] = [0.4, 0.3, 0.2]

            df.to_csv(output_path)

        def _create_biophysical_table(output_path):
            data = {"lucode": [100, 200, 300]}
            df = pandas.DataFrame(data).set_index("lucode")

            df["nesting_soil_availability_index"] = [0.7, 0.4, 0.6]
            df["nesting_wood_availability_index"] = [0.1, 0.2, 0.6]
            df["floral_resources_spring_index"] = [0.3, 0.3, 0.8]
            df["floral_resources_summer_index"] = [0.9, 0.6, 0.3]

            df.to_csv(output_path)

        def _create_farm_vector(output_path):
            from shapely import Polygon

            shapely_geometry_list = [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
            ]

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(26910)
            projection_wkt = srs.ExportToWkt()

            fields = {"crop_type": ogr.OFTString, "half_sat": ogr.OFTReal,
                      "season": ogr.OFTString, "fr_spring": ogr.OFTReal,
                      "fr_summer": ogr.OFTReal,
                      "n_soil": ogr.OFTReal, "n_wood": ogr.OFTReal,
                      "p_dep": ogr.OFTReal,
                      "p_managed": ogr.OFTReal}

            attribute_list = [{
                "crop_type": "barley", "half_sat": 0.5, "season": "spring",
                "fr_spring": 0.8, "fr_summer": 0.3, "n_soil": 0.7,
                "n_wood": 0.5, "p_dep": 0.9, "p_managed": 0.4},
               {"crop_type": "almonds", "half_sat": 0.7, "season": "summer",
                "fr_spring": 0.4, "fr_summer": 0.9, "n_soil": 0.5,
                "n_wood": 0.6, "p_dep": 0.8, "p_managed": 0.6}]

            pygeoprocessing.shapely_geometry_to_vector(
                shapely_geometry_list, output_path, projection_wkt,
                "ESRI Shapefile", fields, attribute_list,
                ogr_geom_type=ogr.wkbPolygon)

        def _generate_output_dict():
            return {'season_list': ['spring', 'summer'],
                    'substrate_list': ['soil', 'wood'],
                    'species_list': ['bee_a', 'bee_b', 'butterfly_c'],
                    'alpha_value': {'bee_a': 150., 'bee_b': 200., 'butterfly_c': 120.},
                    'species_abundance': {'bee_a': 0.44444444444444453,
                                          'bee_b': 0.33333333333333337,
                                          'butterfly_c': 0.22222222222222227},
                    'species_foraging_activity': {
                        ('bee_a', 'spring'): 0.5263157894736842,
                        ('bee_a', 'summer'): 0.4736842105263158,
                        ('bee_b', 'spring'): 0.5714285714285715,
                        ('bee_b', 'summer'): 0.4285714285714286,
                        ('butterfly_c', 'spring'): 0.6666666666666667,
                        ('butterfly_c', 'summer'): 0.33333333333333337},
                    'landcover_substrate_index': {
                        'soil': {100: .7, 200: .4, 300: 0.6},
                        'wood': {100: 0.1, 200: 0.2, 300: .6}},
                    'landcover_floral_resources': {
                        'spring': {100: 0.3, 200: 0.3, 300: 0.8},
                        'summer': {100: 0.9, 200: 0.6, 300: 0.3}},
                    'species_substrate_index': {
                        'bee_a': {'soil': 0.8, 'wood': 0.3},
                        'bee_b': {'soil': 0.5, 'wood': 0.2},
                        'butterfly_c': {'soil': 0.2, 'wood': 0.4}},
                    'foraging_activity_index': {
                        ('bee_a', 'spring'): 1.0, ('bee_a', 'summer'): 0.9,
                        ('bee_b', 'spring'): 0.8, ('bee_b', 'summer'): 0.6,
                        ('butterfly_c', 'spring'): 0.6,
                        ('butterfly_c', 'summer'): 0.3}}

        args = {'guild_table_path':
                os.path.join(self.workspace_dir, "guild_table.csv"),
                'landcover_biophysical_table_path':
                os.path.join(self.workspace_dir, "biophysical_table.csv"),
                'farm_vector_path': os.path.join(self.workspace_dir, "farm.shp")}

        _create_guild_table_csv(args['guild_table_path'])
        _create_biophysical_table(args['landcover_biophysical_table_path'])
        _create_farm_vector(args['farm_vector_path'])

        actual_dict = _parse_scenario_variables(args)
        expected_dict = _generate_output_dict()

        self.assertDictEqual(actual_dict, expected_dict)

    def test_calculate_habitat_nesting_index(self):
        """Test `_calculate_habitat_nesting_index`"""
        from natcap.invest.pollination import _calculate_habitat_nesting_index

        substrate_path_map = {
            "wood": os.path.join(self.workspace_dir, "wood.tif"),
            "soil": os.path.join(self.workspace_dir, "soil.tif")
        }

        make_simple_raster(substrate_path_map["wood"],
                           numpy.array([[5, 6], [3, 2]]))
        make_simple_raster(substrate_path_map["soil"],
                           numpy.array([[2, 12], [1, 0]]))

        species_substrate_index_map = {"wood": 0.8, "soil": 0.5}

        target_habitat_nesting_index_path = os.path.join(
            self.workspace_dir, "habitat_nesting.tif")

        _calculate_habitat_nesting_index(
            substrate_path_map, species_substrate_index_map,
            target_habitat_nesting_index_path)

        # read habitat nesting tif
        habitat_raster = gdal.Open(target_habitat_nesting_index_path)
        band = habitat_raster.GetRasterBand(1)
        actual_array = band.ReadAsArray()
        expected_array = numpy.array([[4, 6], [2.4, 1.6]])

        numpy.testing.assert_allclose(actual_array, expected_array)


class PollinationValidationTests(unittest.TestCase):
    """Tests for the Pollination Model MODEL_SPEC and validation."""

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
