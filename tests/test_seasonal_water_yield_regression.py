"""InVEST Seasonal water yield model tests that use the InVEST sample data."""
import glob
import unittest
import tempfile
import shutil
import os

import numpy
from osgeo import ogr

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'seasonal_water_yield')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'seasonal_water_yield')


class SeasonalWaterYieldUnusualDataTests(unittest.TestCase):
    """Tests for InVEST Seasonal Water Yield model that cover cases where
    input data are in an unusual corner case"""

    def setUp(self):
        """Make tmp workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace after test is done."""
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def test_ambiguous_precip_data(self):
        """SWY test case where there are more than 12 precipitation files"""

        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        test_precip_dir = os.path.join(self.workspace_dir, 'test_precip_dir')
        shutil.copytree(
            os.path.join(SAMPLE_DATA, 'precip_dir'), test_precip_dir)
        shutil.copy(
            os.path.join(test_precip_dir, 'precip_mm_3.tif'),
            os.path.join(test_precip_dir, 'bonus_precip_mm_3.tif'))

        # A placeholder args that has the property that the aoi_path will be
        # the same name as the output aggregate vector
        args = {
            'workspace_dir': self.workspace_dir,
            'aoi_path': os.path.join(SAMPLE_DATA, 'watersheds.shp'),
            'alpha_m': '1/12',
            'beta_i': '1.0',
            'biophysical_table_path': os.path.join(
                SAMPLE_DATA, 'biophysical_table.csv'),
            'dem_raster_path': os.path.join(SAMPLE_DATA, 'dem.tif'),
            'et0_dir': os.path.join(SAMPLE_DATA, 'eto_dir'),
            'gamma': '1.0',
            'lulc_raster_path': os.path.join(SAMPLE_DATA, 'lulc.tif'),
            'precip_dir': test_precip_dir,  # test constructed one
            'rain_events_table_path': os.path.join(
                SAMPLE_DATA, 'rain_events_table.csv'),
            'soil_group_path': os.path.join(SAMPLE_DATA, 'soil_group.tif'),
            'threshold_flow_accumulation': '1000',
            'user_defined_climate_zones': False,
            'user_defined_local_recharge': False,
            'monthly_alpha': False,
        }

        with self.assertRaises(ValueError):
            seasonal_water_yield.execute(args)

    def test_precip_data_missing(self):
        """SWY test case where there is a missing precipitation file"""

        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        test_precip_dir = os.path.join(self.workspace_dir, 'test_precip_dir')
        shutil.copytree(
            os.path.join(SAMPLE_DATA, 'precip_dir'), test_precip_dir)
        os.remove(os.path.join(test_precip_dir, 'precip_mm_3.tif'))

        # A placeholder args that has the property that the aoi_path will be
        # the same name as the output aggregate vector
        args = {
            'workspace_dir': self.workspace_dir,
            'aoi_path': os.path.join(SAMPLE_DATA, 'watersheds.shp'),
            'alpha_m': '1/12',
            'beta_i': '1.0',
            'biophysical_table_path': os.path.join(
                SAMPLE_DATA, 'biophysical_table.csv'),
            'dem_raster_path': os.path.join(SAMPLE_DATA, 'dem.tif'),
            'et0_dir': os.path.join(SAMPLE_DATA, 'eto_dir'),
            'gamma': '1.0',
            'lulc_raster_path': os.path.join(SAMPLE_DATA, 'lulc.tif'),
            'precip_dir': test_precip_dir,  # test constructed one
            'rain_events_table_path': os.path.join(
                SAMPLE_DATA, 'rain_events_table.csv'),
            'soil_group_path': os.path.join(SAMPLE_DATA, 'soil_group.tif'),
            'threshold_flow_accumulation': '1000',
            'user_defined_climate_zones': False,
            'user_defined_local_recharge': False,
            'monthly_alpha': False,
        }

        with self.assertRaises(ValueError):
            seasonal_water_yield.execute(args)

    def test_aggregate_vector_preexists(self):
        """SWY test that model deletes a preexisting aggregate output result"""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # Set up data so there is enough code to do an aggregate over the
        # rasters but the output vector already exists
        for file_path in glob.glob(os.path.join(SAMPLE_DATA, "watershed.*")):
            shutil.copy(file_path, self.workspace_dir)
        aoi_path = os.path.join(SAMPLE_DATA, 'watershed.shp')
        l_path = os.path.join(REGRESSION_DATA, 'L.tif')
        aggregate_vector_path = os.path.join(
            self.workspace_dir, 'watershed.shp')
        seasonal_water_yield._aggregate_recharge(
            aoi_path, l_path, l_path, aggregate_vector_path)

        # test if aggregate is expected
        tolerance_places = 1  # this was an experimentally acceptable value
        agg_results_base_path = os.path.join(
            REGRESSION_DATA, 'l_agg_results.csv')
        result_vector = ogr.Open(aggregate_vector_path)
        result_layer = result_vector.GetLayer()
        incorrect_value_list = []
        with open(agg_results_base_path, 'rb') as agg_result_file:
            for line in agg_result_file:
                fid, vri_sum, qb_val = [float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                for field, value in [('vri_sum', vri_sum), ('qb', qb_val)]:
                    if not numpy.isclose(
                            feature.GetField(field), value, rtol=1e-6):
                        incorrect_value_list.append(
                            'Unexpected value on feature %d, '
                            'expected %f got %f' % (
                                fid, value, feature.GetField(field)))
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None

        if incorrect_value_list:
            raise AssertionError('\n' + '\n'.join(incorrect_value_list))

    def test_duplicate_aoi_assertion(self):
        """SWY ensure model halts when AOI path identical to output vector"""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # A placeholder args that has the property that the aoi_path will be
        # the same name as the output aggregate vector
        args = {
            'workspace_dir': self.workspace_dir,
            'aoi_path': os.path.join(
                self.workspace_dir, 'aggregated_results_foo.shp'),
            'results_suffix': 'foo',

            'alpha_m': '1/12',
            'beta_i': '1.0',
            'biophysical_table_path': os.path.join(
                SAMPLE_DATA, 'biophysical_table.csv'),
            'dem_raster_path': os.path.join(SAMPLE_DATA, 'dem.tif'),
            'et0_dir': os.path.join(SAMPLE_DATA, 'eto_dir'),
            'gamma': '1.0',
            'lulc_raster_path': os.path.join(SAMPLE_DATA, 'lulc.tif'),
            'precip_dir': os.path.join(SAMPLE_DATA, 'precip_dir'),
            'rain_events_table_path': os.path.join(
                SAMPLE_DATA, 'rain_events_table.csv'),
            'soil_group_path': os.path.join(SAMPLE_DATA, 'soil_group.tif'),
            'threshold_flow_accumulation': '1000',
            'user_defined_climate_zones': False,
            'user_defined_local_recharge': False,
            'monthly_alpha': False,
        }
        with self.assertRaises(ValueError):
            seasonal_water_yield.execute(args)



class SeasonalWaterYieldRegressionTests(unittest.TestCase):
    """Regression tests for InVEST Seasonal Water Yield model"""

    def setUp(self):
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across all three regression
        tests"""
        args = {
            'alpha_m': '1/12',
            'aoi_path': os.path.join(SAMPLE_DATA, 'watershed.shp'),
            'beta_i': '1.0',
            'biophysical_table_path': os.path.join(
                SAMPLE_DATA, 'biophysical_table.csv'),
            'dem_raster_path': os.path.join(SAMPLE_DATA, 'dem.tif'),
            'et0_dir': os.path.join(SAMPLE_DATA, 'eto_dir'),
            'gamma': '1.0',
            'lulc_raster_path': os.path.join(SAMPLE_DATA, 'lulc.tif'),
            'precip_dir': os.path.join(SAMPLE_DATA, 'precip_dir'),
            'rain_events_table_path': os.path.join(
                SAMPLE_DATA, 'rain_events_table.csv'),
            'results_suffix': '',
            'soil_group_path': os.path.join(SAMPLE_DATA, 'soil_group.tif'),
            'threshold_flow_accumulation': '1000',
            'workspace_dir': workspace_dir,
        }
        return args

    def test_base_regression(self):
        """SWY base regression test on sample data

        Executes SWY in default mode and checks that the output files are
        generated and that the aggregate shapefile fields are the same as the
        regression case."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)
        # make args explicit that this is a base run of SWY
        args['user_defined_climate_zones'] = False
        args['user_defined_local_recharge'] = False
        args['monthly_alpha'] = False
        args['results_suffix'] = ''

        seasonal_water_yield.execute(args)

        SeasonalWaterYieldRegressionTests._assert_regression_results_equal(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_base.txt'),
            os.path.join(args['workspace_dir'], 'aggregated_results.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_base.csv'))

    def test_monthly_alpha_regression(self):
        """SWY monthly alpha values regression test on sample data

        Executes SWY using the monthly alpha table and checks that the output
        files are generated and that the aggregate shapefile fields are the
        same as the regression case."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)
        # make args explicit that this is a base run of SWY
        args['user_defined_climate_zones'] = False
        args['user_defined_local_recharge'] = False
        args['monthly_alpha'] = True
        args['monthly_alpha_path'] = os.path.join(
            SAMPLE_DATA, 'monthly_alpha.csv')
        args['results_suffix'] = ''

        seasonal_water_yield.execute(args)

        SeasonalWaterYieldRegressionTests._assert_regression_results_equal(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_base.txt'),
            os.path.join(args['workspace_dir'], 'aggregated_results.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_base.csv'))

    def test_climate_zones_regression(self):
        """SWY climate zone regression test on sample data

        Executes SWY in climate zones mode and checks that the output files are
        generated and that the aggregate shapefile fields are the same as the
        regression case."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)
        # modify args to account for climate zones defined
        args['climate_zone_raster_path'] = os.path.join(
            SAMPLE_DATA, 'climate_zones.tif')
        args['climate_zone_table_path'] = os.path.join(
            SAMPLE_DATA, 'climate_zone_events.csv')
        args['user_defined_climate_zones'] = True
        args['user_defined_local_recharge'] = False
        args['monthly_alpha'] = False
        args['results_suffix'] = 'cz'

        seasonal_water_yield.execute(args)

        SeasonalWaterYieldRegressionTests._assert_regression_results_equal(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_cz.txt'),
            os.path.join(
                args['workspace_dir'], 'aggregated_results_cz.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_cz.csv'))

    def test_user_recharge(self):
        """SWY user recharge regression test on sample data

        Executes SWY in user defined local recharge mode and checks that the
        output files are generated and that the aggregate shapefile fields
        are the same as the regression case."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)
        # modify args to account for user recharge
        args['user_defined_climate_zones'] = False
        args['user_defined_local_recharge'] = True
        args['monthly_alpha'] = False
        args['results_suffix'] = ''
        args['l_path'] = os.path.join(REGRESSION_DATA, 'L.tif')

        seasonal_water_yield.execute(args)

        SeasonalWaterYieldRegressionTests._assert_regression_results_equal(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_user_recharge.txt'),
            os.path.join(args['workspace_dir'], 'aggregated_results.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_base.csv'))

    @staticmethod
    def _assert_regression_results_equal(
            workspace_dir, file_list_path, result_vector_path,
            agg_results_path):
        """Test the state of the workspace against the expected list of files
        and aggregated results.

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

         # Test that the workspace has the same files as we expect
        SeasonalWaterYieldRegressionTests._test_same_files(
            file_list_path, workspace_dir)

        # we expect a file called 'aggregated_results.shp'
        result_vector = ogr.Open(result_vector_path)
        result_layer = result_vector.GetLayer()

        # The tolerance of 3 digits after the decimal was determined by
        # experimentation on the application with the given range of numbers.
        # This is an apparently reasonable approach as described by ChrisF:
        # http://stackoverflow.com/a/3281371/42897
        # and even more reading about picking numerical tolerance (it's hard):
        # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        tolerance_places = 3

        with open(agg_results_path, 'rb') as agg_result_file:
            for line in agg_result_file:
                fid, vri_sum, qb_val = [float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                for field, value in [('vri_sum', vri_sum), ('qb', qb_val)]:
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
        """Assert that the files listed in `base_list_path` are also in the
        directory pointed to by `directory_path`.

        Parameters:
            base_list_path (string): a path to a file that has one relative
                file path per line.
            directory_path (string): a path to a directory whose contents will
                be checked against the files listed in `base_list_file`

        Returns:
            None

        Raises:
            AssertionError when there are files listed in `base_list_file`
                that don't exist in the directory indicated by `path`"""

        missing_files = []
        with open(base_list_path, 'r') as file_list:
            for file_path in file_list:
                full_path = os.path.join(directory_path, file_path.rstrip())
                if full_path == '':
                    #skip blank lines
                    continue
                if not os.path.isfile(full_path):
                    missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '\n'.join(missing_files))
