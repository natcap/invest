"""Module for Regression Testing the InVEST Hydropower module."""
import unittest
import tempfile
import shutil
import os

import pandas

import pygeoprocessing.testing

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'hydropower')


class HydropowerTests(unittest.TestCase):
    """Regression Tests for Annual Water Yield Hydropower Model."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across regression tests."""
        args = {
            'workspace_dir': workspace_dir,
            'lulc_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'landuse_90'),
            'depth_to_root_rest_layer_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater',
                'depth_to_root_rest_layer'),
            'precipitation_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'precip'),
            'pawc_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'pawc'),
            'eto_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'eto'),
            'watersheds_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'watersheds.shp'),
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'biophysical_table.csv'),
            'seasonality_constant': 5
        }
        return args

    def test_water_yield_subshed(self):
        """Hydro: testing water yield component only w/ subwatershed."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerTests.generate_base_args(self.workspace_dir)
        # TODO: delete this
        args['workspace_dir'] = 'hydropower_regression/test_water_yield_subshed'

        args['sub_watersheds_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp')
        args['results_suffix'] = 'test'
        hydropower_water_yield.execute(args)

        raster_results = ['aet_test.tif', 'fractp_test.tif', 'wyield_test.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(
                    REGRESSION_DATA, raster_path.replace('_test', '')))

        vector_results = ['watershed_results_wyield_test.shp',
                          'subwatershed_results_wyield_test.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(
                    REGRESSION_DATA, 'water_yield', vector_path.replace(
                        '_test', '')), 1e-3)

        table_results = ['watershed_results_wyield_test.csv',
                         'subwatershed_results_wyield_test.csv']
        for table_path in table_results:
            base_table = pandas.read_csv(
                os.path.join(args['workspace_dir'], 'output', table_path))
            expected_table = pandas.read_csv(
                os.path.join(
                    REGRESSION_DATA, 'water_yield',
                    table_path.replace('_test', '')))
            pandas.testing.assert_frame_equal(base_table, expected_table)

    def test_scarcity_subshed(self):
        """Hydro: testing Scarcity component w/ subwatershed."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerTests.generate_base_args(self.workspace_dir)
        # TODO: delete this
        args['workspace_dir'] = 'hydropower_regression/test_scarcity_subshed'

        args['calculate_water_scarcity'] = True
        args['demand_table_uri'] = os.path.join(
            SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv')
        args['sub_watersheds_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp')

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'scarcity', vector_path),
                1e-3)

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            base_table = pandas.read_csv(
                os.path.join(args['workspace_dir'], 'output', table_path))
            expected_table = pandas.read_csv(
                os.path.join(REGRESSION_DATA, 'scarcity', table_path))
            pandas.testing.assert_frame_equal(base_table, expected_table)

    def test_valuation_subshed(self):
        """Hydro: testing Valuation component w/ subwatershed."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerTests.generate_base_args(self.workspace_dir)
        # TODO: delete this
        args['workspace_dir'] = 'hydropower_regression/test_valuation_subshed'

        args['calculate_water_scarcity'] = True
        args['demand_table_uri'] = os.path.join(
            SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv')
        args['valuation_container'] = True
        args['valuation_table_uri'] = os.path.join(
            SAMPLE_DATA, 'Hydropower', 'input',
            'hydropower_valuation_table.csv')
        args['sub_watersheds_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp')

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'valuation', vector_path),
                1e-3)

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            base_table = pandas.read_csv(
                os.path.join(args['workspace_dir'], 'output', table_path))
            expected_table = pandas.read_csv(
                os.path.join(REGRESSION_DATA, 'valuation', table_path))
            pandas.testing.assert_frame_equal(base_table, expected_table)

    def test_validation(self):
        """Hydro: test failure cases on the validation function."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerTests.generate_base_args(
            self.workspace_dir)

        # default args should be fine
        self.assertEqual(hydropower_water_yield.validate(args), [])

        args_bad_vector = args.copy()
        args_bad_vector['watersheds_uri'] = args_bad_vector['eto_uri']
        bad_vector_list = hydropower_water_yield.validate(args_bad_vector)
        self.assertEqual(bad_vector_list[0][1], 'not a vector')

        args_bad_raster = args.copy()
        args_bad_raster['eto_uri'] = args_bad_raster['watersheds_uri']
        bad_raster_list = hydropower_water_yield.validate(args_bad_raster)
        self.assertEqual(bad_raster_list[0][1], 'not a raster')

        args_bad_file = args.copy()
        args_bad_file['eto_uri'] = 'non_existant_file.tif'
        bad_file_list = hydropower_water_yield.validate(args_bad_file)
        self.assertEqual(bad_file_list[0][1], 'not found on disk')

        args_missing_key = args.copy()
        del args_missing_key['eto_uri']
        with self.assertRaises(KeyError):
            hydropower_water_yield.validate(args_missing_key)
