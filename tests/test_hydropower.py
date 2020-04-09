"""Module for Regression Testing the InVEST Hydropower module."""
import unittest
import tempfile
import shutil
import os

import pandas
import pygeoprocessing.testing

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'hydropower',
    'input')
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
            'lulc_path': os.path.join(SAMPLE_DATA, 'lulc.tif'),
            'depth_to_root_rest_layer_path': os.path.join(
                SAMPLE_DATA,
                'depth_to_root_rest_layer.tif'),
            'precipitation_path': os.path.join(SAMPLE_DATA, 'precip.tif'),
            'pawc_path': os.path.join(SAMPLE_DATA, 'pawc.tif'),
            'eto_path': os.path.join(SAMPLE_DATA, 'eto.tif'),
            'watersheds_path': os.path.join(SAMPLE_DATA, 'watersheds.shp'),
            'biophysical_table_path': os.path.join(
                SAMPLE_DATA, 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'n_workers': -1,
        }
        return args

    def test_invalid_lulc_veg(self):
        """Hydro: catching invalid LULC_veg values."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerTests.generate_base_args(self.workspace_dir)

        new_lulc_veg_path = os.path.join(self.workspace_dir,
                                         'new_lulc_veg.csv')

        table_df = pandas.read_csv(args['biophysical_table_path'])
        table_df['LULC_veg'] = ['']*len(table_df.index)
        table_df.to_csv(new_lulc_veg_path)
        args['biophysical_table_path'] = new_lulc_veg_path

        with self.assertRaises(ValueError) as cm:
            hydropower_water_yield.execute(args)
        self.assertTrue('veg value must be either 1 or 0' in str(cm.exception))

        table_df = pandas.read_csv(args['biophysical_table_path'])
        table_df['LULC_veg'] = ['-1']*len(table_df.index)
        table_df.to_csv(new_lulc_veg_path)
        args['biophysical_table_path'] = new_lulc_veg_path

        with self.assertRaises(ValueError) as cm:
            hydropower_water_yield.execute(args)
        self.assertTrue('veg value must be either 1 or 0' in str(cm.exception))

    def test_water_yield_subshed(self):
        """Hydro: testing water yield component only w/ subwatershed."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerTests.generate_base_args(self.workspace_dir)
        args['sub_watersheds_path'] = os.path.join(
            SAMPLE_DATA, 'subwatersheds.shp')
        args['results_suffix'] = 'test'
        hydropower_water_yield.execute(args)

        raster_results = ['aet_test.tif', 'fractp_test.tif', 'wyield_test.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(
                    REGRESSION_DATA, raster_path.replace('_test', '')),
                1e-6)

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
        args['demand_table_path'] = os.path.join(
            SAMPLE_DATA, 'water_demand_table.csv')
        args['sub_watersheds_path'] = os.path.join(
            SAMPLE_DATA, 'subwatersheds.shp')

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(REGRESSION_DATA, raster_path),
                1e-6)

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
        args['demand_table_path'] = os.path.join(
            SAMPLE_DATA, 'water_demand_table.csv')
        args['valuation_table_path'] = os.path.join(
            SAMPLE_DATA, 'hydropower_valuation_table.csv')
        args['sub_watersheds_path'] = os.path.join(
            SAMPLE_DATA, 'subwatersheds.shp')

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel',
                    raster_path), os.path.join(REGRESSION_DATA, raster_path),
                1e-6)

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

        args = HydropowerTests.generate_base_args(self.workspace_dir)

        # default args should be fine
        self.assertEqual(hydropower_water_yield.validate(args), [])

        args_bad_vector = args.copy()
        args_bad_vector['watersheds_path'] = args_bad_vector['eto_path']
        bad_vector_list = hydropower_water_yield.validate(args_bad_vector)
        self.assertTrue('not be opened as a GDAL vector'
                        in bad_vector_list[0][1])

        args_bad_raster = args.copy()
        args_bad_raster['eto_path'] = args_bad_raster['watersheds_path']
        bad_raster_list = hydropower_water_yield.validate(args_bad_raster)
        self.assertTrue('not be opened as a GDAL raster'
                        in bad_raster_list[0][1])

        args_bad_file = args.copy()
        args_bad_file['eto_path'] = 'non_existant_file.tif'
        bad_file_list = hydropower_water_yield.validate(args_bad_file)
        self.assertTrue('File not found' in bad_file_list[0][1])

        args_missing_key = args.copy()
        del args_missing_key['eto_path']
        validation_warnings = hydropower_water_yield.validate(
            args_missing_key)
        self.assertEqual(
            validation_warnings,
            [(['eto_path'], 'Key is missing from the args dict')])

        # ensure that a missing landcover code in the biophysical table will
        # raise an exception that's helpful
        args_bad_biophysical_table = args.copy()
        bad_biophysical_path = os.path.join(
            self.workspace_dir, 'bad_biophysical_table.csv')
        with open(bad_biophysical_path, 'wb') as bad_biophysical_file:
            with open(args['biophysical_table_path'], 'rb') as (
                    biophysical_table_file):
                lines_to_write = 2
                for line in biophysical_table_file.readlines():
                    bad_biophysical_file.write(line)
                    lines_to_write -= 1
                    if lines_to_write == 0:
                        break
        args_bad_biophysical_table['biophysical_table_path'] = (
            bad_biophysical_path)
        with self.assertRaises(ValueError) as cm:
            hydropower_water_yield.execute(args_bad_biophysical_table)
        actual_message = str(cm.exception)
        self.assertTrue(
            'did not have corresponding entries in the biophysical table' in
            actual_message, actual_message)

        # ensure that a missing landcover code in the demand table will
        # raise an exception that's helpful
        args_bad_biophysical_table = args.copy()
        bad_biophysical_path = os.path.join(
            self.workspace_dir, 'bad_biophysical_table.csv')
        with open(bad_biophysical_path, 'wb') as bad_biophysical_file:
            with open(args['biophysical_table_path'], 'rb') as (
                    biophysical_table_file):
                lines_to_write = 2
                for line in biophysical_table_file.readlines():
                    bad_biophysical_file.write(line)
                    lines_to_write -= 1
                    if lines_to_write == 0:
                        break
        args_bad_demand_table = args.copy()
        bad_demand_path = os.path.join(
            self.workspace_dir, 'bad_demand_table.csv')
        args_bad_demand_table['demand_table_path'] = (
            bad_demand_path)
        with open(bad_demand_path, 'wb') as bad_demand_file:
            with open(os.path.join(
                SAMPLE_DATA, 'water_demand_table.csv'), 'rb') as (
                    demand_table_file):
                lines_to_write = 2
                for line in demand_table_file.readlines():
                    bad_demand_file.write(line)
                    lines_to_write -= 1
                    if lines_to_write == 0:
                        break

        # ensure that a missing watershed id the valuation table will
        # raise an exception that's helpful
        with self.assertRaises(ValueError) as cm:
            hydropower_water_yield.execute(args_bad_demand_table)
        actual_message = str(cm.exception)
        self.assertTrue(
            'did not have corresponding entries in the water demand table' in
            actual_message, actual_message)

        args_bad_valuation_table = args.copy()
        bad_valuation_path = os.path.join(
            self.workspace_dir, 'bad_valuation_table.csv')
        args_bad_valuation_table['valuation_table_path'] = (
            bad_valuation_path)
        # args contract requires a demand table if there is a valuation table
        args_bad_valuation_table['demand_table_path'] = os.path.join(
            SAMPLE_DATA, 'water_demand_table.csv')

        with open(bad_valuation_path, 'wb') as bad_valuation_file:
            with open(os.path.join(
                SAMPLE_DATA, 'hydropower_valuation_table.csv'), 'rb') as (
                    valuation_table_file):
                lines_to_write = 2
                for line in valuation_table_file.readlines():
                    bad_valuation_file.write(line)
                    lines_to_write -= 1
                    if lines_to_write == 0:
                        break

        with self.assertRaises(ValueError) as cm:
            hydropower_water_yield.execute(args_bad_valuation_table)
        actual_message = str(cm.exception)
        self.assertTrue(
            'but are not found in the valuation table' in
            actual_message, actual_message)
