"""Module for Regression Testing the InVEST Hydropower module."""
import unittest
import tempfile
import shutil
import os
import csv

from osgeo import ogr
import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'hydropower')


class HydropowerUnitTests(unittest.TestCase):
    """Unit tests for Annual Water Yield Hydropower Model."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_filter_dictionary(self):
        """Hydro: testing 'filter_dictionary' function."""
        from natcap.invest.hydropower import hydropower_water_yield

        test_dict = {
            0: {'field_1': 'hi', 'field_2': 'bye', 'field_3': 0},
            1: {'field_1': 'aloha', 'field_2': 'aloha', 'field_3': 1},
            2: {'field_1': 'hola', 'field_2': 'adios', 'field_3': 2}}

        keep_fields = ['field_1', 'field_3']

        results = hydropower_water_yield.filter_dictionary(test_dict, keep_fields)

        exp_results = {
            0: {'field_1': 'hi', 'field_3': 0},
            1: {'field_1': 'aloha', 'field_3': 1},
            2: {'field_1': 'hola', 'field_3': 2}}

        self.assertDictEqual(results, exp_results)

    def test_write_new_table(self):
        """Hydro: testing 'write_new_table' function."""
        from natcap.invest.hydropower import hydropower_water_yield

        temp_dir = self.workspace_dir
        filename = os.path.join(temp_dir, 'test_csv.csv')

        fields = ['id', 'precip', 'volume']

        data = {0: {'id':1, 'precip': 100, 'volume': 150},
                1: {'id':2, 'precip': 150, 'volume': 350},
                2: {'id':3, 'precip': 170, 'volume': 250}}

        hydropower_water_yield.write_new_table(filename, fields, data)
        # expected results as a dictionary, note that reading from csv will
        # leave values as strings
        exp_data = {0: {'id': '1', 'precip': '100', 'volume': '150'},
                    1: {'id': '2', 'precip': '150', 'volume': '350'},
                    2: {'id': '3', 'precip': '170', 'volume': '250'}}

        # to test the CSV was written correctly, we'll read back the data
        csv_file = open(filename, 'rb')
        reader = csv.DictReader(csv_file)
        # assert fieldnames are the same
        if fields != reader.fieldnames:
            raise AssertionError(
                "The fields from the CSV file are not correct. "
                "Expected vs Returned: %s vs %s", (fields, reader.fieldnames))

        data_row = 0
        for row in reader:
            # we expect there to only be 3 rows, as indicated by the dict
            # keys 0,1,2. If we come across more than 3 rows give assertion
            # error, instead of letting it error on 'keyerror'
            if data_row > 2:
                raise AssertionError(
                    "Expected 3 rows, got at least 4 returned. "
                    "4th row found is the following: %s", row)
            self.assertDictEqual(row, exp_data[data_row])
            data_row += 1

        csv_file.close()

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_add_dict_to_shape(self):
        """Hydro: testing 'add_dict_to_shape' function."""
        from natcap.invest.hydropower import hydropower_water_yield

        # 'two_poly_shape.shp was created with fields:
        # ['ws_id': 'int', 'wyield_mn': 'real', 'wyield_vol': 'real']
        # and with values of:
        # {'ws_id': 1, 'wyield_mn': 1000, 'wyield_vol': 1000},
        # {'ws_id': 2, 'wyield_mn': 1000, 'wyield_vol': 800}
        # using the script 'create_polygon_shapefile.py'
        shape_uri = os.path.join(
            REGRESSION_DATA, 'two_polygon_shape.shp')

        temp_dir = self.workspace_dir
        vector_uri = os.path.join(temp_dir, 'vector.shp')
        # make a copy of the shapefile that can take edits
        pygeoprocessing.geoprocessing.copy_datasource_uri(shape_uri, vector_uri)

        field_dict = {1: 50.0, 2: 10.5}
        field_name = 'precip'
        key = 'ws_id'

        hydropower_water_yield.add_dict_to_shape(
            vector_uri, field_dict, field_name, key)

        expected_results = {1: {'precip': 50.0}, 2: {'precip': 10.5}}

        # open the shapefile and check that the edits were made correctly.
        shape = ogr.Open(vector_uri)
        layer_count = shape.GetLayerCount()

        for layer_num in range(layer_count):
            layer = shape.GetLayer(layer_num)

            feat = layer.GetNextFeature()
            while feat is not None:
                ws_id = feat.GetField('ws_id')

                try:
                    field_val = feat.GetField(field_name)
                    pygeoprocessing.testing.assert_close(
                        expected_results[ws_id][field_name], field_val,
                        tolerance=1e-9)
                except ValueError:
                    raise AssertionError(
                        'Could not find field %s' % field_name)

                feat = layer.GetNextFeature()

class HydropowerRegressionTests(unittest.TestCase):
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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation(self):
        """Hydro: testing valuation component with no subwatershed."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerRegressionTests.generate_base_args(self.workspace_dir)
        args['water_scarcity_container'] = True
        args['demand_table_uri'] = os.path.join(
            SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv')
        args['valuation_container'] = True
        args['valuation_table_uri'] = os.path.join(
            SAMPLE_DATA, 'Hydropower', 'input',
            'hydropower_valuation_table.csv')

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(REGRESSION_DATA, raster_path),
                tolerance=1e-9)

        vector_results = ['watershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'valuation', vector_path),
                field_tolerance=1e-9)

        table_results = ['watershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'valuation', table_path),
                tolerance=1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_water_yield(self):
        """Hydro: testing water yield component only."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerRegressionTests.generate_base_args(self.workspace_dir)

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(REGRESSION_DATA, raster_path), tolerance=1e-9)

        vector_results = ['watershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'water_yield', vector_path),
                field_tolerance=1e-9)

        table_results = ['watershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'water_yield', table_path),
                tolerance=1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_water_yield_subshed(self):
        """Hydro: testing water yield component only w/ subwatershed."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerRegressionTests.generate_base_args(self.workspace_dir)

        args['sub_watersheds_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp')

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(REGRESSION_DATA, raster_path),
                tolerance=1e-9)

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'water_yield', vector_path),
                field_tolerance=1e-9)

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'water_yield', table_path),
                tolerance=1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_scarcity(self):
        """Hydro: testing Scarcity component."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerRegressionTests.generate_base_args(self.workspace_dir)

        args['water_scarcity_container'] = True
        args['demand_table_uri'] = os.path.join(
            SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv')

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(REGRESSION_DATA, raster_path),
                tolerance=1e-9)

        vector_results = ['watershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'scarcity', vector_path),
                field_tolerance=1e-9)

        table_results = ['watershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'scarcity', table_path),
                tolerance=1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_scarcity_subshed(self):
        """Hydro: testing Scarcity component w/ subwatershed."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerRegressionTests.generate_base_args(self.workspace_dir)

        args['water_scarcity_container'] = True
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
                os.path.join(REGRESSION_DATA, raster_path),
                tolerance=1e-9)

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'scarcity', vector_path),
                field_tolerance=1e-9)

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'scarcity', table_path),
                tolerance=1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation_subshed(self):
        """Hydro: testing Valuation component w/ subwatershed."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerRegressionTests.generate_base_args(self.workspace_dir)

        args['water_scarcity_container'] = True
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
                os.path.join(REGRESSION_DATA, raster_path),
                tolerance=1e-9)

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'valuation', vector_path),
                field_tolerance=1e-9)

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'valuation', table_path),
                tolerance=1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_suffix(self):
        """Hydro: testing that the suffix is handled correctly."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = {
            'workspace_dir': self.workspace_dir,
            'lulc_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'lulc_smoke.tif'),
            'depth_to_root_rest_layer_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'dtr_smoke.tif'),
            'precipitation_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'precip_smoke.tif'),
            'pawc_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'pawc_smoke.tif'),
            'eto_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'eto_smoke.tif'),
            'watersheds_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'watershed_smoke.shp'),
            'biophysical_table_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'biophysical_smoke.csv'),
            'seasonality_constant': 5,
            'water_scarcity_container': True,
            'demand_table_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'demand_smoke.csv'),
            'valuation_container': True,
            'valuation_table_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'valuation_params_smoke.csv'),
            'sub_watersheds_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'subwatershed_smoke.shp'),
            'results_suffix': 'test'
        }

        hydropower_water_yield.execute(args)

        raster_results = ['aet_test.tif', 'fractp_test.tif', 'wyield_test.tif']
        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel',
                    raster_path)))

        vector_results = ['watershed_results_wyield_test.shp',
                          'subwatershed_results_wyield_test.shp']
        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

        table_results = ['watershed_results_wyield_test.csv',
                         'subwatershed_results_wyield_test.csv']
        for table_path in table_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', table_path)))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_suffix_underscore(self):
        """Hydro: testing that a suffix w/ underscore is handled correctly."""
        from natcap.invest.hydropower import hydropower_water_yield

        args = {
            'workspace_dir': self.workspace_dir,
            'lulc_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'lulc_smoke.tif'),
            'depth_to_root_rest_layer_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'dtr_smoke.tif'),
            'precipitation_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'precip_smoke.tif'),
            'pawc_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'pawc_smoke.tif'),
            'eto_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'eto_smoke.tif'),
            'watersheds_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'watershed_smoke.shp'),
            'biophysical_table_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'biophysical_smoke.csv'),
            'seasonality_constant': 5,
            'water_scarcity_container': True,
            'demand_table_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'demand_smoke.csv'),
            'valuation_container': True,
            'valuation_table_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'valuation_params_smoke.csv'),
            'sub_watersheds_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'subwatershed_smoke.shp'),
            'results_suffix': '_test'
        }

        hydropower_water_yield.execute(args)

        raster_results = ['aet_test.tif', 'fractp_test.tif', 'wyield_test.tif']
        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel',
                    raster_path)))

        vector_results = ['watershed_results_wyield_test.shp',
                          'subwatershed_results_wyield_test.shp']
        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

        table_results = ['watershed_results_wyield_test.csv',
                         'subwatershed_results_wyield_test.csv']
        for table_path in table_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', table_path)))
