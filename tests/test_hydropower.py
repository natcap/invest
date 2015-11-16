"""Module for Regression Testing the InVEST Hydropower module."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', 'hydropower')


class HydropowerRegressionTests(unittest.TestCase):
    """Regression Test for Hydropower running Scarcity and Valuation."""

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
        """Regression test for Valuation component with no subwatershed."""
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
                os.path.join(REGRESSION_DATA, raster_path))

        vector_results = ['watershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'valuation', vector_path))

        table_results = ['watershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'valuation', table_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_water_yield(self):
        """Regression test for Biophysical component, Water Yield only.

        No subwatershed is used.
        """
        from natcap.invest.hydropower import hydropower_water_yield

        args = HydropowerRegressionTests.generate_base_args(self.workspace_dir)

        hydropower_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

        vector_results = ['watershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'water_yield', vector_path))

        table_results = ['watershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'water_yield', table_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_water_yield_subshed(self):
        """
        Regression test for Biophysical component, Water Yield.

        Subwatershed is included.
        """
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
                os.path.join(REGRESSION_DATA, raster_path))

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'water_yield', vector_path))

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'water_yield', table_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_scarcity(self):
        """
        Regression test for Biophysical component, Scarcity.

        No subwatershed is used.
        """
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
                os.path.join(REGRESSION_DATA, raster_path))

        vector_results = ['watershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'scarcity', vector_path))

        table_results = ['watershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'scarcity', table_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_scarcity_subshed(self):
        """
        Regression test for Biophysical component, Scarcity.

        Subwatershed is included.
        """
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
                os.path.join(REGRESSION_DATA, raster_path))

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'scarcity', vector_path))

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'scarcity', table_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation_subshed(self):
        """Regression test for Valuation component, subwatershed included."""
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
                os.path.join(REGRESSION_DATA, raster_path))

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'valuation', vector_path))

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'valuation', table_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_suffix(self):
        """Regression test for checking that the suffix is handled correctly."""
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
        args['results_suffix'] = 'test'

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
        """
        Regression test for checking that the suffix is handled correctly.

        Given an underscore in the suffix input, check that it is not
            duplicated in the file output.
        """
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
        args['results_suffix'] = '_test'

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
