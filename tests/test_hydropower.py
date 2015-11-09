"""Module for Regression Testing the InVEST Hydropower module."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', 'hydropower')


class ValuationTest(unittest.TestCase):
    """
    Regression Test for Hydropower running Scarcity and Valuation components,
    """
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for Valuation component with no subwatershed.
        """
        from natcap.invest.hydropower import hydropower_water_yield
        args = {
            'workspace_dir': tempfile.mkdtemp(),
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
            'seasonality_constant': 5,
            'water_scarcity_container': True,
            'demand_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv'),
            'valuation_container': True,
            'valuation_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input',
                'hydropower_valuation_table.csv'),
        }
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

        shutil.rmtree(args['workspace_dir'])

class WaterYieldTest(unittest.TestCase):
    """
    Regression Test for Hydropower running Water Yield component.
    """
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for Biophysical component, Water Yield only,
            no subwatershed.
        """
        from natcap.invest.hydropower import hydropower_water_yield
        args = {
            'workspace_dir': tempfile.mkdtemp(),
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
            'seasonality_constant': 5,
            'results_suffix': '',
            'water_scarcity_container': False,
            'valuation_container': False
        }
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

        shutil.rmtree(args['workspace_dir'])

class WaterYieldSubwatershedTest(unittest.TestCase):
    """
    Regression Test for Hydropower running Water Yield component with
        subwatershed.
    """
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for Biophysical component, Water Yield,
            subwatershed included.
        """
        from natcap.invest.hydropower import hydropower_water_yield
        args = {
            'workspace_dir': tempfile.mkdtemp(),
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
            'sub_watersheds_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp'),
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
            'water_scarcity_container': False,
            'valuation_container': False
        }
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

        shutil.rmtree(args['workspace_dir'])

class ScarcityTest(unittest.TestCase):
    """
    Regression Test for Hydropower running Scarcity component.
    """
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for Biophysical component, Scarcity, no subwatershed.
        """
        from natcap.invest.hydropower import hydropower_water_yield
        args = {
            'workspace_dir': tempfile.mkdtemp(),
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
            'seasonality_constant': 5,
            'results_suffix': '',
            'water_scarcity_container': True,
            'demand_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv'),
            'valuation_container': False
        }
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

        shutil.rmtree(args['workspace_dir'])

class ScarcitySubwatershedTest(unittest.TestCase):
    """
    Regression Test for Hydropower running Scarcity components with
        subwatershed.
    """
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for Biophysical component, Scarcity,
            subwatershed included.
        """
        from natcap.invest.hydropower import hydropower_water_yield
        args = {
            'workspace_dir': tempfile.mkdtemp(),
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
            'sub_watersheds_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp'),
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
            'water_scarcity_container': True,
            'demand_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv'),
            'valuation_container': False
        }
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

        shutil.rmtree(args['workspace_dir'])

class ValuationSubwatershedTest(unittest.TestCase):
    """
    Regression Test for Hydropower running Scarcity and Valuation components
        with subwatershed.
    """
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for Valuation component, subwatershed included.
        """
        from natcap.invest.hydropower import hydropower_water_yield
        args = {
            'workspace_dir': tempfile.mkdtemp(),
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
            'sub_watersheds_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp'),
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
            'water_scarcity_container': True,
            'demand_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv'),
            'valuation_container': True,
            'valuation_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input',
                'hydropower_valuation_table.csv'),
        }
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

        shutil.rmtree(args['workspace_dir'])

class SuffixTest(unittest.TestCase):
    """
    Regression Test for Hydropower running Scarcity and Valuation components
        testing presence of a suffix in the filenames.
    """
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for checking that the suffix is handled correctly.
        """
        from natcap.invest.hydropower import hydropower_water_yield
        args = {
            'workspace_dir': tempfile.mkdtemp(),
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
            'sub_watersheds_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp'),
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': 'test',
            'water_scarcity_container': True,
            'demand_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv'),
            'valuation_container': True,
            'valuation_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input',
                'hydropower_valuation_table.csv')
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

        shutil.rmtree(args['workspace_dir'])

class SuffixUnderscoreTest(unittest.TestCase):
    """
    Regression Test for Hydropower running Scarcity and Valuation components
        testing presence of a suffix in the filenames.
    """
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for checking that the suffix is handled correctly when
            given extra underscore.
        """
        from natcap.invest.hydropower import hydropower_water_yield
        args = {
            'workspace_dir': tempfile.mkdtemp(),
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
            'sub_watersheds_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Freshwater', 'subwatersheds.shp'),
            'biophysical_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '_test',
            'water_scarcity_container': True,
            'demand_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input', 'water_demand_table.csv'),
            'valuation_container': True,
            'valuation_table_uri': os.path.join(
                SAMPLE_DATA, 'Hydropower', 'input',
                'hydropower_valuation_table.csv'),
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

        shutil.rmtree(args['workspace_dir'])
