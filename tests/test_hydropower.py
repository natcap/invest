import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', 'hydropower')


class ValuationTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import hydropower
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'landuse_90'),
            'depth_to_root_rest_layer_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'depth_to_root_rest_layer'),
            'precipitaion_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'precip'),
            'pawc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'pawc'),
            'eto_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'eto'),
            'watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'watersheds.shp'),
            'biophysical_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
	    'water_scarcity_container': True,
            'demand_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'water_demand_table.csv'),
	    'valuation_container': True,
            'valuation_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'hydropower_valuation_table.csv'),
        }
        hydropower_water_yield.execute(args)

	raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
	for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	vector_results = ['watershed_results_wyield.shp']
	for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))

	    
	table_results = ['watershed_results_wyield.csv']
	for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))


        shutil.rmtree(args['workspace_dir'])

class WaterYieldTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import hydropower
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'landuse_90'),
            'depth_to_root_rest_layer_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'depth_to_root_rest_layer'),
            'precipitaion_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'precip'),
            'pawc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'pawc'),
            'eto_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'eto'),
            'watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'watersheds.shp'),
            'biophysical_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
	    'water_scarcity_container': False,
	    'valuation_container': False
        }
        hydropower_water_yield.execute(args)

	raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
	for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	vector_results = ['watershed_results_wyield.shp']
	for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))

	    
	table_results = ['watershed_results_wyield.csv']
	for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))


        shutil.rmtree(args['workspace_dir'])

class WaterYieldSubwatershedTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import hydropower
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'landuse_90'),
            'depth_to_root_rest_layer_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'depth_to_root_rest_layer'),
            'precipitaion_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'precip'),
            'pawc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'pawc'),
            'eto_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'eto'),
            'watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'watersheds.shp'),
            'sub_watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'subwatersheds.shp'),
            'biophysical_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
	    'water_scarcity_container': False,
	    'valuation_container': False
        }
        hydropower_water_yield.execute(args)

	raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
	for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	vector_results = ['watershed_results_wyield.shp', 'subwatershed_results_wyield.shp']
	for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))

	    
	table_results = ['watershed_results_wyield.csv', 'subwatersheds_results_wyield.csv']
	for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))


        shutil.rmtree(args['workspace_dir'])

class ScarcityTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import hydropower
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'landuse_90'),
            'depth_to_root_rest_layer_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'depth_to_root_rest_layer'),
            'precipitaion_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'precip'),
            'pawc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'pawc'),
            'eto_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'eto'),
            'watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'watersheds.shp'),
            'biophysical_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
	    'water_scarcity_container': True,
            'demand_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'water_demand_table.csv'),
	    'valuation_container': False
        }
        hydropower_water_yield.execute(args)

	raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
	for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	vector_results = ['watershed_results_wyield.shp']
	for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))

	    
	table_results = ['watershed_results_wyield.csv']
	for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))


        shutil.rmtree(args['workspace_dir'])

class ScarcitySubwatershedTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import hydropower
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'landuse_90'),
            'depth_to_root_rest_layer_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'depth_to_root_rest_layer'),
            'precipitaion_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'precip'),
            'pawc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'pawc'),
            'eto_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'eto'),
            'watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'watersheds.shp'),
            'sub_watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'subwatersheds.shp'),
            'biophysical_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
	    'water_scarcity_container': True,
            'demand_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'water_demand_table.csv'),
	    'valuation_container': False
        }
        hydropower_water_yield.execute(args)

	raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
	for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	vector_results = ['watershed_results_wyield.shp', 'subwatershed_results_wyield.shp']
	for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))

	    
	table_results = ['watershed_results_wyield.csv', 'subwatersheds_results_wyield.csv']
	for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))


        shutil.rmtree(args['workspace_dir'])

class ValuationSubwatershedTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import hydropower
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'landuse_90'),
            'depth_to_root_rest_layer_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'depth_to_root_rest_layer'),
            'precipitaion_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'precip'),
            'pawc_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'pawc'),
            'eto_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'eto'),
            'watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'watersheds.shp'),
            'sub_watersheds_uri': os.path.join(SAMPLE_DATA, 'Base_Data',
                              'Freshwater', 'subwatersheds.shp'),
            'biophysical_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'results_suffix': '',
	    'water_scarcity_container': True,
            'demand_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'water_demand_table.csv'),
	    'valuation_container': True,
            'valuation_table_uri': os.path.join(SAMPLE_DATA, 'Hydropower',
                              'input', 'hydropower_valuation_table.csv'),
        }
        hydropower_water_yield.execute(args)

	raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
	for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	vector_results = ['watershed_results_wyield.shp', 'subwatershed_results_wyield.shp']
	for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))

	    
	table_results = ['watershed_results_wyield.csv', 'subwatersheds_results_wyield.csv']
	for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))


        shutil.rmtree(args['workspace_dir'])
