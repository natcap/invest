import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', '_example_model')


class ValuationAvgGridDistWindSchedTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wind_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wind_data_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
	    'aoi_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'New_England_US_Aoi.shp'),
	    'bathymetry_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 
		    'global_dem'),
	    'land_polygon_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 
		    'global_polygon.shp'),
	    'global_wind_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'global_wind_energy_parameters.csv'),
	    'suffix': '',
	    'turbine_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    '3_6_turbine.csv'),
	    'number_of_turbines': 80,
	    'min_depth': 3,
	    'max_depth': 60,
	    'min_distance': 0,
	    'max_distance': 200000,
	    'valuation_container': True,
	    'foundation_cost': 2,
	    'discount_rate': 0.07,
	    'grid_points_uri': '',
	    'avg_grid_distance': 4,
	    'price_table': True,
	    'wind_schedule': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'price_table_example.csv')

        }
        
	wind_energy.execute(args)

        raster_results = ['carbon_emissions_tons.tif', 'density_W_per_m2.tif',
			'harvested_energy_MWhr_per_yr.tif',
			'levelized_cost_price_per_kWh.tif', 
			'npv_US_millions.tif']
	
	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
            	os.path.join(REGRESSION_DATA, raster_path))

	vector_results = [
			'example_size_and_orientation_of_a_possible_wind_farm.shp',
			'wind_energy_points.shp']
	
	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	shutil.rmtree(args['workspace_dir'])

class NoAoiTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wind_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wind_data_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
	    'bathymetry_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 
		    'global_dem'),
	    'global_wind_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'global_wind_energy_parameters.csv'),
	    'suffix': '',
	    'turbine_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    '3_6_turbine.csv'),
	    'number_of_turbines': 80,
	    'min_depth': 3,
	    'max_depth': 60,
	    'valuation_container': False

        }
        
	wind_energy.execute(args)

        raster_results = ['density_W_per_m2.tif',
			'harvested_energy_MWhr_per_yr.tif']
	
	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
            	os.path.join(REGRESSION_DATA, raster_path))

	vector_results = [
			'example_size_and_orientation_of_a_possible_wind_farm.shp',
			'wind_energy_points.shp']
	
	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	shutil.rmtree(args['workspace_dir'])

class NoLandPolyTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wind_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wind_data_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
	    'aoi_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'New_England_US_Aoi.shp'),
	    'bathymetry_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 
		    'global_dem'),
	    'global_wind_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'global_wind_energy_parameters.csv'),
	    'suffix': '',
	    'turbine_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    '3_6_turbine.csv'),
	    'number_of_turbines': 80,
	    'min_depth': 3,
	    'max_depth': 60,
	    'valuation_container': False

        }
        
	wind_energy.execute(args)

        raster_results = ['density_W_per_m2.tif',
			'harvested_energy_MWhr_per_yr.tif']
	
	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
            	os.path.join(REGRESSION_DATA, raster_path))

	vector_results = [
			'example_size_and_orientation_of_a_possible_wind_farm.shp',
			'wind_energy_points.shp']
	
	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	shutil.rmtree(args['workspace_dir'])

class NoDistancesTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wind_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wind_data_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
	    'aoi_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'New_England_US_Aoi.shp'),
	    'bathymetry_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 
		    'global_dem'),
	    'land_polygon_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 
		    'global_polygon.shp'),
	    'global_wind_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'global_wind_energy_parameters.csv'),
	    'suffix': '',
	    'turbine_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    '3_6_turbine.csv'),
	    'number_of_turbines': 80,
	    'min_depth': 3,
	    'max_depth': 60,
	    'valuation_container': False

        }
        
	wind_energy.execute(args)

        raster_results = ['density_W_per_m2.tif',
			'harvested_energy_MWhr_per_yr.tif']
	
	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
            	os.path.join(REGRESSION_DATA, raster_path))

	vector_results = [
			'example_size_and_orientation_of_a_possible_wind_farm.shp',
			'wind_energy_points.shp']
	
	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	shutil.rmtree(args['workspace_dir'])

class NoValuationTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wind_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wind_data_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
	    'aoi_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'New_England_US_Aoi.shp'),
	    'bathymetry_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 
		    'global_dem'),
	    'land_polygon_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 
		    'global_polygon.shp'),
	    'global_wind_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'global_wind_energy_parameters.csv'),
	    'suffix': '',
	    'turbine_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    '3_6_turbine.csv'),
	    'number_of_turbines': 80,
	    'min_depth': 3,
	    'max_depth': 60,
	    'min_distance': 0,
	    'max_distance': 200000,
	    'valuation_container': False

        }
        
	wind_energy.execute(args)

        raster_results = ['carbon_emissions_tons.tif', 'density_W_per_m2.tif',
			'harvested_energy_MWhr_per_yr.tif',
			'levelized_cost_price_per_kWh.tif', 
			'npv_US_millions.tif']
	
	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
            	os.path.join(REGRESSION_DATA, raster_path))

	vector_results = [
			'example_size_and_orientation_of_a_possible_wind_farm.shp',
			'wind_energy_points.shp']
	
	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	shutil.rmtree(args['workspace_dir'])

class ValuationGridPtsWindSchedTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wind_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wind_data_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
	    'aoi_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'New_England_US_Aoi.shp'),
	    'bathymetry_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 
		    'global_dem'),
	    'land_polygon_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 
		    'global_polygon.shp'),
	    'global_wind_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'global_wind_energy_parameters.csv'),
	    'suffix': '',
	    'turbine_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    '3_6_turbine.csv'),
	    'number_of_turbines': 80,
	    'min_depth': 3,
	    'max_depth': 60,
	    'min_distance': 0,
	    'max_distance': 200000,
	    'valuation_container': True,
	    'foundation_cost': 2,
	    'discount_rate': 0.07,
	    'grid_points_uri': os.path.join(
		    	SAMPLE_DATA, 'WindEnergy', 'input',
			'NE_sub_pts.csv'),
	    'avg_grid_distance': 4,
	    'price_table': True,
	    'wind_schedule': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'price_table_example.csv')

        }
        
	wind_energy.execute(args)

        raster_results = ['carbon_emissions_tons.tif', 'density_W_per_m2.tif',
			'harvested_energy_MWhr_per_yr.tif',
			'levelized_cost_price_per_kWh.tif', 
			'npv_US_millions.tif']
	
	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
            	os.path.join(REGRESSION_DATA, raster_path))

	vector_results = [
			'example_size_and_orientation_of_a_possible_wind_farm.shp',
			'wind_energy_points.shp']
	
	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	shutil.rmtree(args['workspace_dir'])

class ValuationAvgGridDistWindPriceTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wind_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wind_data_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
	    'aoi_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'New_England_US_Aoi.shp'),
	    'bathymetry_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 
		    'global_dem'),
	    'land_polygon_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 
		    'global_polygon.shp'),
	    'global_wind_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'global_wind_energy_parameters.csv'),
	    'suffix': '',
	    'turbine_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    '3_6_turbine.csv'),
	    'number_of_turbines': 80,
	    'min_depth': 3,
	    'max_depth': 60,
	    'min_distance': 0,
	    'max_distance': 200000,
	    'valuation_container': True,
	    'foundation_cost': 2,
	    'discount_rate': 0.07,
	    'grid_points_uri': '',
	    'avg_grid_distance': 4,
	    'price_table': False,
	    'wind_price': 0.187,
	    'rate_change': 0.2

        }
        
	wind_energy.execute(args)

        raster_results = ['carbon_emissions_tons.tif', 'density_W_per_m2.tif',
			'harvested_energy_MWhr_per_yr.tif',
			'levelized_cost_price_per_kWh.tif', 
			'npv_US_millions.tif']
	
	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
            	os.path.join(REGRESSION_DATA, raster_path))

	vector_results = [
			'example_size_and_orientation_of_a_possible_wind_farm.shp',
			'wind_energy_points.shp']
	
	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	shutil.rmtree(args['workspace_dir'])

class ValuationGridPtsWindPriceTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wind_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wind_data_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
	    'aoi_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'New_England_US_Aoi.shp'),
	    'bathymetry_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 
		    'global_dem'),
	    'land_polygon_uri': os.path.join(
		    SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 
		    'global_polygon.shp'),
	    'global_wind_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'global_wind_energy_parameters.csv'),
	    'suffix': '',
	    'turbine_parameters_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    '3_6_turbine.csv'),
	    'number_of_turbines': 80,
	    'min_depth': 3,
	    'max_depth': 60,
	    'min_distance': 0,
	    'max_distance': 200000,
	    'valuation_container': True,
	    'foundation_cost': 2,
	    'discount_rate': 0.07,
	    'grid_points_uri': os.path.join(
		    SAMPLE_DATA, 'WindEnergy', 'input',
		    'NE_sub_pts.csv')
	    'avg_grid_distance': 4,
	    'price_table': False,
	    'wind_price': 0.187,
	    'rate_change': 0.2

        }
        
	wind_energy.execute(args)

        raster_results = ['carbon_emissions_tons.tif', 'density_W_per_m2.tif',
			'harvested_energy_MWhr_per_yr.tif',
			'levelized_cost_price_per_kWh.tif', 
			'npv_US_millions.tif']
	
	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
            	os.path.join(REGRESSION_DATA, raster_path))

	vector_results = [
			'example_size_and_orientation_of_a_possible_wind_farm.shp',
			'wind_energy_points.shp']
	
	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	shutil.rmtree(args['workspace_dir'])

