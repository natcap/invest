import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', 'wave-energy')


class ValuationTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wave_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wave_base_data_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'WaveData'),
            'analysis_area_uri': 'West Coast of North America and Hawaii',
            'aoi_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'AOI_WCVI.shp'),
            'machine_perf_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Performance.csv'),
            'machine_param_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Parameters.csv'),
            'dem_uri': os.path.join(SAMPLE_DATA, 'Base_Data', 'Marine',
                                         'DEMs', 'global_dem'),
	    'suffix': '',
	    'valuation_container': True,
            'land_gridPts_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'LandGridPts_WCVI.csv'),
            'machine_econ_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Economic.csv'),
	    'number_of_machine': 28


        }
        wave_energy.execute(args)

	raster_results = ['wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif',
			'capwe_mwh.tif', 'npv_rc.tif', 'npv_usd.tif']

	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	vector_results = ['GridPts_prj.shp', 'LandPts_prj.shp']

	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	table_results = ['capwe_rc.csv', 'wp_rc.csv', 'npv_rc.csv']

	for table_path in table_results:
	    pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))
	    
	shutil.rmtree(args['workspace_dir'])

class AoiTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wave_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wave_base_data_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'WaveData'),
            'analysis_area_uri': 'West Coast of North America and Hawaii',
            'aoi_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'AOI_WCVI.shp'),
            'machine_perf_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Performance.csv'),
            'machine_param_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Parameters.csv'),
            'dem_uri': os.path.join(SAMPLE_DATA, 'Base_Data', 'Marine',
                                         'DEMs', 'global_dem'),
	    'suffix': '',
	    'valuation_container': False 

        }
        wave_energy.execute(args)

	raster_results = ['wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif',
			'capwe_mwh.tif']

	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	table_results = ['capwe_rc.csv', 'wp_rc.csv']

	for table_path in table_results:
	    pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))
	    
	shutil.rmtree(args['workspace_dir'])

class NoAoiTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wave_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wave_base_data_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'WaveData'),
            'analysis_area_uri': 'West Coast of North America and Hawaii',
            'machine_perf_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Performance.csv'),
            'machine_param_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Parameters.csv'),
            'dem_uri': os.path.join(SAMPLE_DATA, 'Base_Data', 'Marine',
                                         'DEMs', 'global_dem'),
	    'suffix': '',
	    'valuation_container': False 

        }
        wave_energy.execute(args)

	raster_results = ['wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif',
			'capwe_mwh.tif']

	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	table_results = ['capwe_rc.csv', 'wp_rc.csv']

	for table_path in table_results:
	    pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))
	    
	shutil.rmtree(args['workspace_dir'])

class ValuationSuffixTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
        """
        from natcap.invest import wave_energy
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'wave_base_data_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'WaveData'),
            'analysis_area_uri': 'West Coast of North America and Hawaii',
            'aoi_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'AOI_WCVI.shp'),
            'machine_perf_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Performance.csv'),
            'machine_param_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Parameters.csv'),
            'dem_uri': os.path.join(SAMPLE_DATA, 'Base_Data', 'Marine',
                                         'DEMs', 'global_dem'),
	    'suffix': 'val',
	    'valuation_container': True,
            'land_gridPts_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'LandGridPts_WCVI.csv'),
            'machine_econ_uri': os.path.join(SAMPLE_DATA, 'WaveEnergy',
                                         'input', 'Machine_Pelamis_Economic.csv'),
	    'number_of_machine': 28


        }
        wave_energy.execute(args)

	raster_results = ['wp_rc_val.tif', 'wp_kw_val.tif', 'capwe_rc_val.tif',
			'capwe_mwh_val.tif', 'npv_rc_val.tif', 'npv_usd_val.tif']

	for raster_path in raster_results:
	    pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

	vector_results = ['GridPts_prj_val.shp', 'LandPts_prj_val.shp']

	for vector_path in vector_results:
	    pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, vector_path))
        
	table_results = ['capwe_rc_val.csv', 'wp_rc_val.csv', 'npv_rc_val.csv']

	for table_path in table_results:
	    pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))
	    
	shutil.rmtree(args['workspace_dir'])
