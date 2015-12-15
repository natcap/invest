import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', 'wave-energy')

class WaveEnergyRegressionTests(unittest.TestCase):
    """Regression tests for the Wave Energy module."""

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
            'wave_base_data_uri': os.path.join(
                SAMPLE_DATA, 'WaveEnergy', 'input', 'WaveData'),
            'analysis_area_uri': 'West Coast of North America and Hawaii',
            'machine_perf_uri': os.path.join(
                SAMPLE_DATA, 'WaveEnergy', 'input',
                'Machine_Pelamis_Performance.csv'),
            'machine_param_uri': os.path.join(
                SAMPLE_DATA, 'WaveEnergy', 'input',
                'Machine_Pelamis_Parameters.csv'),
            'dem_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 'global_dem')
        }
        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation(unittest.TestCase):
        """WaveEnergy: testing valuation component."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'AOI_WCVI.shp')
        args['valuation_container'] = True
        args['land_gridPts_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'LandGridPts_WCVI.csv')
        args['machine_econ_uri'] =  os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'Machine_Pelamis_Economic.csv')
        args['number_of_machine'] = 28

        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif',
            'npv_rc.tif', 'npv_usd.tif']

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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_aoi(unittest.TestCase):
        """WaveEnergy: testing Biophysical component with an AOI."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'AOI_WCVI.shp')

        args['machine_perf_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input',
            'Machine_Pelamis_Performance.csv')
        args['machine_param_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input',
            'Machine_Pelamis_Parameters.csv')
        args['dem_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 'global_dem')
        args['valuation_container'] = False

        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

        table_results = ['capwe_rc.csv', 'wp_rc.csv']

        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_no_aoi(unittest.TestCase):
        """WaveEnergy: testing Biophysical component with no AOI."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, raster_path))

        table_results = ['capwe_rc.csv', 'wp_rc.csv']

        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, table_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation_suffix(unittest.TestCase):
        """WaveEnergy: testing suffix through Valuation."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'AOI_WCVI.shp')
        args['valuation_container'] = True
        args['land_gridPts_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'LandGridPts_WCVI.csv')
        args['machine_econ_uri'] =  os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'Machine_Pelamis_Economic.csv')
        args['number_of_machine'] = 28
        args['suffix'] = 'val'

        wave_energy.execute(args)

        raster_results = [
            'wp_rc_val.tif', 'wp_kw_val.tif', 'capwe_rc_val.tif',
            'capwe_mwh_val.tif', 'npv_rc_val.tif', 'npv_usd_val.tif']

        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', raster_path)))

        vector_results = ['GridPts_prj_val.shp', 'LandPts_prj_val.shp']

        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

        table_results = ['capwe_rc_val.csv', 'wp_rc_val.csv', 'npv_rc_val.csv']

        for table_path in table_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', table_path)))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation_suffix_underscore(unittest.TestCase):
        """WaveEnergy: testing suffix with an underscore through Valuation."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'AOI_WCVI.shp')
        args['valuation_container'] = True
        args['land_gridPts_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'LandGridPts_WCVI.csv')
        args['machine_econ_uri'] =  os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'Machine_Pelamis_Economic.csv')
        args['number_of_machine'] = 28
        args['suffix'] = '_val'

        wave_energy.execute(args)

        raster_results = [
            'wp_rc_val.tif', 'wp_kw_val.tif', 'capwe_rc_val.tif',
            'capwe_mwh_val.tif', 'npv_rc_val.tif', 'npv_usd_val.tif']

        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', raster_path)))

        vector_results = ['GridPts_prj_val.shp', 'LandPts_prj_val.shp']

        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

        table_results = ['capwe_rc_val.csv', 'wp_rc_val.csv', 'npv_rc_val.csv']

        for table_path in table_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', table_path)))
