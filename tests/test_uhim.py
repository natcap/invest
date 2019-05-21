"""InVEST Urban Heat Island Mitigation model tests."""
import unittest
import tempfile
import shutil
import os

import numpy
from osgeo import gdal

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'uhim')


class UHIMTests(unittest.TestCase):
    """Regression tests for InVEST SDR model."""

    def setUp(self):
        """Initialize SDRRegression tests."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_uhim_regression(self):
        """UHIM regression."""
        import natcap.invest.urban_heat_island_mitigation
        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': 'test_suffix',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(REGRESSION_DATA, "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(REGRESSION_DATA, "Biophysical table_UHI.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'do_valuation': True,
            't_air_average_radius': "1000.0",
            'building_vector_path': os.path.join(REGRESSION_DATA, "buildings_clip.gpkg"),
            'energy_consumption_table_path': os.path.join(REGRESSION_DATA, "Energy.csv"),
            'avg_rel_humidity': '30.0',
            'cc_weight_shade': '0.6',
            'cc_weight_albedo': '0.2',
            'cc_weight_eti': '0.2',
            'n_workers': -1,
            }

        natcap.invest.urban_heat_island_mitigation.execute(args)
        results_vector = gdal.OpenEx(os.path.join(
            args['workspace_dir'],
            'uhi_results_%s.shp' % args['results_suffix']))
        results_layer = results_vector.GetLayer()
        results_feature = results_layer.GetFeature(1)

        expected_results = {
            'avg_cc': 0.222150472947109,
            'avg_tmp_v': 37.306552793522201,
            'avg_tmp_an': 2.306552793522201,
            'avd_eng_cn': 9212.488475267766262,
            'avg_wbgt_v': 37.306552793522201,
            'avg_ltls_v': 74.654377880184327,
            'avg_hvls_v': 74.654377880184327,
        }

        for key, expected_value in expected_results.items():
            actual_value = float(results_feature.GetField(key))
            self.assertAlmostEqual(
                actual_value, expected_value,
                msg='%s should be close to %f, actual: %f' % (
                    key, expected_value, actual_value))

    def test_bad_building_type(self):
        """UHIM regression."""
        import natcap.invest.urban_heat_island_mitigation
        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': 'test_suffix',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(REGRESSION_DATA, "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(REGRESSION_DATA, "Biophysical table_UHI.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'do_valuation': True,
            't_air_average_radius': "1000.0",
            'building_vector_path': os.path.join(REGRESSION_DATA, "buildings_clip.gpkg"),
            'energy_consumption_table_path': os.path.join(REGRESSION_DATA, "Energy.csv"),
            'avg_rel_humidity': '30.0',
            'cc_weight_shade': '0.6',
            'cc_weight_albedo': '0.2',
            'cc_weight_eti': '0.2',
            'n_workers': -1,
            }

        gpkg_driver = gdal.GetDriverByName('GPKG')
        building_vector = gdal.OpenEx(
            args['building_vector_path'], gdal.OF_VECTOR)
        bad_building_vector_path = os.path.join(
            self.workspace_dir, 'bad_building_vector.gpkg')
        bad_building_vector = gpkg_driver.CreateCopy(
            bad_building_vector_path, building_vector)
        bad_building_layer = bad_building_vector.GetLayer()
        feature = next(bad_building_layer)
        feature.SetField('type', -999)
        bad_building_layer.SetFeature(feature)
        bad_building_layer.SyncToDisk()
        bad_building_layer = None
        bad_building_vector = None

        args['building_vector_path'] = bad_building_vector_path
        with self.assertRaises(ValueError) as context:
            natcap.invest.urban_heat_island_mitigation.execute(args)
        self.assertTrue(
            "ValueError: Encountered a building 'type' of",
            context.exception)
