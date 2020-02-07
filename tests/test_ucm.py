"""InVEST Urban Heat Island Mitigation model tests."""
import unittest
import tempfile
import shutil
import os

import numpy
from osgeo import gdal
import pandas

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'ucm')


class UCMTests(unittest.TestCase):
    """Regression tests for InVEST Urban Cooling Model."""

    def setUp(self):
        """Initialize UCM Regression tests."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_ucm_regression_factors(self):
        """UCM: regression: CC Factors."""
        import natcap.invest.urban_cooling_model
        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'test_suffix',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(REGRESSION_DATA, "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'factors',
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

        natcap.invest.urban_cooling_model.execute(args)
        results_vector = gdal.OpenEx(os.path.join(
            args['workspace_dir'],
            'uhi_results_%s.shp' % args['results_suffix']))
        results_layer = results_vector.GetLayer()
        results_feature = results_layer.GetFeature(1)

        expected_results = {
            'avg_cc': 0.222150472947109,
            'avg_tmp_v': 37.325275675470998,
            'avg_tmp_an': 2.325275675470998,
            'avd_eng_cn': 9019.152329608312357,
            'avg_wbgt_v': 32.60417266705069,
            'avg_ltls_v': 75.000000000000000,
            'avg_hvls_v': 75.000000000000000,
        }

        try:
            for key, expected_value in expected_results.items():
                actual_value = float(results_feature.GetField(key))
                self.assertAlmostEqual(
                    actual_value, expected_value,
                    msg='%s should be close to %f, actual: %f' % (
                        key, expected_value, actual_value))
        finally:
            results_layer = None
            results_vector = None

        # Assert that the decimal value of the energy savings value is what we
        # expect.
        expected_energy_sav = 9361.431821463711
        energy_sav = 0.0
        n_nonetype = 0
        stats_vector_path = (
            os.path.join(args['workspace_dir'],
                         ('buildings_with_stats_%s.shp' %
                          args['results_suffix'])))
        try:
            buildings_vector = gdal.OpenEx(stats_vector_path)
            buildings_layer = buildings_vector.GetLayer()
            for building_feature in buildings_layer:
                try:
                    energy_sav += building_feature.GetField('energy_sav')
                except TypeError:
                    # When energy_sav is NoneType
                    n_nonetype += 1

            self.assertAlmostEqual(energy_sav, expected_energy_sav, msg=(
                '%f should be close to %f' % (
                    energy_sav, expected_energy_sav)))
            self.assertEqual(n_nonetype, 119)
        finally:
            buildings_layer = None
            buildings_vector = None

        # Now, re-run the model with the cost column and verify cost sum is
        # reasonable.  Re-running within the same test function allows us to
        # take advantage of taskgraph.
        new_csv_path = os.path.join(self.workspace_dir, 'cost_csv.csv')
        multiplier = 3.0
        df = pandas.read_csv(args['energy_consumption_table_path'])
        df['Cost'] = pandas.Series([multiplier], index=df.index)
        df.to_csv(new_csv_path)
        args['energy_consumption_table_path'] = new_csv_path

        natcap.invest.urban_cooling_model.execute(args)

        # Assert that the decimal value of the energy savings value is what we
        # expect.  Because we're re-running the model in the same workspace,
        # taskgraph should only re-compute the output vector step.
        expected_energy_sav = expected_energy_sav * multiplier
        energy_sav = 0.0
        n_nonetype = 0
        try:
            buildings_vector = gdal.OpenEx(os.path.join(
                args['workspace_dir'], ('buildings_with_stats_%s.shp' %
                                        args['results_suffix'])))
            buildings_layer = buildings_vector.GetLayer()
            for building_feature in buildings_layer:
                try:
                    energy_sav += building_feature.GetField('energy_sav')
                except TypeError:
                    # When energy_sav is Nonetype
                    n_nonetype += 1

            self.assertAlmostEqual(energy_sav, expected_energy_sav, msg=(
                '%f should be close to %f' % (
                    energy_sav, expected_energy_sav)))
            self.assertEqual(n_nonetype, 119)
        finally:
            buildings_layer = None
            buildings_vector = None

    def test_ucm_regression_intensity(self):
        """UCM: regression: CC Building Intensity."""
        import natcap.invest.urban_cooling_model
        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'test_suffix',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(
                REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(
                REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'intensity',  # main difference in the reg. tests
            'do_valuation': True,
            't_air_average_radius': "1000.0",
            'building_vector_path': os.path.join(
                REGRESSION_DATA, "buildings_clip.gpkg"),
            'energy_consumption_table_path': os.path.join(
                REGRESSION_DATA, "Energy.csv"),
            'avg_rel_humidity': '30.0',
            'cc_weight_shade': '0.6',
            'cc_weight_albedo': '0.2',
            'cc_weight_eti': '0.2',
            'n_workers': -1,
        }

        natcap.invest.urban_cooling_model.execute(args)
        results_vector = gdal.OpenEx(os.path.join(
            args['workspace_dir'],
            'uhi_results_%s.shp' % args['results_suffix']))
        results_layer = results_vector.GetLayer()
        results_feature = results_layer.GetFeature(1)

        expected_results = {
            'avg_cc': 0.428302583240327,
            'avg_tmp_v': 36.60869797039769,
            'avg_tmp_an': 1.608697970397692,
            'avd_eng_cn': 18787.273592787547,
            'avg_wbgt_v': 31.91108630952381,
            'avg_ltls_v': 28.744239631336406,
            'avg_hvls_v': 75.000000000000000,
        }
        try:
            for key, expected_value in expected_results.items():
                actual_value = float(results_feature.GetField(key))
                self.assertAlmostEqual(
                    actual_value, expected_value,
                    msg='%s should be close to %f, actual: %f' % (
                        key, expected_value, actual_value))
        finally:
            results_layer = None
            results_vector = None

    def test_bad_building_type(self):
        """UCM: error on bad building type."""
        import natcap.invest.urban_cooling_model
        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': 'test_suffix',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(REGRESSION_DATA, "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'factors',
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
        bad_building_vector_path = os.path.join(
            self.workspace_dir, 'bad_building_vector.gpkg')

        shutil.copyfile(args['building_vector_path'], bad_building_vector_path)
        bad_building_vector = gdal.OpenEx(bad_building_vector_path,
                                          gdal.OF_VECTOR | gdal.GA_Update)
        bad_building_layer = bad_building_vector.GetLayer()
        feature = next(bad_building_layer)
        feature.SetField('type', -999)
        bad_building_layer.SetFeature(feature)
        bad_building_layer.SyncToDisk()
        bad_building_layer = None
        bad_building_vector = None

        args['building_vector_path'] = bad_building_vector_path
        with self.assertRaises(ValueError) as context:
            natcap.invest.urban_cooling_model.execute(args)
        self.assertTrue(
            "Encountered a building 'type' of:" in
            str(context.exception))

    def test_bad_args(self):
        """UCM: test validation of bad arguments."""
        import natcap.invest.urban_cooling_model
        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': 'test_suffix',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(REGRESSION_DATA, "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'factors',
            'do_valuation': True,
            't_air_average_radius': "1000.0",
            'building_vector_path': os.path.join(REGRESSION_DATA, "buildings_clip.gpkg"),
            'energy_consumption_table_path': os.path.join(REGRESSION_DATA, "Energy.csv"),
            'avg_rel_humidity': '30.0',
            # Explicitly leaving CC weight parameters out.
            'n_workers': -1,
            }

        del args['t_ref']
        warnings = natcap.invest.urban_cooling_model.validate(args)
        self.assertTrue('Key is missing from the args dict' in warnings[0][1])

        args['t_ref'] = ''
        result = natcap.invest.urban_cooling_model.validate(args)
        self.assertEqual(result[0][1], "Input is required but has no value")

        args['t_ref'] = 35.0
        args['cc_weight_shade'] = -0.6
        result = natcap.invest.urban_cooling_model.validate(args)
        self.assertEqual(result[0][1], "Value does not meet condition value > 0")

        args['cc_weight_shade'] = "not a number"
        result = natcap.invest.urban_cooling_model.validate(args)
        self.assertEqual(result[0][1], ("Value 'not a number' could not be "
                                        "interpreted as a number"))

        args['cc_method'] = 'nope'
        result = natcap.invest.urban_cooling_model.validate(args)
        self.assertEqual(
            result[0][1], ("Value must be one of: ['factors', "
                           "'intensity']"))

        args['cc_method'] = 'intensity'
        args['cc_weight_shade'] = 0.2  # reset this arg

        # Create a new table like the original one, but without the building
        # intensity column.
        old_df = pandas.read_csv(args['biophysical_table_path'])
        new_df = old_df.drop('building_intensity', axis='columns')

        args['biophysical_table_path'] = os.path.join(
            self.workspace_dir, 'new_csv.csv')
        new_df.to_csv(args['biophysical_table_path'])

        result = natcap.invest.urban_cooling_model.validate(args)
        self.assertTrue(
            'Fields are missing from this table' in result[0][1])

    def test_flat_disk_kernel(self):
        """UCM: test flat disk kernel."""
        import natcap.invest.urban_cooling_model

        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')
        natcap.invest.urban_cooling_model.flat_disk_kernel(
            1000, kernel_filepath)

        kernel_raster = gdal.OpenEx(kernel_filepath, gdal.OF_RASTER)
        kernel_band = kernel_raster.GetRasterBand(1)
        self.assertAlmostEqual(
            numpy.sum(kernel_band.ReadAsArray())/1000,
            numpy.ceil(1000**2*numpy.pi/1000),
            places=0)
