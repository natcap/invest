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
            't_obs_raster_path': os.path.join(
                REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(
                REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA,
                "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'factors',
            'do_energy_valuation': True,
            'do_productivity_valuation': True,
            't_air_average_radius': "1000.0",
            'building_vector_path': os.path.join(
                REGRESSION_DATA, "buildings_clip.gpkg"),
            'energy_consumption_table_path': os.path.join(
                REGRESSION_DATA, "Energy.csv"),
            'avg_rel_humidity': '30.0',
            'cc_weight_shade': '',  # to trigger default of 0.6
            'cc_weight_albedo': None,  # to trigger default of 0.2
            # Purposefully excluding cc_weight_eti to trigger default of 0.2
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
            'avd_eng_cn': 3520213.280928277,
            'avg_wbgt_v': 32.60417266705069,
            'avg_ltls_v': 75.000000000000000,
            'avg_hvls_v': 75.000000000000000,
        }

        try:
            for key, expected_value in expected_results.items():
                actual_value = float(results_feature.GetField(key))
                # These accumulated values (esp. avd_eng_cn) are accumulated
                # and may differ past about 4 decimal places.
                self.assertAlmostEqual(
                    actual_value, expected_value, places=4,
                    msg='%s should be close to %f, actual: %f' % (
                        key, expected_value, actual_value))
        finally:
            results_layer = None
            results_vector = None

        # Assert that the decimal value of the energy savings value is what we
        # expect.
        expected_energy_sav = 3564034.496484185

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

            # Expected energy savings is an accumulated value and may differ
            # past about 4 decimal places.
            self.assertAlmostEqual(
                energy_sav, expected_energy_sav, places=4, msg=(
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

            # These accumulated values are accumulated
            # and may differ past about 4 decimal places.
            self.assertAlmostEqual(
                energy_sav, expected_energy_sav, places=4, msg=(
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
                REGRESSION_DATA,
                "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'intensity',  # main difference in the reg. tests
            'do_energy_valuation': True,
            'do_productivity_valuation': True,
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
            'avd_eng_cn': 7240015.1958200345,
            'avg_wbgt_v': 31.91108630952381,
            'avg_ltls_v': 28.744239631336406,
            'avg_hvls_v': 75.000000000000000,
        }
        try:
            for key, expected_value in expected_results.items():
                actual_value = float(results_feature.GetField(key))
                # These accumulated values (esp. avd_eng_cn) are accumulated
                # and may differ past about 4 decimal places.
                self.assertAlmostEqual(
                    actual_value, expected_value, places=4,
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
            't_obs_raster_path': os.path.join(
                REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(
                REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA,
                "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'factors',
            'do_energy_valuation': True,
            'do_productivity_valuation': True,
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

        bad_building_vector_path = os.path.join(
            self.workspace_dir, 'bad_building_vector.gpkg')

        shutil.copyfile(args['building_vector_path'], bad_building_vector_path)
        bad_building_vector = gdal.OpenEx(bad_building_vector_path,
                                          gdal.OF_VECTOR | gdal.GA_Update)
        bad_building_layer = bad_building_vector.GetLayer()
        feature = bad_building_layer.GetNextFeature()
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

    def test_missing_lulc_value_in_table(self):
        """UCM: error on missing lulc value in biophysical table."""
        import natcap.invest.urban_cooling_model
        import pandas

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': 'test_suffix',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(
                REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(
                REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA,
                "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'factors',
            'do_energy_valuation': True,
            'do_productivity_valuation': True,
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

        # remove a row from the biophysical table so that lulc value is missing
        bad_biophysical_path = os.path.join(
            self.workspace_dir, 'bad_biophysical_table.csv')

        bio_df = pandas.read_csv(args['biophysical_table_path'])
        bio_df = bio_df[bio_df['lucode'] != 10]
        bio_df.to_csv(bad_biophysical_path)
        bio_df = None

        args['biophysical_table_path'] = bad_biophysical_path
        with self.assertRaises(ValueError) as context:
            natcap.invest.urban_cooling_model.execute(args)
        self.assertTrue(
            "The missing values found in the LULC raster but not the table"
            " are: [10]" in str(context.exception))

    def test_bad_args(self):
        """UCM: test validation of bad arguments."""
        from natcap.invest import urban_cooling_model, validation
        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': 'test_suffix',
            't_ref': 35.0,
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(
                REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA,
                "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'factors',
            'do_energy_valuation': True,
            'do_productivity_valuation': True,
            't_air_average_radius': "1000.0",
            'building_vector_path': os.path.join(
                REGRESSION_DATA, "buildings_clip.gpkg"),
            'energy_consumption_table_path': os.path.join(
                REGRESSION_DATA, "Energy.csv"),
            'avg_rel_humidity': '30.0',
            # Explicitly leaving CC weight parameters out.
            'n_workers': -1,
            }

        del args['t_ref']
        warnings = urban_cooling_model.validate(args)
        expected_warning = (['t_ref'], validation.MESSAGES['MISSING_KEY'])
        self.assertTrue(expected_warning in warnings)

        args['t_ref'] = ''
        result = urban_cooling_model.validate(args)
        self.assertEqual(result[0][1], validation.MESSAGES['MISSING_VALUE'])

        args['t_ref'] = 35.0
        args['cc_weight_shade'] = -0.6
        result = urban_cooling_model.validate(args)
        self.assertEqual(
            result[0][1], validation.MESSAGES['NOT_WITHIN_RANGE'].format(
                value=args['cc_weight_shade'], range='[0, 1]'))

        args['cc_weight_shade'] = "not a number"
        result = urban_cooling_model.validate(args)
        self.assertEqual(
            result[0][1],
            validation.MESSAGES['NOT_A_NUMBER'].format(value=args['cc_weight_shade']))

        args['cc_method'] = 'nope'
        result = urban_cooling_model.validate(args)
        self.assertEqual(
            result[0][1],
            validation.MESSAGES['INVALID_OPTION'].format(
                option_list=['factors', 'intensity']))

        args['cc_method'] = 'intensity'
        args['cc_weight_shade'] = 0.2  # reset this arg

        # Create a new table like the original one, but without the green area
        # column.
        old_df = pandas.read_csv(args['biophysical_table_path'])
        new_df = old_df.drop('Green_area', axis='columns')

        args['biophysical_table_path'] = os.path.join(
            self.workspace_dir, 'new_csv.csv')
        new_df.to_csv(args['biophysical_table_path'])

        result = urban_cooling_model.validate(args)
        expected = [(
            ['biophysical_table_path'],
            validation.MESSAGES['MATCHED_NO_HEADERS'].format(
                header='column', header_name='green_area'))]
        self.assertEqual(result, expected)

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

    def test_do_energy_valuation_option(self):
        """UCM: test separate valuation options."""
        import natcap.invest.urban_cooling_model
        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': '',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(
                REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(
                REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA,
                "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'regression',
            'do_energy_valuation': True,
            'do_productivity_valuation': False,
            't_air_average_radius': "1000.0",
            'building_vector_path': os.path.join(
                REGRESSION_DATA, "buildings_clip.gpkg"),
            'energy_consumption_table_path': os.path.join(
                REGRESSION_DATA, "Energy.csv"),
            'cc_weight_shade': '0.6',
            'cc_weight_albedo': '0.2',
            'cc_weight_eti': '0.2',
            'n_workers': -1,
        }
        natcap.invest.urban_cooling_model.execute(args)
        intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')

        wbgt_path = os.path.join(
            intermediate_dir, f'wbgt.tif')
        light_work_loss_path = os.path.join(
            intermediate_dir, f'light_work_loss_percent.tif')
        heavy_work_loss_path = os.path.join(
            intermediate_dir, f'heavy_work_loss_percent.tif')
        wbgt_stats_pickle_path = os.path.join(
            intermediate_dir, 'wbgt_stats.pickle')
        light_loss_stats_pickle_path = os.path.join(
            intermediate_dir, 'light_loss_stats.pickle')
        heavy_loss_stats_pickle_path = os.path.join(
            intermediate_dir, 'heavy_loss_stats.pickle')
        intermediate_building_vector_path = os.path.join(
            intermediate_dir, f'reprojected_buildings.shp')
        t_air_stats_pickle_path = os.path.join(
            intermediate_dir, 't_air_stats.pickle')
        energy_consumption_vector_path = os.path.join(
            args['workspace_dir'], f'buildings_with_stats.shp')

        # make sure the energy valuation outputs are there,
        # and the productivity valuation outputs aren't
        for path in [intermediate_building_vector_path, t_air_stats_pickle_path,
                     energy_consumption_vector_path]:
            self.assertTrue(os.path.exists(path))

        for path in [wbgt_path, light_work_loss_path, heavy_work_loss_path,
                     wbgt_stats_pickle_path, light_loss_stats_pickle_path, heavy_loss_stats_pickle_path]:
            self.assertFalse(os.path.exists(path))

    def test_do_productivity_valuation_option(self):
        """UCM: test separate valuation options."""
        import natcap.invest.urban_cooling_model
        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': '',
            't_ref': 35.0,
            't_obs_raster_path': os.path.join(
                REGRESSION_DATA, "Tair_Sept.tif"),
            'lulc_raster_path': os.path.join(
                REGRESSION_DATA, "LULC_SFBA.tif"),
            'ref_eto_raster_path': os.path.join(
                REGRESSION_DATA, "ETo_SFBA.tif"),
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA,
                "watersheds_clippedDraft_Watersheds_SFEI.gpkg"),
            'biophysical_table_path': os.path.join(
                REGRESSION_DATA, "biophysical_table_ucm.csv"),
            'green_area_cooling_distance': 1000.0,
            'uhi_max': 3,
            'cc_method': 'regression',
            'do_energy_valuation': False,
            'do_productivity_valuation': True,
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
        intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')

        wbgt_path = os.path.join(
            intermediate_dir, f'wbgt.tif')
        light_work_loss_path = os.path.join(
            intermediate_dir, f'light_work_loss_percent.tif')
        heavy_work_loss_path = os.path.join(
            intermediate_dir, f'heavy_work_loss_percent.tif')
        wbgt_stats_pickle_path = os.path.join(
            intermediate_dir, 'wbgt_stats.pickle')
        light_loss_stats_pickle_path = os.path.join(
            intermediate_dir, 'light_loss_stats.pickle')
        heavy_loss_stats_pickle_path = os.path.join(
            intermediate_dir, 'heavy_loss_stats.pickle')
        intermediate_building_vector_path = os.path.join(
            intermediate_dir, f'reprojected_buildings.shp')
        t_air_stats_pickle_path = os.path.join(
            intermediate_dir, 't_air_stats.pickle')
        energy_consumption_vector_path = os.path.join(
            args['workspace_dir'], f'buildings_with_stats.shp')

        # make sure the productivity valuation outputs are there,
        # and the energy valuation outputs aren't
        for path in [wbgt_path, light_work_loss_path, heavy_work_loss_path,
                     wbgt_stats_pickle_path, light_loss_stats_pickle_path, heavy_loss_stats_pickle_path]:
            self.assertTrue(os.path.exists(path))
        for path in [intermediate_building_vector_path, t_air_stats_pickle_path,
                     energy_consumption_vector_path]:
            self.assertFalse(os.path.exists(path))
