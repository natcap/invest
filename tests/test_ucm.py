"""InVEST Urban Heat Island Mitigation model tests."""
import os
import shutil
import tempfile
import unittest

import numpy
import pandas
from osgeo import gdal, osr, ogr
import pygeoprocessing
from shapely import Polygon

gdal.UseExceptions()
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'ucm')


def make_simple_vector(path_to_shp):
    """
    Generate shapefile with one rectangular polygon
    Args:
        path_to_shp (str): path to target shapefile
    Returns:
        None
    """
    # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)
    shapely_geometry_list = [
        Polygon([(461251, 4923195), (461501, 4923195),
                 (461501, 4923445), (461251, 4923445),
                 (461251, 4923195)])
    ]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    vector_format = "ESRI Shapefile"
    fields = {"id": ogr.OFTReal}
    attribute_list = [{"id": 0}]

    pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                               path_to_shp, projection_wkt,
                                               vector_format, fields,
                                               attribute_list)


def make_simple_raster(base_raster_path, array):
    """Create a raster on designated path with arbitrary values.
    Args:
        base_raster_path (str): the raster path for making the new raster.
    Returns:
        None.
    """
    # UTM Zone 10N
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    origin = (461251, 4923445)
    pixel_size = (30, -30)
    no_data = -1

    pygeoprocessing.numpy_array_to_raster(
        array, no_data, pixel_size, origin, projection_wkt,
        base_raster_path)


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
            'avg_cc': 0.221991,
            'avg_tmp_v': 37.303789,
            'avg_tmp_an': 2.303789,
            'avd_eng_cn': 3602851.784639,
            'avg_wbgt_v': 32.585935,
            'avg_ltls_v': 75.000000000000000,
            'avg_hvls_v': 75.000000000000000,
        }

        try:
            for key, expected_value in expected_results.items():
                actual_value = float(results_feature.GetField(key))
                # These accumulated values (esp. avd_eng_cn) are accumulated
                # and may differ slightly from expected regression values.
                numpy.testing.assert_allclose(actual_value, expected_value,
                                              rtol=1e-4)
        finally:
            results_layer = None
            results_vector = None

        # Assert that the decimal value of the energy savings value is what we
        # expect.
        expected_energy_sav = 3641030.461044

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
            numpy.testing.assert_allclose(energy_sav, expected_energy_sav, rtol=1e-4)
            self.assertEqual(n_nonetype, 136)
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
            numpy.testing.assert_allclose(energy_sav, expected_energy_sav,
                                          rtol=1e-4)
            self.assertEqual(n_nonetype, 136)
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
            'avg_cc': 0.422250,
            'avg_tmp_v': 36.621779,
            'avg_tmp_an': 1.621779,
            'avd_eng_cn': 7148968.928616,
            'avg_wbgt_v': 31.92365,
            'avg_ltls_v': 29.380548,
            'avg_hvls_v': 75.000000000000000,
        }
        try:
            for key, expected_value in expected_results.items():
                actual_value = float(results_feature.GetField(key))
                # These accumulated values (esp. avd_eng_cn) are accumulated
                # and may differ slightly.
                numpy.testing.assert_allclose(actual_value, expected_value,
                                              rtol=1e-4)
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
        from natcap.invest import urban_cooling_model
        from natcap.invest import validation
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

    def test_cc_rasters(self):
        """Test that `execute` creates correct cooling coefficient rasters with
        synthetic data"""
        from natcap.invest import urban_cooling_model

        args = {}
        args['workspace_dir'] = self.workspace_dir
        args['results_suffix'] = "_01"
        args['t_ref'] = 23.5
        args['lulc_raster_path'] = os.path.join(self.workspace_dir,
                                                "lulc.tif")
        args['ref_eto_raster_path'] = os.path.join(self.workspace_dir,
                                                   "evapotranspiration.tif")
        args['aoi_vector_path'] = os.path.join(self.workspace_dir, "aoi.shp")
        args['biophysical_table_path'] = os.path.join(self.workspace_dir,
                                                      "biophysical_table.csv")
        args['green_area_cooling_distance'] = 90
        args['t_air_average_radius'] = 300
        args['uhi_max'] = 2.05
        args['do_energy_valuation'] = False
        args['do_productivity_valuation'] = False
        args['avg_rel_humidity'] = ''
        args['building_vector_path'] = os.path.join(self.workspace_dir,
                                                    "buildings.shp")
        args['energy_consumption_table_path'] = os.path.join(self.workspace_dir,
                                                             "ucm_energy.csv")
        args['cc_method'] = "factors"
        args['cc_weight_shade'] = ''  # 0.6
        args['cc_weight_albedo'] = ''  # 0.2
        args['cc_weight_eti'] = ''  # 0.2

        def _make_input_data(args):
            """ Create aoi shapefile, biophysical table csv, lulc tif,
            and evapotranspiration tif"""

            make_simple_vector(args['aoi_vector_path'])

            biophysical_table = pandas.DataFrame({
                "lucode": [1, 2, 3, 4, 5],
                "lu_desc": ["water", "forest", "grassland", "urban", "barren"],
                "green_area": [0, 1, 1, 0, 0],
                "kc": [1, 1.1, .9, .3, .2],
                "albedo": [.05, .1, .2, .3, .4],
                "shade": [0, 1, 0, 0.2, 0]
            })

            biophysical_csv_path = args['biophysical_table_path']
            biophysical_table.to_csv(biophysical_csv_path, index=False)

            lulc_array = numpy.array([
                [2, 3, 1, 5, 5, 5],
                [3, 3, 1, 1, 4, 5],
                [5, 5, 4, 2, 3, 1],
                [4, 1, 4, 2, 2, 1],
                [1, 5, 4, 1, 1, 2]
            ], dtype=numpy.float32)
            make_simple_raster(args['lulc_raster_path'], lulc_array)

            et_array = numpy.array([
                [800, 799, 567, 234, 422, 422],
                [765, 867, 765, 654, 456, 677],
                [556, 443, 456, 265, 876, 890],
                [433, 266, 677, 776, 900, 687],
                [456, 832, 234, 234, 234, 554]
            ], dtype=numpy.float32)
            make_simple_raster(args['ref_eto_raster_path'], et_array)

        _make_input_data(args)
        urban_cooling_model.execute(args)

        # This array was generated by manually running through calculations
        # Equation: cc = 0.6 * shade + 0.2 * albedo + 0.2 * eti
        cc_array = numpy.array(
            [[0.815556, 0.1998, 0.136, 0.0904, 0.098756, 0.098756],
             [0.193, 0.2134, 0.18, 0.155333, 0.2104, 0.110089],
             [0.10471112, 0.0996889, 0.2104, 0.6847778, 0.2152, 0.20777778],
             [0.20886667, 0.06911112, 0.22513334, 0.8096889,  0.84, 0.1626668],
             [0.11133333, 0.11697778, 0.1956, 0.062, 0.062, 0.75542223]]
            )

        cc_tif = gdal.Open(os.path.join(args["workspace_dir"], "intermediate",
                                        f"cc{args['results_suffix']}.tif"))
        band_cc = cc_tif.GetRasterBand(1)
        actual_cc = band_cc.ReadAsArray()

        numpy.testing.assert_allclose(actual_cc, cc_array, atol=1e-6)

        # Check CC_park
        cc_park_tif = gdal.Open(
            os.path.join(args["workspace_dir"], "intermediate",
                         f"cc_park{args['results_suffix']}.tif"))
        band = cc_park_tif.GetRasterBand(1)
        actual_cc_park = band.ReadAsArray()

        # This array was created by running `convolve_2d_by_exponential`
        # using a manually-calculated signal raster (cc * green_area)
        expected_cc_park = numpy.array(
            [[0.183541, 0.160865, 0.144615, 0.137877, 0.134337, 0.133191],
             [0.15858, 0.152832, 0.147821, 0.150366, 0.14931, 0.146379],
             [0.138345, 0.141573, 0.153495, 0.176796, 0.176097, 0.167245],
             [0.12765204, 0.13550324, 0.15505734, 0.18876731, 0.20118835, 0.18913657],
             [0.12391507, 0.13292464, 0.1511357, 0.17420742, 0.1924037, 0.21190241]])

        numpy.testing.assert_allclose(
            actual_cc_park, expected_cc_park, atol=1e-6)
