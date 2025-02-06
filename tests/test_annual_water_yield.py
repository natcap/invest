"""Module for Regression Testing the InVEST Annual Water Yield module."""
import os
import shutil
import tempfile
import unittest

import numpy
from shapely.geometry import Polygon

import pandas
import pygeoprocessing
from osgeo import gdal, ogr, osr

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'annual_water_yield')
SAMPLE_DATA = os.path.join(REGRESSION_DATA, 'input')
gdal.UseExceptions()


def make_watershed_vector(path_to_shp):
    """
    Generate watershed results shapefile with two polygons

    Args:
        path_to_shp (str): path to store watershed results vector

    Outputs:
        None
    """
    shapely_geometry_list = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
    ]
    projection_wkt = osr.GetUserInputAsWKT("EPSG:4326")
    vector_format = "ESRI Shapefile"
    fields = {"hp_energy": ogr.OFTReal, "hp_val": ogr.OFTReal,
              "ws_id": ogr.OFTReal, "rsupply_vl": ogr.OFTReal,
              "wyield_mn": ogr.OFTReal, "wyield_vol": ogr.OFTReal,
              "consum_mn": ogr.OFTReal, "consum_vol": ogr.OFTReal}
    attribute_list = [
        {"hp_energy": 1, "hp_val": 1, "ws_id": 0, "rsupply_vl": 2},
        {"hp_energy": 11, "hp_val": 3, "ws_id": 1, "rsupply_vl": 52}
        ]

    pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                               path_to_shp, projection_wkt,
                                               vector_format, fields,
                                               attribute_list)


class AnnualWaterYieldTests(unittest.TestCase):
    """Regression Tests for Annual Water Yield Model."""

    def setUp(self):
        """Overriding setUp func. to create temporary workspace directory."""
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
            'lulc_path': os.path.join(SAMPLE_DATA, 'lulc.tif'),
            'depth_to_root_rest_layer_path': os.path.join(
                SAMPLE_DATA,
                'depth_to_root_rest_layer.tif'),
            'precipitation_path': os.path.join(SAMPLE_DATA, 'precip.tif'),
            'pawc_path': os.path.join(SAMPLE_DATA, 'pawc.tif'),
            'eto_path': os.path.join(SAMPLE_DATA, 'eto.tif'),
            'watersheds_path': os.path.join(SAMPLE_DATA, 'watersheds.shp'),
            'biophysical_table_path': os.path.join(
                SAMPLE_DATA, 'biophysical_table.csv'),
            'seasonality_constant': 5,
            'n_workers': -1,
        }
        return args

    def test_invalid_lulc_veg(self):
        """Hydro: catching invalid LULC_veg values."""
        from natcap.invest import annual_water_yield

        args = AnnualWaterYieldTests.generate_base_args(self.workspace_dir)

        new_lulc_veg_path = os.path.join(self.workspace_dir,
                                         'new_lulc_veg.csv')

        table_df = pandas.read_csv(args['biophysical_table_path'])
        table_df['LULC_veg'] = ['']*len(table_df.index)
        table_df.to_csv(new_lulc_veg_path)
        args['biophysical_table_path'] = new_lulc_veg_path

        with self.assertRaises(ValueError) as cm:
            annual_water_yield.execute(args)
        self.assertTrue('veg value must be either 1 or 0' in str(cm.exception))

        table_df = pandas.read_csv(args['biophysical_table_path'])
        table_df['LULC_veg'] = ['-1']*len(table_df.index)
        table_df.to_csv(new_lulc_veg_path)
        args['biophysical_table_path'] = new_lulc_veg_path

        with self.assertRaises(ValueError) as cm:
            annual_water_yield.execute(args)
        self.assertTrue('veg value must be either 1 or 0' in str(cm.exception))

    def test_missing_lulc_value(self):
        """Hydro: catching missing LULC value in Biophysical table."""
        from natcap.invest import annual_water_yield

        args = AnnualWaterYieldTests.generate_base_args(self.workspace_dir)

        # remove a row from the biophysical table so that lulc value is missing
        bad_biophysical_path = os.path.join(
            self.workspace_dir, 'bad_biophysical_table.csv')

        bio_df = pandas.read_csv(args['biophysical_table_path'])
        bio_df = bio_df[bio_df['lucode'] != 2]
        bio_df.to_csv(bad_biophysical_path)
        bio_df = None

        args['biophysical_table_path'] = bad_biophysical_path

        with self.assertRaises(ValueError) as cm:
            annual_water_yield.execute(args)
        self.assertTrue(
            "The missing values found in the LULC raster but not the table"
            " are: [2]" in str(cm.exception))

    def test_missing_lulc_demand_value(self):
        """Hydro: catching missing LULC value in Demand table."""
        from natcap.invest import annual_water_yield

        args = AnnualWaterYieldTests.generate_base_args(self.workspace_dir)

        args['demand_table_path'] = os.path.join(
            SAMPLE_DATA, 'water_demand_table.csv')
        args['sub_watersheds_path'] = os.path.join(
            SAMPLE_DATA, 'subwatersheds.shp')

        # remove a row from the biophysical table so that lulc value is missing
        bad_demand_path = os.path.join(
            self.workspace_dir, 'bad_demand_table.csv')

        demand_df = pandas.read_csv(args['demand_table_path'])
        demand_df = demand_df[demand_df['lucode'] != 2]
        demand_df.to_csv(bad_demand_path)
        demand_df = None

        args['demand_table_path'] = bad_demand_path

        with self.assertRaises(ValueError) as cm:
            annual_water_yield.execute(args)
        self.assertTrue(
            "The missing values found in the LULC raster but not the table"
            " are: [2]" in str(cm.exception))

    def test_water_yield_subshed(self):
        """Hydro: testing water yield component only w/ subwatershed."""
        from natcap.invest import annual_water_yield
        from natcap.invest import utils

        args = AnnualWaterYieldTests.generate_base_args(self.workspace_dir)
        args['sub_watersheds_path'] = os.path.join(
            SAMPLE_DATA, 'subwatersheds.shp')
        args['results_suffix'] = 'test'
        annual_water_yield.execute(args)

        raster_results = ['aet_test.tif', 'fractp_test.tif', 'wyield_test.tif']
        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(
                    REGRESSION_DATA, raster_path.replace('_test', '')))
            numpy.testing.assert_allclose(model_array, reg_array, rtol=1e-03)

        vector_results = ['watershed_results_wyield_test.shp',
                          'subwatershed_results_wyield_test.shp']
        for vector_path in vector_results:
            utils._assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(
                    REGRESSION_DATA, 'water_yield', vector_path.replace(
                        '_test', '')))

        table_results = ['watershed_results_wyield_test.csv',
                         'subwatershed_results_wyield_test.csv']
        for table_path in table_results:
            base_table = pandas.read_csv(
                os.path.join(args['workspace_dir'], 'output', table_path))
            expected_table = pandas.read_csv(
                os.path.join(
                    REGRESSION_DATA, 'water_yield',
                    table_path.replace('_test', '')))
            pandas.testing.assert_frame_equal(base_table, expected_table)

    def test_scarcity_subshed(self):
        """Hydro: testing Scarcity component w/ subwatershed."""
        from natcap.invest import annual_water_yield
        from natcap.invest import utils

        args = AnnualWaterYieldTests.generate_base_args(self.workspace_dir)
        args['demand_table_path'] = os.path.join(
            SAMPLE_DATA, 'water_demand_table.csv')
        args['sub_watersheds_path'] = os.path.join(
            SAMPLE_DATA, 'subwatersheds.shp')

        annual_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, raster_path))
            numpy.testing.assert_allclose(model_array, reg_array, rtol=1e-03)

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            utils._assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'scarcity', vector_path))

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            base_table = pandas.read_csv(
                os.path.join(args['workspace_dir'], 'output', table_path))
            expected_table = pandas.read_csv(
                os.path.join(REGRESSION_DATA, 'scarcity', table_path))
            pandas.testing.assert_frame_equal(base_table, expected_table)

    def test_valuation_subshed(self):
        """Hydro: testing Valuation component w/ subwatershed."""
        from natcap.invest import annual_water_yield
        from natcap.invest import utils

        args = AnnualWaterYieldTests.generate_base_args(self.workspace_dir)
        args['demand_table_path'] = os.path.join(
            SAMPLE_DATA, 'water_demand_table.csv')
        args['valuation_table_path'] = os.path.join(
            SAMPLE_DATA, 'hydropower_valuation_table.csv')
        args['sub_watersheds_path'] = os.path.join(
            SAMPLE_DATA, 'subwatersheds.shp')

        annual_water_yield.execute(args)

        raster_results = ['aet.tif', 'fractp.tif', 'wyield.tif']
        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, raster_path))
            numpy.testing.assert_allclose(model_array, reg_array, 1e-03)

        vector_results = ['watershed_results_wyield.shp',
                          'subwatershed_results_wyield.shp']
        for vector_path in vector_results:
            utils._assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'valuation', vector_path))

        table_results = ['watershed_results_wyield.csv',
                         'subwatershed_results_wyield.csv']
        for table_path in table_results:
            base_table = pandas.read_csv(
                os.path.join(args['workspace_dir'], 'output', table_path))
            expected_table = pandas.read_csv(
                os.path.join(REGRESSION_DATA, 'valuation', table_path))
            pandas.testing.assert_frame_equal(base_table, expected_table)

    def test_validation(self):
        """Hydro: test failure cases on the validation function."""
        from natcap.invest import annual_water_yield
        from natcap.invest import validation

        args = AnnualWaterYieldTests.generate_base_args(self.workspace_dir)

        # default args should be fine
        self.assertEqual(annual_water_yield.validate(args), [])

        args_bad_vector = args.copy()
        args_bad_vector['watersheds_path'] = args_bad_vector['eto_path']
        bad_vector_list = annual_water_yield.validate(args_bad_vector)
        self.assertTrue('not be opened as a GDAL vector'
                        in bad_vector_list[0][1])

        args_bad_raster = args.copy()
        args_bad_raster['eto_path'] = args_bad_raster['watersheds_path']
        bad_raster_list = annual_water_yield.validate(args_bad_raster)
        self.assertTrue('not be opened as a GDAL raster'
                        in bad_raster_list[0][1])

        args_bad_file = args.copy()
        args_bad_file['eto_path'] = 'non_existant_file.tif'
        bad_file_list = annual_water_yield.validate(args_bad_file)
        self.assertTrue('File not found' in bad_file_list[0][1])

        args_missing_key = args.copy()
        del args_missing_key['eto_path']
        validation_warnings = annual_water_yield.validate(
            args_missing_key)
        self.assertEqual(
            validation_warnings,
            [(['eto_path'], validation.MESSAGES['MISSING_KEY'])])

        # ensure that a missing landcover code in the biophysical table will
        # raise an exception that's helpful
        args_bad_biophysical_table = args.copy()
        bad_biophysical_path = os.path.join(
            self.workspace_dir, 'bad_biophysical_table.csv')
        with open(bad_biophysical_path, 'wb') as bad_biophysical_file:
            with open(args['biophysical_table_path'], 'rb') as (
                    biophysical_table_file):
                lines_to_write = 2
                for line in biophysical_table_file.readlines():
                    bad_biophysical_file.write(line)
                    lines_to_write -= 1
                    if lines_to_write == 0:
                        break
        args_bad_biophysical_table['biophysical_table_path'] = (
            bad_biophysical_path)
        with self.assertRaises(ValueError) as cm:
            annual_water_yield.execute(args_bad_biophysical_table)
        actual_message = str(cm.exception)
        self.assertTrue(
            "The missing values found in the LULC raster but not the table"
            " are: [2 3]" in actual_message, actual_message)

        # ensure that a missing landcover code in the demand table will
        # raise an exception that's helpful
        args_bad_biophysical_table = args.copy()
        bad_biophysical_path = os.path.join(
            self.workspace_dir, 'bad_biophysical_table.csv')
        with open(bad_biophysical_path, 'wb') as bad_biophysical_file:
            with open(args['biophysical_table_path'], 'rb') as (
                    biophysical_table_file):
                lines_to_write = 2
                for line in biophysical_table_file.readlines():
                    bad_biophysical_file.write(line)
                    lines_to_write -= 1
                    if lines_to_write == 0:
                        break
        args_bad_demand_table = args.copy()
        bad_demand_path = os.path.join(
            self.workspace_dir, 'bad_demand_table.csv')
        args_bad_demand_table['demand_table_path'] = (
            bad_demand_path)
        with open(bad_demand_path, 'wb') as bad_demand_file:
            with open(os.path.join(
                SAMPLE_DATA, 'water_demand_table.csv'), 'rb') as (
                    demand_table_file):
                lines_to_write = 2
                for line in demand_table_file.readlines():
                    bad_demand_file.write(line)
                    lines_to_write -= 1
                    if lines_to_write == 0:
                        break

        # ensure that a missing watershed id the valuation table will
        # raise an exception that's helpful
        with self.assertRaises(ValueError) as cm:
            annual_water_yield.execute(args_bad_demand_table)
        actual_message = str(cm.exception)
        self.assertTrue(
            "The missing values found in the LULC raster but not the table"
            " are: [2 3]" in actual_message, actual_message)

        args_bad_valuation_table = args.copy()
        bad_valuation_path = os.path.join(
            self.workspace_dir, 'bad_valuation_table.csv')
        args_bad_valuation_table['valuation_table_path'] = (
            bad_valuation_path)
        # args contract requires a demand table if there is a valuation table
        args_bad_valuation_table['demand_table_path'] = os.path.join(
            SAMPLE_DATA, 'water_demand_table.csv')

        with open(bad_valuation_path, 'wb') as bad_valuation_file:
            with open(os.path.join(
                SAMPLE_DATA, 'hydropower_valuation_table.csv'), 'rb') as (
                    valuation_table_file):
                lines_to_write = 2
                for line in valuation_table_file.readlines():
                    bad_valuation_file.write(line)
                    lines_to_write -= 1
                    if lines_to_write == 0:
                        break

        with self.assertRaises(ValueError) as cm:
            annual_water_yield.execute(args_bad_valuation_table)
        actual_message = str(cm.exception)
        self.assertTrue(
            'but are not found in the valuation table' in
            actual_message, actual_message)
        # if the demand table is missing but the valuation table is present,
        # make sure we have a validation error.
        args_missing_demand_table = args.copy()
        args_missing_demand_table['demand_table_path'] = ''
        args_missing_demand_table['valuation_table_path'] = (
            os.path.join(SAMPLE_DATA, 'hydropower_valuation_table.csv'))
        validation_warnings = annual_water_yield.validate(
            args_missing_demand_table)
        self.assertEqual(len(validation_warnings), 1)
        self.assertEqual(
            validation_warnings[0],
            (['demand_table_path'], 'Input is required but has no value'))

    def test_fractp_op(self):
        """Test `fractp_op`"""
        from natcap.invest.annual_water_yield import fractp_op

        # generate fake data
        kc = numpy.array([[1, .1, .1], [.6, .6, .1]])
        eto = numpy.array([[1000, 900, 900], [1100, 1005, 1000]])
        precip = numpy.array([[100, 1000, 10], [500, 800, 1100]])
        root = numpy.array([[99, 300, 400], [5, 500, 800]])
        soil = numpy.array([[600, 700, 700], [800, 900, 600]])
        pawc = numpy.array([[.11, .11, .12], [.55, .55, .19]])
        veg = numpy.array([[1, 1, 0], [0, 1, 0]])
        nodata_dict = {'eto': None, 'precip': None, 'depth_root': None,
                       'pawc': None, 'out_nodata': None}
        seasonality_constant = 6

        actual_fractp = fractp_op(kc, eto, precip, root, soil, pawc, veg,
                                  nodata_dict, seasonality_constant)

        # generated by running fractp_op
        expected_fractp = numpy.array([[0.9345682, 0.06896508, 1.],
                                       [1., 0.6487423, 0.09090909]],
                                       dtype=numpy.float32)

        numpy.testing.assert_allclose(actual_fractp, expected_fractp,
                                      err_msg="Fractp does not match expected")

    def test_compute_watershed_valuation(self):
        """Test `compute_watershed_valuation`, `compute_rsupply_volume`
        and `compute_water_yield_volume`"""
        from natcap.invest import annual_water_yield

        def _create_watershed_results_vector(path_to_shp):
            """Generate a fake watershed results vector file."""
            shapely_geometry_list = [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
            ]
            projection_wkt = osr.GetUserInputAsWKT("EPSG:4326")
            vector_format = "ESRI Shapefile"
            fields = {"ws_id": ogr.OFTReal, "wyield_mn": ogr.OFTReal,
                      "consum_mn": ogr.OFTReal, "consum_vol": ogr.OFTReal}
            attribute_list = [{"ws_id": 0, "wyield_mn": 990000,
                               "consum_mn": 500, "consum_vol": 50},
                              {"ws_id": 1, "wyield_mn": 800000,
                               "consum_mn": 600, "consum_vol": 70}]

            pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                                       path_to_shp,
                                                       projection_wkt,
                                                       vector_format, fields,
                                                       attribute_list)

        def _validate_fields(vector_path, field_name, expected_values, error_msg):
            """
            Validate a specific field in the watershed results vector
            by comparing actual to expected values. Expected values generated
            by running the function.

            Args:
                vector path (str): path to watershed shapefile
                field_name (str): attribute field to check
                expected values (list): list of expected values for field
                error_msg (str): what to print if assertion fails

            Returns:
                None
            """
            with gdal.OpenEx(vector_path, gdal.OF_VECTOR | gdal.GA_Update) as ws_ds:
                ws_layer = ws_ds.GetLayer()
                actual_values = [ws_feat.GetField(field_name)
                                 for ws_feat in ws_layer]
                self.assertEqual(actual_values, expected_values, msg=error_msg)

        # generate fake watershed results vector
        watershed_results_vector_path = os.path.join(self.workspace_dir,
                                                     "watershed_results.shp")
        _create_watershed_results_vector(watershed_results_vector_path)

        # generate fake val_df
        val_df = pandas.DataFrame({'efficiency': [.7, .8], 'height': [12, 50],
                                   'fraction': [.9, .7], 'discount': [60, 20],
                                   'time_span': [10, 10], 'cost': [100, 200],
                                   'kw_price': [15, 20]})

        # test water yield volume
        annual_water_yield.compute_water_yield_volume(
            watershed_results_vector_path)
        _validate_fields(watershed_results_vector_path, "wyield_vol",
                         [990.0, 800.0],
                         "Error with water yield volume calculation.")

        # test rsupply volume
        annual_water_yield.compute_rsupply_volume(
            watershed_results_vector_path)
        _validate_fields(watershed_results_vector_path, "rsupply_vl",
                         [940.0, 730.0],
                         "Error calculating total realized water supply volume.")

        # test compute watershed valuation
        annual_water_yield.compute_watershed_valuation(
            watershed_results_vector_path, val_df)
        _validate_fields(watershed_results_vector_path, "hp_energy",
                         [19.329408, 55.5968],
                         "Error calculating energy.")
        _validate_fields(watershed_results_vector_path, "hp_val",
                         [501.9029748723, 4587.91946857059],
                         "Error calculating net present value.")
