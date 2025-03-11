# coding=UTF-8
"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os
import re

from osgeo import gdal
from osgeo import osr
import numpy
import numpy.random
import numpy.testing

import pygeoprocessing

gdal.UseExceptions()


def make_simple_raster(base_raster_path, fill_val, nodata_val):
    """Create a 10x10 raster on designated path with fill value.

    Args:
        base_raster_path (str): the raster path for making the new raster.
        fill_val (int): the value used for filling the raster.
        nodata_val (int or None): for defining a band's nodata value.

    Returns:
        lulc_path (str): the path of the raster file.

    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # UTM Zone 10N
    projection_wkt = srs.ExportToWkt()
    # origin hand-picked for this epsg:
    geotransform = [461261, 1.0, 0.0, 4923265, 0.0, -1.0]

    n = 10
    gtiff_driver = gdal.GetDriverByName('GTiff')
    new_raster = gtiff_driver.Create(
        base_raster_path, n, n, 1, gdal.GDT_Int32, options=[
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
    new_raster.SetProjection(projection_wkt)
    new_raster.SetGeoTransform(geotransform)
    new_band = new_raster.GetRasterBand(1)
    array = numpy.empty((n, n))
    array.fill(fill_val)
    new_band.WriteArray(array)
    if nodata_val is not None:
        new_band.SetNoDataValue(nodata_val)
    new_raster.FlushCache()
    new_band = None
    new_raster = None


def assert_raster_equal_value(base_raster_path, val_to_compare):
    """Assert that the entire output raster has the same value as specified.

    Args:
        base_raster_path (str): the filepath of the raster to be asserted.
        val_to_compare (float): the value to be filled in the array to compare.

    Returns:
        None.

    """
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    base_band = base_raster.GetRasterBand(1)
    base_array = base_band.ReadAsArray()

    array_to_compare = numpy.empty(base_array.shape)
    array_to_compare.fill(val_to_compare)
    numpy.testing.assert_allclose(base_array, array_to_compare,
                                  rtol=0, atol=1e-3)


def make_pools_csv(pools_csv_path):
    """Create a carbon pools csv file with simplified land cover types.

    Args:
        pools_csv_path (str): the path of carbon pool csv.

    Returns:
        None.

    """
    with open(pools_csv_path, 'w') as open_table:
        open_table.write('C_above,C_below,C_soil,C_dead,lucode,LULC_Name\n')
        open_table.write('15,10,60,1,1,"lulc code 1"\n')
        # total change from 1 -> 2: -58 metric tons per hectare
        open_table.write('5,3,20,0,2,"lulc code 2"\n')
        # total change from 1 -> 3: -78 metric tons per hectare
        open_table.write('2,1,5,0,3,"lulc code 3"\n')


def assert_aggregate_result_equal(html_report_path, stat_name, val_to_compare):
    """Assert that the given stat in the HTML report has a specific value.

    Args:
        html_report_path (str): path to the HTML report generated by the model.
        stat_name (str): name of the stat to find. Must match the name listed
            in the HTML.
        val_to_compare (float): the value to check against.

    Returns:
        None.
    """
    with open(html_report_path) as file:
        report = file.read()
        pattern = (r'data-summary-stat="'
                   + stat_name
                   + r'">(\-?\d+\.\d{2})</td>')
        match = re.search(pattern, report)
        stat_str = match.groups()[0]
        assert float(stat_str) == val_to_compare


class CarbonTests(unittest.TestCase):
    """Tests for the Carbon Model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp(suffix='\U0001f60e')  # smiley

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_carbon_full(self):
        """Carbon: full model run."""
        from natcap.invest import carbon

        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': True,
            'price_per_metric_ton_of_c': 43.0,
            'rate_change': 2.8,
            'lulc_bas_year': 2016,
            'lulc_alt_year': 2030,
            'discount_rate': -7.1,
            'n_workers': -1,
        }

        # Create LULC rasters and pools csv in workspace and add them to args.
        lulc_names = ['lulc_bas_path', 'lulc_alt_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, -1)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)

        # Ensure every pixel has the correct total C value.
        # Baseline: 15 + 10 + 60 + 1 = 86 Mg/ha
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'c_storage_bas.tif'), 86)
        # Alternate: 5 + 3 + 20 + 0 = 28 Mg/ha
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'c_storage_alt.tif'), 28)

        # Ensure c_changes are correct.
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'c_change_bas_alt.tif'), -58)

        # Ensure NPV calculations are correct.
        # Valuation constant based on provided args is 59.00136.
        # Alternate: 59.00136 * -58 Mg/ha = -3422.079 Mg/ha
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_alt.tif'), -3422.079)

        # Ensure aggregate results are correct.
        report_path = os.path.join(args['workspace_dir'], 'report.html')
        # Raster size is 100 m^2; therefore, raster total is as follows:
        # (x Mg / 1 ha) * (1 ha / 10000 m^2) * (100 m^2) = (x / 100) Mg
        for (stat, expected_value) in [
                ('Baseline Carbon Storage', 0.86),
                ('Alternate Carbon Storage', 0.28),
                ('Change in C for Alternate', -0.58),
                ('Net present value from bas to alt', -34.22),
                ]:
            assert_aggregate_result_equal(report_path, stat, expected_value)

    def test_carbon_zero_rates(self):
        """Carbon: test with 0 discount and rate change."""
        from natcap.invest import carbon

        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': True,
            'price_per_metric_ton_of_c': 43.0,
            'rate_change': 0.0,
            'lulc_bas_year': 2016,
            'lulc_alt_year': 2030,
            'discount_rate': 0.0,
            'n_workers': -1,
        }

        # Create LULC rasters and pools csv in workspace and add them to args.
        lulc_names = ['lulc_bas_path', 'lulc_alt_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, -1)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)

        # Add assertions for npv for alternate scenario.
        # carbon change from bas to alt:
        # -58 Mg/ha * 43 $/Mg = -2494 $/ha
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_alt.tif'), -2494)

    def test_carbon_alternate_scenario(self):
        """Carbon: regression testing for alternate scenario"""
        from natcap.invest import carbon
        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': True,
            'price_per_metric_ton_of_c': 43.0,
            'rate_change': 2.8,
            'lulc_bas_year': 2016,
            'lulc_alt_year': 2030,
            'discount_rate': -7.1,
            'n_workers': -1,
        }

        lulc_names = ['lulc_bas_path', 'lulc_alt_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, -1)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)

        # Ensure NPV calculations are correct.
        # Valuation constant based on provided args is 59.00136.
        # Alternate: 59.00136 * -58 Mg/ha = -3422.079 Mg/ha
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_alt.tif'), -3422.079)

    def test_carbon_missing_landcover_values(self):
        """Carbon: testing expected exception on missing LULC codes."""
        from natcap.invest import carbon
        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': False,
            'n_workers': -1,
        }

        lulc_names = ['lulc_bas_path', 'lulc_alt_path']
        for fill_val, lulc_name in enumerate(lulc_names, 200):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, -1)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        # Value error should be raised with lulc code 200
        with self.assertRaises(ValueError) as cm:
            carbon.execute(args)

        self.assertTrue(
            "The missing values found in the LULC raster but not the table"
            " are: [200]" in str(cm.exception))

    def test_carbon_full_undefined_nodata(self):
        """Carbon: full model run when input raster nodata is None."""
        from natcap.invest import carbon

        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': True,
            'price_per_metric_ton_of_c': 43.0,
            'rate_change': 2.8,
            'lulc_bas_year': 2016,
            'lulc_alt_year': 2030,
            'discount_rate': -7.1,
            'n_workers': -1,
        }

        # Create LULC rasters and pools csv in workspace and add them to args.
        lulc_names = ['lulc_bas_path', 'lulc_alt_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, None)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)

        # Ensure NPV calculations are correct.
        # Valuation constant based on provided args is 59.00136.
        # Alternate: 59.00136 * -58 Mg/ha = -3422.079 Mg/ha
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_alt.tif'), -3422.079)

    def test_generate_carbon_map(self):
        """Test `_generate_carbon_map`"""
        from natcap.invest.carbon import _generate_carbon_map

        def _make_simple_lulc_raster(base_raster_path):
            """Create a raster on designated path with arbitrary values.
            Args:
                base_raster_path (str): the raster path for making the new raster.
            Returns:
                None.
            """

            array = numpy.array([[1, 1], [2, 3]], dtype=numpy.int32)

            # UTM Zone 10N
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(26910)
            projection_wkt = srs.ExportToWkt()

            origin = (461251, 4923245)
            pixel_size = (1, 1)
            no_data = -999

            pygeoprocessing.numpy_array_to_raster(
                array, no_data, pixel_size, origin, projection_wkt,
                base_raster_path)

        # generate a fake lulc raster
        lulc_path = os.path.join(self.workspace_dir, "lulc.tif")
        _make_simple_lulc_raster(lulc_path)

        # make fake carbon pool dict
        carbon_pool_by_type = {1: 5000, 2: 60, 3: 120}

        out_carbon_stock_path = os.path.join(self.workspace_dir,
                                             "carbon_stock.tif")

        _generate_carbon_map(lulc_path, carbon_pool_by_type,
                             out_carbon_stock_path)

        # open output carbon stock raster and check values
        actual_carbon_stock = pygeoprocessing.raster_to_numpy_array(
            out_carbon_stock_path)

        expected_carbon_stock = numpy.array([[5000, 5000], [60, 120]],
                                            dtype=numpy.float32)

        numpy.testing.assert_array_equal(actual_carbon_stock,
                                         expected_carbon_stock)

    def test_calculate_valuation_constant(self):
        """Test `_calculate_valuation_constant`"""
        from natcap.invest.carbon import _calculate_valuation_constant

        valuation_constant = _calculate_valuation_constant(lulc_bas_year=2010,
                                                           lulc_alt_year=2012,
                                                           discount_rate=50,
                                                           rate_change=5,
                                                           price_per_metric_ton_of_c=50)
        expected_valuation = 40.87302
        self.assertEqual(round(valuation_constant, 5), expected_valuation)


class CarbonValidationTests(unittest.TestCase):
    """Tests for the Carbon Model MODEL_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'lulc_bas_path',
            'carbon_pools_path',
        ]

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys(self):
        """Carbon Validate: assert missing required keys."""
        from natcap.invest import carbon
        from natcap.invest import validation

        validation_errors = carbon.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_sequestration(self):
        """Carbon Validate: assert missing calc_sequestration keys."""
        from natcap.invest import carbon
        from natcap.invest import validation

        args = {'calc_sequestration': True}
        validation_errors = carbon.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['lulc_alt_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_valuation(self):
        """Carbon Validate: assert missing do_valuation keys."""
        from natcap.invest import carbon
        from natcap.invest import validation

        args = {'do_valuation': True}
        validation_errors = carbon.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['calc_sequestration',
             'price_per_metric_ton_of_c',
             'discount_rate',
             'rate_change',
             'lulc_bas_year',
             'lulc_alt_year'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_invalid_lulc_years(self):
        """Test Alternate LULC year < Baseline LULC year raises a ValueError"""
        from natcap.invest import carbon

        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': True,
            'lulc_bas_year': 2025,
            'lulc_alt_year': 2023,
        }

        with self.assertRaises(ValueError):
            carbon.execute(args)
