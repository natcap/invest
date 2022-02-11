# coding=UTF-8
"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
from osgeo import osr
import numpy
import numpy.random
import numpy.testing
import pygeoprocessing


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
    numpy.testing.assert_allclose(base_array, array_to_compare, rtol=0, atol=1e-6)


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
            'lulc_cur_year': 2016,
            'lulc_fut_year': 2030,
            'discount_rate': -7.1,
            'n_workers': -1,
        }

        # Create LULC rasters and pools csv in workspace and add them to args.
        lulc_names = ['lulc_cur_path', 'lulc_fut_path', 'lulc_redd_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, -1)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)

        # Add assertions for npv for future and REDD scenarios.
        # The npv was calculated based on _calculate_npv in carbon.py.
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_fut.tif'), -0.3422078)
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_redd.tif'), -0.4602106)

    def test_carbon_zero_rates(self):
        """Carbon: test with 0 discount and rate change."""
        from natcap.invest import carbon

        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': True,
            'price_per_metric_ton_of_c': 43.0,
            'rate_change': 0.0,
            'lulc_cur_year': 2016,
            'lulc_fut_year': 2030,
            'discount_rate': 0.0,
            'n_workers': -1,
        }

        # Create LULC rasters and pools csv in workspace and add them to args.
        lulc_names = ['lulc_cur_path', 'lulc_fut_path', 'lulc_redd_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, -1)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)

        # Add assertions for npv for future and REDD scenarios.
        # carbon change from cur to fut:
        # -58 Mg/ha * .0001 ha/pixel * 43 $/Mg = -0.2494 $/pixel
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_fut.tif'), -0.2494)
        # carbon change from cur to redd:
        # -78 Mg/ha * .0001 ha/pixel * 43 $/Mg = -0.3354 $/pixel
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_redd.tif'), -0.3354)

    def test_carbon_future(self):
        """Carbon: regression testing future scenario."""
        from natcap.invest import carbon
        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': True,
            'price_per_metric_ton_of_c': 43.0,
            'rate_change': 2.8,
            'lulc_cur_year': 2016,
            'lulc_fut_year': 2030,
            'discount_rate': -7.1,
            'n_workers': -1,
        }

        lulc_names = ['lulc_cur_path', 'lulc_fut_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, -1)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)
        # Add assertions for npv for the future scenario.
        # The npv was calculated based on _calculate_npv in carbon.py.
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_fut.tif'), -0.3422078)

    def test_carbon_missing_landcover_values(self):
        """Carbon: testing expected exception on missing LULC codes."""
        from natcap.invest import carbon
        args = {
            'workspace_dir': self.workspace_dir,
            'do_valuation': False,
            'n_workers': -1,
        }

        lulc_names = ['lulc_cur_path', 'lulc_fut_path']
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
            'lulc_cur_year': 2016,
            'lulc_fut_year': 2030,
            'discount_rate': -7.1,
            'n_workers': -1,
        }

        # Create LULC rasters and pools csv in workspace and add them to args.
        lulc_names = ['lulc_cur_path', 'lulc_fut_path', 'lulc_redd_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val, None)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)

        # Add assertions for npv for future and REDD scenarios.
        # The npv was calculated based on _calculate_npv in carbon.py.
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_fut.tif'), -0.3422078)
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_redd.tif'), -0.4602106)


class CarbonValidationTests(unittest.TestCase):
    """Tests for the Carbon Model ARGS_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'lulc_cur_path',
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
            ['lulc_fut_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_redd(self):
        """Carbon Validate: assert missing do_redd keys."""
        from natcap.invest import carbon
        from natcap.invest import validation

        args = {'do_redd': True}
        validation_errors = carbon.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['calc_sequestration',
             'lulc_redd_path'])
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
             'lulc_cur_year',
             'lulc_fut_year'])
        self.assertEqual(invalid_keys, expected_missing_keys)
