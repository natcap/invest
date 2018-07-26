"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
from osgeo import osr
import numpy
import pygeoprocessing.testing


def make_simple_raster(raster_path, fill_val):
    """Create a 10x10 raster with designated path and fill value.

    Parameters:
        raster_path (str): a raster path for making the new raster.
        fill_val (int): the value used for filling the raster.

    Returns:
        lulc_path (str): the path of the raster file.

    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # UTM Zone 10N
    projection_wkt = srs.ExportToWkt()

    lulc_array = numpy.empty((10, 10))
    lulc_array.fill(fill_val)
    pygeoprocessing.testing.create_raster_on_disk(
        [lulc_array],
        (461261, 4923265),  # Origin based on the projection
        projection_wkt,
        -1,
        (1, -1),
        filename=raster_path)


def assert_raster_equal_value(base_raster_path, val_to_compare):
    """Assert that the entire output raster has the same value as specified.

    Parameters:
        base_raster_path (str): the filepath of a raster.
        val_to_compare (float): the value to be filled in the array to compare.

    Returns:
        None.

    """
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    base_band = base_raster.GetRasterBand(1)
    base_array = base_band.ReadAsArray()

    array_to_compare = numpy.empty(base_array.shape)
    array_to_compare.fill(val_to_compare)
    numpy.testing.assert_almost_equal(base_array, array_to_compare)


def make_pools_csv(pools_csv_path):
    """Create a carbon pools csv file with simplified land cover types.

    Parameters:
        pools_csv_path (str): the path of carbon pool csv.

    Returns:
        None.

    """
    with open(pools_csv_path, 'w') as open_table:
        open_table.write('C_above,C_below,C_soil,C_dead,lucode,LULC_Name\n')
        open_table.write('15,10,60,1,1,"lulc code 1"\n')
        open_table.write('5,3,20,0,2,"lulc code 2"\n')
        open_table.write('2,1,5,0,3,"lulc code 3"\n')


class CarbonTests(unittest.TestCase):
    """Tests for the Carbon Model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_carbon_full(self):
        """Carbon: full model run."""
        from natcap.invest import carbon

        args = {
            u'workspace_dir': self.workspace_dir,
            u'do_valuation': True,
            u'price_per_metric_ton_of_c': 43.0,
            u'rate_change': 2.8,
            u'lulc_cur_year': 2016,
            u'lulc_fut_year': 2030,
            u'discount_rate': -7.1,
        }

        # Create LULC rasters and pools csv in workspace and add them to args.
        lulc_names = ['lulc_cur_path', 'lulc_fut_path', 'lulc_redd_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)

        # Add assertions for npv for future and REDD scenarios
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_fut.tif'), -0.3422078)
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_redd.tif'), -0.4602106)

    def test_carbon_future(self):
        """Carbon: regression testing future scenario."""
        from natcap.invest import carbon
        args = {
            u'workspace_dir': self.workspace_dir,
            u'do_valuation': True,
            u'price_per_metric_ton_of_c': 43.0,
            u'rate_change': 2.8,
            u'lulc_cur_year': 2016,
            u'lulc_fut_year': 2030,
            u'discount_rate': -7.1,
        }

        lulc_names = ['lulc_cur_path', 'lulc_fut_path']
        for fill_val, lulc_name in enumerate(lulc_names, 1):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        carbon.execute(args)
        # Add assertions for npv for the future scenario
        assert_raster_equal_value(
            os.path.join(args['workspace_dir'], 'npv_fut.tif'), -0.3422078)

    def test_carbon_missing_landcover_values(self):
        """Carbon: testing expected exception on missing LULC codes."""
        from natcap.invest import carbon
        args = {
            u'workspace_dir': self.workspace_dir,
            u'do_valuation': False,
        }

        lulc_names = ['lulc_cur_path', 'lulc_fut_path']
        for fill_val, lulc_name in enumerate(lulc_names, 200):
            args[lulc_name] = os.path.join(args['workspace_dir'],
                                           lulc_name + '.tif')
            make_simple_raster(args[lulc_name], fill_val)

        args['carbon_pools_path'] = os.path.join(args['workspace_dir'],
                                                 'pools.csv')
        make_pools_csv(args['carbon_pools_path'])

        # Value error should be raised with lulc code 200
        with self.assertRaises(ValueError):
            carbon.execute(args)
