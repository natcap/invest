# -*- coding: utf-8 -*-
"""Tests for Crop Production Model."""
import unittest
import os
import shutil
import csv

import numpy as np
from osgeo import gdal
from pygeoprocessing import geoprocessing as geoprocess
import pygeoprocessing.testing as pygeotest
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')

NODATA_INT = -9999

lookup_table_list = \
    [['name', 'code', 'is_crop'],
     ['apple', '0', 'true'],
     ['roads', '1', 'false'],
     ['wheat', '2', 'true'],
     ['maize', '3', 'true'],
     ['parks', '4', 'false']]
nutrient_table_list = \
    [['crop', 'fraction_refuse', 'Protein', 'Lipid', 'Energy'],
     ['apple', '0.5', '100.1', '50.5', '20000.'],
     ['wheat', '0.5', '100.1', '50.5', '20000.'],
     ['maize', '0.5', '100.1', '50.5', '20000.']]
economics_table_list = \
    [['crop', 'price_per_ton', 'cost_nitrogen_per_kg',
        'cost_phosphorus_per_kg', 'cost_potash_per_kg', 'cost_labor_per_ha',
        'cost_machine_per_ha', 'cost_seed_per_ha', 'cost_irrigation_per_ha'],
     ['apple', '10', '1', '1', '1', '1', '1', '1', '1'],
     ['wheat', '10', '1', '1', '1', '1', '1', '1', '1'],
     ['maize', '10', '1', '1', '1', '1', '1', '1', '1']]
percentile_yield_table_list = \
    [['climate_bin', 'yield_25th', 'yield_50th', 'yield_75th', 'yield_95th'],
     ['0', '1.', '1.', '1.', '1.'],
     ['1', '1.', '1.', '1.', '2.'],
     ['2', '1.', '1.', '1.', '3.'],
     ['3', '1.', '1.', '1.', '4.']]
regression_yield_table_list = \
    [['climate_bin', 'yield_ceiling', 'b_nut', 'b_K2O', 'c_N', 'c_P2O5',
      'c_K2O', 'yield_ceiling_rf'],
     ['1', '3', '0.64', '0.44654', '0.03563', '0.17544', '0.34482', '2.7366'],
     ['2', '3', '0.64', '0.44654', '0.03563', '0.17544', '0.34482', '2.7366'],
     ['3', '3', '0.64', '0.44654', '0.03563', '0.17544', '0.34482', '2.7366']]


def _read_array(raster_path):
    """Read raster as array."""
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray()
    ds = None
    return a


def _create_table(uri, rows_list):
    """Create csv file from list of lists."""
    with open(uri, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows_list)
    return uri


def _create_raster(template_path, dst_path):
    """Create raster.

    Args:
        template_path (str): path to template raster.
        dst_path (str): path to destination raster.
    """
    match_ds = gdal.Open(template_path, 0)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    block_size = [256, 256]
    opt = ['TILED=YES',
           'BLOCKXSIZE=%d' % block_size[0],
           'BLOCKYSIZE=%d' % block_size[1]]
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(dst_path, wide, high, 1, gdal.GDT_Float32, options=opt)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)


def _write_to_raster(output_raster, array, xoff, yoff):
    """Write numpy array to raster block.

    Args:
        output_raster (str): filepath to output raster
        array (np.array): block to save to raster
        xoff (int): offset index for x-dimension
        yoff (int): offset index for y-dimension
    """
    ds = gdal.Open(output_raster, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    band.WriteArray(array, xoff, yoff)
    ds = None


def _create_workspace():
    """Create workspace directory."""
    path = os.path.dirname(os.path.realpath(__file__))
    workspace = os.path.join(path, 'workspace')
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.mkdir(workspace)
    return workspace


def _create_fertilizer_rasters(workspace_dir, aoi_raster):
    """Create fertilizer rasters."""
    fertilizer_dir = os.path.join(workspace_dir, 'fertilizers')
    if os.path.exists(fertilizer_dir):
        shutil.rmtree(fertilizer_dir)
    os.mkdir(fertilizer_dir)

    fert_list = [
        ('nitrogen.tif', 1.), ('potash.tif', 2.), ('phosphorus.tif', 3.)]
    for fert, app_rate in fert_list:
        dst_path = os.path.join(fertilizer_dir, fert)
        _create_raster(aoi_raster, dst_path)
        for offset_dict, block in geoprocess.iterblocks(dst_path):
            block[block == block] = app_rate
            _write_to_raster(
                dst_path, block, offset_dict['xoff'], offset_dict['yoff'])

    return fertilizer_dir


def _create_global_maps(path, lookup_table_path, aoi_raster):
    lookup_dict = geoprocess.get_lookup_from_table(lookup_table_path, 'code')
    for k, v in lookup_dict.items():
        if v['is_crop'] == 'true':
            dst_path = os.path.join(path, v['name'] + '_.tif')
            _create_raster(aoi_raster, dst_path)
            for offset_dict, block in geoprocess.iterblocks(dst_path):
                block[block == block] = float(
                    v['code']) if float(v['code']) != 0 else 1.
                _write_to_raster(
                    dst_path, block, offset_dict['xoff'], offset_dict['yoff'])


def _create_global_tables(path, lookup_table_path, table):
    lookup_dict = geoprocess.get_lookup_from_table(lookup_table_path, 'code')
    for k, v in lookup_dict.items():
        if v['is_crop'] == 'true':
            _create_table(os.path.join(path, v['name'] + '_.csv'), table)


def _create_dataset(workspace_dir, aoi_raster, lookup_table_path):
    """Create global dataset."""
    dataset_dir = os.path.join(workspace_dir, 'dataset')
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.mkdir(dataset_dir)

    subdirs = {
        'climate_bin_maps': 'climate_bin_maps',
        'observed': 'observed_yield',
        'percentile': 'climate_percentile_yield',
        'regression': 'climate_regression_yield',
    }
    for i, dirname in subdirs.items():
        os.mkdir(os.path.join(dataset_dir, dirname))

    # make dataset
    for i in ['climate_bin_maps', 'observed_yield']:
        _create_global_maps(
            os.path.join(dataset_dir, i), lookup_table_path, aoi_raster)

    for folder, table in [
            ('climate_percentile_yield', percentile_yield_table_list),
            ('climate_regression_yield', regression_yield_table_list)]:
        _create_global_tables(
            os.path.join(dataset_dir, folder), lookup_table_path, table)

    return dataset_dir


def _get_args():
    """Create and return arguments for CBC main model.

    Returns:
        args (dict): main model arguments.
    """
    band = np.ones((4, 4)) * [1, 0, 2, 3]
    band[0, 0] = 5
    band_matrices = [band]
    srs = pygeotest.sampledata.SRS_WILLAMETTE

    workspace_dir = _create_workspace()
    lookup_table_path = _create_table(
        os.path.join(workspace_dir, 'lookup.csv'), lookup_table_list)
    nutrient_table_path = _create_table(os.path.join(
        workspace_dir, 'nutrient_contents.csv'), nutrient_table_list)
    economics_table_path = _create_table(os.path.join(
        workspace_dir, 'economics_table.csv'), economics_table_list)

    aoi_raster_path = pygeotest.create_raster_on_disk(
        band_matrices,
        srs.origin,
        srs.projection,
        NODATA_INT,
        srs.pixel_size(100),
        datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace_dir, 'aoi_raster.tif'))

    irrigation_raster_path = pygeotest.create_raster_on_disk(
        [np.ones((4, 4))],
        srs.origin,
        srs.projection,
        NODATA_INT,
        srs.pixel_size(100),
        datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace_dir, 'irrigation_raster.tif'))

    fertilizer_dir = _create_fertilizer_rasters(workspace_dir, aoi_raster_path)
    dataset_dir = _create_dataset(
        workspace_dir, aoi_raster_path, lookup_table_path)

    args = {
        'workspace_dir': workspace_dir,
        'results_suffix': 'scenario_name',
        'lookup_table': lookup_table_path,
        'aoi_raster': aoi_raster_path,
        'dataset_dir': dataset_dir,
        'yield_function': 'regression',
        'percentile_column': 'yield_95th',
        'fertilizer_dir': fertilizer_dir,
        'irrigation_raster': irrigation_raster_path,
        'compute_nutritional_contents': 'true',
        'nutrient_table': nutrient_table_path,
        'compute_financial_analysis': 'true',
        'economics_table': economics_table_path
    }

    return args


# class TestFunctions(unittest.TestCase):
#     """Test Crop Production model functions.
#
#
#     """
#
#     def setUp(self):
#         """Create arguments."""
#         self.args = _get_args()
#
#     def test_func(self):
#         """Crop Production: Test ___."""
#         pass
#
#     def test_func(self):
#         """Crop Production: Test ___."""
#         pass
#
#     def test_func(self):
#         """Crop Production: Test ___."""
#         pass
#
#     def test_func(self):
#         """Crop Production: Test ___."""
#         pass
#
#     def tearDown(self):
#         """Remove workspace."""
#         shutil.rmtree(self.args['workspace_dir'])


class TestModel(unittest.TestCase):
    """Test Crop Production model."""

    def setUp(self):
        """Create arguments."""
        self.args = _get_args()

    def test_model_run_observed(self):
        """Crop Production: Test main model for observed yield."""
        from natcap.invest.crop_production import crop_production
        self.args['yield_function'] = 'observed'
        crop_production.execute(self.args)
        a = _read_array(os.path.join(
            self.args['workspace_dir'], 'output', 'observed_yield.tif'))
        np.testing.assert_array_almost_equal(a[0], [0, 1., 2., 3.])

    def test_model_run_percentile(self):
        """Crop Production: Test main model for percentile yield."""
        from natcap.invest.crop_production import crop_production
        self.args['yield_function'] = 'percentile'
        crop_production.execute(self.args)
        a = _read_array(os.path.join(
            self.args['workspace_dir'], 'output', 'percentile_yield.tif'))
        np.testing.assert_array_almost_equal(a[0], [0, 2., 3., 4.])

    def test_model_run_regression(self):
        """Crop Production: Test main model for regression yield."""
        from natcap.invest.crop_production import crop_production
        self.args['yield_function'] = 'regression'
        crop_production.execute(self.args)
        a = _read_array(os.path.join(
            self.args['workspace_dir'], 'output', 'regression_yield.tif'))
        b = a[:, 1:]
        np.testing.assert_array_almost_equal(b, np.ones((4, 3)) * 1.14720523)

    def test_model_run_clear_cache_dir(self):
        """Crop Production: Test main model for observed yield."""
        from natcap.invest.crop_production import crop_production
        self.args['yield_function'] = 'observed'
        os.makedirs(os.path.join(self.args['workspace_dir'], 'intermediate'))
        crop_production.execute(self.args)

    # @scm.skip_if_data_missing(SAMPLE_DATA)
    # def test_binary(self):
    #     """Crop Production: Test main model run against InVEST-Data."""
    #     from natcap.invest.crop_production \
    #         import crop_production
    #     sample_data_path = os.path.join(SAMPLE_DATA, 'CropProduction')
    #     args = {}
    #     crop_production.execute(args)

    def tearDown(self):
        """Remove workspace."""
        shutil.rmtree(self.args['workspace_dir'])


if __name__ == '__main__':
    unittest.main()
