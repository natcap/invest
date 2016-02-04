# -*- coding: utf-8 -*-
"""Tests for IO Functions."""
import unittest
import pprint
import os
import shutil
import csv

import numpy as np
from osgeo import gdal
from pygeoprocessing import geoprocessing as geoprocess
import pygeoprocessing.testing as pygeotest
from pygeoprocessing.testing import scm

pp = pprint.PrettyPrinter(indent=4)


lulc_lookup_list = \
    [['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
     ['N', '0', 'False'],
     ['X', '1', 'True'],
     ['Y', '1', 'True'],
     ['Z', '1', 'True']]

lulc_transition_matrix_list = \
    [['lulc-class', 'N', 'X', 'Y', 'Z'],
     ['N', 'NCC', 'accum', 'accum', 'accum'],
     ['X', 'med-impact-dist', 'accum', 'accum', 'accum'],
     ['Y', 'med-impact-dist', 'accum', 'accum', 'accum'],
     ['Z', 'med-impact-dist', 'accum', 'accum', 'accum']]

carbon_pool_initial_list = \
    [['code', 'lulc-class', 'biomass', 'soil', 'litter'],
     ['0', 'N', '0', '0', '0'],
     ['1', 'X', '5', '5', '0.5'],
     ['2', 'Y', '10', '10', '0.5'],
     ['3', 'Z', '20', '20', '0.5']]

carbon_pool_transient_list = \
    [['code', 'lulc-class', 'pool', 'half-life', 'med-impact-dist',
        'yearly_accumulation'],
     ['0', 'N', 'biomass', '0', '0', '0'],
     ['0', 'N', 'soil', '0', '0', '0'],
     ['1', 'X', 'biomass', '1', '0.5', '1'],
     ['1', 'X', 'soil', '1', '0.5', '1'],
     ['2', 'Y', 'biomass', '1', '0.5', '1'],
     ['2', 'Y', 'soil', '1', '0.5', '1'],
     ['3', 'Z', 'biomass', '1', '0.5', '1'],
     ['3', 'Z', 'soil', '1', '0.5', '1']]


def create_table(uri, rows_list):
    """Create csv file from list of lists."""
    with open(uri, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows_list)
    return uri


def get_args():
    band_matrices = [np.ones((2, 2))]
    srs = pygeotest.sampledata.SRS_WILLAMETTE

    path = os.path.dirname(os.path.realpath(__file__))
    workspace = os.path.join(path, 'workspace')
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.mkdir(workspace)
    lulc_lookup_uri = create_table(
        os.path.join(workspace, 'lulc_lookup.csv'), lulc_lookup_list)
    lulc_transition_matrix_uri = create_table(
        os.path.join(workspace, 'lulc_transition_matrix.csv'),
        lulc_transition_matrix_list)
    carbon_pool_initial_uri = create_table(
        os.path.join(workspace, 'carbon_pool_initial.csv'),
        carbon_pool_initial_list)
    carbon_pool_transient_uri = create_table(
        os.path.join(workspace, 'carbon_pool_transient.csv'),
        carbon_pool_transient_list)
    raster_0_uri = pygeotest.create_raster_on_disk(
        band_matrices, srs.origin, srs.projection, -1, srs.pixel_size(100),
        datatype=gdal.GDT_Int32, filename=os.path.join(
            workspace, 'raster_0.tif'))
    raster_1_uri = pygeotest.create_raster_on_disk(
        band_matrices, srs.origin, srs.projection, -1, srs.pixel_size(100),
        datatype=gdal.GDT_Int32, filename=os.path.join(
            workspace, 'raster_1.tif'))
    raster_2_uri = pygeotest.create_raster_on_disk(
        band_matrices, srs.origin, srs.projection, -1, srs.pixel_size(100),
        datatype=gdal.GDT_Int32, filename=os.path.join(
            workspace, 'raster_2.tif'))
    lulc_baseline_map_uri = raster_0_uri
    lulc_transition_maps_list = [raster_1_uri, raster_2_uri]

    args = {
        'workspace_dir': workspace,
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_uri,
        'lulc_transition_matrix_uri': lulc_transition_matrix_uri,
        'lulc_baseline_map_uri': raster_0_uri,
        'lulc_transition_maps_list': [raster_1_uri, raster_2_uri],
        'lulc_transition_years_list': [2000, 2005],
        'analysis_year': 2010,
        'carbon_pool_initial_uri': carbon_pool_initial_uri,
        'carbon_pool_transient_uri': carbon_pool_transient_uri,
        'do_economic_analysis': True,
        'do_price_table': False,
        'price': 2.,
        'interest_rate': 5.,
        'price_table_uri': None,
        'discount_rate': 2.
    }

    return args


def get_preprocessor_args(args_choice):
    band_matrices_zeros = [np.zeros((2, 2))]
    band_matrices_ones = [np.ones((2, 2))]
    srs = pygeotest.sampledata.SRS_WILLAMETTE

    path = os.path.dirname(os.path.realpath(__file__))
    workspace = os.path.join(path, 'workspace')
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.mkdir(workspace)

    lulc_lookup_uri = create_table(
        os.path.join(workspace, 'lulc_lookup.csv'),
        lulc_lookup_list)

    raster_0_uri = pygeotest.create_raster_on_disk(
        band_matrices_ones, srs.origin, srs.projection, -1,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_0.tif'))
    raster_1_uri = pygeotest.create_raster_on_disk(
        band_matrices_ones, srs.origin, srs.projection, -1,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_1.tif'))
    raster_2_uri = pygeotest.create_raster_on_disk(
        band_matrices_ones, srs.origin, srs.projection, -1,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_2.tif'))
    raster_3_uri = pygeotest.create_raster_on_disk(
        band_matrices_zeros, srs.origin, srs.projection, -1,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_3.tif'))

    args = {
        'workspace_dir': workspace,
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_uri,
        'lulc_snapshot_list': [raster_0_uri, raster_1_uri, raster_2_uri]
    }

    args2 = {
        'workspace_dir': workspace,
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_uri,
        'lulc_snapshot_list': [raster_0_uri, raster_1_uri, raster_3_uri]
    }

    if args_choice == 1:
        return args
    else:
        return args2


class TestIO(unittest.TestCase):

    """Test io library functions."""

    def setUp(self):
        pass

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_get_inputs(self):
        from natcap.invest.coastal_blue_carbon import io
        args = get_args()
        d = io.get_inputs(args)
        self.assertTrue(d['lulc_to_Hb'][0] == 0.0)
        self.assertTrue(d['lulc_to_Hb'][1] == 1.0)
        self.assertTrue(len(d['price_t']) == 11)
        self.assertTrue(len(d['snapshot_years']) == 3)
        self.assertTrue(len(d['transition_years']) == 2)
        shutil.rmtree(args['workspace_dir'])

    def tearDown(self):
        pass


class TestModel(unittest.TestCase):

    """Test main model functions."""

    def setUp(self):
        pass

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_model_run(self):
        """Test main model 'run' function."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc
        args = get_args()
        cbc.execute(args)
        output_raster = os.path.join(
            os.path.split(os.path.realpath(__file__))[0],
            'workspace/outputs_core/net_present_value_test.tif')
        ds = gdal.Open(output_raster)
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray()
        ds = None
        np.testing.assert_almost_equal(a[0, 0], 45.731491, decimal=5)
        shutil.rmtree(args['workspace_dir'])

    def tearDown(self):
        pass


class TestPreprocessor(unittest.TestCase):

    """Test preprocessor library functions."""

    def setUp(self):
        pass

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_preprocessor(self):
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = get_preprocessor_args(1)
        preprocessor.execute(args)
        trans_csv = os.path.join(
            args['workspace_dir'],
            'outputs_preprocessor',
            'transitions_test.csv')
        with open(trans_csv, 'r') as f:
            lines = f.readlines()
        self.assertTrue(lines[2][:].startswith('Z,,accum'))
        shutil.rmtree(args['workspace_dir'])

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_preprocessor_2(self):
        from natcap.invest.coastal_blue_carbon import preprocessor
        args2 = get_preprocessor_args(2)
        preprocessor.execute(args2)
        trans_csv = os.path.join(
            args2['workspace_dir'],
            'outputs_preprocessor',
            'transitions_test.csv')
        with open(trans_csv, 'r') as f:
            lines = f.readlines()
        self.assertTrue(lines[2][:].startswith('Z,disturb,accum'))
        shutil.rmtree(args2['workspace_dir'])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
