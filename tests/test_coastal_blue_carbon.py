# -*- coding: utf-8 -*-
"""Tests for Coastal Blue Carbon Functions."""
import unittest
import os
import shutil
import csv
import logging
import tempfile
import functools
import copy
import pprint

import numpy
from osgeo import gdal
import pygeoprocessing.testing as pygeotest
from natcap.invest import utils

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'coastal_blue_carbon')
LOGGER = logging.getLogger(__name__)


lulc_lookup_list = \
    [['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
     ['n', '0', 'False'],
     ['x', '1', 'True'],
     ['y', '2', 'True'],
     ['z', '3', 'True']]

lulc_lookup_list_unreadable = \
    [['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
     ['n', '0', ''],
     ['x', '1', 'True'],
     ['y', '2', 'True'],
     ['z', '3', 'True']]

lulc_lookup_list_no_ones = \
    [['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
     ['n', '0', 'False'],
     ['y', '2', 'True'],
     ['z', '3', 'True']]

lulc_transition_matrix_list = \
    [['lulc-class', 'n', 'x', 'y', 'z'],
     ['n', 'NCC', 'accum', 'accum', 'accum'],
     ['x', 'med-impact-disturb', 'accum', 'accum', 'accum'],
     ['y', 'med-impact-disturb', 'accum', 'accum', 'accum'],
     ['z', 'med-impact-disturb', 'accum', 'accum', 'accum']]

carbon_pool_initial_list = \
    [['code', 'lulc-class', 'biomass', 'soil', 'litter'],
     ['0', 'n', '0', '0', '0'],
     ['1', 'x', '5', '5', '0.5'],
     ['2', 'y', '10', '10', '0.5'],
     ['3', 'z', '20', '20', '0.5']]

carbon_pool_transient_list = \
    [['code', 'lulc-class', 'biomass-half-life', 'biomass-med-impact-disturb',
        'biomass-yearly-accumulation',
        'soil-half-life',
        'soil-med-impact-disturb',
        'soil-yearly-accumulation'],
     ['0', 'n', '0', '0', '0', '0', '0', '0'],
     ['1', 'x', '1', '0.5', '1', '1', '0.5', '1.1'],
     ['2', 'y', '1', '0.5', '2', '1', '0.5', '2.1'],
     ['3', 'z', '1', '0.5', '1', '1', '0.5', '1.1']]

price_table_list = \
    [['year', 'price'],
     [2000, 20]]

NODATA_INT = -9999


def _read_array(raster_path):
    """"Read raster as array."""
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


def _create_workspace():
    """Create workspace directory."""
    return tempfile.mkdtemp()


def _get_args(workspace, num_transitions=2, valuation=True):
    """Create and return arguments for CBC main model.

    Parameters:
        workspace(string): A path to a folder on disk.  Generated inputs will
            be saved to this directory.
        num_transitions=2 (int): The number of transitions to synthesize.
        valuation=True (bool): Whether to include parameters related to
            valuation in the args dict.

    Returns:
        args (dict): main model arguments.
    """
    band_matrices = [numpy.ones((2, 2))]
    band_matrices_two = [numpy.ones((2, 2)) * 2]
    band_matrices_with_nodata = [numpy.ones((2, 2))]
    band_matrices_with_nodata[0][0][0] = NODATA_INT
    srs = pygeotest.sampledata.SRS_WILLAMETTE

    lulc_lookup_uri = _create_table(
        os.path.join(workspace, 'lulc_lookup.csv'), lulc_lookup_list)
    lulc_transition_matrix_uri = _create_table(
        os.path.join(workspace, 'lulc_transition_matrix.csv'),
        lulc_transition_matrix_list)
    carbon_pool_initial_uri = _create_table(
        os.path.join(workspace, 'carbon_pool_initial.csv'),
        carbon_pool_initial_list)
    carbon_pool_transient_uri = _create_table(
        os.path.join(workspace, 'carbon_pool_transient.csv'),
        carbon_pool_transient_list)
    raster_0_uri = pygeotest.create_raster_on_disk(
        band_matrices,
        srs.origin,
        srs.projection,
        NODATA_INT,
        srs.pixel_size(100),
        datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_0.tif'))
    raster_1_uri = pygeotest.create_raster_on_disk(
        band_matrices_with_nodata,
        srs.origin,
        srs.projection,
        NODATA_INT,
        srs.pixel_size(100),
        datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_1.tif'))
    raster_2_uri = pygeotest.create_raster_on_disk(
        band_matrices_two,
        srs.origin,
        srs.projection,
        NODATA_INT,
        srs.pixel_size(100),
        datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_2.tif'))

    possible_transitions = [raster_1_uri, raster_2_uri]
    possible_transition_years = [2000, 2005]

    args = {
        'workspace_dir': os.path.join(workspace, 'workspace'),
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_uri,
        'lulc_transition_matrix_uri': lulc_transition_matrix_uri,
        'lulc_baseline_map_uri': raster_0_uri,
        'lulc_baseline_year': 1995,
        'lulc_transition_maps_list': possible_transitions[:num_transitions+1],
        'lulc_transition_years_list': possible_transition_years[:num_transitions+1],
        'analysis_year': 2010,
        'carbon_pool_initial_uri': carbon_pool_initial_uri,
        'carbon_pool_transient_uri': carbon_pool_transient_uri,
        'do_economic_analysis': False,
    }

    utils.make_directories([args['workspace_dir']])

    if valuation:
        args.update({
            'do_economic_analysis': True,
            'do_price_table': False,
            'price': 2.,
            'inflation_rate': 5.,
            'price_table_uri': None,
            'discount_rate': 2.
        })

    return args


def _get_preprocessor_args(args_choice, workspace):
    """Create and return arguments for preprocessor model.

    Args:
        args_choice (int): which arguments to return
        workspace (string): The path to a workspace directory.

    Returns:
        args (dict): preprocessor arguments
    """
    band_matrices_zeros = [numpy.zeros((2, 2))]
    band_matrices_ones = [numpy.ones((2, 3))]  # tests alignment
    band_matrices_nodata = [numpy.ones((2, 2)) * NODATA_INT]
    srs = pygeotest.sampledata.SRS_WILLAMETTE

    lulc_lookup_uri = _create_table(
        os.path.join(workspace, 'lulc_lookup.csv'), lulc_lookup_list)

    raster_0_uri = pygeotest.create_raster_on_disk(
        band_matrices_ones, srs.origin, srs.projection, NODATA_INT,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_0.tif'))
    raster_1_uri = pygeotest.create_raster_on_disk(
        band_matrices_ones, srs.origin, srs.projection, NODATA_INT,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_1.tif'))
    raster_2_uri = pygeotest.create_raster_on_disk(
        band_matrices_ones, srs.origin, srs.projection, NODATA_INT,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_2.tif'))
    raster_3_uri = pygeotest.create_raster_on_disk(
        band_matrices_zeros, srs.origin, srs.projection, NODATA_INT,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_3.tif'))
    raster_4_uri = pygeotest.create_raster_on_disk(
        band_matrices_zeros, srs.origin, srs.projection, -1,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_4.tif'))
    raster_nodata_uri = pygeotest.create_raster_on_disk(
        band_matrices_nodata, srs.origin, srs.projection, NODATA_INT,
        srs.pixel_size(100), datatype=gdal.GDT_Int32,
        filename=os.path.join(workspace, 'raster_4.tif'))

    args = {
        'workspace_dir': os.path.join(workspace, 'workspace'),
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_uri,
        'lulc_snapshot_list': [raster_0_uri, raster_1_uri, raster_2_uri]
    }

    args2 = {
        'workspace_dir': os.path.join(workspace, 'workspace'),
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_uri,
        'lulc_snapshot_list': [raster_0_uri, raster_1_uri, raster_3_uri]
    }

    args3 = {
        'workspace_dir': os.path.join(workspace, 'workspace'),
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_uri,
        'lulc_snapshot_list': [raster_0_uri, raster_nodata_uri, raster_3_uri]
    }

    args4 = {
        'workspace_dir': os.path.join(workspace, 'workspace'),
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_uri,
        'lulc_snapshot_list': [raster_0_uri, raster_nodata_uri, raster_4_uri]
    }

    if args_choice == 1:
        return args
    elif args_choice == 2:
        return args2
    elif args_choice == 3:
        return args3
    else:
        return args4


class TestPreprocessor(unittest.TestCase):
    """Test Coastal Blue Carbon preprocessor library functions."""

    def setUp(self):
        """Create a temp directory for the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_create_carbon_pool_transient_table_template(self):
        """Coastal Blue Carbon: Test creation of transient table template."""
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(1, self.workspace_dir)
        filepath = os.path.join(self.workspace_dir,
                                'transient_temp.csv')
        code_to_lulc_dict = {1: 'one', 2: 'two', 3: 'three'}
        preprocessor._create_carbon_pool_transient_table_template(
            filepath, code_to_lulc_dict)
        transient_dict = utils.build_lookup_from_csv(filepath, 'code')
        # demonstrate that output table contains all input land cover classes
        for i in [1, 2, 3]:
            self.assertTrue(i in transient_dict.keys())

    def test_preprocessor_ones(self):
        """Coastal Blue Carbon: Test entire run of preprocessor (ones).

        All rasters contain ones.
        """
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(1, self.workspace_dir)
        preprocessor.execute(args)
        trans_csv = os.path.join(
            args['workspace_dir'],
            'outputs_preprocessor',
            'transitions_test.csv')
        with open(trans_csv, 'r') as f:
            lines = f.readlines()
        # just a regression test.  this tests that an output file was
        # successfully created, and demonstrates that one land class transition
        # does not occur and the other is set in the right direction.
        self.assertTrue(lines[2].startswith('x,,accum'))

    def test_preprocessor_zeros(self):
        """Coastal Blue Carbon: Test entire run of preprocessor (zeroes).

        First two rasters contain ones, last contains zeros.
        """
        from natcap.invest.coastal_blue_carbon import preprocessor
        args2 = _get_preprocessor_args(2, self.workspace_dir)
        preprocessor.execute(args2)
        trans_csv = os.path.join(
            args2['workspace_dir'],
            'outputs_preprocessor',
            'transitions_test.csv')
        with open(trans_csv, 'r') as f:
            lines = f.readlines()

        # just a regression test.  this tests that an output file was
        # successfully created, and that two particular land class transitions
        # occur and are set in the right directions.
        self.assertTrue(lines[2][:].startswith('x,disturb,accum'))

    def test_preprocessor_nodata(self):
        """Coastal Blue Carbon: Test run of preprocessor (various values).

        First raster contains ones, second nodata, third zeros.
        """
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(3, self.workspace_dir)
        preprocessor.execute(args)
        trans_csv = os.path.join(
            args['workspace_dir'],
            'outputs_preprocessor',
            'transitions_test.csv')
        with open(trans_csv, 'r') as f:
            lines = f.readlines()
        # just a regression test.  this tests that an output file was
        # successfully created, and that two particular land class transitions
        # occur and are set in the right directions.
        self.assertTrue(lines[2][:].startswith('x,,'))

    def test_preprocessor_user_defined_nodata(self):
        """Coastal Blue Carbon: Test preprocessor with user-defined nodata.

        First raster contains ones, second nodata, third zeros.
        """
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(4, self.workspace_dir)
        preprocessor.execute(args)
        trans_csv = os.path.join(
            self.workspace_dir,
            'workspace',
            'outputs_preprocessor',
            'transitions_test.csv')
        with open(trans_csv, 'r') as f:
            lines = f.readlines()
        # just a regression test.  this tests that an output file was
        # successfully created, and that two particular land class transitions
        # occur and are set in the right directions.
        self.assertTrue(lines[2][:].startswith('x,,'))

    def test_lookup_parsing_exception(self):
        """Coastal Blue Carbon: Test lookup table parsing exception."""
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(1, self.workspace_dir)
        _create_table(args['lulc_lookup_uri'], lulc_lookup_list_unreadable)
        with self.assertRaises(ValueError):
            preprocessor.execute(args)

    def test_raster_validation(self):
        """Coastal Blue Carbon: Test raster validation."""
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(1, self.workspace_dir)
        OTHER_NODATA = -1
        srs = pygeotest.sampledata.SRS_WILLAMETTE
        band_matrices_with_nodata = [numpy.ones((2, 2)) * OTHER_NODATA]
        raster_wrong_nodata = pygeotest.create_raster_on_disk(
            band_matrices_with_nodata,
            srs.origin,
            srs.projection,
            OTHER_NODATA,
            srs.pixel_size(100),
            datatype=gdal.GDT_Int32,
            filename=os.path.join(
                self.workspace_dir, 'raster_wrong_nodata.tif'))
        args['lulc_snapshot_list'][0] = raster_wrong_nodata
        with self.assertRaises(ValueError):
            preprocessor.execute(args)

    def test_raster_values_not_in_lookup_table(self):
        """Coastal Blue Carbon: Test raster values not in lookup table."""
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(1, self.workspace_dir)
        _create_table(args['lulc_lookup_uri'], lulc_lookup_list_no_ones)
        with self.assertRaises(ValueError):
            preprocessor.execute(args)

    def test_mark_transition_type(self):
        """Coastal Blue Carbon: Test mark_transition_type."""
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(1, self.workspace_dir)

        band_matrices_zero = [numpy.zeros((2, 2))]
        srs = pygeotest.sampledata.SRS_WILLAMETTE
        raster_zeros = pygeotest.create_raster_on_disk(
            band_matrices_zero,
            srs.origin,
            srs.projection,
            NODATA_INT,
            srs.pixel_size(100),
            datatype=gdal.GDT_Int32,
            filename=os.path.join(
                self.workspace_dir, 'raster_1.tif'))
        args['lulc_snapshot_list'][0] = raster_zeros

        preprocessor.execute(args)
        trans_csv = os.path.join(
            self.workspace_dir,
            'workspace',
            'outputs_preprocessor',
            'transitions_test.csv')
        with open(trans_csv, 'r') as f:
            lines = f.readlines()
        self.assertTrue(lines[1][:].startswith('n,NCC,accum'))

    def test_mark_transition_type_nodata_check(self):
        """Coastal Blue Carbon: Test mark_transition_type with nodata check."""
        from natcap.invest.coastal_blue_carbon import preprocessor
        args = _get_preprocessor_args(1, self.workspace_dir)

        band_matrices_zero = [numpy.zeros((2, 2))]
        srs = pygeotest.sampledata.SRS_WILLAMETTE
        raster_zeros = pygeotest.create_raster_on_disk(
            band_matrices_zero,
            srs.origin,
            srs.projection,
            NODATA_INT,
            srs.pixel_size(100),
            datatype=gdal.GDT_Int32,
            filename=os.path.join(
                self.workspace_dir, 'raster_1.tif'))
        args['lulc_snapshot_list'][0] = raster_zeros

        preprocessor.execute(args)

    def test_binary(self):
        """Coastal Blue Carbon: Test preprocessor run against InVEST-Data."""
        from natcap.invest.coastal_blue_carbon import preprocessor

        raster_0_uri = os.path.join(
            REGRESSION_DATA, 'inputs/GBJC_2010_mean_Resample.tif')
        raster_1_uri = os.path.join(
            REGRESSION_DATA, 'inputs/GBJC_2030_mean_Resample.tif')
        raster_2_uri = os.path.join(
            REGRESSION_DATA, 'inputs/GBJC_2050_mean_Resample.tif')
        args = {
            'workspace_dir': _create_workspace(),
            'results_suffix': '150225',
            'lulc_lookup_uri': os.path.join(
                REGRESSION_DATA, 'inputs', 'lulc_lookup.csv'),
            'lulc_snapshot_list': [raster_0_uri, raster_1_uri, raster_2_uri]
        }
        preprocessor.execute(args)

        # walk through all files in the workspace and assert that outputs have
        # the file suffix.
        non_suffixed_files = []
        for root_dir, dirnames, filenames in os.walk(args['workspace_dir']):
            for filename in filenames:
                if not filename.lower().endswith('.txt'):  # ignore logfile
                    basename, extension = os.path.splitext(filename)
                    if not basename.endswith('_150225'):
                        path_rel_to_workspace = os.path.relpath(
                            os.path.join(root_dir, filename),
                            self.args['workspace_dir'])
                        non_suffixed_files.append(path_rel_to_workspace)

        if non_suffixed_files:
            self.fail('%s files are missing suffixes: %s' %
                      (len(non_suffixed_files),
                       pprint.pformat(non_suffixed_files)))


class TestIO(unittest.TestCase):
    """Test Coastal Blue Carbon io library functions."""

    def setUp(self):
        """Create arguments."""
        self.workspace_dir = tempfile.mkdtemp()
        self.args = _get_args(self.workspace_dir)

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_get_inputs(self):
        """Coastal Blue Carbon: Test get_inputs function in IO module."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        d = cbc.get_inputs(self.args)
        # check several items in the data dictionary to check that the inputs
        # are properly fetched.
        self.assertEqual(d['lulc_to_Hb'][0], 0.0)
        self.assertEqual(d['lulc_to_Hb'][1], 1.0)
        self.assertEqual(len(d['price_t']), 16)
        self.assertEqual(len(d['snapshot_years']), 4)
        self.assertEqual(len(d['transition_years']), 2)

    def test_get_price_table_exception(self):
        """Coastal Blue Carbon: Test price table exception."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc
        self.args['price_table_uri'] = os.path.join(
            self.args['workspace_dir'], 'price.csv')
        self.args['do_price_table'] = True
        self.args['price_table_uri'] = _create_table(
            self.args['price_table_uri'], price_table_list)
        with self.assertRaises(KeyError):
            cbc.get_inputs(self.args)

    def test_chronological_order_exception(self):
        """Coastal Blue Carbon: Test exception checking chronological order."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc
        self.args['lulc_transition_years_list'] = [2005, 2000]
        with self.assertRaises(ValueError):
            cbc.get_inputs(self.args)

    def test_chronological_order_exception_analysis_year(self):
        """Coastal Blue Carbon: Test exception checking analysis year order."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc
        self.args['analysis_year'] = 2000
        with self.assertRaises(ValueError):
            cbc.get_inputs(self.args)

    def test_create_transient_dict(self):
        """Coastal Blue Carbon: Read transient table."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc
        biomass_transient_dict, soil_transient_dict = \
            cbc._create_transient_dict(self.args['carbon_pool_transient_uri'])
        # check that function can properly parse table of transient carbon pool
        # values.
        self.assertTrue(1 in biomass_transient_dict.keys())
        self.assertTrue(1 in soil_transient_dict.keys())

    def test_get_lulc_trans_to_D_dicts(self):
        """Coastal Blue Carbon: Read transient table (disturbed)."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc
        biomass_transient_dict, soil_transient_dict = \
            cbc._create_transient_dict(self.args['carbon_pool_transient_uri'])
        lulc_transition_uri = self.args['lulc_transition_matrix_uri']
        lulc_lookup_uri = self.args['lulc_lookup_uri']
        lulc_trans_to_Db, lulc_trans_to_Ds = cbc._get_lulc_trans_to_D_dicts(
            lulc_transition_uri,
            lulc_lookup_uri,
            biomass_transient_dict,
            soil_transient_dict)

        # check that function can properly parse table of transient carbon pool
        # values.
        self.assertTrue((3.0, 0.0) in lulc_trans_to_Db.keys())
        self.assertTrue((3.0, 0.0) in lulc_trans_to_Ds.keys())


class TestModel(unittest.TestCase):
    """Test Coastal Blue Carbon main model functions."""

    def setUp(self):
        """Create arguments."""
        self.workspace_dir = tempfile.mkdtemp()
        self.args = _get_args(workspace=self.workspace_dir)

    def tearDown(self):
        """Remove workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_model_run(self):
        """Coastal Blue Carbon: Test run function in main model."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        self.args['suffix'] = 'xyz'
        self.args['lulc_baseline_year'] = 2000
        self.args['lulc_transition_years_list'] = [2005, 2010]
        self.args['analysis_year'] = None

        cbc.execute(self.args)
        netseq_output_raster = os.path.join(
            self.args['workspace_dir'],
            'outputs_core/total_net_carbon_sequestration_test.tif')
        npv_output_raster = os.path.join(
            self.args['workspace_dir'],
            'outputs_core/net_present_value_at_2010_test.tif')
        netseq_array = _read_array(netseq_output_raster)
        npv_array = _read_array(npv_output_raster)

        # (Explanation for why netseq is 31.)
        # LULC Code: Baseline: 1 --> Year 2000: 1, Year 2005: 2,  Year 2010: 2
        # Initial Stock from Baseline: 5+5=10
        # Sequest:
        #    2000-->2005: (1+1.1)*5=10.5, 2005-->2010: (2+2.1)*5=20.5
        #       Total: 10.5 + 20.5 = 31.
        netseq_test = numpy.array([[cbc.NODATA_FLOAT, 31.], [31., 31.]])
        npv_test = numpy.array(
            [[cbc.NODATA_FLOAT, 60.27801514], [60.27801514, 60.27801514]])

        # just a simple regression test.  this demonstrates that a NaN value
        # will properly propagate across the model. the npv raster was chosen
        # because the values are determined by multiple inputs, and any changes
        # in those inputs would propagate to this raster.
        numpy.testing.assert_array_almost_equal(
            netseq_array, netseq_test, decimal=4)
        numpy.testing.assert_array_almost_equal(
            npv_array, npv_test, decimal=4)

    def test_model_run_2(self):
        """Coastal Blue Carbon: Test CBC without analysis year."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        self.args['analysis_year'] = None
        self.args['lulc_baseline_year'] = 2000
        self.args['lulc_transition_maps_list'] = [self.args['lulc_transition_maps_list'][0]]
        self.args['lulc_transition_years_list'] = [2005]

        cbc.execute(self.args)
        netseq_output_raster = os.path.join(
            self.args['workspace_dir'],
            'outputs_core/total_net_carbon_sequestration_test.tif')
        netseq_array = _read_array(netseq_output_raster)

        # (Explanation for why netseq is 10.5.)
        # LULC Code: Baseline: 1 --> Year 2000: 1, Year 2005: 2
        # Initial Stock from Baseline: 5+5=10
        # Sequest:
        #    2000-->2005: (1+1.1)*5=10.5
        netseq_test = numpy.array([[cbc.NODATA_FLOAT, 10.5], [10.5, 10.5]])

        # just a simple regression test.  this demonstrates that a NaN value
        # will properly propagate across the model. the npv raster was chosen
        # because the values are determined by multiple inputs, and any changes
        # in those inputs would propagate to this raster.
        numpy.testing.assert_array_almost_equal(
            netseq_array, netseq_test, decimal=4)

    def test_model_no_valuation(self):
        """Coastal Blue Carbon: Test main model without valuation."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        self.args = _get_args(valuation=False, workspace=self.workspace_dir)
        self.args['lulc_baseline_year'] = 2000
        self.args['lulc_transition_years_list'] = [2005, 2010]
        self.args['analysis_year'] = None

        cbc.execute(self.args)
        netseq_output_raster = os.path.join(
            self.args['workspace_dir'],
            'outputs_core/total_net_carbon_sequestration_test.tif')
        netseq_array = _read_array(netseq_output_raster)

        # (Explanation for why netseq is 31.)
        # LULC Code: Baseline: 1 --> Year 2000: 1, Year 2005: 2,  Year 2010: 2
        # Initial Stock from Baseline: 5+5=10
        # Sequest:
        #    2000-->2005: (1+1.1)*5=10.5, 2005-->2010: (2+2.1)*5=20.5
        #       Total: 10.5 + 20.5 = 31.
        netseq_test = numpy.array([[cbc.NODATA_FLOAT, 31.], [31., 31.]])

        # just a simple regression test.  this demonstrates that a NaN value
        # will properly propagate across the model. the npv raster was chosen
        # because the values are determined by multiple inputs, and any changes
        # in those inputs would propagate to this raster.
        numpy.testing.assert_array_almost_equal(
            netseq_array, netseq_test, decimal=4)

    def test_binary(self):
        """Coastal Blue Carbon: Test CBC model against InVEST-Data."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        args = {
            'workspace_dir': self.args['workspace_dir'],
            'carbon_pool_initial_uri': os.path.join(
                REGRESSION_DATA,
                'outputs_preprocessor/carbon_pool_initial_sample.csv'),
            'carbon_pool_transient_uri': os.path.join(
                REGRESSION_DATA,
                'outputs_preprocessor/carbon_pool_transient_sample.csv'),
            'discount_rate': 6.0,
            'do_economic_analysis': True,
            'do_price_table': True,
            'inflation_rate': 3.0,
            'lulc_lookup_uri': os.path.join(
                REGRESSION_DATA, 'inputs', 'lulc_lookup.csv'),
            'lulc_baseline_map_uri': os.path.join(
                REGRESSION_DATA, 'inputs/GBJC_2010_mean_Resample.tif'),
            'lulc_baseline_year': 2010,
            'lulc_transition_maps_list': [
                os.path.join(
                    REGRESSION_DATA, 'inputs/GBJC_2030_mean_Resample.tif'),
                os.path.join(
                    REGRESSION_DATA, 'inputs/GBJC_2050_mean_Resample.tif')],
            'lulc_transition_years_list': [2030, 2050],
            'price_table_uri': os.path.join(
                REGRESSION_DATA, 'inputs/Price_table_SCC3.csv'),
            'lulc_transition_matrix_uri': os.path.join(
                REGRESSION_DATA, 'outputs_preprocessor/transitions_sample.csv'),
            'price': 10.0,
            'results_suffix': '150225'
        }
        cbc.execute(args)
        npv_raster = os.path.join(
            os.path.join(
                args['workspace_dir'],
                'outputs_core/net_present_value_at_2050_150225.tif'))
        npv_array = _read_array(npv_raster)

        # this is just a regression test, but it will capture all values
        # in the net present value raster.  the npv raster was chosen because
        # the values are determined by multiple inputs, and any changes in
        # those inputs would propagate to this raster.
        u = numpy.unique(npv_array)
        u.sort()
        a = numpy.array([-76992.05, -40101.57, -34930., -34821.32,
                         0., 108.68, 6975.94, 7201.22, 7384.99],
                        dtype=numpy.float32)

        a.sort()
        numpy.testing.assert_array_almost_equal(u, a, decimal=2)

        # walk through all files in the workspace and assert that outputs have
        # the file suffix.
        non_suffixed_files = []
        for root_dir, dirnames, filenames in os.walk(self.args['workspace_dir']):
            for filename in filenames:
                if not filename.lower().endswith('.txt'):  # ignore logfile
                    basename, extension = os.path.splitext(filename)
                    if not basename.endswith('_150225'):
                        path_rel_to_workspace = os.path.relpath(
                            os.path.join(root_dir, filename),
                            self.args['workspace_dir'])
                        non_suffixed_files.append(path_rel_to_workspace)

        if non_suffixed_files:
            self.fail('%s files are missing suffixes: %s' %
                      (len(non_suffixed_files),
                       pprint.pformat(non_suffixed_files)))

    def test_1_transition_passes(self):
        """Coastal Blue Carbon: Test model runs with only 1 transition.

        This is a regression test addressing issue #3572
        (see: https://bitbucket.org/natcap/invest/issues/3572)
        """
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        self.args['lulc_transition_maps_list'] = \
            [self.args['lulc_transition_maps_list'][0]]
        self.args['lulc_transition_years_list'] = \
            [self.args['lulc_transition_years_list'][0]]
        self.args['analysis_year'] = None
        try:
            cbc.execute(self.args)
        except AttributeError as error:
            LOGGER.exception("Here's the traceback encountered: %s" % error)
            self.fail('CBC should not crash when only 1 transition provided')


class CBCRefactorTest(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def create_args(workspace, transition_tuples=None, analysis_year=None):
        """Create a default args dict with the given transition matrices.

        Arguments:
            workspace (string): The path to the workspace directory on disk.
                Files will be saved to this location.
            transition_tuples (list or None): A list of tuples, where the first
                element of the tuple is a numpy matrix of the transition values,
                and the second element of the tuple is the year of the transition.
                Provided years must be in chronological order.
                If ``None``, the transition parameters will be ignored.
            analysis_year (int or None): The year of the final analysis.  If
                provided, it must be greater than the last year within the
                transition tuples (unless ``transition_tuples`` is None, in which
                case ``analysis_year`` can be anything greater than 2000, the
                baseline year).

        Returns:
            A dict of the model arguments.
        """
        from pygeoprocessing.testing import sampledata

        args = {
            'workspace_dir': workspace,
            'lulc_lookup_uri': os.path.join(workspace, 'lulc_lookup.csv'),
            'lulc_transition_matrix_uri': os.path.join(workspace,
                                                       'transition_matrix.csv'),
            'carbon_pool_initial_uri': os.path.join(workspace,
                                                    'carbon_pool_initial.csv'),
            'carbon_pool_transient_uri': os.path.join(workspace,
                                                      'carbon_pool_transient.csv'),
            'lulc_baseline_map_uri': os.path.join(workspace, 'lulc.tif'),
            'lulc_baseline_year': 2000,
            'do_economic_analysis': False,
        }
        _create_table(args['lulc_lookup_uri'], lulc_lookup_list)
        _create_table(
            args['lulc_transition_matrix_uri'],
            lulc_transition_matrix_list)
        _create_table(
            args['carbon_pool_initial_uri'],
            carbon_pool_initial_list)
        _create_table(
            args['carbon_pool_transient_uri'],
            carbon_pool_transient_list)

        # Only parameters needed are band_matrices and filename
        make_raster = functools.partial(
            sampledata.create_raster_on_disk,
            origin=sampledata.SRS_WILLAMETTE.origin,
            projection_wkt=sampledata.SRS_WILLAMETTE.projection,
            nodata=-1, pixel_size=sampledata.SRS_WILLAMETTE.pixel_size(100))

        known_matrix_size = None
        if transition_tuples:
            args['lulc_transition_maps_list'] = []
            args['lulc_transition_years_list'] = []

            for band_matrix, transition_year in transition_tuples:
                known_matrix_size = band_matrix.shape
                filename = os.path.join(workspace,
                                        'transition_%s.tif' % transition_year)
                make_raster(band_matrices=[band_matrix], filename=filename)

                args['lulc_transition_maps_list'].append(filename)
                args['lulc_transition_years_list'].append(transition_year)

        # Make the lulc
        lulc_shape = (10, 10) if not known_matrix_size else known_matrix_size
        make_raster(band_matrices=[numpy.ones(lulc_shape)],
                    filename=args['lulc_baseline_map_uri'])

        if analysis_year:
            args['analysis_year'] = analysis_year

        return args

    def test_no_transitions(self):
        """Coastal Blue Carbon: Verify model can run without transitions."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        args = CBCRefactorTest.create_args(
            workspace=self.workspace_dir, transition_tuples=None,
            analysis_year=None)

        cbc.execute(args)

    def test_no_transitions_with_analysis_year(self):
        """Coastal Blue Carbon: Model can run w/o trans., w/analysis yr."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        args = CBCRefactorTest.create_args(
            workspace=self.workspace_dir, transition_tuples=None,
            analysis_year=2010)

        cbc.execute(args)

    def test_one_transition(self):
        """Coastal Blue Carbon: Verify model can run with 1 transition."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        transition_tuples = [
            (numpy.ones((10, 10)), 2010),
        ]

        args = CBCRefactorTest.create_args(
            workspace=self.workspace_dir,
            transition_tuples=transition_tuples,
            analysis_year=None)

        cbc.execute(args)

    def test_transient_dict_extraction(self):
        """Coastal Blue Carbon: Verify extraction of transient dictionary."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        transient_file = _create_table(
            os.path.join(self.workspace_dir, 'transient.csv'),
            carbon_pool_transient_list[:3])

        biomass_dict, soil_dict = cbc._create_transient_dict(transient_file)

        expected_biomass_dict = {
            0: {
                'lulc-class': 'n',
                'half-life': 0.0,
                'med-impact-disturb': 0.0,
                'yearly-accumulation': 0.0,
            },
            1: {
                'lulc-class': 'x',
                'half-life': 1,
                'med-impact-disturb': 0.5,
                'yearly-accumulation': 1,
            }
        }

        expected_soil_dict = copy.deepcopy(expected_biomass_dict)
        expected_soil_dict[1]['yearly-accumulation'] = 1.1

        self.assertEqual(biomass_dict, expected_biomass_dict)
        self.assertEqual(soil_dict, expected_soil_dict)

    def test_reclass_invalid_nodata(self):
        """Coastal Blue Carbon: verify handling of incorrect nodata values."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        # In this case, the nodata provided (-1) cannot be represented by
        # numpy.uint16 datatype.
        lulc_nodata = -1
        lulc_matrix = numpy.array([
            [1, 2, 3],
            [1, 2, 3],
            [-1, -1, -1]], numpy.uint16)

        reclass_map = {
            1: 1.1,
            2: 2.2,
            3: 3.3
        }

        expected_array = numpy.array([
            [1.1, 2.2, 3.3],
            [1.1, 2.2, 3.3],
            [numpy.nan, numpy.nan, numpy.nan]], numpy.float32)

        reclassified_array = cbc.reclass(
            lulc_matrix, reclass_map, out_dtype=numpy.float32,
            nodata_mask=lulc_nodata)

        numpy.testing.assert_almost_equal(reclassified_array, expected_array)


class CBCValidationTests(unittest.TestCase):
    """Tests for Coastal Blue Carbon Model ARGS_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'lulc_lookup_uri',
            'lulc_transition_matrix_uri',
            'carbon_pool_initial_uri',
            'carbon_pool_transient_uri',
            'lulc_baseline_map_uri',
            'lulc_baseline_year',
        ]

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys(self):
        """CBC Validate: assert missing required keys."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        from natcap.invest import validation

        validation_errors = coastal_blue_carbon.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_do_valuation(self):
        """CBC Validate: assert missing required for valuation."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        from natcap.invest import validation

        validation_errors = coastal_blue_carbon.validate(
            {'do_economic_analysis': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['price',
             'inflation_rate',
             'discount_rate'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_do_valuation_table(self):
        """CBC Validate: assert missing required for valuation with table."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        from natcap.invest import validation

        validation_errors = coastal_blue_carbon.validate(
            {'do_economic_analysis': True,
             'do_price_table': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['discount_rate',
             'price_table_uri'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_transition_years(self):
        """CBC Validate: assert maps required if years given."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        from natcap.invest import validation

        validation_errors = coastal_blue_carbon.validate(
            {'lulc_transition_years_list': [1999]})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['lulc_transition_maps_list'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_transition_maps(self):
        """CBC Validate: assert years required if maps given."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        from natcap.invest import validation

        validation_errors = coastal_blue_carbon.validate(
            {'lulc_transition_maps_list': ['foo.tif']})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['lulc_transition_years_list'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_transitions_map_year_mismatch(self):
        """CBC Validate: assert transition maps and years are equal length."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        from natcap.invest import validation

        validation_warnings = coastal_blue_carbon.validate(
            {'lulc_transition_years_list': [1999],
             'lulc_transition_maps_list': ['foo.tif', 'bar.tif']})
        invalid_keys = validation.get_invalid_keys(validation_warnings)
        expected_missing_keys = set(
            self.base_required_keys +
            ['lulc_transition_maps_list',
             'lulc_transition_years_list'])
        self.assertEqual(invalid_keys, expected_missing_keys)
        expected_message = 'Must have the same number of elements.'
        actual_messages = set()
        for keys, error_strings in validation_warnings:
            actual_messages.add(error_strings)
        self.assertTrue(expected_message in actual_messages)
