# -*- coding: utf-8 -*-
"""Tests for Coastal Blue Carbon Functions."""
import unittest
import os
import shutil
import csv
import logging
import tempfile
import copy
import pprint

import numpy
from osgeo import gdal, osr
import pygeoprocessing
from natcap.invest import utils
import scipy.sparse

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


def _create_workspace():
    """Create workspace directory."""
    return tempfile.mkdtemp()


def _get_args(workspace, num_transitions=2, valuation=True):
    """Create and return arguments for CBC main model.

    Args:
        workspace(string): A path to a folder on disk.  Generated inputs will
            be saved to this directory.
        num_transitions=2 (int): The number of transitions to synthesize.
        valuation=True (bool): Whether to include parameters related to
            valuation in the args dict.

    Returns:
        args (dict): main model arguments.
    """
    band_matrices = numpy.ones((2, 2))
    band_matrices_two = numpy.ones((2, 2)) * 2
    band_matrices_with_nodata = numpy.ones((2, 2))
    band_matrices_with_nodata[0][0] = NODATA_INT

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3157)
    projection_wkt = srs.ExportToWkt()
    origin = (443723.127327877911739, 4956546.905980412848294)

    lulc_lookup_path = _create_table(
        os.path.join(workspace, 'lulc_lookup.csv'), lulc_lookup_list)
    lulc_transition_matrix_path = _create_table(
        os.path.join(workspace, 'lulc_transition_matrix.csv'),
        lulc_transition_matrix_list)
    carbon_pool_initial_path = _create_table(
        os.path.join(workspace, 'carbon_pool_initial.csv'),
        carbon_pool_initial_list)
    carbon_pool_transient_path = _create_table(
        os.path.join(workspace, 'carbon_pool_transient.csv'),
        carbon_pool_transient_list)
    raster_0_path = os.path.join(workspace, 'raster_0.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices, NODATA_INT, (100, -100), origin, projection_wkt,
        raster_0_path)
    raster_1_path = os.path.join(workspace, 'raster_1.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices_with_nodata, NODATA_INT, (100, -100), origin,
        projection_wkt, raster_1_path)
    raster_2_path = os.path.join(workspace, 'raster_2.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices_two, NODATA_INT, (100, -100), origin,
        projection_wkt, raster_2_path)

    possible_transitions = [raster_1_path, raster_2_path]
    possible_transition_years = [2000, 2005]

    args = {
        'workspace_dir': os.path.join(workspace, 'workspace'),
        'results_suffix': 'test',
        'lulc_lookup_uri': lulc_lookup_path,
        'lulc_transition_matrix_uri': lulc_transition_matrix_path,
        'lulc_baseline_map_uri': raster_0_path,
        'lulc_baseline_year': 1995,
        'lulc_transition_maps_list': possible_transitions[:num_transitions+1],
        'lulc_transition_years_list': possible_transition_years[
                                                        :num_transitions+1],
        'analysis_year': 2010,
        'carbon_pool_initial_uri': carbon_pool_initial_path,
        'carbon_pool_transient_uri': carbon_pool_transient_path,
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
    band_matrices_zeros = numpy.zeros((2, 2))
    band_matrices_ones = numpy.ones((2, 3))  # tests alignment
    band_matrices_nodata = numpy.ones((2, 2)) * NODATA_INT

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3157)
    projection_wkt = srs.ExportToWkt()
    origin = (443723.127327877911739, 4956546.905980412848294)

    lulc_lookup_path = _create_table(
        os.path.join(workspace, 'lulc_lookup.csv'), lulc_lookup_list)

    raster_0_path = os.path.join(workspace, 'raster_0.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices_ones, NODATA_INT, (100, -100), origin, projection_wkt,
        raster_0_path)
    raster_1_path = os.path.join(workspace, 'raster_1.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices_ones, NODATA_INT, (100, -100), origin, projection_wkt,
        raster_1_path)
    raster_2_path = os.path.join(workspace, 'raster_2.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices_ones, NODATA_INT, (100, -100), origin, projection_wkt,
        raster_2_path)
    raster_3_path = os.path.join(workspace, 'raster_3.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices_zeros, NODATA_INT, (100, -100), origin, projection_wkt,
        raster_3_path)
    raster_4_path = os.path.join(workspace, 'raster_4.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices_zeros, NODATA_INT, (100, -100), origin, projection_wkt,
        raster_4_path)
    raster_nodata_path = os.path.join(workspace, 'raster_4.tif')
    pygeoprocessing.numpy_array_to_raster(
        band_matrices_nodata, NODATA_INT, (100, -100), origin, projection_wkt,
        raster_nodata_path)

    snapshot_csv_path = os.path.join(workspace, 'snapshot_csv.csv')
    def _write_snapshot_csv(raster_path_list):
        with open(snapshot_csv_path, 'w') as snapshot_csv:
            snapshot_csv.write('snapshot_year,raster_path\n')

            # The actual years in the snapshots CSV don't matter except that
            # they are numeric and unique.  Hardcoding them here should be
            # fine.
            for year, raster_path in enumerate(raster_path_list, start=2000):
                snapshot_csv.write(f'{year},{raster_path}\n')

    workspace_path = os.path.join(workspace, 'workspace')
    args = {
        'workspace_dir': workspace_path,
        'results_suffix': 'test',
        'lulc_lookup_table_path': lulc_lookup_path,
        'landcover_snapshot_csv': snapshot_csv_path,
    }
    if args_choice == 1:
        _write_snapshot_csv([raster_0_path, raster_1_path, raster_2_path])
    elif args_choice == 2:
        _write_snapshot_csv([raster_0_path, raster_1_path, raster_3_path])
    elif args_choice == 3:
        _write_snapshot_csv([raster_0_path, raster_nodata_path, raster_3_path])
    elif args_choice == 4:
        _write_snapshot_csv([raster_0_path, raster_nodata_path, raster_4_path])
    else:
        raise ValueError("Invalid args_choice value")

    return args


class TestPreprocessor(unittest.TestCase):
    """Test Coastal Blue Carbon preprocessor library functions."""

    def setUp(self):
        """Create a temp directory for the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_sample_data(self):
        """CBC Preprocessor: Test on sample data."""
        from natcap.invest.coastal_blue_carbon import preprocessor

        snapshot_csv_path = os.path.join(REGRESSION_DATA, 'inputs', 'snapshots.csv')

        args = {
            'workspace_dir': _create_workspace(),
            'results_suffix': '150225',
            'lulc_lookup_table_path': os.path.join(
                REGRESSION_DATA, 'inputs', 'lulc_lookup.csv'),
            'landcover_snapshot_csv': snapshot_csv_path,
        }
        preprocessor.execute(args)

        # walk through all files in the workspace and assert that outputs have
        # the file suffix.
        non_suffixed_files = []
        outputs_dir = os.path.join(
            args['workspace_dir'], 'outputs_preprocessor')
        for root_dir, dirnames, filenames in os.walk(outputs_dir):
            for filename in filenames:
                if not filename.lower().endswith('.txt'):  # ignore logfile
                    basename, extension = os.path.splitext(filename)
                    if not basename.endswith('_150225'):
                        path_rel_to_workspace = os.path.relpath(
                            os.path.join(root_dir, filename),
                            args['workspace_dir'])
                        non_suffixed_files.append(path_rel_to_workspace)

        if non_suffixed_files:
            self.fail('%s files are missing suffixes: %s' %
                      (len(non_suffixed_files),
                       pprint.pformat(non_suffixed_files)))

        expected_landcover_codes = set(range(0, 24))
        found_landcover_codes = set(utils.build_lookup_from_csv(
            os.path.join(outputs_dir,
                         'carbon_biophysical_table_template_150225.csv'),
            'code').keys())
        self.assertEqual(expected_landcover_codes, found_landcover_codes)

    def test_transition_table(self):
        """CBC Preprocessor: Test creation of transition table."""
        from natcap.invest.coastal_blue_carbon import preprocessor

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        matrix_a = numpy.array([
            [0, 1],
            [0, 1],
            [0, 1]], dtype=numpy.int16)
        filename_a = os.path.join(self.workspace_dir, 'raster_a.tif')
        snapshot_a = pygeoprocessing.numpy_array_to_raster(
            matrix_a, -1, (100, -100), origin, projection_wkt, filename_a)

        matrix_b = numpy.array([
            [0, 1],
            [1, 0],
            [-1, -1]], dtype=numpy.int16)
        filename_b = os.path.join(self.workspace_dir, 'raster_b.tif')
        snapshot_b = pygeoprocessing.numpy_array_to_raster(
            matrix_b, -1, (100, -100), origin, projection_wkt, filename_b)

        landcover_table_path = os.path.join(self.workspace_dir,
                                            'lulc_table.csv')
        with open(landcover_table_path, 'w') as lulc_csv:
            lulc_csv.write('code,lulc-class,is_coastal_blue_carbon_habitat\n')
            lulc_csv.write('0,mangrove,True\n')
            lulc_csv.write('1,parking lot,False\n')

        landcover_table = utils.build_lookup_from_csv(
            landcover_table_path, 'code')
        target_table_path = os.path.join(self.workspace_dir,
                                         'transition_table.csv')

        # Remove landcover code 1 from the table; expect error.
        del landcover_table[1]
        with self.assertRaises(ValueError) as context:
            preprocessor._create_transition_table(
                landcover_table, [filename_a, filename_b], target_table_path)

        self.assertIn('missing a row with the landuse code 1',
                      str(context.exception))

        # Re-load the landcover table
        landcover_table = utils.build_lookup_from_csv(
            landcover_table_path, 'code')
        preprocessor._create_transition_table(
            landcover_table, [filename_a, filename_b], target_table_path)

        with open(target_table_path) as transition_table:
            self.assertEqual(
                transition_table.readline(),
                'lulc-class,mangrove,parking lot\n')
            self.assertEqual(
                transition_table.readline(),
                'mangrove,accum,disturb\n')
            self.assertEqual(
                transition_table.readline(),
                'parking lot,accum,NCC\n')

            # After the above lines is a blank line, then the legend.
            # Deliberately not testing the legend.
            self.assertEqual(transition_table.readline(), '\n')

class TestIO(unittest.TestCase):
    """Test Coastal Blue Carbon io library functions."""

    def setUp(self):
        """Create arguments."""
        self.workspace_dir = tempfile.mkdtemp()
        self.args = _get_args(self.workspace_dir)

    def tearDown(self):
        """Clean up workspace when finished."""
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
        lulc_transition_path = self.args['lulc_transition_matrix_uri']
        lulc_lookup_path = self.args['lulc_lookup_uri']
        lulc_trans_to_Db, lulc_trans_to_Ds = cbc._get_lulc_trans_to_D_dicts(
            lulc_transition_path,
            lulc_lookup_path,
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
        numpy.testing.assert_allclose(
            netseq_array, netseq_test, rtol=0, atol=1e-4)
        numpy.testing.assert_allclose(
            npv_array, npv_test, rtol=0, atol=1e-4)

    def test_model_run_2(self):
        """Coastal Blue Carbon: Test CBC without analysis year."""
        from natcap.invest.coastal_blue_carbon \
            import coastal_blue_carbon as cbc

        self.args['analysis_year'] = None
        self.args['lulc_baseline_year'] = 2000
        self.args['lulc_transition_maps_list'] = [
            self.args['lulc_transition_maps_list'][0]]
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
        numpy.testing.assert_allclose(
            netseq_array, netseq_test, rtol=0, atol=1e-4)

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
        numpy.testing.assert_allclose(
            netseq_array, netseq_test, rtol=0, atol=1e-4)

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
                REGRESSION_DATA,
                'outputs_preprocessor/transitions_sample.csv'),
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
        numpy.testing.assert_allclose(u, a, rtol=0, atol=1e-2)

        # walk through all files in the workspace and assert that outputs have
        # the file suffix.
        non_suffixed_files = []
        for root_dir, dirnames, filenames in os.walk(
                self.args['workspace_dir']):
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
    """CBC Refactor Tests."""
    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary workspace when done."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def create_args(workspace, transition_tuples=None, analysis_year=None):
        """Create a default args dict with the given transition matrices.

        Arguments:
            workspace (string): The path to the workspace directory on disk.
                Files will be saved to this location.
            transition_tuples (list or None): A list of tuples, where the first
                element of the tuple is a numpy matrix of the transition
                values, and the second element of the tuple is the year of the
                transition. Provided years must be in chronological order.
                If ``None``, the transition parameters will be ignored.
            analysis_year (int or None): The year of the final analysis.  If
                provided, it must be greater than the last year within the
                transition tuples (unless ``transition_tuples`` is None, in
                which case ``analysis_year`` can be anything greater than 2000,
                the baseline year).

        Returns:
            A dict of the model arguments.
        """
        import pygeoprocessing

        args = {
            'workspace_dir': workspace,
            'lulc_lookup_uri': os.path.join(workspace, 'lulc_lookup.csv'),
            'lulc_transition_matrix_uri': os.path.join(
                workspace, 'transition_matrix.csv'),
            'carbon_pool_initial_uri': os.path.join(
                workspace, 'carbon_pool_initial.csv'),
            'carbon_pool_transient_uri': os.path.join(
                workspace, 'carbon_pool_transient.csv'),
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

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)

        known_matrix_size = None
        if transition_tuples:
            args['lulc_transition_maps_list'] = []
            args['lulc_transition_years_list'] = []

            for band_matrix, transition_year in transition_tuples:
                known_matrix_size = band_matrix.shape
                filename = os.path.join(
                    workspace, 'transition_%s.tif' % transition_year)
                pygeoprocessing.numpy_array_to_raster(
                    band_matrix, -1, (100, -100), origin, projection_wkt,
                    filename)

                args['lulc_transition_maps_list'].append(filename)
                args['lulc_transition_years_list'].append(transition_year)

        # Make the lulc
        lulc_shape = (10, 10) if not known_matrix_size else known_matrix_size
        pygeoprocessing.numpy_array_to_raster(
            numpy.ones(lulc_shape), -1, (100, -100), origin, projection_wkt,
            args['lulc_baseline_map_uri'])

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

        numpy.testing.assert_allclose(reclassified_array, expected_array, rtol=0, atol=1e-6)


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

        # empty args dict.
        validation_errors = coastal_blue_carbon.validate({})
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


def make_raster_from_array(
        base_array, base_raster_path, nodata_val=-1, gdal_type=gdal.GDT_Int32):
    """Make a raster from an array on a designated path.

    Args:
        base_array (numpy.ndarray): the 2D array for making the raster.
        nodata_val (int; float): nodata value for the raster.
        gdal_type (gdal datatype; int): gdal datatype for the raster.
        base_raster_path (str): the path for the raster to be created.

    Returns:
        None.
    """
    # Projection to user for generated sample data UTM Zone 10N
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    project_wkt = srs.ExportToWkt()
    origin = (1180000, 690000)

    pygeoprocessing.numpy_array_to_raster(
        base_array, nodata_val, (1, -1), origin, project_wkt, base_raster_path)


from natcap.invest.coastal_blue_carbon import coastal_blue_carbon2


class TestCBC2(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_extract_shapshots(self):
        csv_path = os.path.join(self.workspace_dir, 'snapshots.csv')

        transition_years = (2000, 2010, 2020)
        transition_rasters = []
        with open(csv_path, 'w') as transitions_csv:
            # Check that we can interpret varying case.
            transitions_csv.write('snapshot_YEAR,raster_PATH\n')
            for transition_year in transition_years:
                # Write absolute paths.
                transition_file_path = os.path.join(
                    self.workspace_dir, f'{transition_year}.tif)')
                transition_rasters.append(transition_file_path)
                transitions_csv.write(
                    f'{transition_year},{transition_file_path}\n')

            # Make one path relative to the workspace, where the transitions
            # CSV also lives.
            # The expected raster path is absolute.
            transitions_csv.write(f'2030,some_path.tif\n')
            transition_years += (2030,)
            transition_rasters.append(os.path.join(self.workspace_dir,
                                                   'some_path.tif'))

        extracted_transitions = (
            coastal_blue_carbon2._extract_snapshots_from_table(csv_path))

        self.assertEqual(
            extracted_transitions,
            dict(zip(transition_years, transition_rasters)))

    def test_track_latest_transition_year(self):
        """CBC: Track the latest disturbance year."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31 S
        wkt = srs.ExportToWkt()

        current_disturbance_vol_raster = os.path.join(
            self.workspace_dir, 'cur_disturbance.tif')
        current_disturbance_vol_matrix = numpy.array([
            [5.0, 1.0],
            [-1, 3.0]], dtype=numpy.float32)
        pygeoprocessing.numpy_array_to_raster(
            current_disturbance_vol_matrix, -1, (2, -2), (2, -2), wkt,
            current_disturbance_vol_raster)

        known_transition_years_raster = os.path.join(
            self.workspace_dir, 'known_transition_years.tif')
        known_transition_years_matrix = numpy.array([
            [100, 100],
            [5, 6]], dtype=numpy.uint16)
        pygeoprocessing.numpy_array_to_raster(
            known_transition_years_matrix, 100, (2, -2), (2, -2), wkt,
            known_transition_years_raster)

        target_raster_path = os.path.join(
            self.workspace_dir, 'new_tracked_years.tif')
        latest_disturbance_year_matrix = (
            coastal_blue_carbon2._track_latest_transition_year(
                current_disturbance_vol_raster,
                known_transition_years_raster,
                11,  # current "year" being disturbed.
                target_raster_path))

        expected_array = numpy.array([
            [11, 11],
            [5, 11]], dtype=numpy.uint16)
        numpy.testing.assert_allclose(
            gdal.OpenEx(target_raster_path, gdal.OF_RASTER).ReadAsArray(),
            expected_array)

    def test_read_transition_matrix(self):
        """CBC: Test transition matrix reading."""
        # The full biophysical table will have much, much more information.  To
        # keep the test simple, I'm only tracking the columns I know I'll need
        # in this function.
        biophysical_table = {
            1: {'lulc-class': 'a',
                'soil-yearly-accumulation': 2,
                'biomass-yearly-accumulation': 3,
                'soil-high-impact-disturb': 4,
                'biomass-high-impact-disturb': 5,
            },
            2: {'lulc-class': 'b',
                'soil-yearly-accumulation': 6,
                'biomass-yearly-accumulation': 7,
                'soil-high-impact-disturb': 8,
                'biomass-high-impact-disturb': 9,
            },
            3: {'lulc-class': 'c',
                'soil-yearly-accumulation': 10,
                'biomass-yearly-accumulation': 11,
                'soil-high-impact-disturb': 12,
                'biomass-high-impact-disturb': 13,
            }
        }

        transition_csv_path = os.path.join(self.workspace_dir,
                                           'transitions.csv')
        with open(transition_csv_path, 'w') as transition_csv:
            transition_csv.write('lulc-class,a,b,c\n')
            transition_csv.write('a,NCC,accum,high-impact-disturb\n')
            transition_csv.write('b,,NCC,accum\n')
            transition_csv.write('c,accum,,NCC')

        (biomass_disturbance_matrix, soil_disturbance_matrix,
         biomass_accumulation_matrix, soil_accumulation_matrix) = (
             coastal_blue_carbon2._read_transition_matrix(
                 transition_csv_path, biophysical_table))

        expected_biomass_disturbance = numpy.zeros((4,4), dtype=numpy.float32)
        expected_biomass_disturbance[1, 3] = (
            biophysical_table[1]['biomass-high-impact-disturb'])
        numpy.testing.assert_allclose(
            expected_biomass_disturbance,
            biomass_disturbance_matrix.toarray())

        expected_soil_disturbance = numpy.zeros((4,4), dtype=numpy.float32)
        expected_soil_disturbance[1, 3] = (
            biophysical_table[1]['soil-high-impact-disturb'])
        numpy.testing.assert_allclose(
            expected_soil_disturbance,
            soil_disturbance_matrix.toarray())

        expected_biomass_accumulation = numpy.zeros((4,4), dtype=numpy.float32)
        expected_biomass_accumulation[3, 1] = (
            biophysical_table[1]['biomass-yearly-accumulation'])
        expected_biomass_accumulation[1, 2] = (
            biophysical_table[2]['biomass-yearly-accumulation'])
        expected_biomass_accumulation[2, 3] = (
            biophysical_table[3]['biomass-yearly-accumulation'])
        numpy.testing.assert_allclose(
            expected_biomass_accumulation,
            biomass_accumulation_matrix.toarray())

        expected_soil_accumulation = numpy.zeros((4,4), dtype=numpy.float32)
        expected_soil_accumulation[3, 1] = (
            biophysical_table[1]['soil-yearly-accumulation'])
        expected_soil_accumulation[1, 2] = (
            biophysical_table[2]['soil-yearly-accumulation'])
        expected_soil_accumulation[2, 3] = (
            biophysical_table[3]['soil-yearly-accumulation'])
        numpy.testing.assert_allclose(
            expected_soil_accumulation,
            soil_accumulation_matrix.toarray())

    def test_emissions(self):
        """CBC: Check emissions calculations."""
        volume_disturbed_carbon = numpy.array(
            [[5.5, coastal_blue_carbon2.NODATA_FLOAT32]], dtype=numpy.float32)
        year_last_disturbed = numpy.array(
            [[10, coastal_blue_carbon2.NODATA_UINT16]], dtype=numpy.uint16)
        half_life = numpy.array([[7.5, 7.5]], dtype=numpy.float32)
        current_year = 15

        result_matrix = coastal_blue_carbon2._calculate_emissions(
            volume_disturbed_carbon, year_last_disturbed, half_life,
            current_year)

        # Calculated by hand.
        expected_array = numpy.array([
            [0.3058625, coastal_blue_carbon2.NODATA_FLOAT32]],
            dtype=numpy.float32)
        numpy.testing.assert_allclose(
            result_matrix, expected_array, rtol=1E-6)

    def test_add_rasters(self):
        """CBC: Check that we can add two rasters."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31 S
        wkt = srs.ExportToWkt()

        raster_a_path = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.array([[5, 15, 12]], dtype=numpy.uint8),
            15, (2, -2), (2, -2), wkt, raster_a_path)

        raster_b_path = os.path.join(self.workspace_dir, 'b.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.array([[3, 4, 5]], dtype=numpy.uint8),
            5, (2, -2), (2, -2), wkt, raster_b_path)

        target_path = os.path.join(self.workspace_dir, 'output.tif')
        coastal_blue_carbon2._sum_n_rasters(
            [raster_a_path, raster_b_path], target_path)

        nodata = coastal_blue_carbon2.NODATA_FLOAT32
        numpy.testing.assert_allclose(
            gdal.OpenEx(target_path).ReadAsArray(),
            numpy.array([[8, nodata, nodata]], dtype=numpy.float32))

    @staticmethod
    def _create_model_args(target_dir):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31 S
        wkt = srs.ExportToWkt()

        biophysical_table = [
            ['code', 'lulc-class', 'biomass-initial', 'soil-initial',
                'litter-initial', 'biomass-half-life',
                'biomass-low-impact-disturb', 'biomass-med-impact-disturb',
                'biomass-high-impact-disturb', 'biomass-yearly-accumulation',
                'soil-half-life', 'soil-low-impact-disturb',
                'soil-med-impact-disturb', 'soil-high-impact-disturb',
                'soil-yearly-accumulation', 'litter-yearly-accumulation'],
            [1, 'mangrove',
                64, 313, 3,  # initial
                15, 0.5, 0.5, 1, 2,  # biomass
                7.5, 0.3, 0.5, 0.66, 5.35,  # soil
                1],  # litter accum.
            [2, 'parking lot',
                0, 0, 0,  # initial
                0, 0, 0, 0, 0,  # biomass
                0, 0, 0, 0, 0,  # soil
                0],  # litter accum.
        ]
        biophysical_table_path = os.path.join(
            target_dir, 'biophysical.csv')
        with open(biophysical_table_path, 'w') as bio_table:
            for line_list in biophysical_table:
                line = ','.join(str(field) for field in line_list)
                bio_table.write(f'{line}\n')

        transition_matrix = [
            ['lulc-class', 'mangrove', 'parking lot'],
            ['mangrove', 'NCC', 'high-impact-disturb'],
            ['parking lot', 'accum', 'NCC']
        ]
        transition_matrix_path = os.path.join(
            target_dir, 'transitions.csv')
        with open(transition_matrix_path, 'w') as transition_table:
            for line_list in transition_matrix:
                line = ','.join(line_list)
                transition_table.write(f'{line}\n')

        baseline_landcover_raster_path = os.path.join(
            target_dir, 'baseline_lulc.tif')
        baseline_matrix = numpy.array([[1, 2]], dtype=numpy.uint8)
        pygeoprocessing.numpy_array_to_raster(
            baseline_matrix, 255, (2, -2), (2, -2), wkt,
            baseline_landcover_raster_path)

        snapshot_2010_raster_path = os.path.join(
            target_dir, 'snapshot_2010.tif')
        snapshot_2010_matrix = numpy.array([[2, 1]], dtype=numpy.uint8)
        pygeoprocessing.numpy_array_to_raster(
            snapshot_2010_matrix, 255, (2, -2), (2, -2), wkt,
            snapshot_2010_raster_path)

        snapshot_2020_raster_path = os.path.join(
            target_dir, 'snapshot_2020.tif')
        snapshot_2020_matrix = numpy.array([[1, 2]], dtype=numpy.uint8)
        pygeoprocessing.numpy_array_to_raster(
            snapshot_2020_matrix, 255, (2, -2), (2, -2), wkt,
            snapshot_2020_raster_path)

        snapshot_rasters_csv_path = os.path.join(
            target_dir, 'snapshot_rasters.csv')
        baseline_year = 2000
        with open(snapshot_rasters_csv_path, 'w') as snapshot_rasters_csv:
            snapshot_rasters_csv.write('snapshot_year,raster_path\n')
            snapshot_rasters_csv.write(
                f'{baseline_year},{baseline_landcover_raster_path}\n')
            snapshot_rasters_csv.write(
                f'2010,{snapshot_2010_raster_path}\n')
            snapshot_rasters_csv.write(
                f'2020,{snapshot_2020_raster_path}\n')

        args = {
            'landcover_transitions_table': transition_matrix_path,
            'landcover_snapshot_csv': snapshot_rasters_csv_path,
            'biophysical_table_path': biophysical_table_path,
            'analysis_year': 2030,
            'do_economic_analysis': True,
            'use_price_table': True,
            'price_table_path': os.path.join(target_dir,
                                             'price_table.csv'),
            'discount_rate': 4,
        }

        with open(args['price_table_path'], 'w') as price_table:
            price_table.write('year,price\n')
            prior_year_price = 1.0
            for year in range(baseline_year,
                              args['analysis_year']+1):
                price = prior_year_price * 1.04
                price_table.write(f'{year},{price}\n')
        return args

    def test_model(self):
        """CBC: Test the model's execution."""
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = os.path.join(self.workspace_dir, 'workspace')

        coastal_blue_carbon2.execute(args)

        # Sample values calculated by hand.  Pixel 0 only accumulates.  Pixel 1
        # has no accumulation (per the biophysical table) and also has no
        # emissions.
        expected_sequestration_2000_to_2010 = numpy.array(
            [[83.5, 0]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2000-and-2010.tif'))
        numpy.testing.assert_allclose(
            (gdal.OpenEx(raster_path)).ReadAsArray(),
            expected_sequestration_2000_to_2010)

        expected_sequestration_2010_to_2020 = numpy.array(
            [[-176.9792, 73.5]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2010-and-2020.tif'))
        numpy.testing.assert_allclose(
            gdal.OpenEx(raster_path).ReadAsArray(),
            expected_sequestration_2010_to_2020, rtol=1e-6)

        expected_sequestration_2020_to_2030 = numpy.array(
            [[73.5, -25.828205]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2020-and-2030.tif'))
        numpy.testing.assert_allclose(
            gdal.OpenEx(raster_path).ReadAsArray(),
            expected_sequestration_2020_to_2030, rtol=1e-6)

        # Total sequestration is the sum of all the previous sequestration.
        expected_total_sequestration = (
            expected_sequestration_2000_to_2010 +
            expected_sequestration_2010_to_2020 +
            expected_sequestration_2020_to_2030)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            'total-net-carbon-sequestration.tif')
        numpy.testing.assert_allclose(
            gdal.OpenEx(raster_path).ReadAsArray(),
            expected_total_sequestration, rtol=1e-6)

        expected_net_present_value_at_2030 = numpy.array(
            [[-373.67245, 891.60846]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output', 'net-present-value-at-2030.tif')
        numpy.testing.assert_allclose(
            gdal.OpenEx(raster_path).ReadAsArray(),
            expected_net_present_value_at_2030, rtol=1e-6)

    def test_model_no_transitions(self):
        """CBC: Test model without transitions.

        When the model executes without transitions, we still evaluate carbon
        sequestration (accumulation only) for the whole baseline period.
        """
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = os.path.join(self.workspace_dir, 'workspace')

        prior_snapshots = coastal_blue_carbon2._extract_snapshots_from_table(
            args['landcover_snapshot_csv'])
        baseline_year = min(prior_snapshots.keys())
        baseline_raster = prior_snapshots[baseline_year]
        with open(args['landcover_snapshot_csv'], 'w') as snapshot_csv:
            snapshot_csv.write('snapshot_year,raster_path\n')
            snapshot_csv.write(f'{baseline_year},{baseline_raster}\n')
        args['analysis_year'] = baseline_year + 10

        # Use valuation parameters rather than price table.
        args['use_price_table'] = False
        args['inflation_rate'] = 4
        args['price'] = 1.0

        coastal_blue_carbon2.execute(args)

        # Check sequestration raster
        expected_sequestration_2000_to_2010 = numpy.array(
            [[83.5, 0.]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2000-and-2010.tif'))
        numpy.testing.assert_allclose(
            (gdal.OpenEx(raster_path)).ReadAsArray(),
            expected_sequestration_2000_to_2010)

        # Check valuation raster
        # Discount rate here matches the inflation rate, so the value of the 10
        # years' accumulation is just 1*(10 years of accumulation).
        expected_net_present_value_at_2010 = numpy.array(
            [[835.0, 0.]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output', 'net-present-value-at-2010.tif')
        numpy.testing.assert_allclose(
            gdal.OpenEx(raster_path).ReadAsArray(),
            expected_net_present_value_at_2010, rtol=1e-6)

    def test_validation(self):
        """CBC: Test custom validation."""
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = self.workspace_dir

        # verify validation passes on basic set of arguments.
        validation_warnings = coastal_blue_carbon2.validate(args)
        self.assertEqual([], validation_warnings)

        # Now work through the extra validation warnings.
        # Create an invalid transitions table.
        invalid_raster_path = os.path.join(self.workspace_dir,
                                           'invalid_raster.tif')
        with open(invalid_raster_path, 'w') as raster:
            raster.write('not a raster')

        # Write over the landcover snapshot CSV
        prior_snapshots = coastal_blue_carbon2._extract_snapshots_from_table(
            args['landcover_snapshot_csv'])
        baseline_year = min(prior_snapshots)
        with open(args['landcover_snapshot_csv'], 'w') as snapshot_table:
            snapshot_table.write('snapshot_year,raster_path\n')
            snapshot_table.write(
                f'{baseline_year},{prior_snapshots[baseline_year]}\n')
            snapshot_table.write(
                f"{baseline_year + 10},{invalid_raster_path}")

        # analysis year must be >= the last transition year.
        args['analysis_year'] = baseline_year

        validation_warnings = coastal_blue_carbon2.validate(args)
        self.assertEquals(len(validation_warnings), 2)
        self.assertIn(
            f"Raster for snapshot {baseline_year + 10} could not "
            "be validated", validation_warnings[0][1])
        self.assertIn(
            "Analysis year 2000 must be >= the latest snapshot year "
            "(2010)",
            validation_warnings[1][1])
