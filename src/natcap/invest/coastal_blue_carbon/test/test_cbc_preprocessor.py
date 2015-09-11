"""Test Cases for CBC Preprocessor.

python -m unittest test_cbc_preprocessor.TestCBCPreprocessor
"""

import unittest
import os
import pprint
import csv
import shutil

import gdal
from pygeoprocessing.geoprocessing import get_lookup_from_csv

import natcap.invest.coastal_blue_carbon.utilities.io as io
from natcap.invest.coastal_blue_carbon.global_variables import *
from natcap.invest.coastal_blue_carbon.classes.affine import Affine
from natcap.invest.coastal_blue_carbon.classes.raster import Raster
from natcap.invest.coastal_blue_carbon.classes.raster_factory import RasterFactory
import natcap.invest.coastal_blue_carbon.preprocessor as cbc_preprocessor

pp = pprint.PrettyPrinter(indent=4)


class TestCBCPreprocessor(unittest.TestCase):
    def setUp(self):
        # create lookup.csv
        cwd = os.path.dirname(os.path.realpath(__file__))
        self.workspace_dir = os.path.join(cwd, 'workspace')
        self.output_dir = os.path.join(
            self.workspace_dir, 'outputs_preprocessor')
        os.makedirs(self.output_dir)
        table = [
            ['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
            ['seagrass', '1', 'true'],
            ['man-made', '2', 'false'],
            ['marsh', '3', 'true'],
            ['mangrove', '4', 'true']]
        self.lookup_table_uri = os.path.join(self.workspace_dir, 'lookup.csv')
        io.write_csv(self.lookup_table_uri, table)

        # set arguments
        shape = (2, 2)  # (2, 2)  #(1889, 1325)
        affine = Affine(30.0, 0.0, 443723.127328, 0.0, -30.0, 4956546.905980)
        proj = 26910
        datatype = gdal.GDT_Int32
        nodata_val = 255

        # initialize factory
        aoi_int_factory = RasterFactory(
            proj, datatype, nodata_val, shape[0], shape[1], affine=affine)

        # LULC Map
        self.year1_raster = aoi_int_factory.alternating(1, 2)
        self.year2_raster = aoi_int_factory.alternating(2, 1)
        self.year3_raster = aoi_int_factory.alternating(2, 1)
        self.year4_raster = aoi_int_factory.alternating(3, 1)
        self.year5_raster = aoi_int_factory.alternating(4, 1)

        self.args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'lulc_lookup_uri': self.lookup_table_uri,
            'lulc_snapshot_list': [
                self.year1_raster.uri,
                self.year2_raster.uri,
                self.year3_raster.uri,
                self.year4_raster.uri,
                self.year5_raster.uri]
        }

    def test_cbc_preprocessor(self):
        cbc_preprocessor.execute(self.args)
        transition_dict = get_lookup_from_csv(
            os.path.join(
                self.workspace_dir,
                'outputs_preprocessor', 'transitions.csv'), 'lulc-class')
        self.assertEqual(
            transition_dict['seagrass']['seagrass'], 'accum')

    def test_append_legend(self):
        cbc_preprocessor.execute(self.args)
        transition_dict = get_lookup_from_csv(
            os.path.join(
                self.workspace_dir,
                'outputs_preprocessor', 'transitions.csv'), 'lulc-class')
        self.assertTrue('man-made' in transition_dict)

    def tearDown(self):
        # remove lookup.csv
        if os.path.isfile(self.lookup_table_uri):
            os.remove(self.lookup_table_uri)

        # remove workspace
        shutil.rmtree(self.workspace_dir)


class TestValidateTransition(unittest.TestCase):
    def setUp(self):
        pass

    def test_validate_transition(self):
        t1 = (1, 2)
        t2 = (2, NODATA_INT)
        t3 = (NODATA_INT, 1)
        t4 = (NODATA_INT, NODATA_INT)

        cbc_preprocessor._validate_transitions(set([t1]))

        with self.assertRaises(AssertionError):
            cbc_preprocessor._validate_transitions(set([t2]))

        with self.assertRaises(AssertionError):
            cbc_preprocessor._validate_transitions(set([t3]))

        cbc_preprocessor._validate_transitions(set([t4]))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
