'''
python -m unittest test_cbc_preprocessor.TestCBCPreprocessor
'''

import unittest
import os
import pprint
import csv

import gdal
from pygeoprocessing.geoprocessing import get_lookup_from_csv

from natcap.invest.coastal_blue_carbon.utilities.affine import Affine
from natcap.invest.coastal_blue_carbon.utilities.raster import Raster
from natcap.invest.coastal_blue_carbon.utilities.raster_factory import RasterFactory
import natcap.invest.coastal_blue_carbon.cbc_preprocessor as cbc_preprocessor

pp = pprint.PrettyPrinter(indent=4)


def write_csv(filepath, l):
    f = open(filepath, 'wb')
    writer = csv.writer(f)
    for i in l:
        writer.writerow(i)


class TestCBCPreprocessor(unittest.TestCase):
    def setUp(self):
        # create lookup.csv
        cwd = os.path.dirname(os.path.realpath(__file__))
        table = [
            ['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
            ['seagrass', '1', 'true'],
            ['man-made', '2', 'false'],
            ['marsh', '3', 'true'],
            ['mangrove', '4', 'true']]
        self.lookup_table_uri = os.path.join(cwd, 'lookup.csv')
        write_csv(self.lookup_table_uri, table)

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
        self.year3_raster = aoi_int_factory.alternating(3, 1)
        self.year4_raster = aoi_int_factory.alternating(4, 1)

        self.workspace_dir = os.path.join(cwd, 'workspace')

        self.args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'lulc_lookup_uri': self.lookup_table_uri,
            'lulc_snapshot_list': [
                self.year1_raster.uri,
                self.year2_raster.uri,
                self.year3_raster.uri,
                self.year4_raster.uri]
        }

    def test_cbc_preprocessor(self):
        cbc_preprocessor.execute(self.args)
        transition_dict = get_lookup_from_csv(
            os.path.join(self.workspace_dir, 'outputs', 'transition.csv'), 'lulc-classes')
        assert(transition_dict['seagrass']['seagrass'] == 'accumulation')

    def tearDown(self):
        # remove lookup.csv
        if os.path.isfile(self.lookup_table_uri):
            os.remove(self.lookup_table_uri)

        # remove transition.csv
        transition_table_uri = os.path.join(
            self.workspace_dir, 'outputs', 'transition.csv')
        if os.path.isfile(transition_table_uri):
            os.remove(transition_table_uri)

        # remove outputs and workspace
        output_dir = os.path.join(self.workspace_dir, 'outputs')
        if os.path.isdir(output_dir):
            os.removedirs(output_dir)
        if os.path.isdir(self.workspace_dir):
            os.removedirs(self.workspace_dir)


if __name__ == '__main__':
    unittest.main()
