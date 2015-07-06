import unittest
import os
import pprint
import tempfile

import gdal

from natcap.invest.coastal_blue_carbon.utilities.raster import Raster
from natcap.invest.coastal_blue_carbon.utilities.raster_factory import RasterFactory
from natcap.invest.coastal_blue_carbon.cbc_preprocessor import *

pp = pprint.PrettyPrinter(indent=4)


class TestCBCPreprocessor(unittest.TestCase):
    def setUp(self):
        # create lookup table

        # set arguments
        shape = (2, 2)  # (2, 2)  #(1889, 1325)
        affine = Affine(30.0, 0.0, 443723.127328, 0.0, -30.0, 4956546.905980)
        proj = 26910
        datatype = gdal.GDT_Int32
        nodata_val = 255

        # initialize factory
        aoi_int_factory = RasterFactory(proj, datatype, nodata_val, shape[0], shape[1], affine=affine)

        # LULC Map
        year1_raster = aoi_int_factory.alternating(1, 2)
        year2_raster = aoi_int_factory.alternating(2, 1)
        year3_raster = aoi_int_factory.alternating(3, 1)
        year4_raster = aoi_int_factory.alternating(4, 1)

        self.args = {
            'workspace_dir': os.path.join(os.getcwd(), 'workspace'),
            'results_suffix': '',
            'lulc_lookup_table': os.path.join(os.getcwd(), 'lookup.csv'),
            'lulc_snapshot_list': [year1_raster.uri, year2_raster.uri, year3_raster.uri, year4_raster.uri]
        }

    def test_cbc_preprocessor(self):
        # remove lookup.csv

        # remove transition.csv

        pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
