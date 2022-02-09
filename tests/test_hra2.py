import os
import shutil
import tempfile
import unittest

import numpy
import pygeoprocessing
from osgeo import gdal
from osgeo import osr

ORIGIN = (1180000.0, 690000.0)
_SRS = osr.SpatialReference()
_SRS.ImportFromEPSG(26910)  # UTM zone 10N
SRS_WKT = _SRS.ExportToWkt()


class HRATests2(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_calc_criteria(self):
        from natcap.invest import hra2

        habitat_mask = numpy.array([
            [0, 1, 1]], dtype=numpy.uint8)
        habitat_mask_path = os.path.join(self.workspace_dir,
                                         'habitat_mask.tif')
        pygeoprocessing.numpy_array_to_raster(
            habitat_mask, 255, (30, -30), ORIGIN, SRS_WKT, habitat_mask_path)

        rating_array = numpy.array([
            [0, 0.5, 0.25]], dtype=numpy.float32)
        rating_raster_path = os.path.join(self.workspace_dir,
                                          'rating.tif')
        pygeoprocessing.numpy_array_to_raster(
            rating_array, -1, (30, -30), ORIGIN, SRS_WKT, rating_raster_path)

        attributes_list = [
            {'rating': rating_raster_path, 'data_quality': 3, 'weight': 3},
            {'rating': 1, 'data_quality': 2, 'weight': 1},
            {'rating': 2, 'data_quality': 3, 'weight': 3},
            {'rating': 0, 'data_quality': 3, 'weight': 3},
        ]
        target_exposure_path = os.path.join(self.workspace_dir, 'exposure.tif')
        hra2._calc_criteria(attributes_list, habitat_mask_path,
                            target_exposure_path)

        exposure_array = pygeoprocessing.raster_to_numpy_array(
            target_exposure_path)
        nodata = hra2._TARGET_NODATA_FLT
        # These expected values were calculated by hand based on the equation
        # for criteria scores in the user's guide.
        expected_exposure_array = numpy.array([
            [nodata, 1.0769231, 1.0384616]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            exposure_array, expected_exposure_array)
