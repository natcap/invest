import io
import math
import os
import shutil
import tempfile
import textwrap
import unittest

import numpy
import pandas
import pandas.testing
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

        decayed_distance_array = numpy.array([
            [0, 1, 1]], dtype=numpy.float32)
        decayed_distance_raster_path = os.path.join(self.workspace_dir,
                                                    'decayed_dist.tif')
        pygeoprocessing.numpy_array_to_raster(
            decayed_distance_array, -1, (30, -30), ORIGIN, SRS_WKT,
            decayed_distance_raster_path)

        attributes_list = [
            {'rating': rating_raster_path, 'data_quality': 3, 'weight': 3},
            {'rating': 1, 'data_quality': 2, 'weight': 1},
            {'rating': 2, 'data_quality': 3, 'weight': 3},
            {'rating': 0, 'data_quality': 3, 'weight': 3},
        ]
        target_exposure_path = os.path.join(self.workspace_dir, 'exposure.tif')
        hra2._calc_criteria(attributes_list, habitat_mask_path,
                            decayed_distance_raster_path, target_exposure_path)

        exposure_array = pygeoprocessing.raster_to_numpy_array(
            target_exposure_path)
        nodata = hra2._TARGET_NODATA_FLT
        # These expected values were calculated by hand based on the equation
        # for criteria scores in the user's guide.
        expected_exposure_array = numpy.array([
            [nodata, 1.0769231, 1.0384616]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            exposure_array, expected_exposure_array)

    def test_decayed_distance_linear(self):
        from natcap.invest import hra2

        stressor_mask = numpy.array([
            [1, 0, 0, 0, 0, 0]], dtype=numpy.uint8)
        stressor_raster_path = os.path.join(self.workspace_dir, 'stressor.tif')
        pygeoprocessing.numpy_array_to_raster(
            stressor_mask, 255, (30, -30), ORIGIN, SRS_WKT,
            stressor_raster_path)

        # buffer distance is 4*pixelwidth
        buffer_distance = 4*30
        decayed_edt_path = os.path.join(self.workspace_dir, 'decayed_edt.tif')
        hra2._calculate_decayed_distance(
            stressor_raster_path, 'linear', buffer_distance, decayed_edt_path)

        expected_array = numpy.array([
            [1, 0.75, 0.5, 0.25, 0, 0]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            expected_array,
            pygeoprocessing.raster_to_numpy_array(decayed_edt_path))

    def test_decayed_distance_exponential(self):
        from natcap.invest import hra2

        stressor_mask = numpy.array([
            [1, 0, 0, 0, 0, 0]], dtype=numpy.uint8)
        stressor_raster_path = os.path.join(self.workspace_dir, 'stressor.tif')
        pygeoprocessing.numpy_array_to_raster(
            stressor_mask, 255, (30, -30), ORIGIN, SRS_WKT,
            stressor_raster_path)

        # buffer distance is 4*pixelwidth
        buffer_distance = 4*30
        decayed_edt_path = os.path.join(self.workspace_dir, 'decayed_edt.tif')
        hra2._calculate_decayed_distance(
            stressor_raster_path, 'exponential', buffer_distance,
            decayed_edt_path)

        # Values here are represented by e**(-x), where x is pixel distances
        # away from the closest stressor pixel.
        expected_array = numpy.array([
            [1, math.exp(-1), math.exp(-2), math.exp(-3), 0, 0]],
            dtype=numpy.float32)
        numpy.testing.assert_allclose(
            expected_array,
            pygeoprocessing.raster_to_numpy_array(decayed_edt_path))

    def test_info_table_parsing(self):
        from natcap.invest import hra2

        info_table_path = os.path.join(self.workspace_dir, 'info_table.csv')
        with open(info_table_path, 'w') as info_table:
            info_table.write(
                textwrap.dedent(
                    # This leading backslash is important for dedent to parse
                    # the right number of leading spaces from the following
                    # rows.
                    # The paths don't actually need to exist for this test -
                    # this function is merely parsing the table contents.
                    """\
                    NAME,PATH,TYPE,STRESSOR BUFFER (meters)
                    corals,habitat/corals.shp,habitat,
                    oil,stressors/oil.shp,stressor,1000
                    transportation,stressors/transport.shp,stressor,100"""
                ))

        habitats, stressors = hra2._parse_info_table(info_table_path)

        expected_habitats = {
            'corals': {
                'path': 'habitat/corals.shp',
            }
        }
        self.assertEqual(habitats, expected_habitats)

        expected_stressors = {
            'oil': {
                'path': 'stressors/oil.shp',
                'buffer': 1000,
            },
            'transportation': {
                'path': 'stressors/transport.shp',
                'buffer': 100,
            }
        }
        self.assertEqual(stressors, expected_stressors)

    def test_criteria_table_parsing(self):
        from natcap.invest import hra2

        criteria_table_path = os.path.join(self.workspace_dir, 'criteria.csv')
        with open(criteria_table_path, 'w') as criteria_table:
            criteria_table.write(
                textwrap.dedent(
                    """\
                    HABITAT NAME,eelgrass,,,hardbottom,,,CRITERIA TYPE
                    HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    recruitment rate,2,2,2,2,2,2,C
                    connectivity rate,eelgrass_connectivity.shp,2,2,2,2,2,C
                    ,,,,,,,
                    HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                    oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    ,,,,,,,
                    HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                    fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    """
                ))
        target_composite_csv_path = os.path.join(self.workspace_dir,
                                                 'composite.csv')
        hra2._parse_criteria_table(criteria_table_path, ['oil', 'fishing'],
                                   target_composite_csv_path)

        expected_composite_dataframe = pandas.read_csv(
            io.StringIO(textwrap.dedent(
                """\
                habitat,stressor,criterion,rating,dq,weight,e/c
                eelgrass,RESILIENCE,recruitment rate,2,2,2,C
                hardbottom,RESILIENCE,recruitment rate,2,2,2,C
                eelgrass,RESILIENCE,connectivity rate,eelgrass_connectivity.shp,2,2,C
                hardbottom,RESILIENCE,connectivity rate,2,2,2,C
                eelgrass,oil,frequency of disturbance,2,2,3,C
                hardbottom,oil,frequency of disturbance,2,2,3,C
                eelgrass,oil,management effectiveness,2,2,1,E
                hardbottom,oil,management effectiveness,2,2,1,E
                eelgrass,fishing,frequency of disturbance,2,2,3,C
                hardbottom,fishing,frequency of disturbance,2,2,3,C
                eelgrass,fishing,management effectiveness,2,2,1,E
                hardbottom,fishing,management effectiveness,2,2,1,E
                """)))
        composite_dataframe = pandas.read_csv(target_composite_csv_path)
        pandas.testing.assert_frame_equal(
            expected_composite_dataframe, composite_dataframe)
