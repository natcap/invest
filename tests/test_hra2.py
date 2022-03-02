import io
import json
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
import shapely.geometry
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

ORIGIN = (1180000.0, 690000.0)
_SRS = osr.SpatialReference()
_SRS.ImportFromEPSG(26910)  # UTM zone 10N
SRS_WKT = _SRS.ExportToWkt()


class HRAUnitTests(unittest.TestCase):
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
            {'rating': rating_raster_path, 'dq': 3, 'weight': 3},
            {'rating': 1, 'dq': 2, 'weight': 1},
            {'rating': 2, 'dq': 3, 'weight': 3},
            {'rating': 0, 'dq': 3, 'weight': 3},
        ]
        target_exposure_path = os.path.join(self.workspace_dir, 'exposure.tif')
        hra2._calc_criteria(attributes_list, habitat_mask_path,
                            target_exposure_path, decayed_distance_raster_path)

        exposure_array = pygeoprocessing.raster_to_numpy_array(
            target_exposure_path)
        nodata = hra2._TARGET_NODATA_FLOAT32
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

    def test_decayed_distance_no_decay(self):
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
            stressor_raster_path, 'None', buffer_distance,
            decayed_edt_path)

        # All pixels within the buffer distance are as impacted as though the
        # stressor overlapped it directly.
        expected_array = numpy.array([
            [1, 1, 1, 1, 0, 0]], dtype=numpy.float32)
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

        workspace = self.workspace_dir.replace('\\', '/')
        expected_habitats = {
            'corals': {
                'path': f'{workspace}/habitat/corals.shp',
            }
        }
        self.assertEqual(habitats, expected_habitats)

        expected_stressors = {
            'oil': {
                'path': f'{workspace}/stressors/oil.shp',
                'buffer': 1000,
            },
            'transportation': {
                'path': f'{workspace}/stressors/transport.shp',
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

    def test_maximum_reclassified_score(self):
        from natcap.invest import hra2

        nodata = hra2._TARGET_NODATA_BYTE

        habitat_mask = numpy.array(
            [[0, 1, nodata, 1, 1]], dtype=numpy.uint8)

        risk_classes = [
            numpy.array(classes, dtype=numpy.uint8) for classes in [
                [[nodata, 1, nodata, 2, 1]],
                [[nodata, 2, nodata, 2, 1]],
                [[nodata, 3, nodata, 1, 1]]]
        ]
        reclassified_score = hra2._maximum_reclassified_score(
            habitat_mask, *risk_classes)

        expected_risk_classes = numpy.array(
            [[nodata, 3, nodata, 2, 1]], dtype=numpy.uint8)
        numpy.testing.assert_allclose(
            reclassified_score, expected_risk_classes)

    def test_simplify(self):
        from natcap.invest import hra2

        geoms = [
            shapely.geometry.Point((ORIGIN[0] + 50, ORIGIN[1] + 50)).buffer(100),
            shapely.geometry.Point((ORIGIN[0] + 25, ORIGIN[1] + 25)).buffer(50),
        ]
        source_vector_path = os.path.join(self.workspace_dir,
                                          'source_vector.shp')
        pygeoprocessing.shapely_geometry_to_vector(
            geoms, source_vector_path, SRS_WKT, 'ESRI Shapefile')

        target_vector_path = os.path.join(self.workspace_dir,
                                          'target_vector.shp')
        hra2._simplify(source_vector_path, 20, target_vector_path)

        # Expected areas are from eyeballing that the resulting geometry look
        # correctly simplified.
        expected_areas = [28284.271247476, 5000.0]
        target_vector = gdal.OpenEx(target_vector_path)
        target_layer = target_vector.GetLayer()
        self.assertEqual(target_layer.GetFeatureCount(), 2)
        for expected_area, feature in zip(expected_areas, target_layer):
            feature_geom = feature.GetGeometryRef()
            self.assertAlmostEqual(expected_area, feature_geom.Area())

    def test_polygonize(self):
        from natcap.invest import hra2

        source_raster_path = os.path.join(self.workspace_dir, 'source.tif')
        source_array = numpy.array([
            [0, 1, 2],
            [1, 1, 2],
            [0, 1, 2]], dtype=numpy.uint8)

        mask_raster_path = os.path.join(self.workspace_dir, 'mask.tif')
        mask_array = (source_array != 0).astype(numpy.uint8)

        for array, target_path in ((source_array, source_raster_path),
                                   (mask_array, mask_raster_path)):
            pygeoprocessing.numpy_array_to_raster(
                array, 255, (30, -30), ORIGIN, SRS_WKT, target_path)

        target_vector_path = os.path.join(self.workspace_dir, 'target.gpkg')
        hra2._polygonize(source_raster_path, mask_raster_path,
                         target_vector_path, 'source_id')

        try:
            vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)
            layer = vector.GetLayer()
            self.assertEqual(
                [field.GetName() for field in layer.schema],
                ['source_id'])

            # The 0 pixels, which are not in the mask, should not be included
            # in the output vector.
            self.assertEqual(layer.GetFeatureCount(), 2)

            source_id_to_area = {
                1: 900 * 4,  # 4 pixels at 900m2/pixel
                2: 900 * 3,  # 3 pixels at 900m2/pixel
            }
            for feature in layer:
                source_id = feature.GetField('source_id')
                area = feature.GetGeometryRef().Area()
                self.assertEqual(area, source_id_to_area[source_id])
        finally:
            layer = None
            vector = None

    def test_polygonize_mask(self):
        from natcap.invest import hra2

        source_raster_path = os.path.join(self.workspace_dir, 'source.tif')
        nodata = 255
        source_array = numpy.array([
            [nodata, 1, 2],
            [1, 1, 2],
            [nodata, 1, 2]], dtype=numpy.uint8)
        pygeoprocessing.numpy_array_to_raster(
            source_array, nodata, (30, -30), ORIGIN, SRS_WKT,
            source_raster_path)

        mask_raster_path = os.path.join(self.workspace_dir, 'mask.tif')

        hra2._create_mask_for_polygonization(
            source_raster_path, mask_raster_path)

        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(mask_raster_path),
            (source_array != nodata).astype(numpy.uint8)
        )

    def test_summary_statistics_file(self):
        # TODO: test this in the model test?
        pass

    def test_rasterize_aoi_regions(self):
        from natcap.invest import hra2

        habitat_mask_path = os.path.join(
            self.workspace_dir, 'habitat_mask.tif')
        nodata = 255
        habitat_mask_array = numpy.array([
            [0, 1, 1],
            [0, 1, 255],
            [255, 1, 1]], dtype=numpy.uint8)
        pygeoprocessing.numpy_array_to_raster(
            habitat_mask_array, nodata, (30, -30), ORIGIN, SRS_WKT,
            habitat_mask_path)

        bounding_box = pygeoprocessing.get_raster_info(
            habitat_mask_path)['bounding_box']

        # 3 overlapping AOI regions
        aoi_geometries = [shapely.geometry.box(*bounding_box)] * 3
        source_vector_path = os.path.join(self.workspace_dir, 'aoi.shp')

        target_raster_dir = os.path.join(self.workspace_dir, 'rasters')
        target_json = os.path.join(self.workspace_dir, 'aoi_rasters.json')

        # First test: with no subregion names provided, there's only 1
        # subregion raster produced.
        pygeoprocessing.shapely_geometry_to_vector(
            aoi_geometries, source_vector_path, SRS_WKT, 'ESRI Shapefile')
        hra2._rasterize_aoi_regions(
            source_vector_path, habitat_mask_path, target_raster_dir,
            target_json)
        self.assertEqual(
            os.listdir(target_raster_dir),
            ['subregion_set_0.tif'])

        self.assertEqual(
            json.load(open(target_json)),
            {'subregion_rasters': [
                os.path.join(target_raster_dir, 'subregion_set_0.tif')],
             'subregion_names': {'0': 'Total Region'}})

        # Second test: when subregion names are provided, subregions should be
        # treated as distinct, nonoverlapping regions.
        pygeoprocessing.shapely_geometry_to_vector(
            aoi_geometries, source_vector_path, SRS_WKT,
            vector_format='ESRI Shapefile',
            fields={'NamE': ogr.OFTString},
            attribute_list=[
                {'NamE': 'subregion_1'},
                {'NamE': 'subregion_2'},
                {'NamE': 'subregion_3'},
            ])
        hra2._rasterize_aoi_regions(
            source_vector_path, habitat_mask_path, target_raster_dir,
            target_json)
        self.assertEqual(
            os.listdir(target_raster_dir),
            [f'subregion_set_{n}.tif' for n in (0, 1, 2)])
        self.assertEqual(
            json.load(open(target_json)),
            {'subregion_rasters': [
                os.path.join(target_raster_dir, f'subregion_set_{n}.tif')
                for n in (0, 1, 2)],
             'subregion_names': {
                 '0': 'subregion_1',
                 '1': 'subregion_2',
                 '2': 'subregion_3',
             }}
        )










class HRAModelTests(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_model(self):
        from natcap.invest import hra2

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'info_table_path': os.path.join(self.workspace_dir, 'info.csv'),
            'criteria_table_path': os.path.join(self.workspace_dir,
                                                'criteria.csv'),
            'resolution': 250,
            'max_rating': 3,
            'risk_eq': 'Multiplicative',
            'decay_eq': 'linear',
            'aoi_vector_path': 'create a vector',
            'override_max_overlapping_stressors': 2,
        }
