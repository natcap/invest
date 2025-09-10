import glob
import io
import itertools
import json
import math
import os
import shutil
import tempfile
import textwrap
import unittest
import unittest.mock

import numpy
import pandas
import pandas.testing
import pygeoprocessing
import shapely.geometry
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

gdal.UseExceptions()
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
        """HRA: test criteria calculations are correct."""
        from natcap.invest import hra

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
        hra._calc_criteria(attributes_list, habitat_mask_path,
                           target_exposure_path, decayed_distance_raster_path)

        exposure_array = pygeoprocessing.raster_to_numpy_array(
            target_exposure_path)
        nodata = hra._TARGET_NODATA_FLOAT32
        # These expected values were calculated by hand based on the equation
        # for criteria scores in the user's guide.
        expected_exposure_array = numpy.array([
            [nodata, 1.0769231, 1.0384616]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            exposure_array, expected_exposure_array)

    def test_calc_criteria_skip_all_criteria(self):
        """HRA: handle user skipping all criteria."""
        from natcap.invest import hra

        habitat_mask = numpy.array([
            [0, 1, 1]], dtype=numpy.uint8)
        habitat_mask_path = os.path.join(self.workspace_dir,
                                         'habitat_mask.tif')
        pygeoprocessing.numpy_array_to_raster(
            habitat_mask, 255, (30, -30), ORIGIN, SRS_WKT, habitat_mask_path)

        decayed_distance_array = numpy.array([
            [0, 1, 1]], dtype=numpy.float32)
        decayed_distance_raster_path = os.path.join(self.workspace_dir,
                                                    'decayed_dist.tif')
        pygeoprocessing.numpy_array_to_raster(
            decayed_distance_array, -1, (30, -30), ORIGIN, SRS_WKT,
            decayed_distance_raster_path)

        attributes_list = [
            {'rating': 0, 'dq': 3, 'weight': 3},
            {'rating': 0, 'dq': 2, 'weight': 1},
            {'rating': 0, 'dq': 3, 'weight': 3},
            {'rating': 0, 'dq': 3, 'weight': 3},
        ]
        target_exposure_path = os.path.join(self.workspace_dir, 'exposure.tif')
        hra._calc_criteria(attributes_list, habitat_mask_path,
                           target_exposure_path, decayed_distance_raster_path)

        exposure_array = pygeoprocessing.raster_to_numpy_array(
            target_exposure_path)
        nodata = hra._TARGET_NODATA_FLOAT32
        # These expected values were calculated by hand.
        expected_exposure_array = numpy.array([
            [nodata, 0, 0]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            exposure_array, expected_exposure_array)

    def test_decayed_distance_linear(self):
        """HRA: linear decay over a distance."""
        from natcap.invest import hra

        stressor_mask = numpy.array([
            [1, 0, 0, 0, 0, 0]], dtype=numpy.uint8)
        stressor_raster_path = os.path.join(self.workspace_dir, 'stressor.tif')
        pygeoprocessing.numpy_array_to_raster(
            stressor_mask, 255, (30, -30), ORIGIN, SRS_WKT,
            stressor_raster_path)

        # buffer distance is 4*pixelwidth
        buffer_distance = 4*30
        decayed_edt_path = os.path.join(self.workspace_dir, 'decayed_edt.tif')
        hra._calculate_decayed_distance(
            stressor_raster_path, 'linear', buffer_distance, decayed_edt_path)

        expected_array = numpy.array([
            [1, 0.75, 0.5, 0.25, 0, 0]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            expected_array,
            pygeoprocessing.raster_to_numpy_array(decayed_edt_path))

    def test_decayed_distance_exponential(self):
        """HRA: exponential decay over a distance."""
        from natcap.invest import hra

        stressor_mask = numpy.array([
            [1, 0, 0, 0, 0, 0]], dtype=numpy.uint8)
        stressor_raster_path = os.path.join(self.workspace_dir, 'stressor.tif')
        pygeoprocessing.numpy_array_to_raster(
            stressor_mask, 255, (30, -30), ORIGIN, SRS_WKT,
            stressor_raster_path)

        # buffer distance is 4*pixelwidth
        buffer_distance = 4*30
        decayed_edt_path = os.path.join(self.workspace_dir, 'decayed_edt.tif')
        hra._calculate_decayed_distance(
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
        """HRA: weight with no decay out to a distance."""
        from natcap.invest import hra

        stressor_mask = numpy.array([
            [1, 0, 0, 0, 0, 0]], dtype=numpy.uint8)
        stressor_raster_path = os.path.join(self.workspace_dir, 'stressor.tif')
        pygeoprocessing.numpy_array_to_raster(
            stressor_mask, 255, (30, -30), ORIGIN, SRS_WKT,
            stressor_raster_path)

        # buffer distance is 4*pixelwidth
        buffer_distance = 4*30
        decayed_edt_path = os.path.join(self.workspace_dir, 'decayed_edt.tif')
        hra._calculate_decayed_distance(
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
        """HRA: check info table parsing w/ case sensitivity."""
        from natcap.invest import hra

        corals_path = 'habitat/corals.shp'
        oil_path = 'stressors/oil.shp'
        transport_path = 'stressors/transport.shp'
        geoms = [shapely.geometry.Point(ORIGIN).buffer(100)]
        os.makedirs(os.path.join(self.workspace_dir, 'habitat'))
        os.makedirs(os.path.join(self.workspace_dir, 'stressors'))
        for path in [corals_path, oil_path, transport_path]:
            pygeoprocessing.shapely_geometry_to_vector(
                geoms, os.path.join(self.workspace_dir, path),
                SRS_WKT, 'ESRI Shapefile')

        info_table_path = os.path.join(self.workspace_dir, 'info_table.csv')
        with open(info_table_path, 'w') as info_table:
            info_table.write(
                textwrap.dedent(
                    # This leading backslash is important for dedent to parse
                    # the right number of leading spaces from the following
                    # rows.
                    f"""\
                    NAME,PATH,TYPE,STRESSOR BUFFER (meters)
                    Corals,{corals_path},habitat,
                    Oil,{oil_path},stressor,1000
                    Transportation,{transport_path},stressor,100"""
                ))

        habitats, stressors = hra._parse_info_table(info_table_path)

        expected_habitats = {
            'corals': {
                'path': os.path.abspath(f'{self.workspace_dir}/habitat/corals.shp'),
            }
        }
        self.assertEqual(habitats, expected_habitats)

        expected_stressors = {
            'oil': {
                'path': os.path.abspath(f'{self.workspace_dir}/stressors/oil.shp'),
                'buffer': 1000,
            },
            'transportation': {
                'path': os.path.abspath(f'{self.workspace_dir}/stressors/transport.shp'),
                'buffer': 100,
            }
        }
        self.assertEqual(stressors, expected_stressors)

    def test_info_table_overlapping_habs_stressors(self):
        """HRA: error when info table has overlapping habitats, stressors."""
        from natcap.invest import hra

        corals_habitat_path = 'habitat/corals.shp'
        oil_path = 'stressors/oil.shp'
        corals_stressor_path = 'stressors/corals.shp'
        transport_path = 'stressors/transport.shp'
        geoms = [shapely.geometry.Point(ORIGIN).buffer(100)]
        os.makedirs(os.path.join(self.workspace_dir, 'habitat'))
        os.makedirs(os.path.join(self.workspace_dir, 'stressors'))
        for path in [corals_habitat_path, oil_path,
                corals_stressor_path, transport_path]:
            pygeoprocessing.shapely_geometry_to_vector(
                geoms, os.path.join(self.workspace_dir, path),
                SRS_WKT, 'ESRI Shapefile')

        info_table_path = os.path.join(self.workspace_dir, 'info_table.csv')
        with open(info_table_path, 'w') as info_table:
            info_table.write(
                textwrap.dedent(
                    # This leading backslash is important for dedent to parse
                    # the right number of leading spaces from the following
                    # rows.
                    # The paths don't actually need to exist for this test -
                    # this function is merely parsing the table contents.
                    f"""\
                    NAME,PATH,TYPE,STRESSOR BUFFER (meters)
                    corals,{corals_habitat_path},habitat,
                    oil,{oil_path},stressor,1000
                    corals,{corals_stressor_path},stressor,1000
                    transportation,{transport_path},stressor,100"""
                ))

        with self.assertRaises(ValueError) as cm:
            habitats, stressors = hra._parse_info_table(info_table_path)

        self.assertIn("Habitat and stressor names may not overlap",
                      str(cm.exception))

    def test_criteria_table_parsing(self):
        """HRA: check parsing of the criteria table w/ case sensitivity."""
        from natcap.invest import hra

        eelgrass_relpath = 'foo/eelgrass_connectivity.shp'

        criteria_table_path = os.path.join(self.workspace_dir, 'criteria.csv')
        with open(criteria_table_path, 'w') as criteria_table:
            criteria_table.write(
                textwrap.dedent(
                    f"""\
                    HABITAT NAME,Eelgrass,,,Hardbottom,,,CRITERIA TYPE
                    HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    recruitment rate,2,2,2,2,2,2,C
                    connectivity rate,{eelgrass_relpath},2,2,2,2,2,C
                    ,,,,,,,
                    HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                    Oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    ,,,,,,,
                    Fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    """
                ))
        target_composite_csv_path = os.path.join(self.workspace_dir,
                                                 'composite.csv')
        geoms = [
            shapely.geometry.Point(
                (ORIGIN[0] + 50, ORIGIN[1] + 50)).buffer(100),
            shapely.geometry.Point(
                (ORIGIN[0] + 25, ORIGIN[1] + 25)).buffer(50),
        ]
        source_vector_path = os.path.join(self.workspace_dir, 'foo',
                                          'eelgrass_connectivity.shp')
        os.makedirs(os.path.dirname(source_vector_path))
        pygeoprocessing.shapely_geometry_to_vector(
            geoms, source_vector_path, SRS_WKT, 'ESRI Shapefile')
        habitats, stressors = hra._parse_criteria_table(
            criteria_table_path, target_composite_csv_path)
        self.assertEqual(habitats, {'eelgrass', 'hardbottom'})
        self.assertEqual(stressors, {'oil', 'fishing'})

        eelgrass_abspath = os.path.abspath(
            os.path.join(self.workspace_dir, eelgrass_relpath))
        expected_composite_dataframe = pandas.read_csv(
            io.StringIO(textwrap.dedent(
                f"""\
                habitat,stressor,criterion,rating,dq,weight,e/c
                eelgrass,resilience,recruitment rate,2,2,2,C
                hardbottom,resilience,recruitment rate,2,2,2,C
                eelgrass,resilience,connectivity rate,{eelgrass_abspath},2,2,C
                hardbottom,resilience,connectivity rate,2,2,2,C
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

    def test_criteria_table_parsing_with_bom(self):
        """HRA: criteria table - parse a BOM."""
        from natcap.invest import hra

        criteria_table_path = os.path.join(self.workspace_dir, 'criteria.csv')
        with open(criteria_table_path, 'w', encoding='utf-8-sig') as criteria_table:
            criteria_table.write(
                textwrap.dedent(
                    """\
                    HABITAT NAME,eelgrass,,,hardbottom,,,CRITERIA TYPE
                    HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    recruitment rate,2,2,2,2,2,2,C
                    connectivity rate,2,2,2,2,2,2,C
                    ,,,,,,,
                    HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                    oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    ,,,,,,,
                    fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    """
                ))

        # Sanity check: make sure the file has the expected BOM
        # Gotta use binary mode so that python doesn't silently strip the BOM
        with open(criteria_table_path, 'rb') as criteria_table:
            self.assertTrue(criteria_table.read().startswith(b"\xef\xbb\xbf"))

        target_composite_csv_path = os.path.join(self.workspace_dir,
                                                 'composite.csv')
        hra._parse_criteria_table(criteria_table_path,
                                  target_composite_csv_path)

    def test_criteria_table_file_not_found(self):
        """HRA: criteria table - spatial file not found."""
        from natcap.invest import hra

        criteria_table_path = os.path.join(self.workspace_dir, 'criteria.csv')
        with open(criteria_table_path, 'w') as criteria_table:
            criteria_table.write(
                textwrap.dedent(
                    """\
                    HABITAT NAME,eelgrass,,,hardbottom,,,CRITERIA TYPE
                    HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    recruitment rate,2,2,2,2,2,2,C
                    connectivity rate,foo/eelgrass_connectivity.shp,2,2,2,2,2,C
                    ,,,,,,,
                    HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                    oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    ,,,,,,,
                    fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    """
                ))
        target_composite_csv_path = os.path.join(self.workspace_dir,
                                                 'composite.csv')
        with self.assertRaises(ValueError) as cm:
            habitats, stressors = hra._parse_criteria_table(
                criteria_table_path, target_composite_csv_path)
        self.assertIn("Criterion could not be opened as a spatial file",
                      str(cm.exception))

    def test_criteria_table_missing_section_headers(self):
        """HRA: verify exception when a required section is not found."""
        from natcap.invest import hra

        criteria_table_path = os.path.join(self.workspace_dir, 'criteria.csv')
        with open(criteria_table_path, 'w') as criteria_table:
            criteria_table.write(
                textwrap.dedent(  # NOTE: also checking whitespace around
                    """\
                      HABITAT-NAME,eelgrass,,,hardbottom ,,,CRITERIA TYPE
                    HABITAT-FOOOOO-ATTRIBUTES,RATING  ,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    recruitment rate,2,2,2,2,2,2,C
                    connectivity rate,foo/eelgrass_connectivity.shp,2,2,2,2,2,C
                    ,,,,,,,
                    HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                    oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance ,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    ,,,,,,,
                    fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    """
                ))
        target_composite_csv_path = os.path.join(self.workspace_dir,
                                                 'composite.csv')
        with self.assertRaises(AssertionError) as cm:
            habitats, stressors = hra._parse_criteria_table(
                criteria_table_path, target_composite_csv_path)
        self.assertIn('The criteria table is missing these section headers',
                      str(cm.exception))
        self.assertIn('HABITAT NAME', str(cm.exception))
        self.assertIn('HABITAT RESILIENCE ATTRIBUTES', str(cm.exception))

    def test_criteria_table_remote_filepath(self):
        """HRA: correctly parse a remote path in criteria table."""
        from natcap.invest import hra

        criteria_table_path = os.path.join(self.workspace_dir, 'criteria.csv')
        with open(criteria_table_path, 'w') as criteria_table:
            criteria_table.write(
                textwrap.dedent(  # NOTE: also checking whitespace around
                    """\
                    HABITAT NAME,Eelgrass,,,Hardbottom,,,CRITERIA TYPE
                    HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    recruitment rate,2,2,2,2,2,2,C
                    connectivity rate,https://example.com/raster.tif,2,2,2,2,2,C
                    ,,,,,,,
                    HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                    Oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    ,,,,,,,
                    Fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                    frequency of disturbance,2,2,3,2,2,3,C
                    management effectiveness,2,2,1,2,2,1,E
                    """
                ))
        target_composite_csv_path = os.path.join(self.workspace_dir,
                                                 'composite.csv')
        with unittest.mock.patch('pygeoprocessing.get_gis_type',
                                 lambda path: pygeoprocessing.RASTER_TYPE):
            habitats, stressors = hra._parse_criteria_table(
                criteria_table_path, target_composite_csv_path)
        parsed_table = pandas.read_csv(target_composite_csv_path)
        self.assertEqual(parsed_table['rating'][2],
                        '/vsicurl/https://example.com/raster.tif')


    def test_maximum_reclassified_score(self):
        """HRA: check maximum reclassed score given a stack of scores."""
        from natcap.invest import hra

        nodata = hra._TARGET_NODATA_BYTE

        habitat_mask = numpy.array(
            [[0, 1, nodata, 1, 1]], dtype=numpy.uint8)

        risk_classes = [
            numpy.array(classes, dtype=numpy.uint8) for classes in [
                [[nodata, 1, nodata, 2, 1]],
                [[nodata, 2, nodata, 2, 1]],
                [[nodata, 3, nodata, 1, 1]]]
        ]
        reclassified_score = hra._maximum_reclassified_score(
            habitat_mask, *risk_classes)

        expected_risk_classes = numpy.array(
            [[nodata, 3, nodata, 2, 1]], dtype=numpy.uint8)
        numpy.testing.assert_allclose(
            reclassified_score, expected_risk_classes)

    def test_simplify(self):
        """HRA: check geometry simplification routine."""
        from natcap.invest import hra

        geoms = [
            shapely.geometry.Point(
                (ORIGIN[0] + 50, ORIGIN[1] + 50)).buffer(100),
            shapely.geometry.Point(
                (ORIGIN[0] + 25, ORIGIN[1] + 25)).buffer(50),
        ]
        source_vector_path = os.path.join(self.workspace_dir,
                                          'source_vector.shp')
        pygeoprocessing.shapely_geometry_to_vector(
            geoms, source_vector_path, SRS_WKT, 'ESRI Shapefile')

        target_vector_path = os.path.join(self.workspace_dir,
                                          'target_vector.gpkg')
        hra._simplify(source_vector_path, 20, target_vector_path)

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
        """HRA: test polygonization."""
        from natcap.invest import hra

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
        layer_name = 'my_layer'
        hra._polygonize(source_raster_path, mask_raster_path,
                        target_vector_path, 'source_id', layer_name)

        try:
            vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)
            layer = vector.GetLayer(layer_name)
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
        """HRA: test the polygonization mask."""
        from natcap.invest import hra

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

        hra._create_mask_for_polygonization(
            source_raster_path, mask_raster_path)

        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(mask_raster_path),
            (source_array != nodata).astype(numpy.uint8)
        )

    def test_align(self):
        """HRA: test alignment function."""
        from natcap.invest import hra

        habitat_raster_path = os.path.join(
            self.workspace_dir, 'habitat_raster.tif')
        habitat_array = numpy.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=numpy.uint8)
        # pixel size is slightly smaller than the target pixel size in order to
        # force resampling.
        pygeoprocessing.numpy_array_to_raster(
            habitat_array, hra._TARGET_NODATA_BYTE, (29, -29), ORIGIN,
            SRS_WKT, habitat_raster_path)

        criterion_raster_path = os.path.join(
            self.workspace_dir, 'criterion_raster.tif')
        criterion_array = numpy.array([
            [1.1, 1, 0.9, 0.8, 0.7],
            [1.1, 1, 0.9, 0.8, 0.7],
            [1.1, 1, 0.9, 0.8, 0.7],
            [1.1, 1, 0.9, 0.8, 0.7]], dtype=numpy.float32)
        pygeoprocessing.numpy_array_to_raster(
            criterion_array, hra._TARGET_NODATA_FLOAT32, (30, -30), ORIGIN,
            SRS_WKT, criterion_raster_path)

        habitat_vector_path = os.path.join(
            self.workspace_dir, 'habitat_vector.shp')
        habitat_polygons = [
            shapely.geometry.Point(
                (ORIGIN[0] + 50, ORIGIN[1] - 50)).buffer(30)]
        pygeoprocessing.shapely_geometry_to_vector(
            habitat_polygons, habitat_vector_path, SRS_WKT, 'ESRI Shapefile')

        criterion_vector_path = os.path.join(
            self.workspace_dir, 'criterion_vector.shp')
        criterion_polygons = [
            shapely.geometry.Point(
                (ORIGIN[0] - 50, ORIGIN[1] + 50)).buffer(100)]
        pygeoprocessing.shapely_geometry_to_vector(
            criterion_polygons, criterion_vector_path, SRS_WKT,
            vector_format='ESRI Shapefile',
            fields={'RatInG': ogr.OFTReal},  # test case sensitivity.
            attribute_list=[{'RatInG': 0.12}]
        )

        raster_path_map = {
            habitat_raster_path: os.path.join(
                self.workspace_dir, 'aligned_habitat_raster.tif'),
            criterion_raster_path: os.path.join(
                self.workspace_dir, 'aligned_criterion_raster.tif')
        }
        vector_path_map = {
            habitat_vector_path: os.path.join(
                self.workspace_dir, 'aligned_habitat_vector.tif'),
            criterion_vector_path: os.path.join(
                self.workspace_dir, 'aligned_criterion_vector.tif'),
        }

        hra._align(raster_path_map, vector_path_map, (30, -30), SRS_WKT,
                  all_touched_vectors=set([habitat_vector_path]))

        # Calculated by hand given the above spatial inputs and
        # (30, -30) pixels.  All rasters should share the same extents and
        # pixel size.
        expected_bounding_box = [
            ORIGIN[0] - 150,
            ORIGIN[1] - 120,
            ORIGIN[0] + 150,
            ORIGIN[1] + 150
        ]

        # Keeping this in here for debugging, although it's not used for the
        # test.
        pygeoprocessing.geoprocessing.shapely_geometry_to_vector(
            [shapely.geometry.box(*expected_bounding_box)],
            os.path.join(self.workspace_dir, 'expected_bbox.shp'),
            SRS_WKT, vector_format='ESRI Shapefile')

        for aligned_raster_path in itertools.chain(raster_path_map.values(),
                                                   vector_path_map.values()):
            raster_info = pygeoprocessing.get_raster_info(aligned_raster_path)
            self.assertEqual(raster_info['pixel_size'], (30, -30))
            self.assertEqual(raster_info['bounding_box'],
                             expected_bounding_box)

        # The aligned habitat raster should have been rasterized as all 1s on a
        # field of nodata.
        expected_habitat_array = numpy.array([
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 1, 1, 1, 255, 255],
            [255, 255, 255, 255, 255, 1, 1, 1, 255, 255],
            [255, 255, 255, 255, 255, 1, 1, 1, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]],
            dtype=numpy.uint8)
        aligned_habitat_array = pygeoprocessing.raster_to_numpy_array(
            vector_path_map[habitat_vector_path])
        numpy.testing.assert_equal(
            aligned_habitat_array, expected_habitat_array)

        # The aligned criterion raster should have been rasterized from the
        # rating column.
        # This is an ALL_TOUCHED=FALSE rasterization.
        ndta = hra._TARGET_NODATA_FLOAT32
        expected_criterion_array = numpy.array([
            [ndta, ndta, 0.12, 0.12, 0.12, ndta, ndta, ndta, ndta, ndta],
            [ndta, 0.12, 0.12, 0.12, 0.12, 0.12, ndta, ndta, ndta, ndta],
            [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, ndta, ndta, ndta],
            [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, ndta, ndta, ndta],
            [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, ndta, ndta, ndta, ndta],
            [ndta, 0.12, 0.12, 0.12, 0.12, 0.12, ndta, ndta, ndta, ndta],
            [ndta, ndta, 0.12, 0.12, ndta, ndta, ndta, ndta, ndta, ndta],
            [ndta, ndta, ndta, ndta, ndta, ndta, ndta, ndta, ndta, ndta],
            [ndta, ndta, ndta, ndta, ndta, ndta, ndta, ndta, ndta, ndta]],
            dtype=numpy.float32)
        aligned_criterion_array = pygeoprocessing.raster_to_numpy_array(
            vector_path_map[criterion_vector_path])
        numpy.testing.assert_allclose(
            aligned_criterion_array, expected_criterion_array)

    def test_prep_criterion_raster(self):
        """HRA: Test processing of user inputs for consistency."""
        from natcap.invest import hra

        # Test what happens when the raster has a defined nodata value.
        nodata = 255
        criterion_array_with_nodata = numpy.array([
            [-1, 0, 1.67, nodata]], dtype=numpy.float32)
        raster_path = os.path.join(
            self.workspace_dir, 'raster_with_nodata.tif')
        pygeoprocessing.numpy_array_to_raster(
            criterion_array_with_nodata, nodata, (30, -30), ORIGIN, SRS_WKT,
            raster_path)
        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        hra._prep_input_criterion_raster(raster_path, target_raster_path)
        expected_array = numpy.array([
            [hra._TARGET_NODATA_FLOAT32, 0, 1.67,
             hra._TARGET_NODATA_FLOAT32]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(target_raster_path),
            expected_array)

        # Test what happens when the raster does not have a defined nodata
        # value
        criterion_array_without_nodata = numpy.array([
            [-1, 0, 0.33, 2]], dtype=numpy.float32)
        raster_path = os.path.join(
            self.workspace_dir, 'raster_without_nodata.tif')
        pygeoprocessing.numpy_array_to_raster(
            criterion_array_without_nodata, None, (30, -30), ORIGIN, SRS_WKT,
            raster_path)
        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        hra._prep_input_criterion_raster(raster_path, target_raster_path)
        expected_array = numpy.array([
            [hra._TARGET_NODATA_FLOAT32, 0, 0.33, 2]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(target_raster_path),
            expected_array)

    def test_mask_binary_values(self):
        """HRA: test masking of presence/absence."""
        from natcap.invest import hra

        mask_array_1 = numpy.array([
            [0, 1, 255]], dtype=numpy.uint8)
        float32_min = numpy.finfo(numpy.float32).min
        mask_array_2 = numpy.array([
            [1, 0, float32_min]], dtype=numpy.float32)
        mask_array_3 = numpy.array([
            [1024, 0, 1]], dtype=numpy.int32)
        source_paths = []
        for index, (array, nodata) in enumerate([
                (mask_array_1, 255),
                (mask_array_2, float(float32_min)),
                (mask_array_3, None)]):
            path = os.path.join(self.workspace_dir, f'{index}.tif')
            source_paths.append(path)
            pygeoprocessing.numpy_array_to_raster(
                array, nodata, (30, -30), ORIGIN, SRS_WKT, path)

        mask_path = os.path.join(self.workspace_dir, 'mask.tif')
        hra._mask_binary_presence_absence_rasters(source_paths, mask_path)

        expected_mask_array = numpy.array([
            [1, 1, 1]], dtype=numpy.uint8)

        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(mask_path),
            expected_mask_array)

    def test_pairwise_risk(self):
        """HRA: check pairwise risk calculations."""
        from natcap.invest import hra

        byte_nodata = hra._TARGET_NODATA_BYTE
        habitat_mask_path = os.path.join(
            self.workspace_dir, 'habitat_mask.tif')
        habitat_mask_array = numpy.array([
            [1, 1, 1, 0, byte_nodata]], dtype=numpy.uint8)

        float_nodata = hra._TARGET_NODATA_FLOAT32
        exposure_path = os.path.join(self.workspace_dir, 'exposure.tif')
        exposure_array = numpy.array([
            [0.1, 1.1, 1.2, 0.3, float_nodata]], dtype=numpy.float32)

        consequence_path = os.path.join(self.workspace_dir, 'consequence.tif')
        consequence_array = numpy.array([
            [0.2, 1.2, 1.3, 0.4, float_nodata]], dtype=numpy.float32)

        for path, array, nodata in [
                (habitat_mask_path, habitat_mask_array, byte_nodata),
                (exposure_path, exposure_array, float_nodata),
                (consequence_path, consequence_array, float_nodata)]:
            pygeoprocessing.numpy_array_to_raster(
                array, nodata, (30, -30), ORIGIN, SRS_WKT, path)

        multiplicative_risk_path = os.path.join(
            self.workspace_dir, 'multiplicative.tif')
        hra._calculate_pairwise_risk(
            habitat_mask_path, exposure_path, consequence_path,
            'multiplicative', multiplicative_risk_path)

        expected_multiplicative_array = numpy.array([
            [0.02, 1.32, 1.56, float_nodata, float_nodata]],
            dtype=numpy.float32)
        numpy.testing.assert_allclose(
            expected_multiplicative_array,
            pygeoprocessing.raster_to_numpy_array(multiplicative_risk_path))

        euclidean_risk_path = os.path.join(
            self.workspace_dir, 'euclidean.tif')
        hra._calculate_pairwise_risk(
            habitat_mask_path, exposure_path, consequence_path, 'euclidean',
            euclidean_risk_path)

        expected_euclidean_array = numpy.array([
            [0.0, 0.22360685, 0.36055511, float_nodata, float_nodata]],
            dtype=numpy.float32)
        numpy.testing.assert_allclose(
            expected_euclidean_array,
            pygeoprocessing.raster_to_numpy_array(euclidean_risk_path)
        )

        with self.assertRaises(AssertionError) as cm:
            hra._calculate_pairwise_risk(
                habitat_mask_path, exposure_path, consequence_path,
                'bad_risk_type', euclidean_risk_path)

        self.assertIn('Invalid risk equation', str(cm.exception))

    def test_sum_rasters(self):
        """HRA: check summing of rasters."""
        from natcap.invest import hra

        nodata = -1
        risk_array_1 = numpy.array([
            [nodata, 1.3, 2.4]], dtype=numpy.float32)
        risk_array_2 = numpy.array([
            [0.6, nodata, 3.8]], dtype=numpy.float32)
        risk_array_3 = numpy.array([
            [0.1, 0.7, nodata]], dtype=numpy.float32)

        raster_paths = []
        for index, array in enumerate((risk_array_1,
                                       risk_array_2,
                                       risk_array_3)):
            path = os.path.join(self.workspace_dir, f'{index}.tif')
            raster_paths.append(path)
            pygeoprocessing.numpy_array_to_raster(
                array, nodata, (10, -10), ORIGIN, SRS_WKT, path)

        # Test a straight sum
        target_nodata = hra._TARGET_NODATA_FLOAT32
        target_datatype = hra._TARGET_GDAL_TYPE_FLOAT32
        target_raster_path = os.path.join(self.workspace_dir, 'sum.tif')
        hra._sum_rasters(
            raster_paths, target_nodata, target_datatype, target_raster_path)
        expected_array = numpy.array([
            [0.7, 2.0, 6.2]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(target_raster_path),
            expected_array)

        # Test with normalization.
        hra._sum_rasters(raster_paths, target_nodata, target_datatype,
                         target_raster_path, normalize=True)
        expected_array = numpy.array([
            [0.35, 1.0, 3.1]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(target_raster_path),
            expected_array)

    def test_datastack_criteria_table_override(self):
        """HRA: verify we store all data referenced in the criteria table."""
        from natcap.invest import hra

        criteria_table_path = os.path.join(
            self.workspace_dir, 'criteria_table.csv')
        with open(criteria_table_path, 'w') as criteria_table:
            criteria_table.write(textwrap.dedent(
                """\
                HABITAT NAME,eelgrass,,,hardbottom,,,CRITERIA TYPE
                HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                recruitment rate,2,2,2,2,2,2,C
                connectivity rate,eelgrass_connectivity.shp,2,2,2,2,2,C
                ,,,,,,,
                HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,2,2,1,my_data/mgmt1.tif,2,1,E
                ,,,,,,,
                fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,filenotfound,2,3,2,2,3,C
                management effectiveness,my_data/mgmt1.tif,2,1,2,2,1,E
                ,,,,,,,
                transportation,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,2,2,1,my_data/mgmt2.tif,2,1,E
                """
            ))

        eelgrass_path = os.path.join(
            self.workspace_dir, 'eelgrass_connectivity.shp')
        geoms = [shapely.geometry.Point(ORIGIN).buffer(100)]
        pygeoprocessing.shapely_geometry_to_vector(
            geoms, eelgrass_path, SRS_WKT, 'ESRI Shapefile')

        mgmt_path_1 = os.path.join(
            self.workspace_dir, 'my_data', 'mgmt1.tif')
        mgmt_path_2 = os.path.join(
            self.workspace_dir, 'my_data', 'mgmt2.tif')
        for mgmt_path in (mgmt_path_1, mgmt_path_2):
            os.makedirs(os.path.dirname(mgmt_path), exist_ok=True)
            array = numpy.ones((20, 20), dtype=numpy.uint8)
            pygeoprocessing.numpy_array_to_raster(
                array, 255, (10, -10), (ORIGIN[0] - 50, ORIGIN[1] - 50),
                SRS_WKT, mgmt_path)

        data_dir = os.path.join(self.workspace_dir, 'datastack_data')
        known_files = {}

        new_csv_path = hra._override_datastack_archive_criteria_table_path(
            criteria_table_path, data_dir, known_files)
        self.assertEqual(
            new_csv_path, os.path.join(data_dir, 'criteria_table_path.csv'))
        output_criteria_data_dir = os.path.join(
            data_dir, 'criteria_table_path_data')
        self.maxDiff = None

        self.assertEqual(
            known_files, {
                eelgrass_path: os.path.join(
                    output_criteria_data_dir, 'eelgrass_connectivity',
                    'eelgrass_connectivity.shp'),
                mgmt_path_1: os.path.join(
                    output_criteria_data_dir, 'mgmt1', 'mgmt1.tif'),
                mgmt_path_2: os.path.join(
                    output_criteria_data_dir, 'mgmt2', 'mgmt2.tif')
            }
        )
        for copied_filepath in known_files.values():
            self.assertEqual(True, os.path.exists(copied_filepath))
            try:
                spatial_file = gdal.OpenEx(copied_filepath)
                if spatial_file is None:
                    self.fail('Filepath could not be opened by GDAL: '
                              f'{copied_filepath}')
            finally:
                spatial_file = None

    def test_none_decay_distance(self):
        """HRA: Test 0 buffer distance."""
        from natcap.invest import hra
        nodata = -1
        shape = (20, 20)
        stressor_array = numpy.ones(shape, dtype=numpy.uint8)
        stressor_path = os.path.join(self.workspace_dir, 'stressor.tif')
        pygeoprocessing.numpy_array_to_raster(
            stressor_array, nodata, (10, -10), ORIGIN, SRS_WKT, stressor_path)

        target_path = os.path.join(self.workspace_dir, 'decayed.tif')
        hra._calculate_decayed_distance(stressor_path, 'none', 0,  target_path)

        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(target_path),
            stressor_array.astype(numpy.float32))

    def test_exception_invalid_decay(self):
        """HRA: Test invalid decay type."""
        from natcap.invest import hra
        nodata = -1
        shape = (20, 20)
        stressor_array = numpy.ones(shape, dtype=numpy.uint8)
        stressor_path = os.path.join(self.workspace_dir, 'stressor.tif')
        pygeoprocessing.numpy_array_to_raster(
            stressor_array, nodata, (10, -10), ORIGIN, SRS_WKT, stressor_path)

        target_path = os.path.join(self.workspace_dir, 'decayed.tif')
        with self.assertRaises(AssertionError) as cm:
            hra._calculate_decayed_distance(stressor_path, 'bad decay type', 0,
                                            target_path)
        self.assertIn('Invalid decay type bad decay type provided',
                      str(cm.exception))

    def test_summary_stats(self):
        """HRA: test summary stats table."""
        from natcap.invest import hra, file_registry
        e_array = numpy.array([[0, 1, 2, 3]], dtype=numpy.float32)
        c_array = numpy.array([[0.5, 1.5, 2.5, 3.5]], dtype=numpy.float32)
        risk_array = numpy.array([[0, 1.1, 2.2, 3.3]], dtype=numpy.float32)
        pairwise_classes_array = numpy.array([[0, 1, 2, 3]], dtype=numpy.int8)
        # NOTE that if we were running this in the real world with only 1
        # pairwise risk raster, the cumulative risk would match the pairwise
        # risk.  I'm providing a different cumulative risk raster here for the
        # sole purpose of checking table construction, not to provide
        # real-world model results.
        cumulative_classes_array = numpy.array([[2, 3, 2, 3]], dtype=numpy.uint8)

        habitats = ['life']
        stressors = ['industry']

        os.mkdir(os.path.join(self.workspace_dir, 'intermediate_outputs'))
        file_registry = file_registry.FileRegistry(
            hra.MODEL_SPEC.outputs, self.workspace_dir, '')
        nodata = -1
        for array, path in [
                (e_array, 'E_life_industry.tif'),
                (c_array, 'C_life_industry.tif'),
                (risk_array, 'RISK_life_industry.tif'),
                (pairwise_classes_array, 'reclass_life_industry.tif'),
                (cumulative_classes_array, 'reclass_total_risk_life.tif')]:
            pygeoprocessing.numpy_array_to_raster(
                array, nodata, (10, -10), ORIGIN, SRS_WKT,
                os.path.join(self.workspace_dir, 'intermediate_outputs', path))

        target_summary_csv_path = os.path.join(
            self.workspace_dir, 'summary.csv')
        aoi_vector_path = os.path.join(self.workspace_dir, 'aoi.shp')
        subregion_bounding_box = pygeoprocessing.get_raster_info(
            os.path.join(self.workspace_dir, 'intermediate_outputs',
                'reclass_total_risk_life.tif'))['bounding_box']
        subregion_geom = shapely.geometry.box(*subregion_bounding_box)

        def percent_with_risk_class(array, risk_class):
            """Calculate the percent of risk class pixels matching a class.

            Args:
                array (numpy.array): A risk classification array.
                risk_class (int): The integer risk class of interest

            Returns:
                The percentage (0-100) of pixels in ``array`` that match the
                risk class ``risk_class``.
            """
            return (array[array == risk_class].size / array.size) * 100

        # This is a standard record in the summary table, used in both subtests
        # below.
        std_record = {
            'HABITAT': 'life',
            'STRESSOR': 'industry',
            'E_MIN': numpy.min(e_array),
            'E_MAX': numpy.max(e_array),
            'E_MEAN': numpy.sum(e_array) / 4,
            'C_MIN': numpy.min(c_array),
            'C_MAX': numpy.max(c_array),
            'C_MEAN': numpy.sum(c_array) / 4,
            'R_MIN': numpy.min(risk_array),
            'R_MAX': numpy.max(risk_array),
            'R_MEAN': numpy.sum(risk_array) / 4,
            'R_%HIGH': percent_with_risk_class(pairwise_classes_array, 3),
            'R_%MEDIUM': percent_with_risk_class(pairwise_classes_array, 2),
            'R_%LOW': percent_with_risk_class(pairwise_classes_array, 1),
            'R_%NONE': percent_with_risk_class(pairwise_classes_array, 0),
        }

        with self.subTest("multiple subregion names"):
            # 3 subregions, 2 of which have the same name.
            # In cases of overlap, the function double-counts.
            pygeoprocessing.shapely_geometry_to_vector(
                [subregion_geom] * 3, aoi_vector_path, SRS_WKT,
                'ESRI Shapefile', fields={'name': ogr.OFTString},
                attribute_list=[
                    {'name': 'first region'},
                    {'name': 'first region'},
                    {'name': 'second region'}
                ])
            hra._create_summary_statistics_file(
                aoi_vector_path, habitats, stressors, file_registry,
                target_summary_csv_path)
            expected_records = [
                {**std_record,
                 **{'SUBREGION': 'first region',
                    'STRESSOR': '(FROM ALL STRESSORS)'},
                    'R_%HIGH': percent_with_risk_class(
                        cumulative_classes_array, 3),
                    'R_%MEDIUM': percent_with_risk_class(
                        cumulative_classes_array, 2),
                    'R_%LOW': percent_with_risk_class(
                        cumulative_classes_array, 1),
                    'R_%NONE': percent_with_risk_class(
                        cumulative_classes_array, 0),
                },
                {**std_record,
                 **{'SUBREGION': 'second region',
                    'STRESSOR': '(FROM ALL STRESSORS)'},
                    'R_%HIGH': percent_with_risk_class(
                        cumulative_classes_array, 3),
                    'R_%MEDIUM': percent_with_risk_class(
                        cumulative_classes_array, 2),
                    'R_%LOW': percent_with_risk_class(
                        cumulative_classes_array, 1),
                    'R_%NONE': percent_with_risk_class(
                        cumulative_classes_array, 0),
                },
                {**std_record,
                 **{'SUBREGION': 'first region'},
                },
                {**std_record,
                 **{'SUBREGION': 'second region'}
                },
            ]
            created_dataframe = pandas.read_csv(target_summary_csv_path)
            expected_dataframe = pandas.DataFrame.from_records(
                expected_records).reindex(columns=created_dataframe.columns)
            pandas.testing.assert_frame_equal(
                expected_dataframe, created_dataframe,
                check_dtype=False  # ignore float32/float64 type difference.
            )

        with self.subTest("no subregion names"):
            # When no subregion names provided, all subregions are assumed to
            # be in the same region: "Total Region".
            pygeoprocessing.shapely_geometry_to_vector(
                [subregion_geom] * 3, aoi_vector_path, SRS_WKT,
                'ESRI Shapefile')
            hra._create_summary_statistics_file(
                aoi_vector_path, habitats, stressors, file_registry,
                target_summary_csv_path)
            expected_records = [
                {**std_record,
                 **{'SUBREGION': 'Total Region',
                    'STRESSOR': '(FROM ALL STRESSORS)'},
                    'R_%HIGH': 50.0,
                    'R_%MEDIUM': 50.0,
                    'R_%LOW': 0,
                    'R_%NONE': 0,
                },
                {**std_record,
                 **{'SUBREGION': 'Total Region'},
                },
            ]
            created_dataframe = pandas.read_csv(target_summary_csv_path)
            expected_dataframe = pandas.DataFrame.from_records(
                expected_records).reindex(columns=created_dataframe.columns)
            pandas.testing.assert_frame_equal(
                expected_dataframe, created_dataframe,
                check_dtype=False  # ignore float32/float64 type difference.
            )


class HRAModelTests(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_model(self):
        """HRA: end-to-end test of the model, including datastack."""
        from natcap.invest import hra
        from natcap.invest import datastack

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'info_table_path': os.path.join(self.workspace_dir, 'info.csv'),
            'criteria_table_path': os.path.join(self.workspace_dir,
                                                'criteria.csv'),
            'resolution': 250,
            'max_rating': 3,
            'risk_eq': 'multiplicative',
            'decay_eq': 'linear',
            'aoi_vector_path': os.path.join(self.workspace_dir, 'aoi.shp'),
            'n_overlapping_stressors': 2,
            'visualize_outputs': False,
        }
        aoi_geoms = [
            shapely.geometry.box(  # This geometry covers all areas
                *shapely.geometry.Point(ORIGIN).buffer(100).bounds),
            shapely.geometry.box(  # Geometry covers only 1 stressor
                *shapely.geometry.Point(
                    (ORIGIN[0], ORIGIN[1]-200)).buffer(100).bounds)
        ]
        pygeoprocessing.shapely_geometry_to_vector(
            aoi_geoms, args['aoi_vector_path'], SRS_WKT, 'ESRI Shapefile',
            fields={'name': ogr.OFTString}, attribute_list=[
                {'name': 'wholearea'}, {'name': 'noarea'}])

        with open(args['criteria_table_path'], 'w') as criteria_table:
            criteria_table.write(textwrap.dedent(
                """\
                HABITAT NAME,eelgrass,,,hardbottom,,,CRITERIA TYPE
                HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                recruitment rate,2,2,2,2,2,2,C
                connectivity rate,eelgrass_connectivity.shp,2,2,2,2,2,C
                ,,,,,,,
                HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,mgmt_effect.tif,2,1,2,2,1,E
                ,,,,,,,
                fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,2,2,1,2,2,1,E
                ,,,,,,,
                transportation,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,2,2,1,2,2,1,E
                """
            ))

        with open(args['info_table_path'], 'w') as info_table:
            info_table.write(textwrap.dedent(
                """\
                NAME,PATH,TYPE,STRESSOR BUFFER (meters)
                eelgrass,habitats/eelgrass.tif,habitat,
                hardbottom,habitats/hardbottom.shp,habitat
                oil,stressors/oil.tif,stressor,1000
                fishing,stressors/fishing.shp,stressor,500
                transportation,stressors/transport.shp,stressor,100"""))

        # The tuple notation (path, ) is needed here to tell python to
        # interpret the result of itertuples as a tuple.
        spatial_files_to_make = [
            item[0] for item in
            pandas.read_csv(args['info_table_path'])[['PATH']].itertuples(
                index=False)]
        spatial_files_to_make.append('eelgrass_connectivity.shp')
        spatial_files_to_make.append('mgmt_effect.tif')

        for path in spatial_files_to_make:
            full_path = os.path.join(self.workspace_dir, path)
            if not os.path.exists(os.path.dirname(full_path)):
                os.makedirs(os.path.dirname(full_path))

            if path.endswith('.shp'):
                # Make a point that sits at the center of the AOI, buffered by
                # 100m
                geoms = [shapely.geometry.Point(ORIGIN).buffer(100)]

                field_defns = {}
                field_values = [{}]
                if path == 'eelgrass_connectivity.shp':
                    # Make sure we can test field preservation
                    field_defns = {'rating': ogr.OFTInteger}
                    field_values = [{'rating': 1}]

                pygeoprocessing.shapely_geometry_to_vector(
                    geoms, full_path, SRS_WKT, 'ESRI Shapefile',
                    fields=field_defns, attribute_list=field_values)
            else:  # Assume geotiff
                # Raster is centered on the origin, spanning 50m on either
                # side.
                array = numpy.ones((20, 20), dtype=numpy.uint8)
                if path == 'mgmt_effect.tif':
                    array *= 2  # fill with twos
                pygeoprocessing.numpy_array_to_raster(
                    array, 255, (10, -10), (ORIGIN[0] - 50, ORIGIN[1] - 50),
                    SRS_WKT, full_path)

        archive_path = os.path.join(self.workspace_dir, 'datstack.tar.gz')
        datastack.build_datastack_archive(
            args, 'habitat_risk_assessment', archive_path)

        unarchived_path = os.path.join(self.workspace_dir, 'unarchived_data')
        unarchived_args = datastack.extract_datastack_archive(
            archive_path, unarchived_path)
        unarchived_args['workspace_dir'] = os.path.join(
            self.workspace_dir, 'workspace')

        # Confirm no validation warnings
        validation_warnings = hra.validate(unarchived_args)
        self.assertEqual(validation_warnings, [])

        hra.execute(unarchived_args)

        # Ecosystem risk is the sum of all risk values, so a good indicator of
        # whether the model has changed.
        numpy.testing.assert_allclose(
            numpy.array([[10.25], [3.125]], dtype=numpy.float32),
            pygeoprocessing.geoprocessing.raster_to_numpy_array(
                os.path.join(unarchived_args['workspace_dir'], 'outputs',
                             'TOTAL_RISK_Ecosystem.tif')))

        self.assertFalse(os.path.exists(
            os.path.join(unarchived_args['workspace_dir'],
                         'visualization_outputs')))

        # Re-run with vizualizations
        # Also tests the task graph
        unarchived_args['visualize_outputs'] = True
        hra.execute(unarchived_args)

        # Make sure we have some valid geojson files in the viz dir.
        n_geojson_files = 0
        for geojson_file in glob.glob(
                os.path.join(unarchived_args['workspace_dir'],
                             'visualization_outputs', '*.geojson')):
            try:
                vector = gdal.OpenEx(geojson_file)
                self.assertNotEqual(vector, None)
            finally:
                vector = None
            n_geojson_files += 1
        self.assertEqual(n_geojson_files, 6)

        # verify that the rasterized vectors match the source rasters.
        output_dir = os.path.join(args['workspace_dir'], 'outputs')
        intermediate_dir = os.path.join(
            args['workspace_dir'], 'intermediate_outputs')
        viz_dir = os.path.join(args['workspace_dir'], 'visualization_outputs')
        raster_and_vector_versions = {
            os.path.join(output_dir, 'RECLASS_RISK_Ecosystem.tif'):
                (os.path.join(viz_dir, 'RECLASS_RISK_Ecosystem.geojson'),
                 'Risk Score'),
            os.path.join(output_dir, 'RECLASS_RISK_eelgrass.tif'):
                (os.path.join(viz_dir, 'RECLASS_RISK_eelgrass.geojson'),
                 'Risk Score'),
            os.path.join(output_dir, 'RECLASS_RISK_hardbottom.tif'):
                (os.path.join(viz_dir, 'RECLASS_RISK_hardbottom.geojson'),
                 'Risk Score'),
            os.path.join(intermediate_dir, 'aligned_oil.tif'):
                (os.path.join(viz_dir, 'STRESSOR_oil.geojson'),
                 'Stressor'),
            os.path.join(intermediate_dir, 'aligned_fishing.tif'):
                (os.path.join(viz_dir, 'STRESSOR_fishing.geojson'),
                 'Stressor'),
            os.path.join(intermediate_dir, 'aligned_transportation.tif'):
                (os.path.join(viz_dir, 'STRESSOR_transportation.geojson'),
                 'Stressor'),
        }
        for source_raster, (target_vector, attribute) in raster_and_vector_versions.items():
            rasterized_path = os.path.join(
                self.workspace_dir, 'temp_rasterized.tif')
            pygeoprocessing.geoprocessing.new_raster_from_base(
                source_raster, rasterized_path, gdal.GDT_Byte, [255], [255])
            pygeoprocessing.geoprocessing.rasterize(
                target_vector, rasterized_path,
                option_list=[f'ATTRIBUTE={attribute}'])

            numpy.testing.assert_array_equal(
                pygeoprocessing.geoprocessing.raster_to_numpy_array(
                    source_raster),
                pygeoprocessing.geoprocessing.raster_to_numpy_array(
                    rasterized_path))

    def test_model_habitat_mismatch(self):
        """HRA: check errors when habitats are mismatched."""
        from natcap.invest import hra

        eelgrass_conn_path = os.path.join(
            self.workspace_dir, 'eelgrass_connectivity.shp')
        criteria_table_path = os.path.join(self.workspace_dir, 'criteria.csv')
        with open(criteria_table_path, 'w') as criteria_table:
            criteria_table.write(textwrap.dedent(
                f"""\
                HABITAT NAME,eelgrass,,,hardbottom,,,CRITERIA TYPE
                HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                recruitment rate,2,2,2,2,2,2,C
                connectivity rate,{eelgrass_conn_path},2,2,2,2,2,C
                ,,,,,,,
                HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,2,2,1,2,2,1,E
                ,,,,,,,
                fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,2,2,1,2,2,1,E
                """
            ))

        corals_path = 'habitat/corals.shp'
        oil_path = 'stressors/oil.shp'
        transport_path = 'stressors/transport.shp'
        geoms = [shapely.geometry.Point(ORIGIN).buffer(100)]
        os.makedirs(os.path.join(self.workspace_dir, 'habitat'))
        os.makedirs(os.path.join(self.workspace_dir, 'stressors'))
        for path in [
                eelgrass_conn_path, corals_path, oil_path, transport_path]:
            pygeoprocessing.shapely_geometry_to_vector(
                geoms, os.path.join(self.workspace_dir, path),
                SRS_WKT, 'ESRI Shapefile')

        info_table_path = os.path.join(self.workspace_dir, 'info.csv')
        with open(info_table_path, 'w') as info_table:
            info_table.write(textwrap.dedent(
                f"""\
                NAME,PATH,TYPE,STRESSOR BUFFER (meters)
                corals,{corals_path},habitat,
                oil,{oil_path},stressor,1000
                transportation,{transport_path},stressor,100"""))

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'criteria_table_path': criteria_table_path,
            'info_table_path': info_table_path,
            'resolution': 250,
            'max_rating': 3,
            'n_overlapping_stressors': 3,
            'risk_eq': 'multiplicative',
            'decay_eq': 'linear',
            'aoi_vector_path': os.path.join(self.workspace_dir, 'aoi.shp'),
        }

        aoi_geoms = [shapely.geometry.box(
            *shapely.geometry.Point(ORIGIN).buffer(100).bounds)]
        pygeoprocessing.shapely_geometry_to_vector(
            aoi_geoms, args['aoi_vector_path'], SRS_WKT, 'ESRI Shapefile')

        with self.assertRaises(ValueError) as cm:
            hra.execute(args)

        self.assertIn('habitats', str(cm.exception))
        self.assertIn("Missing from info table: eelgrass, hardbottom",
                      str(cm.exception))
        self.assertIn("Missing from criteria table: corals",
                      str(cm.exception))

    def test_model_stressor_mismatch(self):
        """HRA: check stressor mismatch."""
        from natcap.invest import hra

        eelgrass_conn_path = 'eelgrass_connectivity.shp'
        criteria_table_path = os.path.join(self.workspace_dir, 'criteria.csv')
        with open(criteria_table_path, 'w') as criteria_table:
            criteria_table.write(textwrap.dedent(
                f"""\
                HABITAT NAME,eelgrass,,,hardbottom,,,CRITERIA TYPE
                HABITAT RESILIENCE ATTRIBUTES,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                recruitment rate,2,2,2,2,2,2,C
                connectivity rate,{eelgrass_conn_path},2,2,2,2,2,C
                ,,,,,,,
                HABITAT STRESSOR OVERLAP PROPERTIES,,,,,,,
                oil,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,2,2,1,2,2,1,E
                ,,,,,,,
                fishing,RATING,DQ,WEIGHT,RATING,DQ,WEIGHT,E/C
                frequency of disturbance,2,2,3,2,2,3,C
                management effectiveness,2,2,1,2,2,1,E
                """
            ))

        eelgrass_path = 'habitat/eelgrass.shp'
        hardbottom_path = 'habitat/hardbottom.shp'
        oil_path = 'stressors/oil.shp'
        transport_path = 'stressors/transport.shp'
        geoms = [shapely.geometry.Point(ORIGIN).buffer(100)]
        os.makedirs(os.path.join(self.workspace_dir, 'habitat'))
        os.makedirs(os.path.join(self.workspace_dir, 'stressors'))
        for path in [
                eelgrass_conn_path, eelgrass_path, hardbottom_path,
                oil_path, transport_path]:
            pygeoprocessing.shapely_geometry_to_vector(
                geoms, os.path.join(self.workspace_dir, path),
                SRS_WKT, 'ESRI Shapefile')

        info_table_path = os.path.join(self.workspace_dir, 'info.csv')
        with open(info_table_path, 'w') as info_table:
            info_table.write(textwrap.dedent(
                f"""\
                NAME,PATH,TYPE,STRESSOR BUFFER (meters)
                eelgrass,{eelgrass_path},habitat,
                hardbottom,{hardbottom_path},habitat,
                oil,{oil_path},stressor,1000
                transportation,{transport_path},stressor,100"""))

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'criteria_table_path': criteria_table_path,
            'info_table_path': info_table_path,
            'resolution': 250,
            'max_rating': 3,
            'n_overlapping_stressors': 3,
            'risk_eq': 'euclidean',
            'decay_eq': 'linear',
            'aoi_vector_path': os.path.join(self.workspace_dir, 'aoi.shp'),
        }

        aoi_geoms = [shapely.geometry.box(
            *shapely.geometry.Point(ORIGIN).buffer(100).bounds)]
        pygeoprocessing.shapely_geometry_to_vector(
            aoi_geoms, args['aoi_vector_path'], SRS_WKT, 'ESRI Shapefile')

        with self.assertRaises(ValueError) as cm:
            hra.execute(args)
        self.assertIn('stressors', str(cm.exception))
        self.assertIn("Missing from info table: fishing",
                      str(cm.exception))
        self.assertIn("Missing from criteria table: transportation",
                      str(cm.exception))

        args['risk_eq'] = 'some other risk eq'
        with self.assertRaises(ValueError) as cm:
            hra.execute(args)

        self.assertIn("must be either 'Multiplicative' or 'Euclidean'",
                      str(cm.exception))
