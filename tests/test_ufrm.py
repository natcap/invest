# coding=UTF-8
"""Tests for Urban Flood Risk Mitigation Model."""
import os
import shutil
import tempfile
import unittest

import numpy
import pandas
import pygeoprocessing
import shapely.geometry
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

gdal.UseExceptions()

class UFRMTests(unittest.TestCase):
    """Tests for the Urban Flood Risk Mitigation Model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp(suffix='\U0001f60e')  # smiley

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def _make_args(self):
        """Create args list for UFRM."""
        base_dir = os.path.dirname(__file__)
        args = {
            'aoi_watersheds_path': os.path.join(
                base_dir, '..', 'data', 'invest-test-data', 'ufrm',
                'watersheds.gpkg'),
            'built_infrastructure_vector_path': os.path.join(
                base_dir, '..', 'data', 'invest-test-data', 'ufrm',
                'infrastructure.gpkg'),
            'curve_number_table_path': os.path.join(
                base_dir, '..', 'data', 'invest-test-data', 'ufrm',
                'Biophysical_water_SF.csv'),
            'infrastructure_damage_loss_table_path': os.path.join(
                base_dir, '..', 'data', 'invest-test-data', 'ufrm',
                'Damage.csv'),
            'lulc_path': os.path.join(
                base_dir, '..', 'data', 'invest-test-data', 'ufrm',
                'lulc.tif'),
            'rainfall_depth': 40,
            'results_suffix': 'Test1',
            'soils_hydrological_group_raster_path': os.path.join(
                base_dir, '..', 'data', 'invest-test-data', 'ufrm',
                'soilgroup.tif'),
            'workspace_dir': self.workspace_dir,
        }
        return args

    def test_ufrm_regression(self):
        """UFRM: regression test."""
        from natcap.invest import urban_flood_risk_mitigation
        args = self._make_args()
        input_vector = gdal.OpenEx(args['aoi_watersheds_path'],
                                   gdal.OF_VECTOR)
        input_layer = input_vector.GetLayer()
        input_fields = [field.GetName() for field in input_layer.schema]

        urban_flood_risk_mitigation.execute(args)

        result_vector = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'flood_risk_service_Test1.shp'),
            gdal.OF_VECTOR)
        result_layer = result_vector.GetLayer()

        # Check that all expected fields are there.
        output_fields = ['aff_bld', 'serv_blt', 'rnf_rt_idx',
                         'rnf_rt_m3', 'flood_vol']
        output_fields += input_fields
        self.assertEqual(
            set(output_fields),
            set(field.GetName() for field in result_layer.schema))

        result_feature = result_layer.GetNextFeature()
        for fieldname, expected_value in (
                ('aff_bld', 187010830.32202843),
                ('serv_blt', 13253546667257.65),
                ('rnf_rt_idx', 0.70387527942),
                ('rnf_rt_m3', 70870.4765625),
                ('flood_vol', 29815.640625)):
            result_val = result_feature.GetField(fieldname)
            places_to_round = (
                int(round(numpy.log(expected_value)/numpy.log(10)))-6)
            self.assertAlmostEqual(
                result_val, expected_value, places=-places_to_round)

        input_feature = input_layer.GetNextFeature()
        for fieldname in input_fields:
            self.assertEqual(result_feature.GetField(fieldname),
                             input_feature.GetField(fieldname))

        result_feature = None
        result_layer = None
        result_vector = None

    def test_ufrm_regression_no_infrastructure(self):
        """UFRM: regression for no infrastructure."""
        from natcap.invest import urban_flood_risk_mitigation
        args = self._make_args()
        del args['built_infrastructure_vector_path']
        input_vector = gdal.OpenEx(args['aoi_watersheds_path'],
                                   gdal.OF_VECTOR)
        input_layer = input_vector.GetLayer()
        input_fields = [field.GetName() for field in input_layer.schema]

        urban_flood_risk_mitigation.execute(args)

        result_raster = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'Runoff_retention_m3_Test1.tif'),
            gdal.OF_RASTER)
        band = result_raster.GetRasterBand(1)
        array = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        band = None
        result_raster = None
        result_sum = numpy.sum(array[~numpy.isclose(array, nodata)])
        # expected result observed from regression run.
        expected_result = 156070.36
        self.assertAlmostEqual(result_sum, expected_result, places=0)

        result_vector = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'flood_risk_service_Test1.shp'),
            gdal.OF_VECTOR)
        result_layer = result_vector.GetLayer()
        result_feature = result_layer.GetFeature(0)

        # Check that only the expected fields are there.
        output_fields = ['rnf_rt_idx', 'rnf_rt_m3', 'flood_vol']
        output_fields += input_fields
        self.assertEqual(
            set(output_fields),
            set(field.GetName() for field in result_layer.schema))

        for fieldname, expected_value in (
                ('rnf_rt_idx', 0.70387527942),
                ('rnf_rt_m3', 70870.4765625),
                ('flood_vol', 29815.640625)):
            result_val = result_feature.GetField(fieldname)
            places_to_round = (
                int(round(numpy.log(expected_value)/numpy.log(10)))-6)
            self.assertAlmostEqual(
                result_val, expected_value, places=-places_to_round)

    def test_ufrm_value_error_on_bad_soil(self):
        """UFRM: assert exception on bad soil raster values."""
        from natcap.invest import urban_flood_risk_mitigation
        args = self._make_args()

        bad_soil_raster = os.path.join(
            self.workspace_dir, 'bad_soilgroups.tif')
        value_map = {
            1: 1,
            2: 2,
            3: 9,  # only 1, 2, 3, 4 are valid values for this raster.
            4: 4
        }
        pygeoprocessing.reclassify_raster(
            (args['soils_hydrological_group_raster_path'], 1), value_map,
            bad_soil_raster, gdal.GDT_Int16, -9)
        args['soils_hydrological_group_raster_path'] = bad_soil_raster

        with self.assertRaises(ValueError) as cm:
            urban_flood_risk_mitigation.execute(args)

        actual_message = str(cm.exception)
        expected_message = (
            'Check that the Soil Group raster does not contain')
        self.assertTrue(expected_message in actual_message)

    def test_ufrm_value_error_on_bad_lucode(self):
        """UFRM: assert exception on missing lucodes."""
        import pandas
        from natcap.invest import urban_flood_risk_mitigation
        args = self._make_args()

        bad_cn_table_path = os.path.join(
            self.workspace_dir, 'bad_cn_table.csv')
        cn_table = pandas.read_csv(args['curve_number_table_path'])

        # drop a row with an lucode known to exist in lulc raster
        # This is a code that will successfully index into the
        # CN table sparse matrix, but will not return valid data.
        bad_cn_table = cn_table[cn_table['lucode'] != 0]
        bad_cn_table.to_csv(bad_cn_table_path, index=False)
        args['curve_number_table_path'] = bad_cn_table_path

        with self.assertRaises(ValueError) as cm:
            urban_flood_risk_mitigation.execute(args)

        actual_message = str(cm.exception)
        expected_message = (
            f'The biophysical table is missing a row for lucode(s) {[0]}')
        self.assertEqual(expected_message, actual_message)

        # drop rows with lucodes known to exist in lulc raster
        # These are codes that will raise an IndexError on
        # indexing into the CN table sparse matrix. The test
        # LULC raster has values from 0 to 21.
        bad_cn_table = cn_table[cn_table['lucode'] < 15]
        bad_cn_table.to_csv(bad_cn_table_path, index=False)
        args['curve_number_table_path'] = bad_cn_table_path

        with self.assertRaises(ValueError) as cm:
            urban_flood_risk_mitigation.execute(args)

        actual_message = str(cm.exception)
        expected_message = (
            f'The biophysical table is missing a row for lucode(s) '
            f'{[16, 17, 18, 21]}')
        self.assertEqual(expected_message, actual_message)

    def test_ufrm_explicit_zeros_in_table(self):
        """UFRM: assert no exception on row of all zeros."""
        import pandas
        from natcap.invest import urban_flood_risk_mitigation
        args = self._make_args()

        good_cn_table_path = os.path.join(
            self.workspace_dir, 'good_cn_table.csv')
        cn_table = pandas.read_csv(args['curve_number_table_path'])

        # a user may define a row with all 0s
        cn_table.iloc[0] = [0] * cn_table.shape[1]
        cn_table.to_csv(good_cn_table_path, index=False)
        args['curve_number_table_path'] = good_cn_table_path

        try:
            urban_flood_risk_mitigation.execute(args)
        except ValueError:
            self.fail('unexpected ValueError when testing curve number row with all zeros')

    def test_ufrm_string_damage_to_infrastructure(self):
        """UFRM: handle str(int) structure indices.

        This came up on the forums, where a user had provided a string column
        type that contained integer data.  OGR returned these ints as strings,
        leading to a ``KeyError``.  See
        https://github.com/natcap/invest/issues/590.
        """
        from natcap.invest import urban_flood_risk_mitigation

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        aoi_geometry = [
            shapely.geometry.box(pos_x, pos_y, pos_x + 200, pos_y + 200),
        ]

        def _infra_geom(xoff, yoff):
            """Create sample infrastructure geometry at a position offset.

            The geometry will be centered on (x+xoff, y+yoff).

            Parameters:
                xoff (number): The x offset, referenced against ``pos_x`` from
                    the outer scope.
                yoff (number): The y offset, referenced against ``pos_y`` from
                    the outer scope.

            Returns:
                A ``shapely.Geometry`` of a point buffered by ``20`` centered
                on the provided (x+xoff, y+yoff) point.
            """
            return shapely.geometry.Point(
                pos_x + xoff, pos_y + yoff).buffer(20)

        infra_geometries = [
            _infra_geom(x_offset, 100)
            for x_offset in range(0, 200, 40)]

        infra_fields = {'Type': ogr.OFTString}  # THIS IS THE THING TESTED
        infra_attrs = [
            {'Type': str(index)} for index in range(len(infra_geometries))]

        infrastructure_path = os.path.join(
            self.workspace_dir, 'infra_vector.shp')
        pygeoprocessing.shapely_geometry_to_vector(
            infra_geometries, infrastructure_path, projection_wkt,
            'ESRI Shapefile', fields=infra_fields, attribute_list=infra_attrs,
            ogr_geom_type=ogr.wkbPolygon)

        aoi_path = os.path.join(self.workspace_dir, 'aoi.shp')
        pygeoprocessing.shapely_geometry_to_vector(
            aoi_geometry, aoi_path, projection_wkt,
            'ESRI Shapefile', ogr_geom_type=ogr.wkbPolygon)

        structures_damage_table_path = os.path.join(
            self.workspace_dir, 'damage_table_path.csv')
        with open(structures_damage_table_path, 'w') as csv_file:
            csv_file.write('"Type","damage"\n')
            for attr_dict in infra_attrs:
                type_index = int(attr_dict['Type'])
                csv_file.write(f'"{type_index}",1\n')

        aoi_damage_dict = (
            urban_flood_risk_mitigation._calculate_damage_to_infrastructure_in_aoi(
                aoi_path, infrastructure_path, structures_damage_table_path))

        # Total damage is the sum of the area of all infrastructure geometries
        # that intersect the AOI, with each area multiplied by the damage cost.
        # For this test, damage is always 1, so it's just the intersecting
        # area.
        self.assertEqual(len(aoi_damage_dict), 1)
        numpy.testing.assert_allclose(aoi_damage_dict[0], 5645.787282992962)
    
    def test_ufrm_smax(self):
        """UFRM: test _s_max operation."""
        from natcap.invest import urban_flood_risk_mitigation
        
        cn_nodata = -1
        result_nodata = -9999
        # These varying percent slope values should cover all of the slope
        # factor and slope table cases.
        cn_array = numpy.array(
            [[100, 0, 65, 90, 30, cn_nodata]], dtype=numpy.float32)

        smax_result = urban_flood_risk_mitigation._s_max_op(
            cn_array, cn_nodata, result_nodata)

        smax_expected = numpy.array(
            [[0.0, 100000, 136.769, 28.222, 592.667, result_nodata]],
            dtype=numpy.float32)
        numpy.testing.assert_allclose(smax_result, smax_expected, rtol=1e-5)

    def test_validate(self):
        """UFRM: test validate function."""
        from natcap.invest import urban_flood_risk_mitigation
        from natcap.invest import validation
        args = self._make_args()
        validation_warnings = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(len(validation_warnings), 0)

        del args['workspace_dir']
        validation_warnings = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(len(validation_warnings), 1)

        args['workspace_dir'] = ''
        result = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(
            result, [(['workspace_dir'], validation.MESSAGES['MISSING_VALUE'])])

        args = self._make_args()
        args['lulc_path'] = 'fake/path/notfound.tif'
        result = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(
            result, [(['lulc_path'], validation.MESSAGES['FILE_NOT_FOUND'])])

        args = self._make_args()
        args['lulc_path'] = args['aoi_watersheds_path']
        result = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(
            result, [(['lulc_path'], validation.MESSAGES['NOT_GDAL_RASTER'])])

        args = self._make_args()
        args['aoi_watersheds_path'] = args['lulc_path']
        result = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(
            result,
            [(['aoi_watersheds_path'], validation.MESSAGES['NOT_GDAL_VECTOR'])])

        args = self._make_args()
        del args['infrastructure_damage_loss_table_path']
        result = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(
            result,
            [(['infrastructure_damage_loss_table_path'],
                validation.MESSAGES['MISSING_KEY'])])

        args = self._make_args()
        cn_table = pandas.read_csv(args['curve_number_table_path'])
        cn_table = cn_table.drop(columns=[f'CN_{code}' for code in 'ABCD'])
        new_cn_path = os.path.join(self.workspace_dir, 'new_cn_table.csv')
        cn_table.to_csv(new_cn_path, index=False)
        args['curve_number_table_path'] = new_cn_path
        result = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(
            result,
            [(['curve_number_table_path'],
              validation.MESSAGES['MATCHED_NO_HEADERS'].format(
                  header='column', header_name='cn_a'))])

        # test missing CN_X values raise warnings
        args = self._make_args()
        cn_table = pandas.read_csv(args['curve_number_table_path'])
        cn_table.at[0, 'CN_A'] = numpy.nan
        new_cn_path = os.path.join(
            self.workspace_dir, 'cn_missing_value_table.csv')
        cn_table.to_csv(new_cn_path, index=False)
        args['curve_number_table_path'] = new_cn_path
        result = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(
            result,
            [(['curve_number_table_path'],
              'Missing curve numbers for lucode(s) [0]')])
