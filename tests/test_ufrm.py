# coding=UTF-8
"""Tests for Urban Flood Risk Mitigation Model."""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
import numpy
import pygeoprocessing


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
        urban_flood_risk_mitigation.execute(args)

        result_vector = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'flood_risk_service_Test1.shp'),
            gdal.OF_VECTOR)
        result_layer = result_vector.GetLayer()
        result_feature = next(result_layer)
        result_val = result_feature.GetField('serv_bld')
        result_feature = None
        result_layer = None
        result_vector = None
        # expected result observed from regression run.
        expected_result = 13253546667257.65
        places_to_round = (
            int(round(numpy.log(expected_result)/numpy.log(10)))-6)
        self.assertAlmostEqual(
            result_val, expected_result, places=-places_to_round)

    def test_ufrm_regression_no_infrastructure(self):
        """UFRM: regression for no infrastructure."""
        from natcap.invest import urban_flood_risk_mitigation
        args = self._make_args()
        del args['built_infrastructure_vector_path']
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

    def test_ufrm_value_error_on_bad_soil(self):
        """UFRM: assert exception on bad soil raster values."""
        from natcap.invest import urban_flood_risk_mitigation
        args = self._make_args()

        bad_soil_raster = os.path.join(self.workspace_dir, 'bad_soilgroups.tif')
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
            expected_message = 'Check that the Soil Group raster does not contain'
            self.assertTrue(expected_message in actual_message)

    def test_validate(self):
        """UFRM: test validate function."""
        from natcap.invest import urban_flood_risk_mitigation
        args = self._make_args()
        self.assertEqual(
            len(urban_flood_risk_mitigation.validate(args)), 0)

        del args['workspace_dir']
        validation_warnings = urban_flood_risk_mitigation.validate(args)
        self.assertEqual(len(validation_warnings), 1)

        args['workspace_dir'] = ''
        result = urban_flood_risk_mitigation.validate(args)
        self.assertTrue('has no value' in result[0][1])

        args = self._make_args()
        args['lulc_path'] = 'fake/path/notfound.tif'
        result = urban_flood_risk_mitigation.validate(args)
        self.assertTrue('not found' in result[0][1])

        args = self._make_args()
        args['lulc_path'] = args['aoi_watersheds_path']
        result = urban_flood_risk_mitigation.validate(args)
        self.assertTrue('GDAL raster' in result[0][1])

        args = self._make_args()
        args['aoi_watersheds_path'] = args['lulc_path']
        result = urban_flood_risk_mitigation.validate(args)
        self.assertTrue('GDAL vector' in result[0][1])

        args = self._make_args()
        del args['infrastructure_damage_loss_table_path']
        result = urban_flood_risk_mitigation.validate(args)
        self.assertTrue('missing from the args dict' in result[0][1])
