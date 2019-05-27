# coding=UTF-8
"""Tests for Urban Flood Risk Mitigation Model."""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
from osgeo import osr
import numpy


class UFRMTests(unittest.TestCase):
    """Tests for the Urban Flood Risk Mitigation Model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp(suffix=u'\U0001f60e')  # smiley

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_ufrm_regression(self):
        """UFRM: regression test."""
        from natcap.invest import urban_flood_risk_mitigation

        args = {
            'aoi_watersheds_path': './data/invest-test-dat/ufrm/watersheds.gpkg',
            'built_infrastructure_vector_path': './data/invest-test-dat/ufrm/infrastructure.gpkg',
            'curve_number_table_path': './data/invest-test-dat/ufrm/Biophysical_water_SF.csv',
            'infrastructure_damage_loss_table_path': './data/invest-test-dat/ufrm/Damage.csv',
            'lulc_path': './data/invest-test-dat/ufrm/lulc.tif',
            'rainfall_depth': 40,
            'results_suffix': 'test',
            'soils_hydrological_group_raster_path': './data/invest-test-dat/ufrm/soilgroup.tif',
            'workspace_dir': self.workspace_dir,
        }

        urban_flood_risk_mitigation.execute(args)

        result_vector = gdal.OpenEx(os.path.join(
            args['workspace_dir'], 'flood_risk_service_Test1.shp'))
        result_layer = result_vector.GetLayer()
        result_feature = next(result_layer)
        result_val = result_feature.GetField('serv_bld')
        result_feature  = None
        result_layer = None
        result_vector = None

        # expected result observed from regression run.
        expected_result = 13253548128279.762
        self.assertClose(result_val, expected_result)
