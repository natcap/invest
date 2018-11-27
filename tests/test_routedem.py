"""Module for Regression Testing the InVEST Carbon model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
import numpy
from osgeo import gdal
from osgeo import osr

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'routedem')


class RouteDEMTests(unittest.TestCase):
    """Tests for RouteDEM with Pygeoprocessing 1.x routing API."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def _make_dem(target_path):
        # makes a 10x10 DEM with a valley in the middle that flows to row 0.
        elevation = numpy.arange(1, 2, step=0.1).reshape((10, 1))
        dem_array = numpy.tile(
            numpy.concatenate((
                numpy.flip(numpy.arange(5)),
                numpy.arange(1, 5))),
            (10, 1)) + elevation

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)
        srs_wkt = srs.ExportToWkt()

        driver = gdal.GetDriverByName('GTiff')
        dem_raster = driver.Create(
            target_path, dem_array.shape[1], dem_array.shape[0],
            1, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        dem_raster.SetProjection(srs_wkt)
        dem_band = dem_raster.GetRasterBand(1)
        dem_band.SetNoDataValue(-1)
        dem_band.WriteArray(dem_array)
        dem_geotransform = [2, 2, 0, -2, 0, -2]
        dem_raster.SetGeoTransform(dem_geotransform)
        dem_raster = None

    def test_routedem_invalid_algorithm(self):
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'algorithm': 'invalid',
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'results_suffix': 'foo',
        }

        RouteDEMTests._make_dem(args['dem_path'])
        with self.assertRaises(RuntimeError) as cm:
            routedem.execute(args)

        self.assertTrue('Invalid algorithm specified' in str(cm.exception))

    def test_routedem_d8(self):
        from natcap.invest import routedem
        args = {
            'workspace_dir': self.workspace_dir,
            'algorithm': 'd8',
            'dem_path': os.path.join(self.workspace_dir, 'dem.tif'),
            'results_suffix': 'foo',
            'calculate_flow_accumulation': True,
            'calculate_stream_threshold': True,
            'calculate_downstream_distance': True,
            'calculate_slope': True,
            'threshold_flow_accumulation': 4,
        }

        RouteDEMTests._make_dem(args['dem_path'])

        routedem.execute(args)

        

