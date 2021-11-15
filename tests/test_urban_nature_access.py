# coding=UTF-8
"""Tests for the Urban Nature Access Model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing
import numpy
from osgeo import gdal
from osgeo import osr

_DEFAULT_ORIGIN = (444720, 3751320)
_DEFAULT_PIXEL_SIZE = (30, -30)
_DEFAULT_EPSG = 3116


class UNATests(unittest.TestCase):
    """Tests for the Urban Nature Access Model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp(suffix='\U0001f60e')  # smiley

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_resample_population_raster(self):
        """UNA: Test population raster resampling."""
        from natcap.invest import urban_nature_access

        source_population_raster_path = os.path.join(
            self.workspace_dir, 'population.tif')
        population_pixel_size = (90, -90)
        population_array_shape = (10, 10)
        population_array = numpy.full(
            population_array_shape, 100, dtype=numpy.uint32)
        population_srs = osr.SpatialReference()
        population_srs.ImportFromEPSG(_DEFAULT_EPSG)
        population_wkt = population_srs.ExportToWkt()
        pygeoprocessing.numpy_array_to_raster(
            base_array=population_array,
            target_nodata=-1,
            pixel_size=population_pixel_size,
            origin=_DEFAULT_ORIGIN,
            projection_wkt=population_wkt,
            target_path=source_population_raster_path)

        for target_pixel_size in (
                (30, -30),  # 1/3 the pixel size
                (4, -4),  # way smaller
                (100, -100)):  # bigger
            target_population_raster_path = os.path.join(
                self.workspace_dir, 'resampled_population.tif')
            urban_nature_access._resample_population_raster(
                source_population_raster_path=source_population_raster_path,
                target_population_raster_path=target_population_raster_path,
                target_pixel_size=target_pixel_size,
                target_bb=pygeoprocessing.get_raster_info(
                    source_population_raster_path)['bounding_box'],
                target_projection_wkt=population_wkt,
                working_dir=os.path.join(self.workspace_dir, 'working'))

            resampled_population_array = pygeoprocessing.raster_to_numpy_array(
                target_population_raster_path)
            numpy.testing.assert_allclose(
                population_array.sum(), resampled_population_array.sum())
