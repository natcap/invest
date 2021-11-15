# coding=UTF-8
"""Tests for the Urban Nature Access Model."""
import unittest
import tempfile
import shutil
import os
import random

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

        random.seed(-1)  # for our random number generation

        source_population_raster_path = os.path.join(
            self.workspace_dir, 'population.tif')
        population_pixel_size = (90, -90)
        population_array_shape = (10, 10)

        array_of_100s = numpy.full(
            population_array_shape, 100, dtype=numpy.uint32)
        array_of_random_ints = numpy.array(
            random.choices(range(0, 100), k=100),
            dtype=numpy.uint32).reshape(population_array_shape)

        for population_array in (
                array_of_100s, array_of_random_ints):
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
                    source_population_raster_path,
                    target_population_raster_path,
                    target_pixel_size=target_pixel_size,
                    target_bb=pygeoprocessing.get_raster_info(
                        source_population_raster_path)['bounding_box'],
                    target_projection_wkt=population_wkt,
                    working_dir=os.path.join(self.workspace_dir, 'working'))

                resampled_population_array = (
                    pygeoprocessing.raster_to_numpy_array(
                        target_population_raster_path))

                # There should be no significant loss or gain of population due
                # to warping, but the fact that this is aggregating across the
                # whole raster (lots of pixels) means we need to lower the
                # relative tolerance.
                numpy.testing.assert_allclose(
                    population_array.sum(), resampled_population_array.sum(),
                    rtol=1e-3)

    def test_model(self):
        """UNA: Run through the model."""
        from natcap.invest import urban_nature_access

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'suffix',
            'population_raster_path': os.path.join(
                self.workspace_dir, 'population.tif'),
            'lulc_raster_path': os.path.join(self.workspace_dir, 'lulc.tif'),
        }

        random.seed(-1)  # for our random number generation
        population_pixel_size = (90, -90)
        population_array_shape = (10, 10)
        population_array = numpy.array(
            random.choices(range(0, 100), k=100),
            dtype=numpy.int32).reshape(population_array_shape)
        population_srs = osr.SpatialReference()
        population_srs.ImportFromEPSG(_DEFAULT_EPSG)
        population_wkt = population_srs.ExportToWkt()
        pygeoprocessing.numpy_array_to_raster(
            base_array=population_array,
            target_nodata=-1,
            pixel_size=population_pixel_size,
            origin=_DEFAULT_ORIGIN,
            projection_wkt=population_wkt,
            target_path=args['population_raster_path'])

        lulc_pixel_size = _DEFAULT_PIXEL_SIZE
        lulc_array_shape = (30, 30)
        lulc_array = numpy.array(
            random.choices(range(0, 10), k=900),
            dtype=numpy.int32).reshape(lulc_array_shape)
        pygeoprocessing.numpy_array_to_raster(
            base_array=lulc_array,
            target_nodata=-1,
            pixel_size=lulc_pixel_size,
            origin=_DEFAULT_ORIGIN,
            projection_wkt=population_wkt,
            target_path=args['lulc_raster_path'])

        urban_nature_access.execute(args)

        # TODO: Assertions will be added here later when I have something to
        # test.
