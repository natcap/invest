# coding=UTF-8
"""Tests for the Urban Nature Access Model."""
import math
import os
import random
import shutil
import tempfile
import textwrap
import unittest

import numpy
import pygeoprocessing
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
        # the test result
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
                    lulc_pixel_size=target_pixel_size,
                    lulc_bb=pygeoprocessing.get_raster_info(
                        source_population_raster_path)['bounding_box'],
                    lulc_projection_wkt=population_wkt,
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

    def test_dichotomous_decay_simple(self):
        """UNA: Test dichotomous decay on a simple case."""
        from natcap.invest import urban_nature_access

        expected_distance = 5
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')

        urban_nature_access.dichotomous_decay_kernel_raster(
            expected_distance, kernel_filepath)

        expected_array = numpy.array([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=numpy.uint8)

        extracted_kernel_array = pygeoprocessing.raster_to_numpy_array(
            kernel_filepath)
        numpy.testing.assert_array_equal(
            expected_array, extracted_kernel_array)

    def test_dichotomous_decay_large(self):
        """UNA: Test dichotomous decay on a very large pixel radius."""
        from natcap.invest import urban_nature_access

        # kernel with > 268 million pixels.  This is big enough to force my
        # laptop to noticeably hang while swapping memory on an all in-memory
        # implementation.
        expected_distance = 2**13
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')

        urban_nature_access.dichotomous_decay_kernel_raster(
            expected_distance, kernel_filepath)

        expected_shape = (expected_distance*2+1, expected_distance*2+1)
        expected_n_1_pixels = math.pi*expected_distance**2

        kernel_info = pygeoprocessing.get_raster_info(kernel_filepath)
        n_1_pixels = 0
        for _, block in pygeoprocessing.iterblocks((kernel_filepath, 1)):
            n_1_pixels += numpy.count_nonzero(block)

        # 210828417 is only a slight overestimate from the area of the circle
        # at this radius: math.pi*expected_distance**2 = 210828714.13315654
        numpy.testing.assert_allclose(
            n_1_pixels, expected_n_1_pixels, rtol=1e-5)
        self.assertEqual(kernel_info['raster_size'], expected_shape)

    def test_density_decay(self):
        """UNA: Test density decay."""
        from natcap.invest import urban_nature_access

        expected_distance = 200
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')

        urban_nature_access.density_decay_kernel_raster(
            expected_distance, kernel_filepath)

        expected_shape = (expected_distance*2+1,) * 2
        kernel_info = pygeoprocessing.get_raster_info(kernel_filepath)
        kernel_array = pygeoprocessing.raster_to_numpy_array(kernel_filepath)
        self.assertEqual(kernel_info['raster_size'], expected_shape)
        numpy.testing.assert_allclose(
            47123.867,  # obtained from manual inspection
            kernel_array.sum())
        self.assertEqual(0.75, kernel_array.max())
        self.assertEqual(0, kernel_array.min())

    def test_model(self):
        """UNA: Run through the model."""
        from natcap.invest import urban_nature_access

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': 'suffix',
            'population_raster_path': os.path.join(
                self.workspace_dir, 'population.tif'),
            'lulc_raster_path': os.path.join(self.workspace_dir, 'lulc.tif'),
            'lulc_attribute_table': os.path.join(
                self.workspace_dir, 'lulc_attributes.csv'),
            'decay_function': 'gaussian',
            'search_radius': 100.0,  # meters
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

        with open(args['lulc_attribute_table'], 'w') as attr_table:
            attr_table.write(textwrap.dedent(
                """lucode,greenspace
                0,0
                1,1
                2,0
                3,1
                4,0
                5,1
                6,0
                7,1
                8,0
                9,1"""))

        urban_nature_access.execute(args)

        # Since we're doing a semi-manual alignment step, assert that the
        # aligned LULC and population rasters have the same pixel sizes, origin
        # and raster dimensions.
        aligned_lulc_raster_info = pygeoprocessing.get_raster_info(
            os.path.join(args['workspace_dir'], 'intermediate',
                         f"aligned_lulc_{args['results_suffix']}.tif"))
        aligned_population_raster_info = pygeoprocessing.get_raster_info(
            os.path.join(args['workspace_dir'], 'intermediate',
                         f"aligned_population_{args['results_suffix']}.tif"))
        numpy.testing.assert_allclose(
            aligned_lulc_raster_info['pixel_size'],
            aligned_population_raster_info['pixel_size'])
        numpy.testing.assert_allclose(
            aligned_lulc_raster_info['raster_size'],
            aligned_population_raster_info['raster_size'])
        numpy.testing.assert_allclose(
            aligned_lulc_raster_info['geotransform'],
            aligned_population_raster_info['geotransform'])
        numpy.testing.assert_allclose(
            aligned_lulc_raster_info['bounding_box'],
            aligned_population_raster_info['bounding_box'])
