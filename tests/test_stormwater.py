import functools
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy
from osgeo import gdal, ogr, osr
import pandas
import pygeoprocessing


def mock_iterblocks(*args, xoffs=[], xsizes=[], yoffs=[], ysizes=[], **kwargs):
    """Mock function for pygeoprocessing.iterblocks that yields custom blocks.

    Args:
        xoffs (list[int]): list of x-offsets for each block in order
        xsizes (list[int]): list of widths for each block in order
        yoffs (list[int]): list of y-offsets for each block in order
        ysizes (list[int]): list of heights for each block in order

    Yields:
        dictionary with keys 'xoff', 'yoff', 'win_xsize', 'win_ysize'
        that have the same meaning as in pygeoprocessing.iterblocks.
    """
    for yoff, ysize in zip(yoffs, ysizes):
        for xoff, xsize in zip(xoffs, xsizes):
            yield {
                'xoff': xoff, 
                'yoff': yoff,
                'win_xsize': xsize,
                'win_ysize': ysize}

def random_array(shape, low=0, high=1, nodata=None, p_nodata=0.2, precision=2):
    """Generate a random array useful as made-up raster data.

    Args:
        shape (tuple(int)): the shape of the array to make
        low (float): the minimum possible value to include
        high (float): the maximum possible value to include
        nodata (int): If a nodata value is given, set some random elements
            to this nodata value that may be outside the range.
        p_nodata (float): A value in the range [0, 1] representing the
            fraction of elements to make nodata (if a nodata value is provided)
        precision (int): The number of decimal places to include

    Returns:
        numpy.ndarray with the given shape
    """
    magnitude = 10**precision
    # multiplying then dividing by the magnitude gives the desired precision
    array = numpy.random.randint(
        low * magnitude, (high + 1) * magnitude, shape) / magnitude

    if nodata is not None:  # could be zero
        # randomly assign some values to nodata
        nodata_indices = numpy.random.choice(
            [False, True], shape, p=[(1 - p_nodata), p_nodata])
        array[nodata_indices] = nodata
    return array


class StormwaterTests(unittest.TestCase):

    def setUp(self):
        """Create a temp directory for the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def test_basic(self):
        from natcap.invest import stormwater

        # generate 4 unique lucodes
        lucodes = numpy.random.choice(20, size=4, replace=False)
        biophysical_table = pandas.DataFrame()
        biophysical_table['lucode'] = lucodes
        biophysical_table['EMC_pollutant1'] = random_array((4,), high=10)

        # In practice RC_X + IR_X <= 1, but they are independent in the model,
        # so ignoring that constraint for convenience.
        for header in ['RC_A', 'RC_B', 'RC_C', 'RC_D', 'IR_A', 'IR_B', 
                'IR_C', 'IR_D']:
            biophysical_table[header] = random_array((4,))

        biophysical_table = biophysical_table.set_index(['lucode'])
        retention_cost = numpy.random.randint(0, 30)

        lulc_array = numpy.random.choice(lucodes, size=(10, 10)).astype(numpy.int8)
        soil_group_array = numpy.random.choice([1, 2, 3, 4], size=(10, 10)).astype(numpy.int8)
        precipitation_array = random_array((10, 10), high=50)

        lulc_path = os.path.join(self.workspace_dir, 'lulc.tif')
        soil_group_path = os.path.join(self.workspace_dir, 'soil_group.tif')
        precipitation_path = os.path.join(self.workspace_dir, 'precipitation.tif')
        biophysical_table_path = os.path.join(self.workspace_dir, 'biophysical.csv')

        # set up a spatial reference and raster properties
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()
        origin = (-10300000, 5610000)
        pixel_size = (20, -20)
        nodata = -1

        # save each dataset to a file
        for (array, path) in [
            (lulc_array, lulc_path), 
            (soil_group_array, soil_group_path), 
            (precipitation_array, precipitation_path)]:
            pygeoprocessing.numpy_array_to_raster(array, nodata, pixel_size, 
                origin, projection_wkt, path)
        biophysical_table.to_csv(biophysical_table_path)

        args = {
            'workspace_dir': self.workspace_dir,
            'lulc_path': lulc_path,
            'soil_group_path': soil_group_path,
            'precipitation_path': precipitation_path,
            'biophysical_table': biophysical_table_path,
            'adjust_retention_ratios': False,
            'retention_radius': None,
            'road_centerlines_path': None,
            'aggregate_areas_path': None,
            'replacement_cost': retention_cost
        }

        soil_group_codes = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        pixel_area = abs(pixel_size[0] * pixel_size[1])

        stormwater.execute(args)

        retention_volume_path = os.path.join(
            self.workspace_dir, 'retention_volume.tif')
        infiltration_volume_path = os.path.join(
            self.workspace_dir, 'infiltration_volume.tif')
        pollutant_path = os.path.join(
            self.workspace_dir, 'avoided_pollutant_load_pollutant1.tif')
        value_path = os.path.join(self.workspace_dir, 'retention_value.tif')

        retention_raster = gdal.OpenEx(retention_volume_path, gdal.OF_RASTER)
        retention_volume = retention_raster.GetRasterBand(1).ReadAsArray()

        infiltration_raster = gdal.OpenEx(infiltration_volume_path, gdal.OF_RASTER)
        infiltration_volume = infiltration_raster.GetRasterBand(1).ReadAsArray()

        avoided_pollutant_raster = gdal.OpenEx(pollutant_path, gdal.OF_RASTER)
        avoided_pollutant_load = avoided_pollutant_raster.GetRasterBand(1).ReadAsArray()

        retention_value_raster = gdal.OpenEx(value_path, gdal.OF_RASTER)
        retention_value = retention_value_raster.GetRasterBand(1).ReadAsArray()

        for row in range(retention_volume.shape[0]):
            for col in range(retention_volume.shape[1]):

                soil_group = soil_group_array[row, col]
                lulc = lulc_array[row, col]
                precipitation = precipitation_array[row, col]

                rc_value = biophysical_table[f'RC_{soil_group_codes[soil_group]}'][lulc]

                # precipitation (mm/yr) * 0.001 (m/mm) * pixel area (m^2) = m^3/yr
                volume = retention_volume[row, col]
                expected_volume = (1 - rc_value) * precipitation * 0.001 * pixel_area
                self.assertTrue(numpy.isclose(volume, expected_volume), 
                    f'values: {volume}, {expected_volume}')

                # retention (m^3/yr) * cost ($/m^3) = value ($/yr)
                value = retention_value[row, col]
                expected_value = expected_volume * retention_cost
                self.assertTrue(numpy.isclose(value, expected_value), 
                    f'values: {value}, {expected_value}')

        for row in range(infiltration_volume.shape[0]):
            for col in range(infiltration_volume.shape[1]):
                
                soil_group = soil_group_array[row][col]
                lulc = lulc_array[row][col]
                precipitation = precipitation_array[row][col]

                ir_value = biophysical_table[f'IR_{soil_group_codes[soil_group]}'][lulc]

                # precipitation (mm/yr) * 0.001 (m/mm) * pixel area (m^2) = m^3
                expected_volume = (ir_value) * precipitation * 0.001 * pixel_area
                self.assertTrue(
                    numpy.isclose(infiltration_volume[row][col], expected_volume),
                    f'values: {infiltration_volume[row][col]}, {expected_volume}')

        for row in range(avoided_pollutant_load.shape[0]):
            for col in range(avoided_pollutant_load.shape[1]):
                
                lulc = lulc_array[row, col]
                retention = retention_volume[row, col]
                emc = biophysical_table['EMC_pollutant1'][lulc]

                # retention (m^3/yr) * emc (mg/L) * 1000 (L/m^3) * 0.000001 (kg/mg) = kg/yr
                avoided_load = avoided_pollutant_load[row, col]
                expected_avoided_load = retention * emc * 0.001
                self.assertTrue(numpy.isclose(avoided_load, expected_avoided_load),
                    f'values: {avoided_load}, {expected_avoided_load}')

    def test_threshold_array(self):
        from natcap.invest import stormwater

        x_size, y_size = 5, 5
        threshold = numpy.random.rand()  # a random value in [0, 1)
        array = random_array((y_size, x_size), nodata=stormwater.NODATA)

        out = stormwater.threshold_array(array, threshold)

        for y in range(y_size):
            for x in range(x_size):
                if array[x,y] == stormwater.NODATA:
                    self.assertEqual(out[x,y], stormwater.NODATA)
                elif array[x,y] > threshold:
                    self.assertEqual(out[x,y], 0)
                else:
                    self.assertEqual(out[x,y], 1)

    def test_ratio_op(self):
        from natcap.invest import stormwater

        sorted_lucodes = [10, 11, 12, 13]
        lulc_array = numpy.array([
            [13, 12],
            [11, 10]])
        soil_group_array = numpy.array([
            [4, 4],
            [2, 2]])
        ratio_array = numpy.array([
            [0.11, 0.12, 0.13, 0.14],
            [0.21, 0.22, 0.23, 0.24],
            [0.31, 0.32, 0.33, 0.34],
            [0.41, 0.42, 0.43, 0.44]])
        expected_ratios = numpy.array([
            [0.44, 0.34],
            [0.22, 0.12]])
        output_ratios = stormwater.ratio_op(
            lulc_array, soil_group_array, ratio_array, sorted_lucodes)
        self.assertTrue(numpy.array_equal(expected_ratios, output_ratios),
            f'Expected:\n{expected_ratios}\nActual:\n{output_ratios}')

    def test_volume_op(self):
        from natcap.invest import stormwater

        x_size, y_size = 5, 5
        ratio_array = random_array((y_size, x_size), nodata=stormwater.NODATA)
        precip_nodata = -1 * numpy.random.rand()
        precip_array = random_array((y_size, x_size), high=100, nodata=precip_nodata)
        pixel_area = numpy.random.rand() * 1000

        out = stormwater.volume_op(ratio_array, precip_array, precip_nodata, 
            pixel_area)
        # precip (mm/yr) * area (m^2) * 0.001 (m/mm) * ratio = volume (m^3/yr)
        for y in range(y_size):
            for x in range(x_size):
                if (ratio_array[y,x] == stormwater.NODATA or 
                        precip_array[y,x] == precip_nodata):
                    self.assertEqual(out[y,x], stormwater.NODATA)
                else:
                    self.assertTrue(numpy.isclose(out[y,x], 
                        precip_array[y,x] * ratio_array[y,x] * pixel_area / 1000))

    def test_avoided_pollutant_load_op(self):
        from natcap.invest import stormwater

        shape = 5, 5
        lulc_nodata = -1
        lulc_array = random_array(shape, nodata=lulc_nodata, high=3, 
            precision=0).astype(int)
        retention_volume_array = random_array(shape, nodata=stormwater.NODATA, 
            high=1000)
        sorted_lucodes = numpy.array([0, 1, 2, 3])
        emc_array = random_array((4,), high=10)

        out = stormwater.avoided_pollutant_load_op(lulc_array, lulc_nodata,
            retention_volume_array, sorted_lucodes, emc_array)
        for y in range(shape[0]):
            for x in range(shape[1]):
                if (lulc_array[y,x] == lulc_nodata or 
                        retention_volume_array[y,x] == stormwater.NODATA):
                    self.assertEqual(out[y,x], stormwater.NODATA)
                else:
                    emc_value = emc_array[lulc_array[y,x]]
                    expected = emc_value * retention_volume_array[y,x] / 1000
                    self.assertTrue(numpy.isclose(out[y,x], expected),
                        f'Expected: {expected} Actual: {out[y,x]}')

    def test_retention_value_op(self):
        from natcap.invest import stormwater

        shape = 5, 5
        retention_volume_array = random_array(shape, nodata=stormwater.NODATA, 
            high=1000)
        replacement_cost = numpy.random.rand() * 20

        out = stormwater.retention_value_op(retention_volume_array, 
            replacement_cost)
        for y in range(shape[0]):
            for x in range(shape[1]):
                if (retention_volume_array[y,x] == stormwater.NODATA):
                    self.assertEqual(out[y,x], stormwater.NODATA)
                else:
                    self.assertEqual(out[y,x], 
                        retention_volume_array[y,x] * replacement_cost)

    def test_impervious_op(self):
        from natcap.invest import stormwater

        shape = 5, 5
        lulc_nodata = -1
        lulc_array = random_array(shape, nodata=lulc_nodata, high=3, 
            precision=0).astype(int)
        sorted_lucodes = numpy.array([0, 1, 2, 3])
        impervious_lookup_array = random_array((4,), precision=0) # 0s and 1s

        out = stormwater.impervious_op(lulc_array, lulc_nodata, sorted_lucodes,
            impervious_lookup_array)
        for y in range(shape[0]):
            for x in range(shape[1]):
                if (lulc_array[y,x] == lulc_nodata):
                    self.assertEqual(out[y,x], stormwater.NODATA)
                else:
                    is_impervious = impervious_lookup_array[lulc_array[y,x]]
                    self.assertEqual(out[y,x], is_impervious)

    def test_adjust_op(self):
        from natcap.invest import stormwater

        shape = 10, 10
        ratio_array = random_array(shape, nodata=stormwater.NODATA)
         # these are obv not averages from the above array but
        # it doesn't matter for this test
        avg_ratio_array = random_array(shape, nodata=stormwater.NODATA)
        # boolean 0/1 arrays
        near_impervious_lulc_array = random_array(shape, precision=0, 
            nodata=stormwater.NODATA).astype(int)
        near_road_centerline_array = random_array(shape, precision=0, 
            nodata=stormwater.NODATA).astype(int)

        out = stormwater.adjust_op(ratio_array, avg_ratio_array, 
            near_impervious_lulc_array, near_road_centerline_array)
        for y in range(shape[0]):
            for x in range(shape[1]):
                if (ratio_array[y,x] == stormwater.NODATA or
                    avg_ratio_array[y,x] == stormwater.NODATA or
                    near_impervious_lulc_array[y,x] == stormwater.NODATA or
                    near_road_centerline_array[y,x] == stormwater.NODATA):
                    self.assertEqual(out[y,x], stormwater.NODATA)
                else:
                    # equation 2-4: Radj_ij = R_ij + (1 - R_ij) * C_ij
                    adjust_factor = (
                        0 if (
                            near_impervious_lulc_array[y,x] or 
                            near_road_centerline_array[y,x]
                        ) else avg_ratio_array[y,x])
                    adjusted = (
                        ratio_array[y,x] + (1 - ratio_array[y,x]) * adjust_factor)
                    self.assertEqual(out[y,x], adjusted)

    def test_is_near(self):
        from natcap.invest import stormwater
        is_connected_array = numpy.array([
            [0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=numpy.uint8)
        search_kernel = numpy.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=numpy.uint8)
        # convolution sum array:
        # 1, 1, 2, 1, 0, 0
        # 1, 1, 2, 1, 0, 1
        # 1, 0, 1, 0, 1, 1
        # expected is_near array: sum > 0
        expected = numpy.array([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1]
        ])

        connected_path = os.path.join(self.workspace_dir, 'connected.tif')
        out_path = os.path.join(self.workspace_dir, 'near_connected.tif')

        # set up an arbitrary spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()
        pygeoprocessing.numpy_array_to_raster(is_connected_array, -1, 
            (10, -10), (0, 0), projection_wkt, connected_path)

        mocked = functools.partial(mock_iterblocks, 
            yoffs=[0], ysizes=[3], xoffs=[0, 3], xsizes=[3, 3])
        with mock.patch('natcap.invest.stormwater.pygeoprocessing.iterblocks', 
                mocked):
            stormwater.is_near(connected_path, search_kernel, out_path)
            actual = pygeoprocessing.raster_to_numpy_array(out_path)
            self.assertTrue(numpy.array_equal(expected, actual))

    def test_overlap_iterblocks(self):
        from natcap.invest import stormwater

        raster_path = os.path.join(self.workspace_dir, 'iterblocks_array.tif')
        array = numpy.array([
            [1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 2],
            [3, 3, 3, 3, 3, 4]
        ], dtype=numpy.int8)

        # set up an arbitrary spatial reference and save the array to a raster
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()
        pygeoprocessing.numpy_array_to_raster(
            array, -1, (10, -10), (0, 0), projection_wkt, raster_path)

        mocked = functools.partial(mock_iterblocks,
            xoffs=[0], xsizes=[6], yoffs=[0], ysizes=[6])
        with mock.patch(
            'natcap.invest.stormwater.pygeoprocessing.iterblocks', 
            mocked):
            blocks = list(stormwater.overlap_iterblocks(raster_path, 2))
            self.assertEqual(blocks, [{
                'xoff': 0, 'yoff': 0, 'xsize': 6, 'ysize': 6,
                'top_overlap': 0, 'left_overlap': 0, 'bottom_overlap': 0, 'right_overlap': 0,
                'top_padding': 2, 'left_padding': 2, 'bottom_padding': 2, 'right_padding': 2
            }])

        mocked = functools.partial(mock_iterblocks,
            xoffs=[0, 5], xsizes=[5, 1], yoffs=[0, 5], ysizes=[5, 1])
        with mock.patch('natcap.invest.stormwater.pygeoprocessing.iterblocks', 
                mocked):
            blocks = list(stormwater.overlap_iterblocks(raster_path, 2))

            self.assertEqual(blocks, [{
                'xoff': 0, 'yoff': 0, 'xsize': 5, 'ysize': 5,
                'top_overlap': 0, 'left_overlap': 0, 'bottom_overlap': 1, 'right_overlap': 1,
                'top_padding': 2, 'left_padding': 2, 'bottom_padding': 1, 'right_padding': 1
            },
            {
                'xoff': 5, 'yoff': 0, 'xsize': 1, 'ysize': 5,
                'top_overlap': 0, 'left_overlap': 2, 'bottom_overlap': 1, 'right_overlap': 0,
                'top_padding': 2, 'left_padding': 0, 'bottom_padding': 1, 'right_padding': 2
            },
            {
                'xoff': 0, 'yoff': 5, 'xsize': 5, 'ysize': 1,
                'top_overlap': 2, 'left_overlap': 0, 'bottom_overlap': 0, 'right_overlap': 1,
                'top_padding': 0, 'left_padding': 2, 'bottom_padding': 2, 'right_padding': 1
            },
            {
                'xoff': 5, 'yoff': 5, 'xsize': 1, 'ysize': 1,
                'top_overlap': 2, 'left_overlap': 2, 'bottom_overlap': 0, 'right_overlap': 0,
                'top_padding': 0, 'left_padding': 0, 'bottom_padding': 2, 'right_padding': 2
            }])

    def test_make_search_kernel(self):
        from natcap.invest import stormwater

        array = numpy.zeros((10, 10))
        path = os.path.join(self.workspace_dir, 'make_search_kernel.tif')
        # set up an arbitrary spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()
        pygeoprocessing.numpy_array_to_raster(array, -1, (10, -10), 
            (0, 0), projection_wkt, path)

        expected_5 = numpy.array([[1]])
        actual_5 = stormwater.make_search_kernel(path, 5)
        self.assertTrue(numpy.array_equal(expected_5, actual_5))

        expected_9 = numpy.array([[1]])
        actual_9 = stormwater.make_search_kernel(path, 9)
        self.assertTrue(numpy.array_equal(expected_9, actual_9))

        expected_10 = numpy.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]])
        actual_10 = stormwater.make_search_kernel(path, 10)
        self.assertTrue(numpy.array_equal(expected_10, actual_10))

        expected_15 = numpy.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])
        actual_15 = stormwater.make_search_kernel(path, 15)
        self.assertTrue(numpy.array_equal(expected_15, actual_15))

    def test_make_coordinate_rasters(self):
        from natcap.invest import stormwater

        # set up an array (values don't matter) and save as raster
        array = numpy.zeros((512, 512), dtype=numpy.int8)
        pixel_size = (10, -10)
        origin = (15100, 7000)
        raster_path = os.path.join(self.workspace_dir, 'input_array.tif')
        # set up an arbitrary spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()
        pygeoprocessing.numpy_array_to_raster(
            array, -1, pixel_size, origin, projection_wkt, raster_path)

        x_coord_path = os.path.join(self.workspace_dir, 'x_coords.tif')
        y_coord_path = os.path.join(self.workspace_dir, 'y_coords.tif')
        
        # make x- and y- coordinate rasters from the input raster and open them
        stormwater.make_coordinate_rasters(raster_path, 
            x_coord_path, y_coord_path)
        x_coords = pygeoprocessing.raster_to_numpy_array(x_coord_path)
        y_coords = pygeoprocessing.raster_to_numpy_array(y_coord_path)

        # coords should start at the raster origin plus 1/2 a pixel
        x_expected = origin[0] + pixel_size[0] / 2
        y_expected = origin[1] + pixel_size[1] / 2
        first_x_coords_row = x_coords[0]
        first_y_coords_col = y_coords[:,0]

        # x coords should increment by one pixel width
        for x_value in first_x_coords_row:
            self.assertEqual(x_value, x_expected)
            x_expected += pixel_size[0]
        # each row of x_coords should be identical
        for row in x_coords:
            self.assertTrue(numpy.array_equal(row, first_x_coords_row))

        # y coords should increment by one pixel height
        for y_value in first_y_coords_col:
            self.assertEqual(y_value, y_expected)
            y_expected += pixel_size[1]
        # each column of y_coords should be identical
        for col in y_coords.T:
            self.assertTrue(numpy.array_equal(col, first_y_coords_col))

    def test_nearest_linestring(self):
        from natcap.invest import stormwater

        base_path = os.path.join(self.workspace_dir, 'coord_base.tif')
        x_coords_path = os.path.join(self.workspace_dir, 'x_coords.tif')
        y_coords_path = os.path.join(self.workspace_dir, 'y_coords.tif')
        linestring_path = os.path.join(self.workspace_dir, 'linestring.gpkg')
        out_path = os.path.join(self.workspace_dir, 'distance.tif')
        expected_distance_path = os.path.join(self.workspace_dir, 'expected_distance.tif')
        actual_distance_path = os.path.join(self.workspace_dir, 'actual_distance.tif')

        # set up an arbitrary spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()
        raster_size = 1000, 1000
        n_points = 10
        nodata = -1
        radius = numpy.random.randint(1, 1000)
        pixel_size = numpy.random.randint(1, 500)
        origin = (numpy.random.randint(-1000, 1000), numpy.random.randint(-1000, 1000))

        base_array = numpy.zeros(raster_size)
        pygeoprocessing.numpy_array_to_raster(base_array, nodata, 
            (pixel_size, pixel_size), origin, projection_wkt, base_path)
        stormwater.make_coordinate_rasters(base_path, x_coords_path, y_coords_path)

        # Randomly generate some points to make a linestring
        linestring = ogr.Geometry(ogr.wkbLineString)
        for i in range(n_points):
            # make the linestring on the same region and scale as the raster,
            # but not necessarily perfectly overlapping
            linestring.AddPoint(
                numpy.random.randint(-1000*pixel_size, 1000*pixel_size), 
                numpy.random.randint(-1000*pixel_size, 1000*pixel_size))
        # Write the linestring to a gpkg file
        driver = ogr.GetDriverByName('GPKG')
        vector = driver.CreateDataSource(linestring_path)
        layer = vector.CreateLayer("1", geom_type=ogr.wkbLineString)
        feature_def = layer.GetLayerDefn()
        feature = ogr.Feature(feature_def)
        feature.SetGeometry(linestring)
        layer.CreateFeature(feature)
        feature, layer, vector = None, None, None
        
        # This is a simpler implementation of the distance algorithm that
        # doesn't have the spatial index optimization.
        # Note that this and the optimized algorithm both rely on 
        # `iter_linestring_segments` and `line_distance`, which are
        # tested separately.
        # This just tests that the optimized and non-optimized version,
        # when thresholded, give the same result.
        def nearest_linestring_op(x_coords, y_coords, linestring_path):
            """Calculate the distance from each pixel centerpoint to the nearest
            linestring in a vector. This is intended to be used with raster_calculator.

            Args:
                x_coords (numpy.ndarray): 2D array where each pixel value is the 
                    x coordinate of that pixel in the raster coordinate system
                y_coords (numpy.ndarray): 2D array where each pixel value is the 
                    y coordinate of that pixel in the raster coordinate system
                linestring_path (str): path to a linestring/multilinestring vector

            Returns:
                2D numpy.ndarray of the same shape as `x_coords` and `y_coords`.
                Each pixel value is the distance from that pixel's centerpoint to
                the nearest linestring in the vector at ``linestring_path``.
            """
            segment_generator = stormwater.iter_linestring_segments(linestring_path)
            (x1, y1), (x2, y2) = next(segment_generator)
            min_distance = stormwater.line_distance(x_coords, y_coords, x1, y1, x2, y2)

            for (x1, y1), (x2, y2) in segment_generator:
                if x2 == x1 and y2 == y1:
                    continue  # ignore lines with length 0
                distance = stormwater.line_distance(x_coords, y_coords, x1, y1, x2, y2)
                min_distance = numpy.minimum(min_distance, distance)
            return min_distance

        pygeoprocessing.raster_calculator([
                (x_coords_path, 1), 
                (y_coords_path, 1), 
                (linestring_path, 'raw')
            ], nearest_linestring_op, expected_distance_path, gdal.GDT_Float32, nodata)
        stormwater.optimized_linestring_distance(x_coords_path, y_coords_path, 
            linestring_path, radius, actual_distance_path)

        # these are not the same
        expected_distance_array = pygeoprocessing.raster_to_numpy_array(expected_distance_path)
        actual_distance_array = pygeoprocessing.raster_to_numpy_array(actual_distance_path)
        # these should be the same
        expected_thresholded = stormwater.threshold_array(expected_distance_array, radius)
        actual_thresholded = stormwater.threshold_array(actual_distance_array, radius)
        # the expected will not have nodata areas. the actual will because of the
        # optimization skipping some blocks. treat nodata as 0 for this purpose.
        actual_thresholded[actual_thresholded == nodata] = 0
        self.assertTrue(numpy.array_equal(expected_thresholded, actual_thresholded))

    def test_line_distance(self):
        from natcap.invest import stormwater

        x_start, y_start = 100, -100
        x_stop, y_stop = 200, 0

        width, height = 11, 11

        # Generate x- and y- coordinate arrays of shape (height, width)
        x_coord_series = numpy.linspace(x_start, x_stop, num=width)
        y_coord_series = numpy.linspace(y_start, y_stop, num=height)
        # repeat the x-coord series for each row in height
        x_coords = numpy.tile(x_coord_series, (height, 1))
        # repeat the y-coord series for each column in width
        y_coords = numpy.tile(numpy.array([y_coord_series]).T, (1, width))  

        # this is a line segment that's below the coordinate block and
        # parallel to the y-axis. so for all points, their shortest distance 
        # will be the distance to the point (x1, y1).
        (x1, y1), (x2, y2) = (150, 50), (150, 100)
        distances = stormwater.line_distance(
            x_coords, y_coords, x1, y1, x2, y2)
        expected_distances = numpy.hypot(x_coords - x1, y_coords - y1)
        self.assertTrue(numpy.allclose(distances, expected_distances),
            f'Expected:\n{expected_distances}\nActual:\n{distances}')

        # this segment is to the left of the coordinate block and parallel to 
        # the x-axis, so for all points, their shortest distance will be the 
        # distance to the point (x2, y2).
        (x1, y1), (x2, y2) = (0, 50), (50, 50)
        distances = stormwater.line_distance(
            x_coords, y_coords, x1, y1, x2, y2)
        expected_distances = numpy.hypot(x_coords - x2, y_coords - y2)
        self.assertTrue(numpy.allclose(distances, expected_distances),
            f'Expected:\n{expected_distances}\nActual:\n{distances}')

        # this is a line segment that goes diagonally through the cooridnate
        # block. (must be a square)
        (x1, y1), (x2, y2) = (x_start, y_start), (x_stop, y_stop)
        distances = stormwater.line_distance(
            x_coords, y_coords, x1, y1, x2, y2)
        # in a square grid, any point (x, y)'s distance from the diagonal is
        # |y - x| * sqrt(2) / 2
        expected_distances = (numpy.abs(numpy.abs(y_coords - y_start) - 
            numpy.abs(x_coords - x_start)) * math.sqrt(2) / 2)
        self.assertTrue(numpy.allclose(distances, expected_distances),
            f'Expected:\n{expected_distances}\nActual:\n{distances}')

    def test_iter_linestring_segments(self):
        from natcap.invest import stormwater
        # Create a linestring vector
        path = os.path.join(self.workspace_dir, 'linestring.gpkg')
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(3857)
        driver = gdal.GetDriverByName('GPKG')
        linestring_vector = driver.Create(path, 0, 0, 0, gdal.GDT_Unknown)
        layer = linestring_vector.CreateLayer('linestring', 
            spatial_reference, ogr.wkbLineString)
        layer_defn = layer.GetLayerDefn()
        layer.StartTransaction()
        linestring = ogr.Geometry(ogr.wkbLineString)
        # Create a linestring from the list of coords and save it to the vector
        coords = [(100, 1), (105.5, 2), (-7, 0)]
        for coord in coords:
            linestring.AddPoint(*coord)
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(linestring)
        layer.CreateFeature(feature)
        layer.CommitTransaction()
        layer = None
        linestring_vector = None

        # Expect the coordinate pairs are yielded in order
        expected_pairs = [(coords[0], coords[1]), (coords[1], coords[2])]
        output_pairs = list(stormwater.iter_linestring_segments(path))
        self.assertEqual(expected_pairs, output_pairs)

    def test_raster_average(self):
        from natcap.invest import stormwater

        array = numpy.empty((150, 150))
        nodata = -1
        array[:, 0:128] = 10
        array[:, 128:149] = 20
        array[:, 149] = nodata

        data_path = os.path.join(self.workspace_dir, 'data.tif')
        # set up an arbitrary spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()
        pygeoprocessing.numpy_array_to_raster(array, nodata, (10, -10), 
            (0, 0), projection_wkt, data_path)

        search_kernel = numpy.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]])
        n_values_path = os.path.join(self.workspace_dir, 'n_values.tif')
        sum_path = os.path.join(self.workspace_dir, 'sum.tif')
        average_path = os.path.join(self.workspace_dir, 'average.tif')

        stormwater.raster_average(data_path, search_kernel, n_values_path, 
            sum_path, average_path)

        n_values_array = pygeoprocessing.raster_to_numpy_array(n_values_path)
        sum_array = pygeoprocessing.raster_to_numpy_array(sum_path)
        average_array = pygeoprocessing.raster_to_numpy_array(average_path)

        expected_n_values = numpy.full((150, 150), 5)
        expected_n_values[0] = 4
        expected_n_values[-1] = 4
        expected_n_values[:, 0] = 4
        expected_n_values[:, -2] = 4
        expected_n_values[:, -1] = nodata
        expected_n_values[0, 0] = 3
        expected_n_values[0, -2] = 3
        expected_n_values[-1, 0] = 3
        expected_n_values[-1, -2] = 3
        self.assertTrue(numpy.array_equal(n_values_array, expected_n_values))

        expected_average = numpy.empty((150, 150))
        expected_average[:, 0:127] = 10
        expected_average[:, 127] = 12
        expected_average[0, 127] = 12.5
        expected_average[-1, 127] = 12.5
        expected_average[:, 128] = 18
        expected_average[0, 128] = 17.5
        expected_average[-1, 128] = 17.5
        expected_average[:, 129:149] = 20
        expected_average[:, 149] = -1
        self.assertTrue(numpy.allclose(average_array, expected_average))
