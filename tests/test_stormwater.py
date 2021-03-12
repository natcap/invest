import math
import numpy
import os
import pandas
import pygeoprocessing
import random
import tempfile
import unittest

from osgeo import gdal, osr


def random_ratios(k, precision=2):
    """Return k random numbers in the range [0, 1] inclusive.

    Args:
        k (int): number of numbers to generate
        precision (int): how many decimal places to include

    Returns:
        list[float] of length k
    """
    magnitude = 10**precision
    return [n/magnitude for n in random.choices(range(0, magnitude + 1), k=k)]

class StormwaterTests(unittest.TestCase):

    def setUp(self):
        """Create a temp directory for the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def test_basic(self):
        from natcap.invest import stormwater

        biophysical_table = pandas.DataFrame()
        lucodes = random.sample(range(0, 20), 4)
        biophysical_table['lucode'] = lucodes
        biophysical_table['EMC_pollutant1'] = [
            round(random.uniform(0, 5), 2) for _ in range(4)]

        # In practice RC_X + IR_X <= 1, but they are independent in the model,
        # so ignoring that constraint for convenience.
        for header in ['RC_A', 'RC_B', 'RC_C', 'RC_D', 'IR_A', 'IR_B', 
                'IR_C', 'IR_D']:
            biophysical_table[header] = random_ratios(4)

        biophysical_table = biophysical_table.set_index(['lucode'])

        retention_cost = numpy.random.randint(0, 30)


        print(biophysical_table) 

        lulc_array = numpy.random.choice(lucodes, size=(10, 10)).astype(numpy.int8)
        soil_group_array = numpy.random.choice([1, 2, 3, 4], size=(10, 10)).astype(numpy.int8)
        precipitation_array = (numpy.random.random((10, 10)).round() * 50).astype(float)

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


    def test_adjust_op(self):
        from natcap.invest import stormwater

        retention_ratio_array = numpy.array([
            [0.6, 0.4, 0.4, 0.2, 0.2],
            [0.6, 0.4, 0.4, 0.2, 0.2],
            [0.6, 0.4, 0.4, 0.2, 0.2],
            [0.6, 0.4, 0.4, 0.2, 0.2],
            [0.6, 0.4, 0.4, 0.2, 0.2]
        ])
        impervious_array = numpy.array([
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, True]
        ])
        distance_array = numpy.array([
            [20, 20, 20, 20, 10],
            [20, 20, 20, 20, 20],
            [30, 30, 30, 30, 30],
            [40, 40, 40, 40, 40],
            [50, 50, 50, 50, 50]
        ])
        search_kernel = numpy.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        # the diagonal band comes from applying the search kernel to
        # each pixel along the diagonal where impervious_array == True
        # the top right value is 1 because that's the only value in 
        # distance_array that's <= the radius
        is_connected = numpy.array([
            [True, True, False, False, True],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
            [False, False, False, True, True]
        ])
        radius = 10

        # The average of the pixels within the search kernel for each pixel in 
        # retention_ratio_array. 
        # sum of values in the search kernel / # of values in the search kernel
        sum_kernel_values = avg_ratios = numpy.array([
            [1.6, 1.8, 1.4, 1.0, 0.6],
            [2.2, 2.2, 1.8, 1.2, 0.8],
            [2.2, 2.2, 1.8, 1.2, 0.8],
            [2.2, 2.2, 1.8, 1.2, 0.8],
            [1.6, 1.8, 1.4, 1.2, 0.6]
        ])
        n_kernel_values = avg_ratios = numpy.array([
            [3, 4, 4, 4, 3],
            [4, 5, 5, 5, 4],
            [4, 5, 5, 5, 4],
            [4, 5, 5, 5, 4],
            [3, 4, 4, 4, 3]
        ])
        avg_ratios = sum_kernel_values / n_kernel_values

        # C_ij is 0 if pixel (i, j) is not connected; 
        # average of surrounding pixels otherwise
        adjustment_factors = avg_ratios * ~is_connected
        # equation 2-4: Radj_ij = R_ij + (1 - R_ij) * C_ij
        expected_adjusted = (retention_ratio_array + 
            (1 - retention_ratio_array) * adjustment_factors)
        actual_adjusted = stormwater.adjust_op(retention_ratio_array, 
            impervious_array, distance_array, search_kernel, radius)
        self.assertTrue(numpy.allclose(expected_adjusted, actual_adjusted), 
            f'Expected:\n{expected_adjusted}\nActual:\n{actual_adjusted}')

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

    def test_line_distance_op(self):
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

        print(x_coords, y_coords)

        # this is a line segment that's below the coordinate block and
        # parallel to the y-axis. so for all points, their shortest distance 
        # will be the distance to the point (x1, y1).
        (x1, y1), (x2, y2) = (150, 50), (150, 100)
        distances = stormwater.line_distance_op(
            x_coords, y_coords, x1, y1, x2, y2)
        expected_distances = numpy.hypot(x_coords - x1, y_coords - y1)
        self.assertTrue(numpy.allclose(distances, expected_distances),
            f'Expected:\n{expected_distances}\nActual:\n{distances}')

        # this segment is to the left of the coordinate block and parallel to 
        # the x-axis, so for all points, their shortest distance will be the 
        # distance to the point (x2, y2).
        (x1, y1), (x2, y2) = (0, 50), (50, 50)
        distances = stormwater.line_distance_op(
            x_coords, y_coords, x1, y1, x2, y2)
        expected_distances = numpy.hypot(x_coords - x2, y_coords - y2)
        self.assertTrue(numpy.allclose(distances, expected_distances),
            f'Expected:\n{expected_distances}\nActual:\n{distances}')

        # this is a line segment that goes diagonally through the cooridnate
        # block. (must be a square)
        (x1, y1), (x2, y2) = (x_start, y_start), (x_stop, y_stop)
        distances = stormwater.line_distance_op(
            x_coords, y_coords, x1, y1, x2, y2)
        # in a square grid, any point (x, y)'s distance from the diagonal is
        # |y - x| * sqrt(2) / 2
        expected_distances = (numpy.abs(numpy.abs(y_coords - y_start) - 
            numpy.abs(x_coords - x_start)) * math.sqrt(2) / 2)
        self.assertTrue(numpy.allclose(distances, expected_distances),
            f'Expected:\n{expected_distances}\nActual:\n{distances}')


    def test_iter_linestring_segments(self):
        from natcap.invest import stormwater

        coords = [(100, 1), (105, 2), (-7, 0)]
        linestring_path = os.path.join(self.workspace_dir, 'linestring.shp')

        expected_pairs = [
            (coords[0], coords[1]),
            (coords[1], coords[2])
        ]

        driver = gdal.GetDriverByName('GPKG')
        linestring_vector = driver.Create(linestring_path, 0, 0, 0, 
            gdal.GDT_Unknown)
        layer = linestring_vector.CreateLayer(
            'linestring', points_layer.GetSpatialRef(), ogr.OGRwkbGeometryType.wkbLineString)
        snapped_layer.CreateFields(points_layer.schema)
        snapped_layer_defn = snapped_layer.GetLayerDefn()

        snapped_layer.StartTransaction()
        stormwater





