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



