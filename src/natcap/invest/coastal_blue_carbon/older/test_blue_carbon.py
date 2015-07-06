import unittest
import os
import pprint
import tempfile

from numpy import testing
import numpy as np
import gdal
import osr
import rasterio as rio
import tempfile

import blue_carbon

input_dir = '../../test/invest-data/BlueCarbon/input'
input2_dir = '../../test/invest-data/BlueCarbon/input_2'
pp = pprint.PrettyPrinter(indent=4)


# class TestOverallModel1(unittest.TestCase):
#     def setUp(self):
#         workspace_dir = tempfile.mkdtemp("subdir")
#         # filepath = os.path.join(subdir, 'test.tif')

#         self.args = {
#             'workspace_dir': workspace_dir,
#             'lulc_uri_1': os.path.join(
#                 input_dir, 'GBJC_2004_mean_Resample.tif'),
#             'year_1': 2004,
#             'lulc_uri_2': os.path.join(
#                 input_dir, 'GBJC_2050_mean_Resample.tif'),
#             'year_2': 2050,
#             'lulc_uri_3': os.path.join(
#                 input_dir, 'GBJC_2100_mean_Resample.tif'),
#             'year_3': 2100,
#             'analysis_year': 2150,
#             'soil_disturbance_csv_uri': os.path.join(
#                 input_dir, 'soil_disturbance.csv'),
#             'biomass_disturbance_csv_uri': os.path.join(
#                 input_dir, 'biomass_disturbance.csv'),
#             'carbon_pools_uri': os.path.join(input_dir, 'carbon.csv'),
#             'half_life_csv_uri': os.path.join(input_dir, 'half_life.csv'),
#             'transition_matrix_uri': os.path.join(input_dir, 'transition.csv'),
#             'do_private_valuation': True,
#             'discount_rate': 5,
#             'do_price_table': True,
#             'carbon_schedule': os.path.join(input_dir, 'SCC5.csv')
#             #'carbon_value': None,
#             #'rate_change': None,
#         }

#     def test_run(self):
#         blue_carbon.execute(self.args)

#         print os.listdir(self.args['workspace_dir'])

#         with rio.open(os.path.join(
#                 self.args['workspace_dir'], 'stock_2050.tif')) as src:
#             print src.width
#             print src.height
#             print src.count
#             print src.read_band(1)[0:100:5, 0:100:5]


class TestOverallModel2(unittest.TestCase):
    def setUp(self):
        workspace_dir = tempfile.mkdtemp("subdir")
        lulc_uri_1 = os.path.join(
            workspace_dir, 'GBJC_2004_mean_Resample.tif'),
        lulc_uri_2 = os.path.join(
            workspace_dir, 'GBJC_2050_mean_Resample.tif'),
        lulc_uri_3 = os.path.join(
            workspace_dir, 'GBJC_2100_mean_Resample.tif'),

        # create arrays
        a = np.ones([1, 1])
        # a[0, 0] = 0
        orgX, orgY = 0, 100
        pixWidth, pixHeight = 100.0, 100.0  # Cell size in meters?

        # create_raster(
        #     lulc_uri_1[0],
        #     a * 2,
        #     orgX,
        #     orgY,
        #     pixWidth=pixWidth,
        #     pixHeight=pixHeight,
        #     proj=26915)

        # create_raster(
        #     lulc_uri_2[0],
        #     a * 7,
        #     orgX,
        #     orgY,
        #     pixWidth=pixWidth,
        #     pixHeight=pixHeight,
        #     proj=26915)

        # create_raster(
        #     lulc_uri_3[0],
        #     a * 11,
        #     orgX,
        #     orgY,
        #     pixWidth=pixWidth,
        #     pixHeight=pixHeight,
        #     proj=26915)

        create_raster(
            lulc_uri_1[0],
            a * 2,
            orgX,
            orgY,
            pixWidth=pixWidth,
            pixHeight=pixHeight,
            proj=26915,
            nodata=0)

        create_raster(
            lulc_uri_2[0],
            a * 7,
            orgX,
            orgY,
            pixWidth=pixWidth,
            pixHeight=pixHeight,
            proj=26915,
            nodata=0)

        create_raster(
            lulc_uri_3[0],
            a * 11,
            orgX,
            orgY,
            pixWidth=pixWidth,
            pixHeight=pixHeight,
            proj=26915,
            nodata=0)

        with rio.open(lulc_uri_1[0]) as src:
            print src.meta
            print src.crs
            print src.bounds
            print src.read_band(1)

        with rio.open(lulc_uri_2[0]) as src:
            print src.meta
            print src.crs
            print src.bounds
            print src.read_band(1)

        self.args = {
            'workspace_dir': workspace_dir,
            'lulc_uri_1': lulc_uri_1[0],
            'year_1': 2004,
            'lulc_uri_2': lulc_uri_2[0],
            'year_2': 2050,
            'lulc_uri_3': lulc_uri_3[0],
            'year_3': 2100,
            'analysis_year': 2150,
            'soil_disturbance_csv_uri': os.path.join(
                input_dir, 'soil_disturbance.csv'),
            'biomass_disturbance_csv_uri': os.path.join(
                input_dir, 'biomass_disturbance.csv'),
            'carbon_pools_uri': os.path.join(input_dir, 'carbon.csv'),
            'half_life_csv_uri': os.path.join(input_dir, 'half_life.csv'),
            'transition_matrix_uri': os.path.join(input_dir, 'transition.csv'),
            'do_private_valuation': True,
            'discount_rate': 5,
            'do_price_table': True,
            'carbon_schedule': os.path.join(input_dir, 'SCC5.csv')
        }

    def test_run(self):
        blue_carbon.execute(self.args)

        print os.listdir(self.args['workspace_dir'])

        with rio.open(os.path.join(
                self.args['workspace_dir'], 'stock_2050.tif')) as src:
            print src.width
            print src.height
            print src.count
            print src.read_band(1)


# class TestOverallModel3(unittest.TestCase):
#     def setUp(self):
#         workspace_dir = tempfile.mkdtemp("subdir")
#         lulc_uri_1 = os.path.join(
#             workspace_dir, 'GBJC_2004_mean_Resample.tif'),
#         lulc_uri_2 = os.path.join(
#             workspace_dir, 'GBJC_2050_mean_Resample.tif'),
#         lulc_uri_3 = os.path.join(
#             workspace_dir, 'GBJC_2100_mean_Resample.tif'),

#         # create arrays
#         a = np.ones([1, 1])
#         # a[0, 0] = 0
#         orgX, orgY = 0, 1
#         pixWidth, pixHeight = 1000.0, -1000.0  # Cell size in meters?

#         create_raster(
#             lulc_uri_1[0],
#             orgX,
#             orgY,
#             pixWidth,
#             pixHeight,
#             a * 8,
#             proj=26915)

#         create_raster(
#             lulc_uri_2[0],
#             orgX,
#             orgY,
#             pixWidth,
#             pixHeight,
#             a * 17,
#             proj=26915)

#         create_raster(
#             lulc_uri_3[0],
#             orgX,
#             orgY,
#             pixWidth,
#             pixHeight,
#             a * 17,
#             proj=26915)

#         with rio.open(lulc_uri_1[0]) as src:
#             print src.meta
#             print src.crs
#             print src.bounds
#             print src.read_band(1)

#         with rio.open(lulc_uri_2[0]) as src:
#             print src.meta
#             print src.crs
#             print src.bounds
#             print src.read_band(1)

#         self.args = {
#             'workspace_dir': workspace_dir,
#             'lulc_uri_1': lulc_uri_1[0],
#             'year_1': 2004,
#             'lulc_uri_2': lulc_uri_2[0],
#             'year_2': 2050,
#             'lulc_uri_3': lulc_uri_3[0],
#             'year_3': 2100,
#             'analysis_year': 2150,
#             'soil_disturbance_csv_uri': os.path.join(
#                 input_dir, 'soil_disturbance.csv'),
#             'biomass_disturbance_csv_uri': os.path.join(
#                 input_dir, 'biomass_disturbance.csv'),
#             'carbon_pools_uri': os.path.join(input_dir, 'carbon.csv'),
#             'half_life_csv_uri': os.path.join(input_dir, 'half_life.csv'),
#             'transition_matrix_uri': os.path.join(input_dir, 'transition.csv'),
#             'do_private_valuation': True,
#             'discount_rate': 5,
#             'do_price_table': True,
#             'carbon_schedule': os.path.join(input_dir, 'SCC5.csv')
#         }

#     def test_run(self):
#         blue_carbon.execute(self.args)

#         print os.listdir(self.args['workspace_dir'])

#         with rio.open(os.path.join(
#                 self.args['workspace_dir'], 'stock_2050.tif')) as src:
#             print src.width
#             print src.height
#             print src.count
#             print src.read_band(1)

# class TestOverallModel4(unittest.TestCase):
#     def setUp(self):
#         workspace_dir = tempfile.mkdtemp("subdir")
#         lulc_uri_1 = os.path.join(
#             workspace_dir, 'GBJC_2004_mean_Resample.tif'),
#         lulc_uri_2 = os.path.join(
#             workspace_dir, 'GBJC_2050_mean_Resample.tif'),
#         lulc_uri_3 = os.path.join(
#             workspace_dir, 'GBJC_2100_mean_Resample.tif'),

#         # create arrays
#         a = np.ones([1, 1])
#         # a[0, 0] = 0
#         orgX, orgY = 0, 1
#         pixWidth, pixHeight = 1000.0, -1000.0  # Cell size in meters?

#         create_raster(
#             lulc_uri_1[0],
#             orgX,
#             orgY,
#             pixWidth,
#             pixHeight,
#             a * 20,
#             proj=26915)

#         create_raster(
#             lulc_uri_2[0],
#             orgX,
#             orgY,
#             pixWidth,
#             pixHeight,
#             a * 20,
#             proj=26915)

#         create_raster(
#             lulc_uri_3[0],
#             orgX,
#             orgY,
#             pixWidth,
#             pixHeight,
#             a * 20,
#             proj=26915)

#         with rio.open(lulc_uri_1[0]) as src:
#             print src.meta
#             print src.crs
#             print src.bounds
#             print src.read_band(1)

#         with rio.open(lulc_uri_2[0]) as src:
#             print src.meta
#             print src.crs
#             print src.bounds
#             print src.read_band(1)

#         self.args = {
#             'workspace_dir': workspace_dir,
#             'lulc_uri_1': lulc_uri_1[0],
#             'year_1': 2004,
#             'lulc_uri_2': lulc_uri_2[0],
#             'year_2': 2050,
#             'lulc_uri_3': lulc_uri_3[0],
#             'year_3': 2100,
#             'analysis_year': 2150,
#             'soil_disturbance_csv_uri': os.path.join(
#                 input_dir, 'soil_disturbance.csv'),
#             'biomass_disturbance_csv_uri': os.path.join(
#                 input_dir, 'biomass_disturbance.csv'),
#             'carbon_pools_uri': os.path.join(input_dir, 'carbon.csv'),
#             'half_life_csv_uri': os.path.join(input_dir, 'half_life.csv'),
#             'transition_matrix_uri': os.path.join(input_dir, 'transition.csv'),
#             'do_private_valuation': True,
#             'discount_rate': 5,
#             'do_price_table': True,
#             'carbon_schedule': os.path.join(input_dir, 'SCC5.csv')
#         }

#     def test_run(self):
#         blue_carbon.execute(self.args)

#         print os.listdir(self.args['workspace_dir'])

#         with rio.open(os.path.join(
#                 self.args['workspace_dir'], 'stock_2050.tif')) as src:
#             print src.width
#             print src.height
#             print src.count
#             print src.read_band(1)


def create_raster(filepath, array, topleftX, topleftY, pixWidth=1, pixHeight=1, proj=4326, gdal_type=gdal.GDT_Float32, nodata=-9999):
    '''
    Converts a numpy array to a GeoTIFF file

    Args:
        filepath (str): Path to output GeoTIFF file
        array (np.array): Two-dimensional NumPy array
        topleftX (float): Western edge? Left-most edge?
        topleftY (float): Northern edge? Top-most edge?

    Keyword Args:
        pixWidth (float): Width of each pixel in given projection's units
        pixHeight (float): Height of each pixel in given projection's units
        proj (int): EPSG projection, default 4326
        gdal_type (type): A GDAL Datatype, default gdal.GDT_Float32
        nodata ((should match the provided GDT)): nodata value, default -9999.0

    Returns:
        None
    '''
    assert(len(array.shape) == 2)
    assert(topleftY >= array.shape[1])
    assert(topleftX >= 0)

    num_bands = 1
    rotX = 0.0
    rotY = 0.0

    rows = array.shape[1]
    cols = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(filepath, cols, rows, num_bands, gdal_type)
    raster.SetGeoTransform((topleftX, pixWidth, rotX, topleftY, rotY, (-pixHeight)))

    band = raster.GetRasterBand(1)  # Get only raster band
    band.SetNoDataValue(nodata)
    band.WriteArray(array)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(proj)
    raster.SetProjection(raster_srs.ExportToWkt())
    band.FlushCache()

    driver = None
    raster = None
    band = None


if __name__ == '__main__':
    unittest.main()
