"""Tests for natcap.invest.hydropower.hydropower_water_yeild.

    This module is intended for unit testing of the hydropower_water_yield
    model. The goal is to thoroughly test each of the modules functions
    and verify correctness.
"""
import unittest
import tempfile
import shutil
import os
import csv

from osgeo import gdal
from osgeo import ogr
import numpy
import numpy.testing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
from shapely.geometry import Polygon
from nose.tools import nottest

def _create_watershed(fields=None, attributes=None, subshed=True, execute=False):
    """
    Create a watershed shapefile

    This Watershed Shapefile is created with the following characteristis:
        * SRS is in the SRS_WILLAMETTE.
        * Vector type is Polygon
        * Polygons are 100m x 100m

    Parameters:
        fields (dict or None): a python dictionary mapping string fieldname
            to a string datatype representation of the target ogr fieldtype.
            Example: {'ws_id': 'int'}.  See
            ``pygeoprocessing.testing.sampledata.VECTOR_FIELD_TYPES.keys()``
            for the complete list of all allowed types.  If None, the datatype
            will be determined automatically based on the types of the
            attribute values.
        attributes (list of dicts): a list of python dictionary mapping
            fieldname to field value.  The field value's type must match the
            type defined in the fields input.  It is an error if it doesn't.

        subshed (boolean): True or False depicting whether the created vector
            should be representative of a sub watershed or watershed.
            Watershed will return one polygon, sub watershed will return 3.
        execute (boolean): Determines how to construct the sub shed polygons

    Returns:
        A string filepath to the vector on disk
    """

    srs = sampledata.SRS_WILLAMETTE

    pos_x = srs.origin[0]
    pos_y = srs.origin[1]

    if subshed:
        if not execute:
            poly_geoms = {
                 'poly_1': [(pos_x, pos_y), (pos_x + 100, pos_y),
                            (pos_x + 100, pos_y - 100), (pos_x, pos_y - 100), (pos_x, pos_y)],
                 'poly_2': [(pos_x + 100, pos_y), (pos_x + 200, pos_y),
                             (pos_x + 200, pos_y - 100), (pos_x + 100, pos_y - 100), (pos_x + 100, pos_y)],
                 'poly_3': [(pos_x, pos_y - 100), (pos_x + 100, pos_y -100),
                            (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200), (pos_x, pos_y - 100)]}

            geometries = [Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2']),
                        Polygon(poly_geoms['poly_3'])]
        else:
            poly_geoms = {
                 'poly_1': [(pos_x, pos_y), (pos_x + 200, pos_y),
                            (pos_x + 200, pos_y - 100), (pos_x, pos_y - 100), (pos_x, pos_y)],
                 'poly_2': [(pos_x, pos_y - 100), (pos_x + 200, pos_y - 100),
                             (pos_x + 200, pos_y - 200), (pos_x, pos_y - 200), (pos_x, pos_y - 100)]}

            geometries = [Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]
    else:
        poly_geoms = {
                 'poly_1': [(pos_x, pos_y), (pos_x + 200, pos_y),
                            (pos_x + 200, pos_y - 200), (pos_x, pos_y - 200), (pos_x, pos_y)]}

        geometries = [Polygon(poly_geoms['poly_1'])]

    return pygeoprocessing.testing.create_vector_on_disk(
		    geometries, srs.projection, fields, attributes,
		    vector_format='ESRI Shapefile')

def _create_csv(fields, data):
    """Create a new csv table from a dictionary

        filename - a URI path for the new table to be written to disk

        fields - a python list of the column names. The order of the fields in
            the list will be the order in how they are written. ex:
            ['id', 'precip', 'total']

        data - a python dictionary representing the table. The dictionary
            should be constructed with unique numerical keys that point to a
            dictionary which represents a row in the table:
            data = {0 : {'id':1, 'precip':43, 'total': 65},
                    1 : {'id':2, 'precip':65, 'total': 94}}

        returns - nothing
    """
    temp, filename = tempfile.mkstemp(suffix='.csv')
    os.close(temp)

    csv_file = open(filename, 'wb')

    #  Sort the keys so that the rows are written in order
    row_keys = data.keys()
    row_keys.sort()

    csv_writer = csv.DictWriter(csv_file, fields)
    #  Write the columns as the first row in the table
    csv_writer.writerow(dict((fn, fn) for fn in fields))

    # Write the rows from the dictionary
    for index in row_keys:
        csv_writer.writerow(data[index])

    csv_file.close()

    return filename

def _create_raster(matrix=None, dtype=gdal.GDT_Int32, nodata=-1):
    """
    Create a raster for the hydropower_water_yield model.

    This raster is created with the following characteristics:
        * SRS is in the SRS_WILLAMETTE.
        * Nodata is -1.
        * Raster type is `gdal.GDT_Int32`
        * Pixel size is 30m

    Parameters:
        matrix=None (numpy.array): A numpy array to use as a landcover matrix.
            The output raster created will be saved with these pixel values.
            If None, a default matrix will be used.

    Returns:
        A string filepath to a new LULC raster on disk.
    """
    if matrix is None:
        lulc_matrix = numpy.array([
            [0, 1, 2, 3, 4],
            [1, 2, 4, 5, 11],
            [1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4]], numpy.int32)
    else:
        lulc_matrix = matrix
    lulc_nodata = nodata
    srs = sampledata.SRS_WILLAMETTE
    return pygeoprocessing.testing.create_raster_on_disk(
        [lulc_matrix], srs.origin, srs.projection, lulc_nodata,
        srs.pixel_size(100), datatype=dtype)

class HydropowerUnitTests(unittest.TestCase):

    """Tests for natcap.invest.hydropower.hydropower_water_yeild"""
    def test_execute(self):
        """Example execution to ensure correctness when called via execute."""
        from natcap.invest.hydropower import hydropower_water_yield
        lulc_matrix = numpy.array([
            [0, 1],
            [2, 2]], numpy.int32)
        root_matrix = numpy.array([
            [100, 1500],
            [1300, 1300]], numpy.float32)
        precip_matrix = numpy.array([
            [1000, 2000],
            [4000, 2000]], numpy.float32)
        eto_matrix = numpy.array([
            [900, 1000],
            [900, 1300]], numpy.float32)
        pawc_matrix = numpy.array([
            [0.19, 0.13],
            [0.11, 0.11]], numpy.float32)

        fields_ws = {'ws_id': 'int'}
        fields_sub = {'subws_id': 'int'}
        attr_ws = [{'ws_id': 1}]
        attr_sub = [{'subws_id': 1}, {'subws_id': 2}]

        bio_fields = ['lucode', 'Kc', 'root_depth', 'LULC_veg']

        bio_data = {0: {'lucode':0, 'Kc': 0.3, 'root_depth': 500, 'LULC_veg': 0},
                1: {'lucode':1, 'Kc': 0.75, 'root_depth': 5000, 'LULC_veg': 1},
                2: {'lucode':2, 'Kc': 0.85, 'root_depth': 5000, 'LULC_veg': 1}}

        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': _create_raster(lulc_matrix),
            'depth_to_root_rest_layer_uri': _create_raster(root_matrix, gdal.GDT_Float32),
            'precipitation_uri': _create_raster(precip_matrix, gdal.GDT_Float32),
            'pawc_uri': _create_raster(pawc_matrix, gdal.GDT_Float32),
            'eto_uri': _create_raster(eto_matrix, gdal.GDT_Float32),
            'watersheds_uri': _create_watershed(fields=fields_ws, attributes=attr_ws, subshed=False, execute=True),
            'sub_watersheds_uri': _create_watershed(fields=fields_sub, attributes=attr_sub, subshed=True, execute=True),
            'biophysical_table_uri': _create_csv(bio_fields, bio_data),
            'seasonality_constant': 5,
            'water_scarcity_container': False,
            'valuation_container': False
        }
        hydropower_water_yield.execute(args)

        test_files = [
            args['lulc_uri'], args['depth_to_root_rest_layer_uri'],
            args['precipitation_uri'], args['pawc_uri'], args['eto_uri'],
            args['watersheds_uri'], args['sub_watersheds_uri'], args['biophysical_table_uri']]

        wyield_res = numpy.array([
            [730, 1451.94876],
            [3494.87048, 1344.52666]], numpy.float32)
        wyield_path = _create_raster(wyield_res, gdal.GDT_Float32)
        #pixel_files =  ['aet_test.tif', 'fractp_test.tif', 'wyield_test.tif']
        pixel_files =  'wyield_test.tif'

        #for file_path in pixel_files:
        #pygeoprocessing.testing.assert_rasters_equal(
        #    wyield_path,
        raster = gdal.Open(os.path.join(args['workspace_dir'], 'output', 'per_pixel', 'wyield.tif'))
        band = raster.GetRasterBand(1)
        computed_res = band.ReadAsArray()
        print computed_res
        a = wyield_res.flatten()
        b = computed_res.flatten()
        print a
        print b
        for x, y in zip(a,b):
            pygeoprocessing.testing.assert_almost_equal(x, y, places=4)

        computed_res = None
        band = None
        raster = None

        shutil.rmtree(args['workspace_dir'])
        os.remove(wyield_path)
        for filename in test_files:
            os.remove(filename)


    @nottest
    def test_execute_bad_nodata(self):
        """Example execution to ensure correctness when called via execute."""
        from natcap.invest.hydropower import hydropower_water_yield
        lulc_matrix = numpy.array([
            [0, 1],
            [2, 2]], numpy.int32)
        root_matrix = numpy.array([
            [100, 1500],
            [1300, 1300]], numpy.float32)
        precip_matrix = numpy.array([
            [1000, 2000],
            [4000, 2000]], numpy.float32)
        eto_matrix = numpy.array([
            [900, 1000],
            [900, 1300]], numpy.float32)
        pawc_matrix = numpy.array([
            [0.19, 0.13],
            [0.11, 0.11]], numpy.float32)

        fields_ws = {'ws_id': 'int'}
        fields_sub = {'subws_id': 'int'}
        attr_ws = [{'ws_id': 1}]
        attr_sub = [{'subws_id': 1}, {'subws_id': 2}]

        bio_fields = ['lucode', 'Kc', 'root_depth', 'LULC_veg']

        bio_data = {0: {'lucode':0, 'Kc': 0.3, 'root_depth': 500, 'LULC_veg': 0},
                1: {'lucode':1, 'Kc': 0.75, 'root_depth': 5000, 'LULC_veg': 1},
                2: {'lucode':2, 'Kc': 0.85, 'root_depth': 5000, 'LULC_veg': 1}}

        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': _create_raster(lulc_matrix, nodata=None),
            'depth_to_root_rest_layer_uri': _create_raster(root_matrix, gdal.GDT_Float32, nodata=None),
            'precipitation_uri': _create_raster(precip_matrix, gdal.GDT_Float32, nodata=None),
            'pawc_uri': _create_raster(pawc_matrix, gdal.GDT_Float32, nodata=None),
            'eto_uri': _create_raster(eto_matrix, gdal.GDT_Float32, nodata=None),
            'watersheds_uri': _create_watershed(fields=fields_ws, attributes=attr_ws, subshed=False, execute=True),
            'sub_watersheds_uri': _create_watershed(fields=fields_sub, attributes=attr_sub, subshed=True, execute=True),
            'biophysical_table_uri': _create_csv(bio_fields, bio_data),
            'seasonality_constant': 5,
            'water_scarcity_container': False,
            'valuation_container': False
        }
        hydropower_water_yield.execute(args)

        test_files = [
            args['lulc_uri'], args['depth_to_root_rest_layer_uri'],
            args['precipitation_uri'], args['pawc_uri'], args['eto_uri'],
            args['watersheds_uri'], args['sub_watersheds_uri'], args['biophysical_table_uri']]

        shutil.rmtree(args['workspace_dir'])
        for filename in test_files:
            os.remove(filename)
    @nottest
    def test_execute_scarcity(self):
        """Example execution to ensure correctness when called via execute."""
        from natcap.invest.hydropower import hydropower_water_yield
        lulc_matrix = numpy.array([
            [0, 1],
            [2, 2]], numpy.int32)
        root_matrix = numpy.array([
            [100, 1500],
            [1300, 1300]], numpy.float32)
        precip_matrix = numpy.array([
            [1000, 2000],
            [4000, 2000]], numpy.float32)
        eto_matrix = numpy.array([
            [900, 1000],
            [900, 1300]], numpy.float32)
        pawc_matrix = numpy.array([
            [0.19, 0.13],
            [0.11, 0.11]], numpy.float32)

        fields_ws = {'ws_id': 'int'}
        fields_sub = {'subws_id': 'int'}
        attr_ws = [{'ws_id': 1}]
        attr_sub = [{'subws_id': 1}, {'subws_id': 2}]

        bio_fields = ['lucode', 'Kc', 'root_depth', 'LULC_veg']

        bio_data = {0: {'lucode':0, 'Kc': 0.3, 'root_depth': 500, 'LULC_veg': 0},
                1: {'lucode':1, 'Kc': 0.75, 'root_depth': 5000, 'LULC_veg': 1},
                2: {'lucode':2, 'Kc': 0.85, 'root_depth': 5000, 'LULC_veg': 1}}

        demand_fields = ['lucode', 'demand']

        demand_data = {0: {'lucode':0, 'demand': 500},
                1: {'lucode':1, 'demand': 0},
                2: {'lucode':2, 'demand': 0}}

        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': _create_raster(lulc_matrix),
            'depth_to_root_rest_layer_uri': _create_raster(root_matrix, gdal.GDT_Float32),
            'precipitation_uri': _create_raster(precip_matrix, gdal.GDT_Float32),
            'pawc_uri': _create_raster(pawc_matrix, gdal.GDT_Float32),
            'eto_uri': _create_raster(eto_matrix, gdal.GDT_Float32),
            'watersheds_uri': _create_watershed(fields=fields_ws, attributes=attr_ws, subshed=False, execute=True),
            'sub_watersheds_uri': _create_watershed(fields=fields_sub, attributes=attr_sub, subshed=True, execute=True),
            'biophysical_table_uri': _create_csv(bio_fields, bio_data),
            'seasonality_constant': 5,
            'demand_table_uri': _create_csv(demand_fields, demand_data),
            'water_scarcity_container': True,
            'valuation_container': False
        }
        hydropower_water_yield.execute(args)

        test_files = [
            args['lulc_uri'], args['depth_to_root_rest_layer_uri'],
            args['precipitation_uri'], args['pawc_uri'], args['eto_uri'],
            args['watersheds_uri'], args['sub_watersheds_uri'], args['biophysical_table_uri'],
            args['demand_table_uri']]

        shutil.rmtree(args['workspace_dir'])
        for filename in test_files:
            os.remove(filename)
    @nottest
    def test_execute_valuation(self):
        """Example execution to ensure correctness when called via execute."""
        from natcap.invest.hydropower import hydropower_water_yield
        lulc_matrix = numpy.array([
            [0, 1],
            [2, 2]], numpy.int32)
        root_matrix = numpy.array([
            [100, 1500],
            [1300, 1300]], numpy.float32)
        precip_matrix = numpy.array([
            [1000, 2000],
            [4000, 2000]], numpy.float32)
        eto_matrix = numpy.array([
            [900, 1000],
            [900, 1300]], numpy.float32)
        pawc_matrix = numpy.array([
            [0.19, 0.13],
            [0.11, 0.11]], numpy.float32)

        fields_ws = {'ws_id': 'int'}
        fields_sub = {'subws_id': 'int'}
        attr_ws = [{'ws_id': 1}]
        attr_sub = [{'subws_id': 1}, {'subws_id': 2}]

        bio_fields = ['lucode', 'Kc', 'root_depth', 'LULC_veg']

        bio_data = {0: {'lucode':0, 'Kc': 0.3, 'root_depth': 500, 'LULC_veg': 0},
                1: {'lucode':1, 'Kc': 0.75, 'root_depth': 5000, 'LULC_veg': 1},
                2: {'lucode':2, 'Kc': 0.85, 'root_depth': 5000, 'LULC_veg': 1}}

        demand_fields = ['lucode', 'demand']

        demand_data = {0: {'lucode':0, 'demand': 500},
                1: {'lucode':1, 'demand': 0},
                2: {'lucode':2, 'demand': 0}}

        val_fields = ['ws_id', 'time_span', 'discount', 'efficiency',
                        'fraction', 'cost', 'height', 'kw_price', 'desc']

        val_data = {0: {'ws_id': 1, 'time_span': 20, 'discount': 5, 'efficiency': 0.8,
                        'fraction': 0.6, 'cost': 0, 'height': 25, 'kw_price': 0.07,
                        'desc': 'Hydropower1'}}

        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': _create_raster(lulc_matrix),
            'depth_to_root_rest_layer_uri': _create_raster(root_matrix, gdal.GDT_Float32),
            'precipitation_uri': _create_raster(precip_matrix, gdal.GDT_Float32),
            'pawc_uri': _create_raster(pawc_matrix, gdal.GDT_Float32),
            'eto_uri': _create_raster(eto_matrix, gdal.GDT_Float32),
            'watersheds_uri': _create_watershed(fields=fields_ws, attributes=attr_ws, subshed=False, execute=True),
            'sub_watersheds_uri': _create_watershed(fields=fields_sub, attributes=attr_sub, subshed=True, execute=True),
            'biophysical_table_uri': _create_csv(bio_fields, bio_data),
            'seasonality_constant': 5,
            'demand_table_uri': _create_csv(demand_fields, demand_data),
            'valuation_table_uri': _create_csv(val_fields, val_data),
            'water_scarcity_container': True,
            'valuation_container': True
        }
        hydropower_water_yield.execute(args)

        test_files = [
            args['lulc_uri'], args['depth_to_root_rest_layer_uri'],
            args['precipitation_uri'], args['pawc_uri'], args['eto_uri'],
            args['watersheds_uri'], args['sub_watersheds_uri'], args['biophysical_table_uri'],
            args['demand_table_uri'], args['valuation_table_uri']]

        shutil.rmtree(args['workspace_dir'])
        for filename in test_files:
            os.remove(filename)
    @nottest
    def test_execute_with_suffix(self):
        """When a suffix is added, verify it's added correctly."""
        from natcap.invest.hydropower import hydropower_water_yield
        lulc_matrix = numpy.array([
            [0, 1],
            [2, 2]], numpy.int32)
        root_matrix = numpy.array([
            [100, 1500],
            [1300, 1300]], numpy.float32)
        precip_matrix = numpy.array([
            [1000, 2000],
            [4000, 2000]], numpy.float32)
        eto_matrix = numpy.array([
            [900, 1000],
            [900, 1300]], numpy.float32)
        pawc_matrix = numpy.array([
            [0.19, 0.13],
            [0.11, 0.11]], numpy.float32)

        fields_ws = {'ws_id': 'int'}
        fields_sub = {'subws_id': 'int'}
        attr_ws = [{'ws_id': 1}]
        attr_sub = [{'subws_id': 1}, {'subws_id': 2}]

        bio_fields = ['lucode', 'Kc', 'root_depth', 'LULC_veg']

        bio_data = {0: {'lucode':0, 'Kc': 0.3, 'root_depth': 500, 'LULC_veg': 0},
                1: {'lucode':1, 'Kc': 0.75, 'root_depth': 5000, 'LULC_veg': 1},
                2: {'lucode':2, 'Kc': 0.85, 'root_depth': 5000, 'LULC_veg': 1}}

        demand_fields = ['lucode', 'demand']

        demand_data = {0: {'lucode':0, 'demand': 500},
                1: {'lucode':1, 'demand': 0},
                2: {'lucode':2, 'demand': 0}}

        val_fields = ['ws_id', 'time_span', 'discount', 'efficiency',
                        'fraction', 'cost', 'height', 'kw_price']

        val_data = {0: {'ws_id': 1, 'time_span': 20, 'discount': 5, 'efficiency': 0.8,
                        'fraction': 0.6, 'cost': 0, 'height': 25, 'kw_price': 0.07}}

        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': _create_raster(lulc_matrix),
            'depth_to_root_rest_layer_uri': _create_raster(root_matrix, gdal.GDT_Float32),
            'precipitation_uri': _create_raster(precip_matrix, gdal.GDT_Float32),
            'pawc_uri': _create_raster(pawc_matrix, gdal.GDT_Float32),
            'eto_uri': _create_raster(eto_matrix, gdal.GDT_Float32),
            'watersheds_uri': _create_watershed(fields=fields_ws, attributes=attr_ws, subshed=False, execute=True),
            'sub_watersheds_uri': _create_watershed(fields=fields_sub, attributes=attr_sub, subshed=True, execute=True),
            'biophysical_table_uri': _create_csv(bio_fields, bio_data),
            'seasonality_constant': 5,
            'results_suffix': 'test',
            'demand_table_uri': _create_csv(demand_fields, demand_data),
            'valuation_table_uri': _create_csv(val_fields, val_data),
            'water_scarcity_container': True,
            'valuation_container': True
        }
        hydropower_water_yield.execute(args)

        test_files = [
            args['lulc_uri'], args['depth_to_root_rest_layer_uri'],
            args['precipitation_uri'], args['pawc_uri'], args['eto_uri'],
            args['watersheds_uri'], args['sub_watersheds_uri'], args['biophysical_table_uri'],
            args['demand_table_uri'], args['valuation_table_uri']]

        output_files = ['watershed_results_wyield_test.csv','subwatershed_results_wyield_test.csv',
            'watershed_results_wyield_test.shp','subwatershed_results_wyield_test.shp',]
        pixel_files =  ['aet_test.tif', 'fractp_test.tif', 'wyield_test.tif']

        for file_path in output_files:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', file_path)))
        for file_path in pixel_files:
            self.assertTrue(os.path.exists(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', file_path)))

        shutil.rmtree(args['workspace_dir'])
        for filename in test_files:
            os.remove(filename)
    @nottest
    def test_execute_with_suffix_and_underscore(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest.hydropower import hydropower_water_yield
        lulc_matrix = numpy.array([
            [0, 1],
            [2, 2]], numpy.int32)
        root_matrix = numpy.array([
            [100, 1500],
            [1300, 1300]], numpy.float32)
        precip_matrix = numpy.array([
            [1000, 2000],
            [4000, 2000]], numpy.float32)
        eto_matrix = numpy.array([
            [900, 1000],
            [900, 1300]], numpy.float32)
        pawc_matrix = numpy.array([
            [0.19, 0.13],
            [0.11, 0.11]], numpy.float32)

        fields_ws = {'ws_id': 'int'}
        fields_sub = {'subws_id': 'int'}
        attr_ws = [{'ws_id': 1}]
        attr_sub = [{'subws_id': 1}, {'subws_id': 2}]

        bio_fields = ['lucode', 'Kc', 'root_depth', 'LULC_veg']

        bio_data = {0: {'lucode':0, 'Kc': 0.3, 'root_depth': 500, 'LULC_veg': 0},
                1: {'lucode':1, 'Kc': 0.75, 'root_depth': 5000, 'LULC_veg': 1},
                2: {'lucode':2, 'Kc': 0.85, 'root_depth': 5000, 'LULC_veg': 1}}

        demand_fields = ['lucode', 'demand']

        demand_data = {0: {'lucode':0, 'demand': 500},
                1: {'lucode':1, 'demand': 0},
                2: {'lucode':2, 'demand': 0}}

        val_fields = ['ws_id', 'time_span', 'discount', 'efficiency',
                        'fraction', 'cost', 'height', 'kw_price']

        val_data = {0: {'ws_id': 1, 'time_span': 20, 'discount': 5, 'efficiency': 0.8,
                        'fraction': 0.6, 'cost': 0, 'height': 25, 'kw_price': 0.07}}

        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'lulc_uri': _create_raster(lulc_matrix),
            'depth_to_root_rest_layer_uri': _create_raster(root_matrix, gdal.GDT_Float32),
            'precipitation_uri': _create_raster(precip_matrix, gdal.GDT_Float32),
            'pawc_uri': _create_raster(pawc_matrix, gdal.GDT_Float32),
            'eto_uri': _create_raster(eto_matrix, gdal.GDT_Float32),
            'watersheds_uri': _create_watershed(fields=fields_ws, attributes=attr_ws, subshed=False, execute=True),
            'sub_watersheds_uri': _create_watershed(fields=fields_sub, attributes=attr_sub, subshed=True, execute=True),
            'biophysical_table_uri': _create_csv(bio_fields, bio_data),
            'seasonality_constant': 5,
            'results_suffix': '_test',
            'demand_table_uri': _create_csv(demand_fields, demand_data),
            'valuation_table_uri': _create_csv(val_fields, val_data),
            'water_scarcity_container': True,
            'valuation_container': True
        }
        hydropower_water_yield.execute(args)

        test_files = [
            args['lulc_uri'], args['depth_to_root_rest_layer_uri'],
            args['precipitation_uri'], args['pawc_uri'], args['eto_uri'],
            args['watersheds_uri'], args['sub_watersheds_uri'], args['biophysical_table_uri'],
            args['demand_table_uri'], args['valuation_table_uri']]

        output_files = ['watershed_results_wyield_test.csv','subwatershed_results_wyield_test.csv',
            'watershed_results_wyield_test.shp','subwatershed_results_wyield_test.shp',]
        pixel_files =  ['aet_test.tif', 'fractp_test.tif', 'wyield_test.tif']

        for file_path in output_files:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', file_path)))
        for file_path in pixel_files:
            self.assertTrue(os.path.exists(
                os.path.join(
                    args['workspace_dir'], 'output', 'per_pixel', file_path)))

        shutil.rmtree(args['workspace_dir'])
        for filename in test_files:
            os.remove(filename)

    def test_compute_waterhsed_valuation(self):
        """Verify valuation is computed correctly"""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int', 'rsupply_vl': 'real'}

        attributes = [
                {'ws_id': 1, 'rsupply_vl': 1000.0},
                {'ws_id': 2, 'rsupply_vl': 2000.0},
                {'ws_id': 3, 'rsupply_vl': 3000.0}]

        watershed_uri = _create_watershed(fields, attributes)

        val_dict = {
                1: {'efficiency': 0.75, 'fraction': 0.6, 'height': 25.0,
                      'discount': 5.0, 'time_span': 100.0, 'kw_price': 0.07,
                      'cost': 0.0},
                2: {'efficiency': 0.85, 'fraction': 0.7, 'height': 20.0,
                      'discount': 5.0, 'time_span': 100.0, 'kw_price': 0.07,
                      'cost': 0.0},
                3: {'efficiency': 0.9, 'fraction': 0.6, 'height': 30.0,
                      'discount': 5.0, 'time_span': 100.0, 'kw_price': 0.07,
                      'cost': 0.0}}

        hydropower_water_yield.compute_watershed_valuation(watershed_uri, val_dict)

        results = {
            1 : {'hp_energy': 30.6, 'hp_val': 44.63993483},
            2 : {'hp_energy': 64.736, 'hp_val': 94.43826213},
            3 : {'hp_energy': 132.192, 'hp_val': 192.84451846}
        }

        # Check that the 'hp_energy' / 'hp_val' fields were added and are correct
        shape = ogr.Open(watershed_uri)
        # Check that the shapefiles have the same number of layers
        layer_count = shape.GetLayerCount()

        for layer_num in range(layer_count):
            # Get the current layer
            layer = shape.GetLayer(layer_num)

            # Get the first features of the layers and loop through all the
            # features
            feat = layer.GetNextFeature()
            while feat is not None:
                # Check that the field counts for the features are the same
                layer_def = layer.GetLayerDefn()
                field_count = layer_def.GetFieldCount()
                ws_id_index = feat.GetFieldIndex('ws_id')
                ws_id = feat.GetField(ws_id_index)

                for key in ['hp_energy', 'hp_val']:
                    try:
                        key_index = feat.GetFieldIndex(key)
                        key_val = feat.GetField(key_index)
                        pygeoprocessing.testing.assert_almost_equal(
                            results[ws_id][key], key_val)
                    except ValueError:
                        raise AssertionError('Could not find field %s' % key)

                feat = None
                feat = layer.GetNextFeature()

        shape = None
        shape_regression = None

    def test_compute_rsupply_volume(self):
        """Verify the real supply volume is computed correctly"""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int', 'wyield_mn': 'real', 'wyield_vol': 'real',
                  'consum_mn': 'real', 'consum_vol': 'real'}

        attributes = [
                {'ws_id': 1, 'wyield_mn': 400, 'wyield_vol': 1200,
                 'consum_vol': 300, 'consum_mn': 80},
                {'ws_id': 2, 'wyield_mn': 450, 'wyield_vol': 1100,
                 'consum_vol': 300, 'consum_mn': 75},
                {'ws_id': 3, 'wyield_mn': 500, 'wyield_vol': 1000,
                 'consum_vol': 200, 'consum_mn': 50}]

        watershed_uri = _create_watershed(fields, attributes)

        hydropower_water_yield.compute_rsupply_volume(watershed_uri)

        results = {
            1 : {'rsupply_vl': 900, 'rsupply_mn': 320},
            2 : {'rsupply_vl': 800, 'rsupply_mn': 375},
            3 : {'rsupply_vl': 800, 'rsupply_mn': 450}
        }

        # Check that the 'hp_energy' / 'hp_val' fields were added and are correct
        shape = ogr.Open(watershed_uri)
        # Check that the shapefiles have the same number of layers
        layer_count = shape.GetLayerCount()

        for layer_num in range(layer_count):
            # Get the current layer
            layer = shape.GetLayer(layer_num)

            # Get the first features of the layers and loop through all the
            # features
            feat = layer.GetNextFeature()
            while feat is not None:
                # Check that the field counts for the features are the same
                layer_def = layer.GetLayerDefn()
                field_count = layer_def.GetFieldCount()
                ws_id_index = feat.GetFieldIndex('ws_id')
                ws_id = feat.GetField(ws_id_index)

                for key in ['rsupply_vl', 'rsupply_mn']:
                    try:
                        key_index = feat.GetFieldIndex(key)
                        key_val = feat.GetField(key_index)
                        pygeoprocessing.testing.assert_almost_equal(
                            results[ws_id][key], key_val)
                    except ValueError:
                        raise AssertionError('Could not find field %s' % key)

                feat = None
                feat = layer.GetNextFeature()

        shape = None
        shape_regression = None

    def test_extract_datasource_table_by_key(self):
        """Test a function that returns a dictionary base on a Shapefiles attributes"""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int', 'wyield_mn': 'real', 'wyield_vol': 'real',
                  'consum_mn': 'real', 'consum_vol': 'real'}

        attributes = [
                {'ws_id': 1, 'wyield_mn': 1000, 'wyield_vol': 1000,
                 'consum_vol': 100, 'consum_mn': 10000},
                {'ws_id': 2, 'wyield_mn': 1000, 'wyield_vol': 800,
                 'consum_vol': 420, 'consum_mn': 10000},
                {'ws_id': 3, 'wyield_mn': 1000, 'wyield_vol': 600,
                 'consum_vol': 350, 'consum_mn': 10000}]

        watershed_uri = _create_watershed(fields, attributes)

        key_field = 'ws_id'
        wanted_list = ['wyield_vol', 'consum_vol']
        results = hydropower_water_yield.extract_datasource_table_by_key(
                    watershed_uri, key_field, wanted_list)

        expected_res = {1: {'wyield_vol': 1000, 'consum_vol': 100},
                        2: {'wyield_vol': 800, 'consum_vol': 420},
                        3: {'wyield_vol': 600, 'consum_vol': 350}}

        for exp_key in expected_res.keys():
            if exp_key not in results:
                raise AssertionError('Key %s not found in returned results' % exp_key)
            for sub_key in expected_res[exp_key].keys():
                if sub_key not in results[exp_key].keys():
                    raise AssertionError('Key %s not found in returned results' % sub_key)
                pygeoprocessing.testing.assert_almost_equal(
                    expected_res[exp_key][sub_key], results[exp_key][sub_key])

    def test_write_new_table(self):
        """Verify that a new CSV table is written to properly"""
        from natcap.invest.hydropower import hydropower_water_yield

        filename = tempfile.mkstemp()[1]

        fields = ['id', 'precip', 'volume']

        data = {0: {'id':1, 'precip': 100, 'volume': 150},
                1: {'id':2, 'precip': 150, 'volume': 350},
                2: {'id':3, 'precip': 170, 'volume': 250}}

        hydropower_water_yield.write_new_table(filename, fields, data)

        csv_file = open(filename, 'rb')
        reader = csv.DictReader(csv_file)

        for row, key in zip(reader, [0, 1, 2]):
            for sub_key in data[key].keys():
                if sub_key not in row:
                    raise AssertionError('Key %s not found in CSV table' % sub_key)
                pygeoprocessing.testing.assert_almost_equal(
                    float(data[key][sub_key]), float(row[sub_key]))

        csv_file.close()

    def test_compute_water_yield_volume(self):
        """Verify that water yield volume is computed correctly"""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int', 'wyield_mn': 'real', 'num_pixels': 'int'}

        attributes = [{'ws_id': 1, 'wyield_mn': 400, 'num_pixels': 20},
                      {'ws_id': 2, 'wyield_mn': 300, 'num_pixels': 15},
                      {'ws_id': 3, 'wyield_mn': 200, 'num_pixels': 10}]

        watershed_uri = _create_watershed(fields, attributes)

        pixel_area = 100

        hydropower_water_yield.compute_water_yield_volume(watershed_uri, pixel_area)

        results = {
            1: {'wyield_vol': 800},
            2: {'wyield_vol': 450},
            3: {'wyield_vol': 200}
            }

        # Check that the 'hp_energy' / 'hp_val' fields were added and are correct
        shape = ogr.Open(watershed_uri)
        # Check that the shapefiles have the same number of layers
        layer_count = shape.GetLayerCount()

        for layer_num in range(layer_count):
            # Get the current layer
            layer = shape.GetLayer(layer_num)

            # Get the first features of the layers and loop through all the
            # features
            feat = layer.GetNextFeature()
            while feat is not None:
                # Check that the field counts for the features are the same
                layer_def = layer.GetLayerDefn()
                field_count = layer_def.GetFieldCount()
                ws_id_index = feat.GetFieldIndex('ws_id')
                ws_id = feat.GetField(ws_id_index)

                try:
                    key_index = feat.GetFieldIndex('wyield_vol')
                    key_val = feat.GetField(key_index)
                    pygeoprocessing.testing.assert_almost_equal(
                        results[ws_id]['wyield_vol'], key_val)
                except ValueError:
                    raise AssertionError('Could not find field %s' % key)

                feat = None
                feat = layer.GetNextFeature()

        shape = None
        shape_regression = None

    def test_add_dict_to_shape(self):
        """Verify values from a dictionary are added to a shapefile properly."""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int'}

        attributes = [{'ws_id': 1}, {'ws_id': 2}, {'ws_id': 3}]

        watershed_uri = _create_watershed(fields, attributes)

        field_dict = {1: 50.0, 2: 10.5, 3: 15.8}

        field_name = 'precip'

        key = 'ws_id'

        hydropower_water_yield.add_dict_to_shape(watershed_uri, field_dict, field_name, key)

        results = {
            1: {'precip': 50.0},
            2: {'precip': 10.5},
            3: {'precip': 15.8}
            }

        # Check that the 'hp_energy' / 'hp_val' fields were added and are correct
        shape = ogr.Open(watershed_uri)
        # Check that the shapefiles have the same number of layers
        layer_count = shape.GetLayerCount()

        for layer_num in range(layer_count):
            # Get the current layer
            layer = shape.GetLayer(layer_num)

            # Get the first features of the layers and loop through all the
            # features
            feat = layer.GetNextFeature()
            while feat is not None:
                # Check that the field counts for the features are the same
                layer_def = layer.GetLayerDefn()
                field_count = layer_def.GetFieldCount()
                ws_id_index = feat.GetFieldIndex('ws_id')
                ws_id = feat.GetField(ws_id_index)

                try:
                    key_index = feat.GetFieldIndex('precip')
                    key_val = feat.GetField(key_index)
                    pygeoprocessing.testing.assert_almost_equal(
                        results[ws_id]['precip'], key_val)
                except ValueError:
                    raise AssertionError('Could not find field %s' % key)

                feat = None
                feat = layer.GetNextFeature()

        shape = None
        shape_regression = None

