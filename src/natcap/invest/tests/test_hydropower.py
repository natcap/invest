"""Tests for natcap.invest.hydropower_water_yeild.

This module is intended only to demonstrate how a model might be tested given
common needs of InVEST models.  Key things to note:

    * Sample data creation that's repeated outside a test method has its own
      function.
    * The module-under-test (natcap.invest._example_model) is imported in
      each test method.
    * Each test method is responsible for the appropriate cleanup of created
      sample data and workspaces.
"""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
from osgeo import ogr
import numpy
import numpy.testing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
from shapely.geometry import Polygon
from nose.tools import nottest

def _create_watershed(fields=None, attributes=None):
    """
    Create a watershed shapefile of polygons

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

    """

    srs = sampledata.SRS_WILLAMETTE

    pos_x = 443800.0
    pos_y = 4957000.0

    poly_geoms = {
                 'poly_1': [(pos_x, pos_y), (pos_x + 100, pos_y),
                            (pos_x, pos_y - 100), (pos_x + 100, pos_y - 100)],
                 'poly_2': [(pos_x + 100, pos_y), (pos_x + 200, pos_y),
                            (pos_x + 100, pos_y - 100), (pos_x + 200, pos_y - 100)],
                 'poly_3': [(pos_x, pos_y - 100), (pos_x + 100, pos_y -100),
                            (pos_x, pos_y - 200), (pos_x + 100, pos_y - 200)]}




    geometries = [Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2']),
                    Polygon(poly_geoms['poly_3'])]

    return pygeoprocessing.testing.create_vector_on_disk(
		    geometries, srs.projection, fields, attributes,
		    vector_format='ESRI Shapefile')

class HydropowerUnitTests(unittest.TestCase):

    """Tests for natcap.hydropower."""
    @nottest
    def test_execute(self):
        """Example execution to ensure correctness when called via execute."""
        from natcap.invest import _example_model
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': _create_lulc(),
        }
        _example_model.execute(args)

        expected_matrix = numpy.array([
            [0, 6, 7, 8, 9],
            [6, 7, 9, 10, 11],
            [6, -1, -1, -1, -1],
            [0, 6, 7, 8, 9]], numpy.int32)
        expected_raster = _create_lulc(expected_matrix)
        sum_raster = os.path.join(args['workspace_dir'], 'sum.tif')
        pygeoprocessing.testing.assert_rasters_equal(sum_raster,
                                                     expected_raster)

        shutil.rmtree(args['workspace_dir'])
        for filename in [args['example_lulc'], expected_raster]:
            os.remove(filename)
    @nottest
    def test_execute_with_suffix(self):
        """When a suffix is added, verify it's added correctly."""
        from natcap.invest import _example_model
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': _create_lulc(),
            'suffix': 'foo',
        }
        _example_model.execute(args)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
    @nottest
    def test_execute_with_suffix_and_underscore(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import _example_model
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': _create_lulc(),
            'suffix': 'foo',
        }
        _example_model.execute(args)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
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

    @nottest
    def test_extract_datasource_table_by_key(self):
        """Test a function that returns a dictionary base on a Shapefiles attributes"""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int', 'wyield_mn': 'real', 'wyield_vol': 'real',
                  'consum_mn': 'real', 'consum_vol': 'real'}

        attributes = [
                {'ws_id': 1, 'wyield_mn': 1000, 'wyield_vol': 1000,
                 'consum_vol': 10000, 'consum_mn': 10000},
                {'ws_id': 2, 'wyield_mn': 1000, 'wyield_vol': 1000,
                 'consum_vol': 10000, 'consum_mn': 10000},
                {'ws_id': 3, 'wyield_mn': 1000, 'wyield_vol': 1000,
                 'consum_vol': 10000, 'consum_mn': 10000}]

        watershed_uri = _create_watershed(fields, attributes)

        key_field = 'ws_id'
        wanted_list = ['wyield_vol', 'consum_vol']
        hydropower_water_yield.extract_datasource_table_by_key(watershed_uri, key_field, wanted_list)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))

    @nottest
    def test_write_new_table(self):
        """Verify that a new CSV table is written to properly"""
        from natcap.invest.hydropower import hydropower_water_yield

        filename = tempfile.mkstemp()

        fields = ['id', 'precip', 'volume']

        data = {0: {'id':1, 'precip': 100, 'volume': 150},
                1: {'id':2, 'precip': 100, 'volume': 150},
                2: {'id':3, 'precip': 100, 'volume': 150}}

        hydropower_water_yield.write_new_table(filename, fields, data)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))

    @nottest
    def test_compute_water_yield_volume(self):
        """Verify that water yield volume is computed correctly"""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int', 'wyield_mn': 'real', 'num_pixels': 'real'}

        attributes = [{'ws_id': 1, 'wyield_mn': 1000, 'num_pixels': 20},
                      {'ws_id': 2, 'wyield_mn': 1000, 'num_pixels': 15},
                      {'ws_id': 3, 'wyield_mn': 1000, 'num_pixels': 10}]

        watershed_uri = _create_watershed(fields, attributes)

        pixel_area = 100

        hydropower_water_yield.compute_water_yield_volume(shape_uri, pixel_area)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))

    @nottest
    def test_add_dict_to_shape(self):
        """Verify values from a dictionary are added to a shapefile properly."""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int'}

        attributes = [{'ws_id': 1}, {'ws_id': 2}, {'ws_id': 3}]

        watershed_uri = _create_watershed(fields, attributes)

        field_dict = {'1': 5, '2': 10, '3': 15}

        field_name = 'test'

        key = 'ws_id'

        hydropower_water_yield.add_dict_to_shape(shape_uri, field_dict, field_name, key)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
