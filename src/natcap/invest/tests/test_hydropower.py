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
import numpy
import numpy.testing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
from shapely.geometry import Polygon

def _create_lulc(matrix=None):
    """
    Create an LULC for the _example_model.

    This LULC is created with the following characteristics:
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
    lulc_nodata = -1
    srs = sampledata.SRS_WILLAMETTE
    return pygeoprocessing.testing.create_raster_on_disk(
        [lulc_matrix], srs.origin, srs.projection, lulc_nodata,
        srs.pixel_size(30), datatype=gdal.GDT_Int32)

def _create_watershed(fields=None, attributes=None):
    """
    Create a watershed shapefile with 


    """
    pass
    
    projection = sampledata.SRS_WILLAMETTE
    
    pos_x = 443800.0
    pos_y = 4957000.0
    
    geometries = [Polygon([(pos_x, pos_y), (pos_x + 100, pos_y), (pos_x, pos_y - 100),(pos_x + 100, pos_y - 100)]), Polygon([(pos_x + 100, pos_y),(pos_x + 200, pos_y),(pos_x + 100, pos_y - 100),(pos_x + 200, pos_y - 100)]), Polygon([(pos_x, pos_y - 100),(pos_x + 100, pos_y - 100),(pos_x, pos_y - 200),(pos_x + 100, pos_y - 200)])]

   
    return pygeoprocessing.testing.create_vector_on_disk(
		    geometries, projection, fields, attributes,
		    vector_format='ESRI Shapefile')

class HydropowerUnitTests(unittest.TestCase):

    """Tests for natcap.hydropower."""

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

    def test_execute_with_suffix(self):
        """When a suffix is added, verify it's added correctly."""
        from natcap.invest import _example_model
        pass
	args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': _create_lulc(),
            'suffix': 'foo',
        }
        _example_model.execute(args)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))

    def test_execute_with_suffix_and_underscore(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import _example_model
        pass
	args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': _create_lulc(),
            'suffix': 'foo',
        }
        _example_model.execute(args)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
    def test_compute_waterhsed_valuation(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import hydropower
        pass 
	fields = {'ws_id': 'int',
		  'rsupply_vl': 
		  }
	attributes = [{'ws_id': 1, 'rsupply_vl': 1000},
		      {'ws_id': 2, 'rsupply_vl': 2000},
		      {'ws_id': 3, 'rsupply_vl': 3000}]
        
        watershed_uri = _create_watershed(fields, attributes)

	val_dict = {'1': {'efficiency': 0.75, 'fraction': 0.6,'height': 25, 'discount': 5,'time_span': 100,'kw_price': 0.07,'cost': 0}, '2': {'efficiency': 0.85, 'fraction': 0.7,'height': 20, 'discount': 5,'time_span': 100,'kw_price': 0.07,'cost': 0}, '3': {'efficiency': 0.9, 'fraction': 0.6,'height': 30, 'discount': 5,'time_span': 100,'kw_price': 0.07,'cost': 0}}
	
	hydropower.compute_watershed_valuation(watershed_uri, val_dict)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
    def test_compute_rsupply_volume(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import hydropower
        pass
	fields = {'ws_id': 'int',
		  'wyield_mn': , 
		  'wyield_vol': , 
		  'consum_mn_mn': , 
		  'consum_vol':  
		  }
	attributes = [{'ws_id': 1, 'wyield_mn': 1000, 'wyield_vol': 1000, 'consum_vol': 10000, 'consum_mn': 10000}, {'ws_id': 2, 'wyield_mn': 1000, 'wyield_vol': 1000, 'consum_vol': 10000, 'consum_mn': 10000}, {'ws_id': 3, 'wyield_mn': 1000, 'wyield_vol': 1000, 'consum_vol': 10000, 'consum_mn': 10000}]
        
        watershed_uri = _create_watershed(fields, attributes)
	hydropower.compute_rsupply_volume(watershed_uri)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
    def test_extract_datasource_table_by_key(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import hydropower
        pass
	fields = {'ws_id': 'int',
		  'wyield_mn': , 
		  'wyield_vol': , 
		  'consum_mn_mn': , 
		  'consum_vol':  
		  }
	attributes = [{'ws_id': 1, 'wyield_mn': 1000, 'wyield_vol': 1000, 'consum_vol': 10000, 'consum_mn': 10000}, {'ws_id': 2, 'wyield_mn': 1000, 'wyield_vol': 1000, 'consum_vol': 10000, 'consum_mn': 10000}, {'ws_id': 3, 'wyield_mn': 1000, 'wyield_vol': 1000, 'consum_vol': 10000, 'consum_mn': 10000}]
        
        watershed_uri = _create_watershed(fields, attributes)

	key_field = 'ws_id'
	wanted_list = ['wyield_vol', 'consum_vol']
	hydropower.extract_datasource_table_by_key(watershed_uri, key_field, wanted_list)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
    def test_write_new_table(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import hydropower
        pass
	filename = tempfile.tmpfile()
	fields = ['id', 'precip', 'volume']
	data = {0: {'id':1, 'precip': 100, 'volume': 150},
	        1: {'id':2, 'precip': 100, 'volume': 150},
	        2: {'id':3, 'precip': 100, 'volume': 150}}

	hydropower.write_new_table(filename, fields, data)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
    def test_compute_water_yield_volume(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import hydropower
        pass
	
	fields = {'ws_id': 'int',
		  'wyield_mn': , 
		  'num_pixels':  
		  }
	attributes = [{'ws_id': 1, 'wyield_mn': 1000, 'num_pixels': 20}, {'ws_id': 2, 'wyield_mn': 1000, 'num_pixels': 15}, {'ws_id': 3, 'wyield_mn': 1000, 'num_pixels': 10}]
        
        watershed_uri = _create_watershed(fields, attributes)
	pixel_area = 100
	hydropower.compute_water_yield_volume(shape_uri, pixel_area)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
    def test_add_dict_to_shape(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import hydropower 
        pass
	fields = {'ws_id': 'int'
		  }
	attributes = [{'ws_id': 1}, {'ws_id': 2}, {'ws_id': 3}]
        
        watershed_uri = _create_watershed(fields, attributes)

	field_dict = {'1': 5, '2': 10, '3': 15}
	field_name = 'test'
	key = 'ws_id'

	hydropower.add_dict_to_shape(shape_uri, field_dict, field_name, key)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
