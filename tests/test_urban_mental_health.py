# coding=UTF-8
"""Tests for the Urban Mental Health Model."""
import itertools
import math
import os
import random
import shutil
import tempfile
import textwrap
import unittest

import numpy
import pandas
import pygeoprocessing
# import shapely.geometry
from shapely import Polygon
from natcap.invest import utils
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
_DEFAULT_ORIGIN = (444720, 3751320)
_DEFAULT_PIXEL_SIZE = (30, -30)
_DEFAULT_EPSG = 3116
_DEFAULT_SRS = osr.SpatialReference()
_DEFAULT_SRS.ImportFromEPSG(_DEFAULT_EPSG)



def make_simple_vector(path_to_shp, fields={"id": ogr.OFTReal},
                       attribute_list=[{"id": 0}], epsg=26910,
                       shapely_geometry_list = [ # this polygon fits within middle pixel of raster
                           Polygon([(461351, 4923191), (461451, 4923191),
                                    (461451, 4923245), (461351, 4923245),
                                    (461351, 4923191)])]):
    
    #Polygon([(461251, 4923195), (461501, 4923195),
            # (461501, 4923445), (461251, 4923445),
            # (461251, 4923195)])]): --> this is too large! good for last test rn
    """
    Generate shapefile with one rectangular polygon

    Args:
        path_to_shp (str): path to target shapefile
        fields (dict): dictionary with attribute: datatype key/value pairs
        attribute_list (list): list of attribute: value dictionaries 
        epsg (int): EPSG code
        shapely_geometry_list (list): list of shapely geometry elements. For
            polygons, order of elements in list is:
            (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)
            Note: list length must equal length of values in attribute_list dicts.

    Returns:
        None.
    """
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    projection_wkt = srs.ExportToWkt()

    pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                               path_to_shp, projection_wkt,
                                               "ESRI Shapefile", fields,
                                               attribute_list)


def make_raster_from_array(base_raster_path, array):
    """Create a raster on designated path with array values.
    Args:
        base_raster_path (str): the raster path for making the new raster.
    Returns:
        None.
    """
    # UTM Zone 10N
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    origin = (461251, 4923445)
    pixel_size = (100, -100)
    no_data = FLOAT32_NODATA

    pygeoprocessing.numpy_array_to_raster(
        array.astype(numpy.float32), no_data, pixel_size, origin, projection_wkt,
        base_raster_path)

def make_synthetic_data_and_params(workspace_dir):

        # make synthetic input data
        baseline_prevalence_path = os.path.join(workspace_dir, "baseline_prevalence.shp")
        fields={"id": ogr.OFTReal, "risk_rate": ogr.OFTReal}
        attribute_list=[{"id": 0,"risk_rate": 10}]
        make_simple_vector(baseline_prevalence_path, fields, attribute_list,
                           shapely_geometry_list=[
                                Polygon([(461251, 4923195), (461501, 4923195),
                                         (461501, 4923445), (461251, 4923445),
                                         (461251, 4923195)])])

        ndvi_base_array = numpy.array([[.1, .2, .35], [.5, .6, .7], [.8, .9, .10], [.11, .12, FLOAT32_NODATA]])
        ndvi_base_path = os.path.join(workspace_dir, "ndvi_base.tif")
        make_raster_from_array(ndvi_base_path, ndvi_base_array)

        ndvi_alt_array = numpy.array([[.12, .22, .1],
                                      [.2, .3, .8],
                                      [.9, .14, .14],
                                      [.16, .17, .3]])
        ndvi_alt_path = os.path.join(workspace_dir, "ndvi_alt.tif")
        make_raster_from_array(ndvi_alt_path, ndvi_alt_array)

        pop_array = ndvi_alt_array*100
        pop_path = os.path.join(workspace_dir, "population.tif")
        make_raster_from_array(pop_path, pop_array)

        aoi_path = os.path.join(workspace_dir, "aoi.shp")
        make_simple_vector(aoi_path)

        args = {
            'aoi_vector_path': aoi_path,
            'baseline_prevalence_vector': baseline_prevalence_path,
            'effect_size': 0.94,
            'health_cost_rate': None,
            'lulc_alt': '',
            'lulc_attr_csv': '',
            'lulc_base': '',
            'ndvi_alt': ndvi_alt_path,
            'ndvi_base': ndvi_base_path,
            'population_raster': pop_path,
            'results_suffix': 'test1',
            'scenario': 'ndvi',
            'search_radius': 100, # 1 pixel
            'tc_raster': '',
            'tc_target': '',
            'workspace_dir': workspace_dir,
        }

        return args

gdal.UseExceptions()
class UMHTests(unittest.TestCase):
    """Tests for the Urban Mental Health Model."""
   
    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the test result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_ndvi_processing(self):
        """Test ``calc_delta_ndvi``"""
        from natcap.invest import urban_mental_health
        args = make_synthetic_data_and_params(self.workspace_dir)
        urban_mental_health.execute(args)

        intermediate = os.path.join(self.workspace_dir, "intermediate")
        suffix = args['results_suffix']
        # Assert that top row was removed as its outside of search_radius distance of AOI
        expected_base_aligned = numpy.array( #this is copied from base_ndvi without top row
            [[.5, .6, .7], 
            [.8, .9, .10], 
            [.11, .12, FLOAT32_NODATA]])
        actual_base_aligned = pygeoprocessing.raster_to_numpy_array(os.path.join(intermediate, f'ndvi_base_aligned_{suffix}.tif'))
        numpy.testing.assert_allclose(actual_base_aligned, expected_base_aligned, atol=1e-6)

        # Assert that convolution was correct
        expected_base_convolve = numpy.array( #calculated by hand
            [[0.633333, 0.675, 0.4666667],
            [0.5775, 0.504, 0.5666667],
            [0.3433333, 0.3766667, FLOAT32_NODATA]]
        )
        actual_base_convolve = pygeoprocessing.raster_to_numpy_array(os.path.join(intermediate, f'ndvi_base_buffer_mean_{suffix}.tif'))
        numpy.testing.assert_allclose(actual_base_convolve, expected_base_convolve, atol=1e-6)

        expected_alt_convolve = numpy.array(
            [[0.466667, 0.36, 0.413333],
            [0.35, 0.33, 0.345],
            [0.41, 0.1925, 0.203333]])
        
        output_delta_ndvi = os.path.join(intermediate, f'delta_ndvi_{suffix}.tif')

        urban_mental_health.calc_delta_ndvi(
             os.path.join(intermediate, f'ndvi_base_buffer_mean_{suffix}.tif'),
             os.path.join(intermediate, f'ndvi_alt_buffer_mean_{suffix}.tif'),
             output_delta_ndvi)
        actual_delta_ndvi = pygeoprocessing.raster_to_numpy_array(output_delta_ndvi)
        
        expected_delta_ndvi = expected_alt_convolve - expected_base_convolve
        expected_delta_ndvi[2, 2] = FLOAT32_NODATA

        numpy.testing.assert_allclose(actual_delta_ndvi, expected_delta_ndvi, atol=1e-6)

    def test_option3(self):
        "Test umh option 3 (ndvi)"
        from natcap.invest import urban_mental_health
        # from "../src/natcap/invest" import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)

        urban_mental_health.execute(args)

        # expected_kernel = numpy.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]])
        # kernel_path = os.path.join(self.workspace_dir, "intermediate",
        #                            f"kernel_{args['results_suffix']}.tif")
        # actual_kernel = pygeoprocessing.raster_to_numpy_array(kernel_path)
        # numpy.testing.assert_allclose(actual_kernel, expected_kernel)
        expected_baseline_cases = numpy.array([
                            [200, 300, 800],
                            [900, 140, 140],
                            [FLOAT32_NODATA, FLOAT32_NODATA, FLOAT32_NODATA]])
        actual_baseline_cases_path = os.path.join(self.workspace_dir, "intermediate",
                                                  f"baseline_cases_{args['results_suffix']}.tif")
        actual_baseline_cases = pygeoprocessing.raster_to_numpy_array(actual_baseline_cases_path)
        numpy.testing.assert_allclose(actual_baseline_cases, expected_baseline_cases, atol=1e-6)

        expected_preventable_cases = numpy.array([
            [-21.72614, -64.55958, -26.8406096],
            [-136.040285, -15.914164, -20.58118],
            [FLOAT32_NODATA, FLOAT32_NODATA, FLOAT32_NODATA]])
        #^ calculated using (1 - numpy.exp(numpy.log(0.94)*10*actual_delta_ndvi)) * actual_baseline_cases
        # (1 - (exp(ln(RR0.1NE)10*NE))) * bc
        expected_preventable_cases = numpy.array([ # only center pixel left bc AOI is small
            [FLOAT32_NODATA, FLOAT32_NODATA, FLOAT32_NODATA],
            [FLOAT32_NODATA, -15.914164, FLOAT32_NODATA],
            [FLOAT32_NODATA, FLOAT32_NODATA, FLOAT32_NODATA]])
        actual_preventable_cases_path = os.path.join(self.workspace_dir,
                                                     f"output/preventable_cases_{args['results_suffix']}.tif")
        actual_preventable_cases = pygeoprocessing.raster_to_numpy_array(actual_preventable_cases_path)

        numpy.testing.assert_allclose(actual_preventable_cases, expected_preventable_cases, atol=1e-5)

    
    def test_diff_prj_inputs(self):
        """Test model option 3 given inputs of different projections.
        
        Test that creating a target bounding box from inputs with
        different projections still produce expected target bbox
        """
        from natcap.invest import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)

        #create AOI w different projection
        epsg = 5070
        # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)
        # xmin = -2151419.7
        # ymax = 2699283.3
        # xmax = xmin + 1000
        # ymin = ymax - 1000
        #make tiny bbox for AOI
        xmin = -2151375#-2151419.7 +10
        ymax = 2699058.8 #2699283.3 +10
        xmax = xmin + 50
        ymin = ymax - 50
        geom = [Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])]
        new_aoi_path = os.path.join(self.workspace_dir, "AOI_5070.shp")

        make_simple_vector(new_aoi_path, epsg=epsg, shapely_geometry_list=geom)

        args["aoi_vector_path"] = new_aoi_path

        urban_mental_health.execute(args)

        #TODO: add assertion
        
    def test_AOI_too_large(self):
        pass









