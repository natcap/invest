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


def make_simple_vector(path_to_shp, fields={"id": ogr.OFTReal},
                       attribute_list=[{"id": 0}]):
    """
    Generate shapefile with one rectangular polygon
    Args:
        path_to_shp (str): path to target shapefile
    Returns:
        None
    """
    # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)
    shapely_geometry_list = [
        Polygon([(461251, 4923195), (461501, 4923195),
                 (461501, 4923445), (461251, 4923445),
                 (461251, 4923195)])
    ]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    vector_format = "ESRI Shapefile"

    pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                               path_to_shp, projection_wkt,
                                               vector_format, fields,
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
    pixel_size = (30, -30)
    no_data = FLOAT32_NODATA

    pygeoprocessing.numpy_array_to_raster(
        array.astype(numpy.float32), no_data, pixel_size, origin, projection_wkt,
        base_raster_path)

def make_synthetic_data_and_params(workspace_dir):

        # make synthetic input data
        baseline_prevalence_path = os.path.join(workspace_dir, "baseline_prevalence.shp")
        fields={"id": ogr.OFTReal, "risk_rate": ogr.OFTReal}
        attribute_list=[{"id": 0,"risk_rate": 10}]
        make_simple_vector(baseline_prevalence_path, fields, attribute_list)

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

        health_effect_csv_path = os.path.join(workspace_dir, "health_effect.csv")
        df = pandas.DataFrame({"health_indicator":['depression'],
                               "effect_size":[0.94],
                               "exposure_metric_percent":[.1],
                               "exposure_metric_name":['NDVI'],
                               "ratio_type":['risk']})
        df.to_csv(health_effect_csv_path)

        args = {
            'aoi_vector_path': aoi_path,
            'baseline_prevalence_vector': baseline_prevalence_path,
            'effect_size_csv': health_effect_csv_path,
            'health_cost_rate_csv': None,
            'lulc_alt': '',
            'lulc_attr_csv': '',
            'lulc_base': '',
            'ndvi_alt': ndvi_alt_path,
            'ndvi_base': ndvi_base_path,
            'population_raster': pop_path,
            'results_suffix': 'test1',
            'scenario': 'ndvi',
            'search_radius': 30, # 1 pixel
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

    def test_delta_ndvi(self):
        """Test ``calc_delta_ndvi``"""
        from natcap.invest import urban_mental_health
        args = make_synthetic_data_and_params(self.workspace_dir)
        urban_mental_health.execute(args)

        intermediate = os.path.join(self.workspace_dir, "intermediate_outputs")
        output_delta_ndvi = os.path.join(intermediate, f'delta_ndvi_{args['results_suffix']}.tif')

        urban_mental_health.calc_delta_ndvi(
             os.path.join(intermediate, f'ndvi_base_buffer_mean_{args['results_suffix']}.tif'),
             os.path.join(intermediate, f'ndvi_alt_buffer_mean_{args['results_suffix']}.tif'),
             output_delta_ndvi)
        actual_delta_ndvi = pygeoprocessing.raster_to_numpy_array(output_delta_ndvi)
        
        expected_delta_ndvi = numpy.array(
             [[-8.66667000e-02, -1.27500000e-01, -4.33337000e-02],
             [-1.20000000e-01, -2.48000000e-01, -1.02500000e-01],
             [-2.27500000e-01, -1.74000000e-01, -2.21666700e-01],
             [ 6.66667000e-02, -1.84166600e-01,  FLOAT32_NODATA]])

        numpy.testing.assert_allclose(actual_delta_ndvi, expected_delta_ndvi, atol=1e-6)

    def test_option3(self):
        "Test umh option 3 (ndvi)"
        from natcap.invest import urban_mental_health
        # from "../src/natcap/invest" import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)

        urban_mental_health.execute(args)

        expected_kernel = numpy.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]])
        kernel_path = os.path.join(self.workspace_dir, "intermediate_outputs",
                                   f"kernel_{args['results_suffix']}.tif")
        actual_kernel = pygeoprocessing.raster_to_numpy_array(kernel_path)

        expected_mean_base_ndvi = numpy.array(
             [[.2666667, .3125, .4166667],
              [.5, .58, .4375],
              [.5775, .504, .5666667],
              [.3433333, .3766666, FLOAT32_NODATA]])
        actual_mean_base_ndvi_path = os.path.join(self.workspace_dir, "intermediate_outputs",
                                                  f"ndvi_base_buffer_mean_{args['results_suffix']}.tif")
        actual_mean_base_ndvi = pygeoprocessing.raster_to_numpy_array(actual_mean_base_ndvi_path)
        
        expected_mean_alt_ndvi = numpy.array(
             [[0.18, 0.185, 0.373333],
              [0.38, 0.332, 0.335],
              [0.35, 0.33, 0.345],
              [0.41, 0.1925, 0.203333]])
        actual_mean_alt_ndvi_path = actual_mean_base_ndvi_path.replace("_base_", "_alt_")
        actual_mean_alt_ndvi = pygeoprocessing.raster_to_numpy_array(actual_mean_alt_ndvi_path)

        numpy.testing.assert_allclose(actual_kernel, expected_kernel)
        numpy.testing.assert_allclose(actual_mean_base_ndvi, expected_mean_base_ndvi, atol=1e-6)
        numpy.testing.assert_allclose(actual_mean_alt_ndvi, expected_mean_alt_ndvi, atol=1e-6)
         
        expected_baseline_cases = numpy.array([[.12, .22, .1],
                                [.2, .3, .8],
                                [.9, .14, .14],
                                [.16, .17, .3]])*1000
        actual_baseline_cases_path = os.path.join(self.workspace_dir, "intermediate_outputs",
                                                  f"baseline_cases_{args['results_suffix']}.tif")
        actual_baseline_cases = pygeoprocessing.raster_to_numpy_array(actual_baseline_cases_path)
        numpy.testing.assert_allclose(actual_baseline_cases, expected_baseline_cases, atol=1e-6)

        expected_preventable_cases = numpy.array(
             [[-6.61070802, -18.05903398, -2.71753617],
             [-15.4153132, -49.75519078, -52.38134228],
             [-136.04027944, -15.91416352, -20.58117008],
             [6.46576569, -20.51906896, FLOAT32_NODATA]])
        #^ calculated using (1 - numpy.exp(numpy.log(0.94)*10*actual_delta_ndvi)) * actual_baseline_cases
        # (1 - (exp(ln(RR0.1NE)10*NE))) * bc
        actual_preventable_cases_path = os.path.join(self.workspace_dir,
                                                     f"preventable_cases_{args['results_suffix']}.tif")
        actual_preventable_cases = pygeoprocessing.raster_to_numpy_array(actual_preventable_cases_path)

        numpy.testing.assert_allclose(actual_preventable_cases, expected_preventable_cases, atol=1e-6)









