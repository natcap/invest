"""Module for Regression Testing the InVEST Wind Energy module."""
import unittest
import tempfile
import shutil
import os
import csv

import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
import numpy
import numpy.testing
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'wind_energy',
    'input')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'wind_energy')


def _create_vertical_csv(data, file_path):
    """Create a new CSV table where the fields are in the left column.

    This CSV table is created with fields / keys running vertically
        down the first column. The second column has the corresponding
        values. This is how Wind Energy csv inputs are expected.

    Parameters:
        data (dict): a Dictionary where each key is the name
            of a field and set in the first column. The second
            column is set with the value of that key.
        file_path (string): a file path for the new table to be written to disk.

    Returns:
        None
    """
    csv_file = open(file_path, 'wb')

    writer = csv.writer(csv_file)
    for key, val in data.iteritems():
        writer.writerow([key, val])

    csv_file.close()


class WindEnergyUnitTests(unittest.TestCase):
    """Unit tests for the Wind Energy module."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_calculate_distances_land_grid(self):
        """WindEnergy: testing 'calculate_distances_land_grid' function."""
        from natcap.invest import wind_energy

        # Setup parameters for creating point shapefile
        fields = {'id': 'real', 'L2G': 'real'}
        attrs = [{'id': 1, 'L2G': 10}, {'id': 2, 'L2G': 20}]
        srs = sampledata.SRS_WILLAMETTE
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]
        geometries = [
            Point(pos_x + 50, pos_y - 50), Point(pos_x + 50, pos_y - 150)]
        shape_path = os.path.join(self.workspace_dir, 'temp_shape.shp')
        # Create point shapefile to use for testing input
        land_shape_path = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attrs,
            vector_format='ESRI Shapefile', filename=shape_path)

        # Setup parameters for create raster
        matrix = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        raster_path = os.path.join(self.workspace_dir, 'temp_raster.tif')
        # Create raster to use for testing input
        harvested_masked_path = pygeoprocessing.testing.create_raster_on_disk(
            [matrix], srs.origin, srs.projection, -1, srs.pixel_size(100),
            datatype=gdal.GDT_Int32, filename=raster_path)

        tmp_dist_final_path = os.path.join(self.workspace_dir, 'dist_final.tif')
        # Call function to test given testing inputs
        wind_energy.calculate_distances_land_grid(
            land_shape_path, harvested_masked_path, tmp_dist_final_path)

        # Compare the results
        result = gdal.Open(tmp_dist_final_path)
        res_band = result.GetRasterBand(1)
        res_array = res_band.ReadAsArray()
        exp_array = numpy.array([[10, 110, 210, 310], [20, 120, 220, 320]])
        numpy.testing.assert_array_equal(res_array, exp_array)

    def test_point_to_polygon_distance(self):
        """WindEnergy: testing 'point_to_polygon_distance' function."""
        from natcap.invest import wind_energy

        # Setup parameters for creating polygon and point shapefiles
        fields = {'vec_id': 'int'}
        attr_pt = [{'vec_id': 1}, {'vec_id': 2}, {'vec_id': 3}, {'vec_id': 4}]
        attr_poly = [{'vec_id': 1}, {'vec_id': 2}]

        srs = sampledata.SRS_WILLAMETTE
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        poly_geoms = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y - 100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x + 100, pos_y - 150),
                       (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200),
                       (pos_x, pos_y - 150)]}

        poly_geometries = [
            Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]
        poly_file = os.path.join(self.workspace_dir, 'poly_shape.shp')
        # Create polygon shapefile to use as testing input
        poly_ds_path = pygeoprocessing.testing.create_vector_on_disk(
            poly_geometries, srs.projection, fields, attr_poly,
            vector_format='ESRI Shapefile', filename=poly_file)

        point_geometries = [
            Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
            Point(pos_x, pos_y - 100), Point(pos_x + 100, pos_y - 100)]
        point_file = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        point_ds_path = pygeoprocessing.testing.create_vector_on_disk(
            point_geometries, srs.projection, fields, attr_pt,
            vector_format='ESRI Shapefile', filename=point_file)
        # Call function to test
        results = wind_energy.point_to_polygon_distance(
            poly_ds_path, point_ds_path)

        exp_results = [.15, .1, .05, .05]

        for dist_a, dist_b in zip(results, exp_results):
            pygeoprocessing.testing.assert_close(
                dist_a, dist_b)

    def test_add_field_to_shape_given_list(self):
        """WindEnergy: testing 'add_field_to_shape_given_list' function."""
        from natcap.invest import wind_energy

        # Setup parameters for point shapefile
        fields = {'pt_id': 'int'}
        attributes = [{'pt_id': 1}, {'pt_id': 2}, {'pt_id': 3}, {'pt_id': 4}]
        srs = sampledata.SRS_WILLAMETTE
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        geometries = [Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
                      Point(pos_x, pos_y - 100),
                      Point(pos_x + 100, pos_y - 100)]
        point_file = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile for testing input
        shape_ds_path = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attributes,
            vector_format='ESRI Shapefile', filename=point_file)

        value_list = [10, 20, 30, 40]
        field_name = "num_turb"
        # Call function to test
        wind_energy.add_field_to_shape_given_list(
            shape_ds_path, value_list, field_name)

        # Compare results
        results = {1: {'num_turb': 10}, 2: {'num_turb': 20},
                   3: {'num_turb': 30}, 4: {'num_turb': 40}}

        shape = ogr.Open(shape_ds_path)
        layer_count = shape.GetLayerCount()

        for layer_num in range(layer_count):
            layer = shape.GetLayer(layer_num)

            feat = layer.GetNextFeature()
            while feat is not None:
                pt_id = feat.GetField('pt_id')

                try:
                    field_val = feat.GetField(field_name)
                    pygeoprocessing.testing.assert_close(
                        results[pt_id][field_name], field_val)
                except ValueError:
                    raise AssertionError(
                        'Could not find field %s' % field_name)

                feat = layer.GetNextFeature()

    def test_combine_dictionaries(self):
        """WindEnergy: testing 'combine_dictionaries' function."""
        from natcap.invest import wind_energy

        dict_1 = {"name": "bob", "age": 3, "sex": "female"}
        dict_2 = {"hobby": "crawling", "food": "milk"}

        result = wind_energy.combine_dictionaries(dict_1, dict_2)

        expected_result = {"name": "bob", "age": 3, "sex": "female",
                           "hobby": "crawling", "food": "milk"}

        self.assertDictEqual(expected_result, result)

    def test_combine_dictionaries_duplicates(self):
        """WindEnergy: testing 'combine_dictionaries' function w/ duplicates."""
        from natcap.invest import wind_energy

        dict_1 = {"name": "bob", "age": 3, "sex": "female"}
        dict_2 = {"hobby": "crawling", "food": "milk", "age": 4}

        result = wind_energy.combine_dictionaries(dict_1, dict_2)

        expected_result = {"name": "bob", "age": 3, "sex": "female",
                           "hobby": "crawling", "food": "milk"}

        self.assertDictEqual(expected_result, result)

    def test_read_csv_wind_parameters(self):
        """WindEnergy: testing 'read_csv_wind_parameter' function."""
        from natcap.invest import wind_energy

        csv_path = os.path.join(
            SAMPLE_DATA,
            'global_wind_energy_parameters.csv')

        parameter_list = [
            'air_density', 'exponent_power_curve', 'decommission_cost',
            'operation_maintenance_cost', 'miscellaneous_capex_cost']

        result = wind_energy.read_csv_wind_parameters(csv_path, parameter_list)

        expected_result = {
            'air_density': 1.225, 'exponent_power_curve': 2.0,
            'decommission_cost': 0.037000000000000005,
            'operation_maintenance_cost': .035, 'miscellaneous_capex_cost': .05
        }
        self.assertDictEqual(expected_result, result)

    def test_create_wind_farm_box(self):
        """WindEnergy: testing 'create_wind_farm_box' function."""
        from natcap.invest import wind_energy

        # Setup parameters for creating polyline shapefile
        fields = {'id': 'real'}
        attributes = [{'id': 1}]
        srs = sampledata.SRS_WILLAMETTE
        spat_ref = osr.SpatialReference()
        spat_ref.ImportFromWkt(srs.projection)
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        geometries = [LinearRing([(pos_x + 100, pos_y),
                      (pos_x + 100, pos_y + 150), (pos_x + 200, pos_y + 150),
                      (pos_x + 200, pos_y), (pos_x + 100, pos_y)])]

        farm_1 = os.path.join(self.workspace_dir, 'farm_1')
        os.mkdir(farm_1)
        farm_file = os.path.join(farm_1, 'vector.shp')
        # Create polyline shapefile to use to test against
        farm_ds_path = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attributes,
            vector_format='ESRI Shapefile', filename=farm_file)

        start_point = (pos_x + 100, pos_y)
        x_len = 100
        y_len = 150

        farm_2 = os.path.join(self.workspace_dir, 'farm_2')
        os.mkdir(farm_2)
        out_path = os.path.join(farm_2, 'vector.shp')
        # Call the function to test
        wind_energy.create_wind_farm_box(
            spat_ref, start_point, x_len, y_len, out_path)
        # Compare results
        pygeoprocessing.testing.assert_vectors_equal(
            out_path, farm_ds_path, 1E-6)

    def test_get_highest_harvested_geom(self):
        """WindEnergy: testing 'get_highest_harvested_geom' function."""
        from natcap.invest import wind_energy

        # Setup parameters for creating point shapefile
        fields = {'pt_id': 'int', 'Harv_MWhr': 'real'}
        attributes = [{'pt_id': 1, 'Harv_MWhr': 20.5},
                      {'pt_id': 2, 'Harv_MWhr': 24.5},
                      {'pt_id': 3, 'Harv_MWhr': 13},
                      {'pt_id': 4, 'Harv_MWhr': 15}]
        srs = sampledata.SRS_WILLAMETTE
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        geometries = [Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
                      Point(pos_x, pos_y - 100),
                      Point(pos_x + 100, pos_y - 100)]
        point_file = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use for testing input
        shape_ds_path = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attributes,
            vector_format='ESRI Shapefile', filename=point_file)
        # Call function to test
        result = wind_energy.get_highest_harvested_geom(shape_ds_path)

        ogr_point = ogr.Geometry(ogr.wkbPoint)
        ogr_point.AddPoint_2D(443823.12732787791, 4956546.9059804128)

        if not ogr_point.Equals(result):
            raise AssertionError(
                'Expected geometry %s is not equal to the result %s' %
                (ogr_point, result))

    def test_pixel_size_transform(self):
        """WindEnergy: testing pixel size transform helper function.

        Function name is : 'pixel_size_based_on_coordinate_transform_path'.
        """
        from natcap.invest import wind_energy

        srs = sampledata.SRS_WILLAMETTE
        srs_wkt = srs.projection
        spat_ref = osr.SpatialReference()
        spat_ref.ImportFromWkt(srs_wkt)

        # Define a Lat/Long WGS84 projection
        epsg_id = 4326
        reference = osr.SpatialReference()
        proj_result = reference.ImportFromEPSG(epsg_id)
        if proj_result != 0:
            raise RuntimeError('EPSG code %s not recognized' % epsg_id)
        # Get projection as WKT
        latlong_proj = reference.ExportToWkt()
        # Set origin to use for setting up geometries / geotransforms
        latlong_origin = (-70.5, 42.5)
        # Pixel size helper for defining lat/long pixel size
        pixel_size = lambda x: (x, -1. * x)

        # Get a point from the clipped data object to use later in helping
        # determine proper pixel size
        matrix = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        input_path = os.path.join(self.workspace_dir, 'input_raster.tif')
        # Create raster to use as testing input
        raster_path = pygeoprocessing.testing.create_raster_on_disk(
            [matrix], latlong_origin, latlong_proj, -1.0,
            pixel_size(0.033333), filename=input_path)

        raster_gt = pygeoprocessing.geoprocessing.get_raster_info(
            raster_path)['geotransform']
        point = (raster_gt[0], raster_gt[3])
        raster_wkt = latlong_proj

        # Create a Spatial Reference from the rasters WKT
        raster_sr = osr.SpatialReference()
        raster_sr.ImportFromWkt(raster_wkt)

        # A coordinate transformation to help get the proper pixel size of
        # the reprojected raster
        coord_trans = osr.CoordinateTransformation(raster_sr, spat_ref)
        # Call the function to test
        result = wind_energy.pixel_size_based_on_coordinate_transform_path(
            raster_path, coord_trans, point)

        expected_res = (5553.93306384, 1187.37081348)

        # Compare
        for res, exp in zip(result, expected_res):
            pygeoprocessing.testing.assert_close(res, exp)

    def test_calculate_distances_grid(self):
        """WindEnergy: testing 'calculate_distances_grid' function."""
        from natcap.invest import wind_energy

        # Setup parameters to create point shapefile
        fields = {'id': 'real'}
        attrs = [{'id': 1}, {'id': 2}]
        srs = sampledata.SRS_WILLAMETTE
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]
        geometries = [Point(pos_x + 50, pos_y - 50),
                      Point(pos_x + 50, pos_y - 150)]
        point_file = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        land_shape_path = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attrs,
            vector_format='ESRI Shapefile', filename=point_file)

        matrix = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        # Create raster to use as testing input
        harvested_masked_path = pygeoprocessing.testing.create_raster_on_disk(
            [matrix], srs.origin, srs.projection, -1, srs.pixel_size(100),
            datatype=gdal.GDT_Int32, filename=raster_path)

        tmp_dist_final_path = os.path.join(self.workspace_dir, 'dist_final.tif')
        # Call function to test
        wind_energy.calculate_distances_grid(
            land_shape_path, harvested_masked_path, tmp_dist_final_path)

        # Compare
        exp_array = numpy.array([[0, 100, 200, 300], [0, 100, 200, 300]])
        res_raster = gdal.Open(tmp_dist_final_path)
        res_band = res_raster.GetRasterBand(1)
        res_array = res_band.ReadAsArray()
        numpy.testing.assert_array_equal(res_array, exp_array)

    def test_wind_data_to_point_shape(self):
        """WindEnergy: testing 'wind_data_to_point_shape' function."""
        from natcap.invest import wind_energy

        dict_data = {
            (31.79, 123.76): {
                'LONG': 123.76, 'LATI': 31.79, 'Ram-080m': 7.98,
                'K-010m': 1.90}
        }

        layer_name = "datatopoint"
        out_path = os.path.join(self.workspace_dir, 'datatopoint.shp')

        wind_energy.wind_data_to_point_shape(dict_data, layer_name, out_path)

        field_names = ['LONG', 'LATI', 'Ram-080m', 'K-010m']
        ogr_point = ogr.Geometry(ogr.wkbPoint)
        ogr_point.AddPoint_2D(123.76, 31.79)

        shape = ogr.Open(out_path)
        layer = shape.GetLayer()

        feat = layer.GetNextFeature()
        while feat is not None:

            geom = feat.GetGeometryRef()
            if bool(geom.Equals(ogr_point)) is False:
                raise AssertionError(
                    'Geometries are not equal. Expected is: %s, '
                    'but current is %s' % (ogr_point, geom))

            for field in field_names:
                try:
                    field_val = feat.GetField(field)
                    self.assertEqual(
                        dict_data[(31.79, 123.76)][field], field_val)
                except ValueError:
                    raise AssertionError(
                        'Could not find field %s' % field)

            feat = layer.GetNextFeature()

    def test_wind_data_to_point_shape_360(self):
        """WindEnergy: testing 'wind_data_to_point_shape' function.

        This test is to test that when Longitude values range from -360 to 0,
            instead of the normal -180 to 180, they are handled properly.
        """
        from natcap.invest import wind_energy

        # Set up a coordinate with a longitude in the range of -360 to 0.
        dict_data = {
            (31.79, -200.0): {
                'LONG': -200.0, 'LATI': 31.79, 'Ram-080m': 7.98,
                'K-010m': 1.90}
        }

        layer_name = "datatopoint"
        out_path = os.path.join(self.workspace_dir, 'datatopoint.shp')

        wind_energy.wind_data_to_point_shape(dict_data, layer_name, out_path)

        field_names = ['LONG', 'LATI', 'Ram-080m', 'K-010m']
        ogr_point = ogr.Geometry(ogr.wkbPoint)
        # Point geometry should have been converted to the WSG84 norm of
        # -180 to 180
        ogr_point.AddPoint_2D(160.00, 31.79)

        shape = ogr.Open(out_path)
        layer = shape.GetLayer()

        feat = layer.GetNextFeature()
        while feat is not None:

            geom = feat.GetGeometryRef()
            if bool(geom.Equals(ogr_point)) is False:
                raise AssertionError(
                    'Geometries are not equal. Expected is: %s, '
                    'but current is %s' % (ogr_point, geom))

            for field in field_names:
                try:
                    field_val = feat.GetField(field)
                    self.assertEqual(
                        dict_data[(31.79, -200.0)][field], field_val)
                except ValueError:
                    raise AssertionError(
                        'Could not find field %s' % field)

            feat = layer.GetNextFeature()


class WindEnergyRegressionTests(unittest.TestCase):
    """Regression tests for the Wind Energy module."""

    def setUp(self):
        """Override setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across regression tests."""
        args = {
            'workspace_dir': workspace_dir,
            'wind_data_path': os.path.join(
                SAMPLE_DATA, 'resampled_wind_points.csv'),
            'bathymetry_path': os.path.join(
                SAMPLE_DATA, 'resampled_global_dem.tif'),
            'global_wind_parameters_path': os.path.join(
                SAMPLE_DATA, 'global_wind_energy_parameters.csv'),
            'turbine_parameters_path': os.path.join(
                SAMPLE_DATA, '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 60
            }

        return args

    def test_no_aoi(self):
        """WindEnergy: testing base case w/o AOI, distances, or valuation."""
        from natcap.invest import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'noaoi', raster_path), 1E-6)

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'noaoi', vector_path), 1E-6)

    def test_no_land_polygon(self):
        """WindEnergy: testing case w/ AOI but w/o land poly or distances."""
        from natcap.invest import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif',	'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'nolandpoly', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'nolandpoly', vector_path), 1E-6)

    def test_no_distances(self):
        """WindEnergy: testing case w/ AOI and land poly, but w/o distances."""
        from natcap.invest import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'nodistances', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'nodistances', vector_path), 1E-6)

    def test_val_gridpts_windprice(self):
        """WindEnergy: testing Valuation w/ grid pts and wind price."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        # Test that only grid points are provided in grid_points_path
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_grid_pts.csv')
        args['price_table'] = False
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif',	'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricevalgrid', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricevalgrid', vector_path), 1E-6)

    def test_val_land_grid_points(self):
        """WindEnergy: testing Valuation w/ grid/land pts and wind price."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        # there was no sample data that provided landing points, thus for
        # testing, grid points in 'resampled_grid_pts.csv' were duplicated and
        # marked as land points. So the distances will be zero, keeping the
        # result the same but testing that section of code
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_grid_land_pts.csv')
        args['price_table'] = False
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif',	'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricevalgridland', raster_path),
                1E-6)

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricevalgridland', vector_path),
                1E-6)

    def test_grid_points_no_aoi(self):
        """WindEnergy: testing ValueError raised w/ grid points but w/o AOI."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['land_polygon_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        # Provide the grid points but not AOI
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_grid_pts.csv')
        args['price_table'] = False
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2

        self.assertRaises(ValueError, wind_energy.execute, args)

    def test_field_error_missing_bio_param(self):
        """WindEnergy: testing that ValueError raised when missing bio param."""
        from natcap.invest import wind_energy

        # for testing raised exceptions, running on a set of data that was
        # created by hand and has no numerical validity. Helps test the
        # raised exception quicker
        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.csv'),
            'bathymetry_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'global_wind_parameters_path': os.path.join(
                SAMPLE_DATA,
                'global_wind_energy_parameters.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000
        }

        # creating a stand in turbine parameter csv file that is missing
        # a biophysical field / value. This should raise the exception
        tmp, file_path = tempfile.mkstemp(suffix='.csv',
                                          dir=args['workspace_dir'])
        os.close(tmp)
        data = {
            'hub_height': 80, 'cut_in_wspd': 4.0, 'rated_wspd': 12.5,
            'cut_out_wspd': 25.0, 'turbine_rated_pwr': 3.6,
            'turbine_cost': 8.0, 'turbines_per_circuit': 8
        }
        _create_vertical_csv(data, file_path)
        args['turbine_parameters_path'] = file_path

        self.assertRaises(ValueError, wind_energy.execute, args)

    def test_missing_valuation_params(self):
        """WindEnergy: testing that ValueError is thrown when val params miss."""
        from natcap.invest import wind_energy

        # for testing raised exceptions, running on a set of data that was
        # created by hand and has no numerical validity. Helps test the
        # raised exception quicker
        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.csv'),
            'bathymetry_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'global_wind_parameters_path': os.path.join(
                SAMPLE_DATA, 'global_wind_energy_parameters.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2,
            'discount_rate': 0.07,
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': os.path.join(
                SAMPLE_DATA, 'price_table_example.csv'),
            'suffix': '_test'
        }

        # creating a stand in turbine parameter csv file that is missing
        # a valuation field / value. This should raise the exception
        tmp, file_path = tempfile.mkstemp(suffix='.csv',
                                          dir=args['workspace_dir'])
        os.close(tmp)
        data = {
            'hub_height': 80, 'cut_in_wspd': 4.0, 'rated_wspd': 12.5,
            'cut_out_wspd': 25.0, 'turbine_rated_pwr': 3.6,
            'turbines_per_circuit': 8, 'rotor_diameter': 40
        }
        _create_vertical_csv(data, file_path)
        args['turbine_parameters_path'] = file_path

        self.assertRaises(ValueError, wind_energy.execute, args)

    def test_time_period_exceptoin(self):
        """WindEnergy: raise ValueError if 'time' and 'wind_sched' differ."""
        from natcap.invest import wind_energy

        # for testing raised exceptions, running on a set of data that was
        # created by hand and has no numerical validity. Helps test the
        # raised exception quicker
        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.csv'),
            'bathymetry_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'turbine_parameters_path': os.path.join(
                SAMPLE_DATA,
                '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2,
            'discount_rate': 0.07,
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': os.path.join(
                SAMPLE_DATA, 'price_table_example.csv'),
            'suffix': '_test'
        }

        # creating a stand in global wind params table that has a different
        # 'time' value than what is given in the wind schedule table.
        # This should raise the exception
        tmp, file_path = tempfile.mkstemp(suffix='.csv',
                                          dir=args['workspace_dir'])
        os.close(tmp)
        data = {
            'air_density': 1.225, 'exponent_power_curve': 2,
            'decommission_cost': .037, 'operation_maintenance_cost': .035,
            'miscellaneous_capex_cost': .05, 'installation_cost': .20,
            'infield_cable_length': 0.91, 'infield_cable_cost': 0.26,
            'mw_coef_ac': .81, 'mw_coef_dc': 1.09, 'cable_coef_ac': 1.36,
            'cable_coef_dc': .89, 'ac_dc_distance_break': 60,
            'time_period': 10, 'rotor_diameter_factor': 7,
            'carbon_coefficient': 6.8956e-4,
            'air_density_coefficient': 1.194e-4, 'loss_parameter': .05
        }
        _create_vertical_csv(data, file_path)
        args['global_wind_parameters_path'] = file_path

        self.assertRaises(ValueError, wind_energy.execute, args)

    def test_remove_datasources(self):
        """WindEnergy: testing datasources which already exist are removed."""
        from natcap.invest import wind_energy

        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.csv'),
            'bathymetry_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'global_wind_parameters_path': os.path.join(
                SAMPLE_DATA,
                'global_wind_energy_parameters.csv'),
            'turbine_parameters_path': os.path.join(
                SAMPLE_DATA,
                '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2,
            'discount_rate': 0.07,
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': os.path.join(
                SAMPLE_DATA, 'price_table_example.csv'),
        }

        wind_energy.execute(args)

        # Make sure the output files were created.
        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']
        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

        # Run through the model again, which should mean deleting
        # shapefiles that have already been made, but which need
        # to be created again.
        wind_energy.execute(args)

        # For testing, just check to make sure the output files
        # were created again.
        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))
