"""Module for Regression Testing the InVEST Wind Energy module."""
import unittest
import csv
import shutil
import tempfile
import os
import pickle

import numpy
import numpy.testing
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.geometry import Point
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import pygeoprocessing

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

    Args:
        data (dict): a Dictionary where each key is the name
            of a field and set in the first column. The second
            column is set with the value of that key.
        file_path (string): a file path for the new table to be written to
            disk.

    Returns:
        None
    """
    csv_file = open(file_path, 'w')

    writer = csv.writer(csv_file)
    for key, val in data.items():
        writer.writerow([key, val])

    csv_file.close()


class WindEnergyUnitTests(unittest.TestCase):
    """Unit tests for the Wind Energy module."""

    def setUp(self):
        """Overriding setUp func. to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_calculate_distances_land_grid(self):
        """WindEnergy: testing 'calculate_distances_land_grid' function."""
        from natcap.invest import wind_energy

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        # Setup parameters for creating point shapefile
        fields = {'id': ogr.OFTReal, 'L2G': ogr.OFTReal}
        attrs = [{'id': 1, 'L2G': 10}, {'id': 2, 'L2G': 20}]

        geometries = [
            Point(pos_x + 50, pos_y - 50), Point(pos_x + 50, pos_y - 150)]
        land_shape_path = os.path.join(self.workspace_dir, 'temp_shape.shp')
        # Create point shapefile to use for testing input
        pygeoprocessing.shapely_geometry_to_vector(
            geometries, land_shape_path, projection_wkt, 'ESRI Shapefile',
            fields=fields, attribute_list=attrs, ogr_geom_type=ogr.wkbPoint)

        # Setup parameters for create raster
        matrix = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=numpy.int32)
        harvested_masked_path = os.path.join(
            self.workspace_dir, 'temp_raster.tif')
        # Create raster to use for testing input
        pygeoprocessing.numpy_array_to_raster(
            matrix, -1, (100, -100), origin, projection_wkt,
            harvested_masked_path)

        tmp_dist_final_path = os.path.join(
            self.workspace_dir, 'dist_final.tif')
        # Call function to test given testing inputs
        wind_energy._calculate_distances_land_grid(
            land_shape_path, harvested_masked_path, tmp_dist_final_path, '')

        # Compare the results
        res_array = pygeoprocessing.raster_to_numpy_array(tmp_dist_final_path)
        exp_array = numpy.array(
            [[10, 110, 210, 310], [20, 120, 220, 320]], dtype=numpy.int32)
        numpy.testing.assert_allclose(res_array, exp_array)

    def test_calculate_land_to_grid_distance(self):
        """WindEnergy: testing 'point_to_polygon_distance' function."""
        from natcap.invest import wind_energy

        # Setup parameters for creating polygon and point shapefiles
        fields = {'vec_id': ogr.OFTInteger}
        attr_pt = [{'vec_id': 1}, {'vec_id': 2}, {'vec_id': 3}, {'vec_id': 4}]
        attr_poly = [{'vec_id': 1}, {'vec_id': 2}]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        poly_geoms = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y - 100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x + 100, pos_y - 150),
                       (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200),
                       (pos_x, pos_y - 150)]}

        poly_geometries = [
            Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]
        poly_vector_path = os.path.join(self.workspace_dir, 'poly_shape.shp')
        # Create polygon shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            poly_geometries, poly_vector_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attr_poly,
            ogr_geom_type=ogr.wkbPolygon)

        point_geometries = [
            Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
            Point(pos_x, pos_y - 100), Point(pos_x + 100, pos_y - 100)]
        point_vector_path = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            point_geometries, point_vector_path, projection_wkt,
            'ESRI Shapefile', fields=fields, attribute_list=attr_pt,
            ogr_geom_type=ogr.wkbPoint)

        target_point_vector_path = os.path.join(
            self.workspace_dir, 'target_point.shp')
        # Call function to test
        field_name = 'L2G'
        wind_energy._calculate_land_to_grid_distance(
            point_vector_path, poly_vector_path, field_name,
            target_point_vector_path)

        exp_results = [.15, .1, .05, .05]

        point_vector = gdal.OpenEx(target_point_vector_path)
        point_layer = point_vector.GetLayer()
        field_index = point_layer.GetFeature(0).GetFieldIndex(field_name)
        for i, point_feat in enumerate(point_layer):
            result_val = point_feat.GetField(field_index)
            numpy.testing.assert_allclose(result_val, exp_results[i])

    def test_read_csv_wind_parameters(self):
        """WindEnergy: testing 'read_csv_wind_parameter' function."""
        from natcap.invest import wind_energy

        csv_path = os.path.join(
            SAMPLE_DATA,
            'global_wind_energy_parameters.csv')

        parameter_list = [
            'air_density', 'exponent_power_curve', 'decommission_cost',
            'operation_maintenance_cost', 'miscellaneous_capex_cost']

        result = wind_energy._read_csv_wind_parameters(
            csv_path, parameter_list)

        expected_result = {
            'air_density': 1.225,
            'exponent_power_curve': 2,
            'decommission_cost': 0.037,
            'operation_maintenance_cost': .035,
            'miscellaneous_capex_cost': .05
        }
        self.assertDictEqual(expected_result, result)

    def test_wind_data_to_point_vector(self):
        """WindEnergy: testing 'wind_data_to_point_vector' function."""
        from natcap.invest import wind_energy

        wind_data = {
            (31.79, 123.76): {
                'LONG': 123.76, 'LATI': 31.79, 'Ram-080m': 7.98,
                'K-010m': 1.90}
        }
        wind_data_pickle_path = os.path.join(
            self.workspace_dir, 'wind_data.pickle')
        pickle.dump(wind_data, open(wind_data_pickle_path, 'wb'))

        layer_name = "datatopoint"
        out_path = os.path.join(self.workspace_dir, 'datatopoint.shp')

        wind_energy._wind_data_to_point_vector(
            wind_data_pickle_path, layer_name, out_path)

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
                        wind_data[(31.79, 123.76)][field], field_val)
                except ValueError:
                    raise AssertionError(
                        'Could not find field %s' % field)

            feat = layer.GetNextFeature()

    def test_wind_data_to_point_vector_360(self):
        """WindEnergy: testing 'wind_data_to_point_vector' function.

        This test is to test that when Longitude values range from -360 to 0,
            instead of the normal -180 to 180, they are handled properly.
        """
        from natcap.invest import wind_energy

        # Set up a coordinate with a longitude in the range of -360 to 0.
        wind_data = {
            (31.79, -200): {
                'LONG': -200, 'LATI': 31.79, 'Ram-080m': 7.98,
                'K-010m': 1.90}
        }
        wind_data_pickle_path = os.path.join(
            self.workspace_dir, 'wind_data.pickle')
        pickle.dump(wind_data, open(wind_data_pickle_path, 'wb'))

        layer_name = "datatopoint"
        out_path = os.path.join(self.workspace_dir, 'datatopoint.shp')

        wind_energy._wind_data_to_point_vector(
            wind_data_pickle_path, layer_name, out_path)

        field_names = ['LONG', 'LATI', 'Ram-080m', 'K-010m']
        ogr_point = ogr.Geometry(ogr.wkbPoint)
        # Point geometry should have been converted to the WSG84 norm of
        # -180 to 180
        ogr_point.AddPoint_2D(160, 31.79)

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
                        wind_data[(31.79, -200)][field], field_val)
                except ValueError:
                    raise AssertionError(
                        'Could not find field %s' % field)

            feat = layer.GetNextFeature()

    def test_create_distance_raster(self):
        """WindEnergy: testing '_create_distance_raster' function."""
        from natcap.invest import wind_energy

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157) #UTM Zone 10N
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        # Setup and create vector to pass to function
        fields = {'id': ogr.OFTReal}
        attrs = [{'id': 1}]

        # Square polygon that will overlap the 4 pixels of the raster in the 
        # upper left corner
        poly_geometry = [box(pos_x, pos_y - 17, pos_x + 17, pos_y)]
        poly_vector_path = os.path.join(
            self.workspace_dir, 'distance_from_vector.gpkg')
        # Create polygon shapefile to use as testing input
        pygeoprocessing.shapely_geometry_to_vector(
            poly_geometry, poly_vector_path, projection_wkt,
            'GPKG', fields=fields, attribute_list=attrs,
            ogr_geom_type=ogr.wkbPolygon)

        # Create 2x5 raster
        matrix = numpy.array(
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=numpy.float32)
        base_raster_path = os.path.join(self.workspace_dir, 'temp_raster.tif')
        # Create raster to use for testing input
        pygeoprocessing.numpy_array_to_raster(
            matrix, -1, (10, -10), origin, projection_wkt, base_raster_path)

        dist_raster_path = os.path.join(self.workspace_dir, 'dist.tif')
        # Call function to test given testing inputs
        wind_energy._create_distance_raster(
            base_raster_path, poly_vector_path, dist_raster_path, 
            self.workspace_dir)

        # Compare the results
        res_array = pygeoprocessing.raster_to_numpy_array(dist_raster_path)
        exp_array = numpy.array(
            [[0, 0, 10, 20, 30], [0, 0, 10, 20, 30]], dtype=numpy.float32)
        numpy.testing.assert_allclose(res_array, exp_array)

    def test_calculate_npv_levelized_rasters(self):
        """WindEnergy: testing '_calculate_npv_levelized_rasters' function."""
        from natcap.invest import wind_energy

        val_parameters_dict = {
            'air_density': 1.225,
            'exponent_power_curve': 2,
            'decommission_cost': 0.03,
            'operation_maintenance_cost': 0.03,
            'miscellaneous_capex_cost': 0.05,
            'installation_cost': 0.2,
            'infield_cable_length': 0.9,
            'infield_cable_cost': 260000,
            'mw_coef_ac': 810000,
            'mw_coef_dc': 1090000,
            'cable_coef_ac': 1360000,
            'cable_coef_dc': 890000,
            'ac_dc_distance_break': 60,
            'time_period': 5,
            'rotor_diameter_factor': 7,
            'carbon_coefficient': 6.90E-04,
            'air_density_coefficient': 1.19E-04,
            'loss_parameter': 0.05,
            'turbine_cost': 10000,
            'turbine_rated_pwr': 5
            }
        args = {
            'foundation_cost': 1000000,
            'discount_rate': 0.01,
            'number_of_turbines': 10
            }
        price_list = [0.10, 0.10, 0.10, 0.10, 0.10]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157) #UTM Zone 10N
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        pos_x = origin[0]
        pos_y = origin[1]

        # Create harvested raster
        harvest_val = 1000000
        harvest_matrix = numpy.array(
            [[harvest_val, harvest_val + 1e5, harvest_val + 2e5,
                harvest_val + 3e5, harvest_val + 4e5],
             [harvest_val, harvest_val + 1e5, harvest_val + 2e5,
                 harvest_val + 3e5, harvest_val + 4e5],
            ], dtype=numpy.float32)
        base_harvest_path = os.path.join(self.workspace_dir, 'harvest_raster.tif')
        # Create raster to use for testing input
        pygeoprocessing.numpy_array_to_raster(
            harvest_matrix, -1, (10, -10), origin, projection_wkt, base_harvest_path)
        # Create distance raster
        dist_matrix = numpy.array(
            [[0, 10, 20, 30, 40], [0, 10, 20, 30, 40]], dtype=numpy.float32)
        base_distance_path = os.path.join(self.workspace_dir, 'dist_raster.tif')
        # Create raster to use for testing input
        pygeoprocessing.numpy_array_to_raster(
            dist_matrix, -1, (10, -10), origin, projection_wkt, base_distance_path)

        target_npv_raster_path = os.path.join(self.workspace_dir, 'npv.tif')
        target_levelized_raster_path = os.path.join(
            self.workspace_dir, 'levelized.tif')
        # Call function to test given testing inputs
        wind_energy._calculate_npv_levelized_rasters(
            base_harvest_path, base_distance_path,
            target_npv_raster_path, target_levelized_raster_path,
            val_parameters_dict, args, price_list)

        # Compare the results that were "eye" tested.
        desired_npv_array = numpy.array(
            [[309332320.0, 348331200.0, 387330020.0, 426328930.0,
               465327800.0],
             [309332320.0, 348331200.0, 387330020.0, 426328930.0,
               465327800.0]], dtype=numpy.float32)
        actual_npv_array = pygeoprocessing.raster_to_numpy_array(
            target_npv_raster_path)
        numpy.testing.assert_allclose(actual_npv_array, desired_npv_array)

        desired_levelized_array = numpy.array(
            [[0.016496297, 0.015000489, 0.0137539795, 0.01269924, 0.011795178],
             [0.016496297, 0.015000489, 0.0137539795, 0.01269924, 0.011795178]],
            dtype=numpy.float32)
        actual_levelized_array = pygeoprocessing.raster_to_numpy_array(
            target_levelized_raster_path)
        numpy.testing.assert_allclose(
            actual_levelized_array, desired_levelized_array)

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
                SAMPLE_DATA, 'resampled_global_dem_unprojected.tif'),
            'global_wind_parameters_path': os.path.join(
                SAMPLE_DATA, 'global_wind_energy_parameters.csv'),
            'turbine_parameters_path': os.path.join(
                SAMPLE_DATA, '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 180,
            'n_workers': -1
        }

        return args

    def test_no_aoi(self):
        """WindEnergy: testing base case w/o AOI, distances, or valuation."""
        from natcap.invest import wind_energy
        from natcap.invest.utils import _assert_vectors_equal

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        # Also test on input bathymetry that has equal x, y pixel sizes
        args['bathymetry_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_global_dem_equal_pixel.tif')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, 'noaoi', raster_path))
            numpy.testing.assert_allclose(model_array, reg_array)

        vector_path = 'wind_energy_points.shp'

        _assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'noaoi', vector_path))

    def test_no_land_polygon(self):
        """WindEnergy: testing case w/ AOI but w/o land poly or distances."""
        from natcap.invest import wind_energy
        from natcap.invest.utils import _assert_vectors_equal

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, 'nolandpoly', raster_path))
            numpy.testing.assert_allclose(model_array, reg_array)

        vector_path = 'wind_energy_points.shp'

        _assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'nolandpoly', vector_path))

    def test_no_distances(self):
        """WindEnergy: testing case w/ AOI and land poly, but w/o distances."""
        from natcap.invest import wind_energy
        from natcap.invest.utils import _assert_vectors_equal

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_vector_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, 'nodistances', raster_path))
            numpy.testing.assert_allclose(model_array, reg_array)

        vector_path = 'wind_energy_points.shp'

        _assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'nodistances', vector_path))

    def test_val_gridpts_windprice(self):
        """WindEnergy: testing Valuation w/ grid pts and wind price."""
        from natcap.invest import wind_energy
        from natcap.invest.utils import _assert_vectors_equal

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_vector_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2000000
        args['discount_rate'] = 0.07
        # Test that only grid points are provided in grid_points_path
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_grid_pts.csv')
        args['price_table'] = False
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2

        wind_energy.execute(args)

        # Make sure the output files were created.
        vector_path = 'wind_energy_points.shp'
        self.assertTrue(os.path.exists(
            os.path.join(args['workspace_dir'], 'output', vector_path)))

        # Run through the model again, which should mean deleting shapefiles
        # that have already been made, but which need to be created again.
        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif', 'npv.tif']

        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, 'pricevalgrid', raster_path))
            numpy.testing.assert_allclose(model_array, reg_array)

        vector_path = 'wind_energy_points.shp'

        _assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'pricevalgrid', vector_path))

    def test_val_land_grid_points(self):
        """WindEnergy: testing Valuation w/ grid/land pts and wind price."""
        from natcap.invest import wind_energy
        from natcap.invest.utils import _assert_vectors_equal
        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_vector_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2000000
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
            'carbon_emissions_tons.tif', 'levelized_cost_price_per_kWh.tif',
            'npv.tif']

        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, 'pricevalgridland', raster_path))
            # loosened tolerance to pass against GDAL 2.2.4 and 2.4.1
            numpy.testing.assert_allclose(
                model_array, reg_array, rtol=1e-04)

        vector_path = 'wind_energy_points.shp'
        _assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'pricevalgridland', vector_path))

    def test_val_no_grid_land_pts(self):
        """WindEnergy: testing Valuation without grid or land points."""
        from natcap.invest import wind_energy
        from natcap.invest.utils import _assert_vectors_equal
        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        # Also use an already projected bathymetry
        args['bathymetry_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_global_dem_projected.tif')
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_vector_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2000000
        args['discount_rate'] = 0.07
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'price_table_example.csv')
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2
        args['avg_grid_distance'] = 4

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif', 'levelized_cost_price_per_kWh.tif',
            'npv.tif']

        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, 'priceval', raster_path))
            numpy.testing.assert_allclose(model_array, reg_array, rtol=1e-6)

        vector_path = 'wind_energy_points.shp'
        _assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'priceval', vector_path))

    def test_valuation_taskgraph(self):
        """WindEnergy: testing Valuation with async TaskGraph."""
        from natcap.invest import wind_energy
        from natcap.invest.utils import _assert_vectors_equal
        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        # Also use an already projected bathymetry
        args['bathymetry_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_global_dem_projected.tif')
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_vector_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2000000
        args['discount_rate'] = 0.07
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'price_table_example.csv')
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2
        args['avg_grid_distance'] = 4
        args['n_workers'] = 1

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif', 'levelized_cost_price_per_kWh.tif',
            'npv.tif']

        for raster_path in raster_results:
            model_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output', raster_path))
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(REGRESSION_DATA, 'priceval', raster_path))
            numpy.testing.assert_allclose(model_array, reg_array, rtol=1e-6)

        vector_path = 'wind_energy_points.shp'
        _assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'priceval', vector_path))

    def test_field_error_missing_bio_param(self):
        """WindEnergy: test that ValueError raised when missing bio param."""
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
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_vector_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000
        }

        # creating a stand in turbine parameter csv file that is missing
        # the 'cut_out_wspd' entry. This should raise the exception
        tmp, file_path = tempfile.mkstemp(
            suffix='.csv', dir=args['workspace_dir'])
        os.close(tmp)
        data = {
            'hub_height': 80, 'cut_in_wspd': 4, 'rated_wspd': 12.5,
            'turbine_rated_pwr': 3.6, 'turbine_cost': 8
        }
        _create_vertical_csv(data, file_path)
        args['turbine_parameters_path'] = file_path

        self.assertRaises(ValueError, wind_energy.execute, args)

    def test_time_period_exception(self):
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
                SAMPLE_DATA, '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_vector_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2000000,
            'discount_rate': 0.07,
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': os.path.join(
                SAMPLE_DATA, 'price_table_example.csv'),
            'results_suffix': '_test'
        }

        # creating a stand in global wind params table that has a different
        # 'time' value than what is given in the wind schedule table.
        # This should raise the exception
        tmp, file_path = tempfile.mkstemp(
            suffix='.csv', dir=args['workspace_dir'])
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

    def test_clip_vector_value_error(self):
        """WindEnergy: Test AOI doesn't intersect Wind Data points."""
        from natcap.invest import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')

        # Make up some Wind Data points that live outside AOI
        wind_data_csv = os.path.join(
            args['workspace_dir'], 'temp-wind-data.csv')
        with open(wind_data_csv, 'w') as open_table:
            open_table.write('LONG,LATI,LAM,K,REF\n')
            open_table.write('-60.5,25.0,7.59,2.6,10\n')
            open_table.write('-59.5,24.0,7.59,2.6,10\n')
            open_table.write('-58.5,24.5,7.59,2.6,10\n')
            open_table.write('-58.95,24.95,7.59,2.6,10\n')
            open_table.write('-57.95,24.95,7.59,2.6,10\n')
            open_table.write('-57.95,25.95,7.59,2.6,10\n')

        args['wind_data_path'] = wind_data_csv

        # AOI and wind data should not overlap, leading to a ValueError in
        # clip_vector_by_vector
        with self.assertRaises(ValueError) as cm:
            wind_energy.execute(args)

        self.assertTrue(
            "returned 0 features. If an AOI was" in str(cm.exception))


class WindEnergyValidationTests(unittest.TestCase):
    """Tests for the Wind Energy Model ARGS_SPEC and validation."""

    def setUp(self):
        """Setup a list of required keys."""
        self.base_required_keys = [
            'workspace_dir',
            'number_of_turbines',
            'min_depth',
            'max_depth',
            'turbine_parameters_path',
            'bathymetry_path',
            'global_wind_parameters_path',
            'wind_data_path'
        ]

    def test_missing_keys(self):
        """Wind Energy Validate: assert missing required keys."""
        from natcap.invest import wind_energy
        from natcap.invest import validation

        validation_errors = wind_energy.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_with_valuation(self):
        """Wind Energy Validate: assert missing required for valuation."""
        from natcap.invest import wind_energy
        from natcap.invest import validation

        base_required_valuation = ['land_polygon_vector_path',
                                   'min_distance',
                                   'max_distance',
                                   'foundation_cost',
                                   'discount_rate']
        required_no_price_table = ['wind_price', 'rate_change']
        required_no_grid_points = ['avg_grid_distance']
        required_no_grid_distance = ['grid_points_path']

        # Test that many args become required for valuation.
        args = {'valuation_container': True}
        validation_errors = wind_energy.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            base_required_valuation +
            ['price_table'] +
            required_no_price_table +
            required_no_grid_distance +
            required_no_grid_points)
        self.assertEqual(invalid_keys, expected_missing_keys)

        # Test wind_price, rate_change are not required if price_table
        args = {
            'valuation_container': True,
            'price_table': True
        }
        validation_errors = wind_energy.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            base_required_valuation +
            ['wind_schedule'] +  # required when price_table
            required_no_grid_distance +
            required_no_grid_points)
        self.assertEqual(invalid_keys, expected_missing_keys)

        # Test grid_points_path is not required if avg_grid_distance:
        args = {
            'valuation_container': True,
            'avg_grid_distance': 9
        }
        validation_errors = wind_energy.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            base_required_valuation +
            ['price_table'] +
            required_no_price_table)
        self.assertEqual(invalid_keys, expected_missing_keys)

        # TestAOI becomes required when these two args present:
        args = {
            'valuation_container': True,
            'grid_points_path': 'foo.shp'
        }
        validation_errors = wind_energy.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            base_required_valuation +
            ['price_table'] +
            required_no_price_table +
            ['grid_points_path', 'aoi_vector_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)
