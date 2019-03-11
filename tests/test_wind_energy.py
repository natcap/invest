"""Module for Regression Testing the InVEST Wind Energy module."""
import unittest
import csv
import shutil
import tempfile
import os
import pickle
import re

import numpy
import numpy.testing
from shapely.geometry import Polygon
from shapely.geometry import Point
from osgeo import gdal
from osgeo import ogr

import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata

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
        wind_energy._calculate_distances_land_grid(
            land_shape_path, harvested_masked_path, tmp_dist_final_path, '')

        # Compare the results
        result = gdal.Open(tmp_dist_final_path)
        res_band = result.GetRasterBand(1)
        res_array = res_band.ReadAsArray()
        exp_array = numpy.array([[10, 110, 210, 310], [20, 120, 220, 320]])
        numpy.testing.assert_array_equal(res_array, exp_array)

    def test_calculate_land_to_grid_distance(self):
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
        poly_vector_path = pygeoprocessing.testing.create_vector_on_disk(
            poly_geometries, srs.projection, fields, attr_poly,
            vector_format='ESRI Shapefile', filename=poly_file)

        point_geometries = [
            Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
            Point(pos_x, pos_y - 100), Point(pos_x + 100, pos_y - 100)]
        point_file = os.path.join(self.workspace_dir, 'point_shape.shp')
        # Create point shapefile to use as testing input
        point_vector_path = pygeoprocessing.testing.create_vector_on_disk(
            point_geometries, srs.projection, fields, attr_pt,
            vector_format='ESRI Shapefile', filename=point_file)
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
            pygeoprocessing.testing.assert_close(result_val, exp_results[i])

    def test_read_csv_wind_parameters(self):
        """WindEnergy: testing 'read_csv_wind_parameter' function."""
        from natcap.invest import wind_energy

        csv_path = os.path.join(
            SAMPLE_DATA,
            'global_wind_energy_parameters.csv')

        parameter_list = [
            'air_density', 'exponent_power_curve', 'decommission_cost',
            'operation_maintenance_cost', 'miscellaneous_capex_cost']

        result = wind_energy._read_csv_wind_parameters(csv_path, parameter_list)

        expected_result = {
            'air_density': 1.225, 'exponent_power_curve': 2.0,
            'decommission_cost': 0.037000000000000005,
            'operation_maintenance_cost': .035, 'miscellaneous_capex_cost': .05
        }
        self.assertDictEqual(expected_result, result)

    def test_calculate_grid_dist_on_raster(self):
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
        wind_energy._calculate_grid_dist_on_raster(
            land_shape_path, harvested_masked_path, tmp_dist_final_path, '')

        # Compare
        exp_array = numpy.array([[0, 100, 200, 300], [0, 100, 200, 300]])
        res_raster = gdal.Open(tmp_dist_final_path)
        res_band = res_raster.GetRasterBand(1)
        res_array = res_band.ReadAsArray()
        numpy.testing.assert_array_equal(res_array, exp_array)

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
            (31.79, -200.0): {
                'LONG': -200.0, 'LATI': 31.79, 'Ram-080m': 7.98,
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
                        wind_data[(31.79, -200.0)][field], field_val)
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

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        # Also test on input bathymetry that has equal x, y pixel sizes
        args['bathymetry_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_global_dem_equal_pixel.tif')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'noaoi', raster_path), 1E-6)

        vector_path = 'wind_energy_points.shp'

        WindEnergyRegressionTests._assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'noaoi', vector_path))

    def test_no_land_polygon(self):
        """WindEnergy: testing case w/ AOI but w/o land poly or distances."""
        from natcap.invest import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif',	'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'nolandpoly', raster_path))

        vector_path = 'wind_energy_points.shp'

        WindEnergyRegressionTests._assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'nolandpoly', vector_path))

    def test_no_distances(self):
        """WindEnergy: testing case w/ AOI and land poly, but w/o distances."""
        from natcap.invest import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_vector_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'nodistances', raster_path))

        vector_path = 'wind_energy_points.shp'

        WindEnergyRegressionTests._assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'nodistances', vector_path))

    def test_val_gridpts_windprice(self):
        """WindEnergy: testing Valuation w/ grid pts and wind price."""
        from natcap.invest import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_vector_path'] = os.path.join(
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

        # Make sure the output files were created.
        vector_path = 'wind_energy_points.shp'
        self.assertTrue(os.path.exists(
            os.path.join(args['workspace_dir'], 'output', vector_path)))

        # Run through the model again, which should mean deleting shapefiles
        # that have already been made, but which need to be created again.
        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif',	'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricevalgrid', raster_path),
                1E-6)

        vector_path = 'wind_energy_points.shp'

        WindEnergyRegressionTests._assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'pricevalgrid', vector_path))

    def test_val_land_grid_points(self):
        """WindEnergy: testing Valuation w/ grid/land pts and wind price."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp')
        args['land_polygon_vector_path'] = os.path.join(
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

        vector_path = 'wind_energy_points.shp'
        WindEnergyRegressionTests._assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'pricevalgridland', vector_path))

    def test_val_no_grid_land_pts(self):
        """WindEnergy: testing Valuation without grid or land points."""
        from natcap.invest import wind_energy
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
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
                SAMPLE_DATA, 'price_table_example.csv')
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2
        args['avg_grid_distance'] = 4

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif', 'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'priceval', raster_path),
                1E-6)

        vector_path = 'wind_energy_points.shp'
        WindEnergyRegressionTests._assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_path),
            os.path.join(REGRESSION_DATA, 'priceval', vector_path))

    def test_grid_points_no_aoi(self):
        """WindEnergy: testing ValueError raised w/ grid points but w/o AOI."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['land_polygon_vector_path'] = os.path.join(
            SAMPLE_DATA, 'simple_north_america_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        # Provide the grid points but not AOI
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_grid_pts.csv')
        args['aoi_vector_path'] = ''  # Empty AOI value
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
            'aoi_vector_path': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_vector_path': os.path.join(
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

    @staticmethod
    def generate_validate_args(workspace_dir):
        """Generate an args list for testing validate function in the model."""
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
            'aoi_vector_path': os.path.join(
                SAMPLE_DATA, 'New_England_US_Aoi.shp'),
            'land_polygon_vector_path': os.path.join(
                SAMPLE_DATA, 'simple_north_america_polygon.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2,
            'discount_rate': 0.07,
            'grid_points_path': os.path.join(
                SAMPLE_DATA, 'resampled_grid_pts.csv'),
            'price_table': False,
            'wind_price': 0.187,
            'rate_change': 0.2,
            'n_workers': -1
            }

        return args

    def test_validate_missing_distance(self):
        """WindEnergy: testing validate with missing max_distance."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Testing when min_distance is provided, but max_distance is None.
        args['max_distance'] = None
        validation_error_list = wind_energy.validate(args)
        expected_error = (['max_distance'], 'Value must be defined.')
        self.assertTrue(expected_error in validation_error_list)

    def test_validate_invalid_price_table(self):
        """WindEnergy: testing price_table is neither True nor False."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Testing wrong value in price_table
        args['price_table'] = 'wrong_value'
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['price_table'], 'Parameter must be either True or False.')
        self.assertTrue(expected_error in validation_error_list)

    def test_validate_invalid_wind_schedule(self):
        """WindEnergy: testing missing or invalid wind_schedule values."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Testing exception raised when missing wind_schedule, provided
        # price_table is True
        args['price_table'] = True
        with self.assertRaises(AssertionError) as cm:
            wind_energy.validate(args)
        expected_message = (
            "Key wind_schedule (from ['wind_schedule']) must be in args")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        # Testing invalid values in the wind schedule table
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'price_table_bad_example.csv')
        validation_error_list = wind_energy.validate(args)
        expected_errors = [
            (['wind_schedule'], 'Value(s) in Year column is not integer.'),
            (['wind_schedule'], 'Value(s) in Price column is not numeric.')]
        for expected_error in expected_errors:
            self.assertTrue(expected_error in validation_error_list)

        # Testing duplicate years in the wind schedule table
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'price_table_example_duplicate_years.csv')
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['wind_schedule'],
            'The following year(s) showed up more than once: [2008L, 2004L].')
        self.assertTrue(expected_error in validation_error_list)

    def test_validate_missing_grid_distance_keys(self):
        """WindEnergy: testing missing distance keys when valuation checked."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Remove grid_points_path from args so both grid_points_path and
        # avg_grid_distance are missing
        args['grid_points_path'] = 'non_existing_file.csv'
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['grid_points_path'],
            'Either avg_grid_distance or grid_points_path must be provided.')
        self.assertTrue(expected_error in validation_error_list)

    def test_validate_missing_workspace(self):
        """WindEnergy: testing missing workspace_dir."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        args['workspace_dir'] = ''
        validation_error_list = wind_energy.validate(args)
        expected_error = (['workspace_dir'], 'Parameter must have a value.')
        self.assertTrue(expected_error in validation_error_list)

    def test_validate_invalid_aoi_vector(self):
        """WindEnergy: testing invalid aoi_vector_path values in args."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Use a non-compatible vector file for AOI
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp.gz')
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['aoi_vector_path'],
            'Parameter must be a path to an OGR-compatible vector file.')
        self.assertTrue(expected_error in validation_error_list)

        # Use a projection that's not in meters
        args['aoi_vector_path'] = os.path.join(
            SAMPLE_DATA, 'aoi_projected_in_ft.shp')
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['aoi_vector_path'], 'Vector must be projected in meters')
        self.assertTrue(expected_error in validation_error_list)

    def test_validate_invalid_bathymetry_raster(self):
        """WindEnergy: testing GDAL non-compatible bathymetry_path in args."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Use a non-compatible vector file for AOI
        args['bathymetry_path'] = os.path.join(
            SAMPLE_DATA, 'New_England_US_Aoi.shp.gz')
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['bathymetry_path'],
            'Parameter must be a path to a GDAL-compatible raster on disk.')
        self.assertTrue(expected_error in validation_error_list)

    def test_missing_required_fields_in_wind_data(self):
        """WindEnergy: testing missing required fields in wind_data_path."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Use a CSV file that's missing the REF field
        args['wind_data_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_wind_points_missing_ref_field.csv')
        validation_error_list = wind_energy.validate(args)
        expected_error = (['wind_data_path'], 'Missing REF field.')
        self.assertTrue(expected_error in validation_error_list)

        # Use a CSV file that's missing some fields other than REF
        args['wind_data_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_wind_points_missing_other_fields.csv')
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['wind_data_path'], 'CSV missing required fields: lam, k')
        self.assertTrue(expected_error in validation_error_list)

        # Use a CSV file path that doesn't exist
        args['wind_data_path'] = os.path.join(
            SAMPLE_DATA, 'file_not_exist.csv')
        validation_error_list = wind_energy.validate(args)
        expected_error = (['wind_data_path'], 'Could not locate file.')
        self.assertTrue(expected_error in validation_error_list)

        # Use a CSV file that has invalid field values
        args['wind_data_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_wind_points_invalid_field_values.csv')
        validation_error_list = wind_energy.validate(args)
        expected_errors = [
            (['wind_data_path'], 'Ref 10.2 must be an integer.'),
            (['wind_data_path'], 'Ref 10.2 column k must be a number.')]
        for expected_error in expected_errors:
            self.assertTrue(expected_error in validation_error_list)

    def test_validate_csv_file_nonexist(self):
        """WindEnergy: testing validate turbine_parameters_path not existing."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Use a non-compatible vector file for AOI
        args['turbine_parameters_path'] = os.path.join(
            SAMPLE_DATA, 'file_not_exist.csv')
        validation_error_list = wind_energy.validate(args)
        expected_error = (['turbine_parameters_path'], 'File not found.')
        self.assertTrue(expected_error in validation_error_list)

    def test_validate_invalid_number_keys(self):
        """WindEnergy: testing validate invalid values for number keys."""
        from natcap.invest import wind_energy

        # Use a float for number_of_turbines
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)
        args['number_of_turbines'] = 0.5
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['number_of_turbines'], 'Parameter must be an integer.')
        self.assertTrue(expected_error in validation_error_list)

        # Use a string for discount_rate
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)
        args['discount_rate'] = 'string'
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['discount_rate'], 'Parameter must be a number.')
        self.assertTrue(expected_error in validation_error_list)

    def test_missing_required_fields_in_grid_points(self):
        """WindEnergy: testing missing required fields in grid_points_path."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Use a CSV file that's missing the ID field
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_grid_pts_missing_id_field.csv')
        validation_error_list = wind_energy.validate(args)
        expected_error = (['grid_points_path'], 'Missing ID field.')
        self.assertTrue(expected_error in validation_error_list)

        # Use a CSV file that's missing some fields other than ID
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_grid_pts_missing_other_fields.csv')
        validation_error_list = wind_energy.validate(args)
        expected_error = (
            ['grid_points_path'], 'CSV missing required fields: lati, type')
        self.assertTrue(expected_error in validation_error_list)

        # Use a CSV file path that doesn't exist
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'file_not_exist.csv')
        validation_error_list = wind_energy.validate(args)
        expected_error = (['grid_points_path'], 'Could not locate file.')
        self.assertTrue(expected_error in validation_error_list)

        # Use a CSV file that has invalid field values
        args['grid_points_path'] = os.path.join(
            SAMPLE_DATA, 'resampled_grid_pts_invalid_field_values.csv')
        validation_error_list = wind_energy.validate(args)
        expected_errors = [
            (['grid_points_path'], 'ID 10.5 must be an integer.'),
            (['grid_points_path'], 'ID 10.5 column lati must be a number.'),
            (['grid_points_path'], ('ID 30.0 column TYPE must be either ' +
             '"land" or "grid" (case-insensitive)'))]
        for expected_error in expected_errors:
            self.assertTrue(expected_error in validation_error_list)

    def test_validate_missing_land_poly(self):
        """WindEnergy: testing missing land_polygon_vector_path."""
        from natcap.invest import wind_energy
        args = WindEnergyRegressionTests.generate_validate_args(
            self.workspace_dir)

        # Remove land_polygon_vector_path while min_distance, max_distance
        # and valuation_container remain in args
        del args['land_polygon_vector_path']
        with self.assertRaises(AssertionError) as cm:
            wind_energy.validate(args)
        expected_message = (
            "Key land_polygon_vector_path (from ['land_polygon_vector_path'])")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    @staticmethod
    def _assert_vectors_equal(a_vector_path, b_vector_path):
        """Assert that geometries and fields in the two vectors are equal.

        Parameters:
            a_vector_path (str): a path to an OGR vector.
            b_vector_path (str): a path to an OGR vector.

        Returns:
            None.

        Raises:
            AssertionError when the two geometries or field values are not
            equal up to desired precision (default is 6).

        """
        a_shape = ogr.Open(a_vector_path)
        a_layer = a_shape.GetLayer(0)
        a_feat = a_layer.GetNextFeature()

        b_shape = ogr.Open(b_vector_path)
        b_layer = b_shape.GetLayer(0)
        b_feat = b_layer.GetNextFeature()

        while a_feat is not None:
            # Get coordinates from geometry and store them in a list
            a_geom = a_feat.GetGeometryRef()
            a_geom_list = re.findall(r'\d+\.\d+', a_geom.ExportToWkt())
            a_geom_list = [float(x) for x in a_geom_list]

            b_geom = b_feat.GetGeometryRef()
            b_geom_list = re.findall(r'\d+\.\d+', b_geom.ExportToWkt())
            b_geom_list = [float(x) for x in b_geom_list]

            try:
                numpy.testing.assert_array_almost_equal(
                    a_geom_list, b_geom_list)
            except AssertionError:
                a_feature_fid = a_feat.GetFID()
                b_feature_fid = b_feat.GetFID()
                raise AssertionError('Geometries are not equal in feature %s, '
                                     'regression feature %s.' %
                                     (a_feature_fid, b_feature_fid))

            # Get field names/values as dictionaries and compare them without
            # specifying
            a_fields = a_feat.items()
            b_fields = b_feat.items()
            for a_field, a_value in a_fields.iteritems():
                try:
                    b_value = b_fields[a_field]
                except KeyError:
                    raise AssertionError(
                        'Field %s in feature %s does not exist in regression'
                        'feature %s.' % (a_field, a_feature_fid, b_feature_fid))
                try:
                    numpy.testing.assert_almost_equal(a_value, b_value)
                except AssertionError:
                    raise AssertionError(
                        'Values in %s field are not equal in feature %s: %s, '
                        'regression feature %s: %s.' %
                        (a_field, a_feature_fid, a_value, b_feature_fid, b_value))

            a_feat = None
            b_feat = None
            a_feat = a_layer.GetNextFeature()
            b_feat = b_layer.GetNextFeature()

        a_shape = None
        b_shape = None
