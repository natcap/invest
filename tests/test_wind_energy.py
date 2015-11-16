"""Module for Regression Testing the InVEST Wind Energy module."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from nose.tools import nottest

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', 'wind_energy')


class WindEnergyRegressionTests(unittest.TestCase):
    """Regression tests for the Wind Energy module."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across regression tests."""
        args = {
            'workspace_dir': workspace_dir,
            'wind_data_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
            'bathymetry_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs',
                'global_dem'),
            'global_wind_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'global_wind_energy_parameters.csv'),
            'turbine_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 60
            }
        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_avggrid_dist_windsched(self):
        """
        Regression test for Valuation.

        This test uses average grid distance and wind schedule for pricing.
        """
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['avg_grid_distance'] = 4
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv')

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif', 'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricetable', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricetable', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_no_aoi(self):
        """Regression test for Biophysical run through NOT using an AOI."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif',	'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'noaoi', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'noaoi', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_no_land_polygon(self):
        """
        Regression test for Wind Energy Biophysical.

        This test uses an AOI but does Not use a Land Polygon and distances.
        """
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')

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
                os.path.join(REGRESSION_DATA, 'nolandpoly', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_no_distances(self):
        """
        Regression test for Wind Energy Biophysical.

        This test uses an AOI and Land Polygon but does not use distances
            for masking.
        """
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif',	'harvested_energy_MWhr_per_yr.tif']

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
                os.path.join(REGRESSION_DATA, 'nodistances', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_no_valuation(self):
        """
        Regression test for Wind Energy Biophysical.

        This test uses an AOI, Land Polygon, and distances.
        """
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'novaluation', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'novaluation', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_gridpts_windsched(self):
        """
        Regression test for Wind Energy Valuation.

        This test uses grid points and wind schedule pricing for Valuation.
        """
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['grid_points_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'NE_sub_pts.csv')
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv')

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif',	'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricetablegridpts', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricetablegridpts', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_avggriddist_windprice(self):
        """
        Regression test for Wind Energy Valuation.

        This test uses average grid distance and wind price.
        """
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['avg_grid_distance'] = 4
        args['price_table'] = False
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif', 'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'priceval', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'priceval', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_gridpts_windprice(self):
        """
        Regression test for Wind Energy Valuation.

        This test uses grid points and wind price.
        """
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['grid_points_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'NE_sub_pts.csv')
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
                os.path.join(REGRESSION_DATA, 'pricevalgridpts', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricevalgridpts', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_suffix(self):
        """Regression test for suffix handling, running Valuation."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['avg_grid_distance'] = 4
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv')
        args['suffix'] = 'test'

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons_test.tif', 'density_W_per_m2_test.tif',
            'harvested_energy_MWhr_per_yr_test.tif',
            'levelized_cost_price_per_kWh_test.tif', 'npv_US_millions_test.tif']

        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', raster_path)))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm_test.shp',
            'wind_energy_points_test.shp']

        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_suffix_underscore(self):
        """Regression test for suffix handling given an underscore."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['avg_grid_distance'] = 4
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv')
        args['suffix'] = '_test'

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons_test.tif', 'density_W_per_m2_test.tif',
            'harvested_energy_MWhr_per_yr_test.tif',
            'levelized_cost_price_per_kWh_test.tif', 'npv_US_millions_test.tif']

        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', raster_path)))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm_test.shp',
            'wind_energy_points_test.shp']

        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))
