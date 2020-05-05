"""Tests for the Fisheries model."""
import unittest
import tempfile
import shutil
import os
import io

import numpy
from osgeo import gdal
import pandas.testing

TEST_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'fisheries')
SAMPLE_DATA = os.path.join(TEST_DATA, 'input')
HST_INPUTS = os.path.join(SAMPLE_DATA, 'Habitat_Scenario_Tool')


class FisheriesSampleDataTests(unittest.TestCase):
    """Tests for Fisheries that rely on InVEST sample data."""

    def setUp(self):
        """Set up the test environment by creating the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the test environment by removing the workspace."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def get_harvest_info(workspace, filename='results_table.csv'):
        """Extract final harvest info from the results CSV.

        Parameters:
            workspace (string): The path to the output workspace.  The file
                *workspace*/output/results_table.csv must exist.

        Returns:
            A dict with 4 attributes mapping to the 4 columns of the very last
            timestep in the fisheries calculation.  The 4 attributes are:
            'timestep', 'is_equilibrated', 'spawners', and 'harvest'.
        """
        filename = os.path.join(workspace, 'output', filename)
        with io.open(filename, 'r', newline=os.linesep) as results_csv:
            # last_line = results_csv.readlines()[-1].strip().split(',')
            timestep, is_equilibrated, spawners, harvest = \
                    results_csv.readlines()[-1].strip().split(',')
            # try:
            #     timestep, is_equilibrated, spawners, harvest = \
            #         results_csv.readlines()[-1].strip().split(',')
            # except ValueError:  # not enough values to unpack
            #     # A last line with nothing but a newline character
            #     # seems to be ignored in Python 2, with [-1] indexing
            #     # the last line with content. In Python 3 though, [-1]
            #     # returns a line with only '\n', so we resent and get [-2]
            #     results_csv.seek(0)
            #     timestep, is_equilibrated, spawners, harvest = \
            #         results_csv.readlines()[-2].strip().split(',')
            try:
                spawners = float(spawners)
            except:
                # Sometimes, spawners == "(fixed recruitment)"
                pass

            return {
                'timestep': int(timestep),
                'is_equilibrated': is_equilibrated,
                'spawners': spawners,
                'harvest': float(harvest),
            }

    def test_validation(self):
        """Fisheries: Full validation."""
        from natcap.invest.fisheries import fisheries
        args = {
            'alpha': 6050000.0,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_galveston',
                                             'Galveston_Subregion.shp'),
            'beta': 4.14e-08,
            'do_batch': False,
            'harvest_units': 'Weight',
            'migr_cont': False,
            'population_csv_path': os.path.join(SAMPLE_DATA,
                                                 'input_shrimp',
                                                 'population_params.csv'),
            'population_type': 'Stage-Based',
            'recruitment_type': 'Fixed',
            'sexsp': 'No',
            'spawn_units': '',  # should have a value
            'total_init_recruits': 1e5,
            'total_recur_recruits': 2.16e11,
            'total_timesteps': 300,
            'val_cont': False,
            'results_suffix': 'foo',
            'workspace_dir': self.workspace_dir,

        }
        validation_warnings = fisheries.validate(args)
        self.assertEqual(len(validation_warnings), 1)
        self.assertTrue('required but has no value' in validation_warnings[0][1])

    def test_validation_batch(self):
        """Fisheries: Batch parameters (full model validation)."""
        from natcap.invest.fisheries import fisheries
        # Lobster args should be valid, has migration and validation portions
        # enabled.
        args = {
            'alpha': 5.77e6,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_belize',
                                             'Lob_Belize_Subregions.shp'),
            'beta': 2.885e6,
            'do_batch': False,
            'harvest_units': 'Weight',
            'migr_cont': True,
            'migration_dir': os.path.join(SAMPLE_DATA,
                                           'input_lobster', 'Migrations'),
            'population_csv_path': os.path.join(SAMPLE_DATA,
                                                 'input_lobster',
                                                 'population_params.csv'),
            'population_type': 'Age-Based',
            'recruitment_type': 'Beverton-Holt',
            'sexsp': 'No',
            'spawn_units': 'Weight',
            'total_init_recruits': 1e5,
            'total_recur_recruits': 2.16e11,
            'total_timesteps': 100,
            'val_cont': True,
            'frac_post_process': 0.28633258,
            'unit_price': 29.93,
            'workspace_dir': self.workspace_dir,
        }
        validation_warnings = fisheries.validate(args)
        self.assertEqual(len(validation_warnings), 0)

    def test_validation_invalid_aoi(self):
        """Fisheries: Validate AOI vector."""
        from natcap.invest.fisheries import fisheries
        args = {'aoi_vector_path': 'not a vector'}

        validation_warnings = fisheries.validate(
            args, limit_to='aoi_vector_path')
        self.assertEqual(len(validation_warnings), 1)
        self.assertTrue('File not found' in
                        validation_warnings[0][1])

    def test_validation_invalid_batch(self):
        """Fisheries: Validate batch-processing option."""
        from natcap.invest.fisheries import fisheries
        args = {'do_batch': 'foo'}

        validation_warnings = fisheries.validate(args, limit_to='do_batch')
        self.assertEqual(len(validation_warnings), 1)
        self.assertTrue('Value must be either True or False' in
                        validation_warnings[0][1])

    def test_validation_invalid_pop_csv(self):
        """Fisheries: Validate population CSV."""
        from natcap.invest.fisheries import fisheries
        args = {'population_csv_path': 'foo'}

        validation_warnings = fisheries.validate(
            args, limit_to='population_csv_path')
        self.assertEqual(len(validation_warnings), 1)
        self.assertTrue('File not found' in
                        validation_warnings[0][1])

    def test_validation_invalid_aoi_fields(self):
        """Fisheries: Validate AOI fields."""
        from natcap.invest.fisheries import fisheries

        args = {'aoi_vector_path': os.path.join(self.workspace_dir, 'aoi.gpkg')}
        gpkg_driver = gdal.GetDriverByName('GPKG')
        vector = gpkg_driver.Create(
            args['aoi_vector_path'], 0, 0, 0, gdal.GDT_Unknown)
        # Layer has no fields in it.
        layer = vector.CreateLayer('new_layer')

        layer = None
        vector = None

        validation_warnings = fisheries.validate(args, limit_to='aoi_vector_path')
        self.assertEqual(len(validation_warnings), 1)
        self.assertTrue('Fields are missing from the first layer' in
                        validation_warnings[0][1])

    def test_validation_invalid_init_recruits(self):
        """Fisheries: Validate negative initial recruits value."""
        from natcap.invest.fisheries import fisheries
        args = {'total_init_recruits': -100}

        validation_warnings = fisheries.validate(
            args, limit_to='total_init_recruits')
        self.assertEqual(len(validation_warnings), 1)
        self.assertTrue('Value does not meet condition' in
                        validation_warnings[0][1])

    def test_sampledata_shrimp(self):
        """Fisheries: Verify run on Shrimp sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            'alpha': 6050000.0,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_galveston',
                                             'Galveston_Subregion.shp'),
            'beta': 4.14e-08,
            'do_batch': False,
            'harvest_units': 'Weight',
            'migr_cont': False,
            'population_csv_path': os.path.join(SAMPLE_DATA,
                                                 'input_shrimp',
                                                 'population_params.csv'),
            'population_type': 'Stage-Based',
            'recruitment_type': 'Fixed',
            'sexsp': 'No',
            'spawn_units': 'Individuals',
            'total_init_recruits': 1e5,
            'total_recur_recruits': 2.16e11,
            'total_timesteps': 300,
            'val_cont': False,
            'results_suffix': 'foo',
            'workspace_dir': self.workspace_dir,

        }
        fisheries.execute(args)
        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            args['workspace_dir'], 'results_table_foo.csv')
        self.assertEqual(final_timestep_data['spawners'], '(fixed recruitment)')
        self.assertEqual(final_timestep_data['harvest'], 3120557.88)

    def test_sampledata_lobster(self):
        """Fisheries: Verify run on Lobster sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            'alpha': 5.77e6,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_belize',
                                             'Lob_Belize_Subregions.shp'),
            'beta': 2.885e6,
            'do_batch': False,
            'harvest_units': 'Weight',
            'migr_cont': True,
            'migration_dir': os.path.join(SAMPLE_DATA,
                                           'input_lobster', 'Migrations'),
            'population_csv_path': os.path.join(SAMPLE_DATA,
                                                 'input_lobster',
                                                 'population_params.csv'),
            'population_type': 'Age-Based',
            'recruitment_type': 'Beverton-Holt',
            'sexsp': 'No',
            'spawn_units': 'Weight',
            'total_init_recruits': 1e5,
            'total_recur_recruits': 2.16e11,
            'total_timesteps': 100,
            'val_cont': True,
            'frac_post_process': 0.28633258,
            'unit_price': 29.93,
            'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 2846715.12)
        self.assertEqual(final_timestep_data['harvest'], 963108.36)

    def test_sampledata_blue_crab(self):
        """Fisheries: Verify run on Blue Crab sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            'alpha': 6.05e6,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_galveston',
                                             'Galveston_Subregion.shp'),
            'beta': 4.14e-08,
            'do_batch': False,
            'harvest_units': 'Individuals',
            'migr_cont': False,
            'population_csv_path': os.path.join(SAMPLE_DATA,
                                                 'input_blue_crab',
                                                 'population_params.csv'),
            'population_type': 'Age-Based',
            'recruitment_type': 'Ricker',
            'sexsp': 'No',
            'spawn_units': 'Individuals',
            'total_init_recruits': 2e5,
            'total_recur_recruits': 5.0,
            'total_timesteps': 100,
            'val_cont': False,
            'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 42649419.32)
        self.assertEqual(final_timestep_data['harvest'], 24789383.34)

    def test_sampledata_blue_crab_batch(self):
        """Fisheries: Verify run on (batched) Blue Crab sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            'alpha': 6.05e6,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_galveston',
                                             'Galveston_Subregion.shp'),
            'beta': 4.14e-08,
            'do_batch': True,
            'harvest_units': 'Individuals',
            'migr_cont': False,
            'population_csv_dir': os.path.join(SAMPLE_DATA,
                                                'input_blue_crab'),
            'population_type': 'Age-Based',
            'recruitment_type': 'Ricker',
            'sexsp': 'No',
            'spawn_units': 'Individuals',
            'total_init_recruits': 2e5,
            'total_recur_recruits': 5.0,
            'total_timesteps': 100,
            'val_cont': False,
            'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            self.workspace_dir, filename='results_table_population_params.csv')
        self.assertEqual(final_timestep_data['spawners'], 42649419.32)
        self.assertEqual(final_timestep_data['harvest'], 24789383.34)

    def test_sampledata_dungeness_crab(self):
        """Fisheries: Verify run on Dungeness Crab sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            'alpha': 2e6,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_hood_canal',
                                             'DC_HoodCanal_Subregions.shp'),
            'beta': 3.09e-7,
            'do_batch': False,
            'harvest_units': 'Individuals',
            'migr_cont': False,
            'population_csv_path': os.path.join(SAMPLE_DATA,
                                                 'input_dungeness_crab',
                                                 'population_params.csv'),
            'population_type': 'Age-Based',
            'recruitment_type': 'Ricker',
            'sexsp': 'Yes',
            'spawn_units': 'Individuals',
            'total_init_recruits': 2e12,
            'total_recur_recruits': 5.0,
            'total_timesteps': 100,
            'val_cont': False,
            'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 4053119.08)
        self.assertEqual(final_timestep_data['harvest'], 527192.41)

    @staticmethod
    def fecundity_args(workspace):
        """
        Create a base set of args for the fecundity recruitment model.

        The AOI is located in Belize.

        Parameters:
            workspace (string): The path to the workspace on disk.

        Returns:
            A dict with args for the model.
        """
        args = {
            'alpha': 5.77e6,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_belize',
                                             'Lob_Belize_Subregions.shp'),
            'beta': 2.885e6,
            'do_batch': False,
            'harvest_units': 'Weight',
            'migr_cont': False,
            'population_csv_path': os.path.join(TEST_DATA,
                                                 'sample_fecundity_params.csv'),
            'population_type': 'Age-Based',
            'recruitment_type': 'Fecundity',
            'sexsp': 'No',
            'spawn_units': 'Weight',
            'total_init_recruits': 1e5,
            'total_recur_recruits': 2.16e11,
            'total_timesteps': 100,
            'val_cont': False,
            'workspace_dir': workspace,
        }
        return args

    @staticmethod
    def galveston_args(workspace):
        """
        Create a base set of fecundity args for Galveston Bay.

        Parameters:
            workspace (string): The path to the workspace on disk.

        Returns:
            A dict with args for the model.
        """
        args = FisheriesSampleDataTests.fecundity_args(workspace)
        args.update({
            'alpha': 6050000.0,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_galveston',
                                             'Galveston_Subregion.shp'),
            'beta': 4.14e-08,
            'harvest_units': 'Individuals',
            'spawn_units': 'Individuals',
            'total_timesteps': 300,
        })
        return args

    def test_sampledata_fecundity(self):
        """Fisheries: Verify run with fecundity recruitment."""
        # Based on the lobster inputs, but need coverage for fecundity.
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.fecundity_args(self.workspace_dir)
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 594922.52)
        self.assertEqual(final_timestep_data['harvest'], 205666.3)

    def test_sampledata_custom_function(self):
        """Fisheries: Verify results with custom function."""
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.galveston_args(self.workspace_dir)

        args.update({
            'recruitment_type': 'Other',
            # This doesn't model anything real, but it will produce outputs as
            # expected.
            'recruitment_func': lambda x: (numpy.ones((9,)),
                                            numpy.float64(100))
        })

        fisheries.execute(args)
        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 100.0)
        self.assertEqual(final_timestep_data['harvest'], 1.83)

    def test_sampledata_invalid_custom_function(self):
        """Fisheries: Verify exception with invalid custom function."""
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.galveston_args(self.workspace_dir)

        args.update({
            'recruitment_type': 'Other',
            # The custom function must return two values, so this should raise
            # an error.
            'recruitment_func': lambda x: x,
        })
        with self.assertRaises(ValueError):
            fisheries.execute(args)

    def test_sampledata_invalid_recruitment(self):
        """Fisheries: Verify exception with invalid recruitment type."""
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.galveston_args(self.workspace_dir)

        args.update({
            'recruitment_type': 'foo',
        })

        with self.assertRaises(ValueError):
            fisheries.execute(args)

    def test_sampledata_invalid_population_type(self):
        """Fisheries: Verify exception with invalid population type."""
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.galveston_args(self.workspace_dir)

        args.update({
            'population_type': 'INVALID TYPE',
        })

        with self.assertRaises(ValueError):
            fisheries.execute(args)

    def test_sampledata_shrimp_multiple_regions(self):
        """Fisheries: Verify shrimp run on multiple identical regions."""
        from natcap.invest.fisheries import fisheries

        args = {
            'alpha': 6050000.0,
            'beta': 4.14e-08,
            'aoi_vector_path': os.path.join(SAMPLE_DATA,
                                             'shapefile_galveston',
                                             'Galveston_Subregion.shp'),
            'do_batch': False,
            'harvest_units': 'Weight',
            'migr_cont': False,
            'population_csv_path': os.path.join(
                TEST_DATA, 'shrimp_multiregion_pop_params.csv'),
            'population_type': 'Stage-Based',
            'recruitment_type': 'Fixed',
            'sexsp': 'No',
            'spawn_units': 'Individuals',
            'total_init_recruits': 1e5,
            'total_recur_recruits': 2.16e11,
            'total_timesteps': 300,
            'val_cont': False,
            'results_suffix': 'foo',
            'workspace_dir': self.workspace_dir

        }
        fisheries.execute(args)
        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            args['workspace_dir'], 'results_table_foo.csv')
        self.assertEqual(final_timestep_data['spawners'], '(fixed recruitment)')
        self.assertEqual(final_timestep_data['harvest'], 3120557.88)

        # verify that two identical subregions were found.
        in_subregion = False
        subregions = {}
        harvest_table_path = os.path.join(self.workspace_dir, 'output',
                                          'results_table_foo.csv')
        with io.open(harvest_table_path, 'r', newline=os.linesep) as harvest_table:
            for line in harvest_table:
                if in_subregion:
                    if line.lower().startswith('total'):
                        break
                    else:
                        subregion_id, harvest = line.strip().split(',')
                        subregions[int(subregion_id)] = float(harvest)
                else:
                    if line.lower().startswith('subregion'):
                        in_subregion = True

        # we should only have two subregions, and their values should match.
        self.assertEqual(len(subregions), 2)
        self.assertEqual(len(set(subregions.values())), 1)


class FisheriesHSTTest(unittest.TestCase):
    """Tests for the Fisheries Habitat Suitability Tool."""

    def setUp(self):
        """Set up the test environment by creating the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the test environment by removing the workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_regression_sex_neutral(self):
        """Fisheries-HST: Verify outputs of sex-neutral run."""
        from natcap.invest.fisheries import fisheries_hst
        args = {
            'gamma': 0.5,
            'hab_cont': False,
            'habitat_chg_csv_path': os.path.join(HST_INPUTS,
                                                  'habitat_chg_params.csv'),
            'habitat_dep_csv_path': os.path.join(HST_INPUTS,
                                                  'habitat_dep_params.csv'),
            'pop_cont': False,
            'population_csv_path': os.path.join(HST_INPUTS, 'pop_params.csv'),
            'sexsp': 'No',
            'workspace_dir': self.workspace_dir,
        }
        fisheries_hst.execute(args)

        actual_values_df = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'output', 'pop_params_modified.csv'))
        expected_values_df = pandas.read_csv(
            os.path.join(TEST_DATA, 'pop_params_modified.csv'))
        pandas.testing.assert_frame_equal(actual_values_df, expected_values_df)

    def test_regression_sex_specific(self):
        """Fisheries-HST: Verify outputs of sex-specific run."""
        from natcap.invest.fisheries import fisheries_hst
        args = {
            'gamma': 0.5,
            'hab_cont': False,
            'habitat_chg_csv_path': os.path.join(HST_INPUTS,
                                                  'habitat_chg_params.csv'),
            'habitat_dep_csv_path': os.path.join(HST_INPUTS,
                                                  'habitat_dep_params.csv'),
            'pop_cont': False,
            'population_csv_path': os.path.join(TEST_DATA,
                                                 'hst_pop_params_sexsp.csv'),
            'sexsp': 'Yes',
            'workspace_dir': self.workspace_dir,
        }

        fisheries_hst.execute(args)

        actual_values_df = pandas.read_csv(
            os.path.join(args['workspace_dir'], 'output', 'hst_pop_params_sexsp_modified.csv'))
        expected_values_df = pandas.read_csv(
            os.path.join(TEST_DATA, 'hst_pop_params_sexsp_modified.csv'))
        pandas.testing.assert_frame_equal(actual_values_df, expected_values_df)
