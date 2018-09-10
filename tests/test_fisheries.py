"""Tests for the Fisheries model."""
import unittest
import tempfile
import shutil
import os

import numpy
import natcap.invest.pygeoprocessing_0_3_3.testing

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'Fisheries')
HST_INPUTS = os.path.join(SAMPLE_DATA, 'input', 'Habitat_Scenario_Tool')
TEST_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'fisheries')


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
        with open(filename) as results_csv:
            last_line = results_csv.readlines()[-1].strip().split(',')
            timestep, is_equilibrated, spawners, harvest = last_line
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

    def test_sampledata_shrimp(self):
        """Fisheries: Verify run on Shrimp sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 6050000.0,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_galveston',
                                     'Galveston_Subregion.shp'),
            u'beta': 4.14e-08,
            u'do_batch': False,
            u'harvest_units': 'Weight',
            u'migr_cont': False,
            u'population_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                'input_shrimp',
                                                'population_params.csv'),
            u'population_type': 'Stage-Based',
            u'recruitment_type': 'Fixed',
            u'sexsp': 'No',
            u'spawn_units': 'Individuals',
            u'total_init_recruits': 1e5,
            u'total_recur_recruits': 2.16e11,
            u'total_timesteps': 300,
            u'val_cont': False,
            u'results_suffix': 'foo',
            u'workspace_dir': self.workspace_dir,

        }
        fisheries.execute(args)
        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            self.workspace_dir, 'results_table_foo.csv')
        self.assertEqual(final_timestep_data['spawners'], '(fixed recruitment)')
        self.assertEqual(final_timestep_data['harvest'], 3120557.88)

    def test_sampledata_lobster(self):
        """Fisheries: Verify run on Lobster sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 5.77e6,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_belize',
                                     'Lob_Belize_Subregions.shp'),
            u'beta': 2.885e6,
            u'do_batch': False,
            u'harvest_units': 'Weight',
            u'migr_cont': True,
            u'migration_dir': os.path.join(SAMPLE_DATA, 'input',
                                           'input_lobster', 'Migrations'),
            u'population_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                'input_lobster',
                                                'population_params.csv'),
            u'population_type': 'Age-Based',
            u'recruitment_type': 'Beverton-Holt',
            u'sexsp': 'No',
            u'spawn_units': 'Weight',
            u'total_init_recruits': 1e5,
            u'total_recur_recruits': 2.16e11,
            u'total_timesteps': 100,
            u'val_cont': True,
            u'frac_post_process': 0.28633258,
            u'unit_price': 29.93,
            u'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 2846715.12)
        self.assertEqual(final_timestep_data['harvest'], 963108.36)


    def test_sampledata_blue_crab(self):
        """Fisheries: Verify run on Blue Crab sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 6.05e6,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_galveston',
                                     'Galveston_Subregion.shp'),
            u'beta': 4.14e-08,
            u'do_batch': False,
            u'harvest_units': 'Individuals',
            u'migr_cont': False,
            u'population_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                'input_blue_crab',
                                                'population_params.csv'),
            u'population_type': 'Age-Based',
            u'recruitment_type': 'Ricker',
            u'sexsp': 'No',
            u'spawn_units': 'Individuals',
            u'total_init_recruits': 2e5,
            u'total_recur_recruits': 5.0,
            u'total_timesteps': 100,
            u'val_cont': False,
            u'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 42649419.32)
        self.assertEqual(final_timestep_data['harvest'], 24789383.34)

    def test_sampledata_blue_crab_batch(self):
        """Fisheries: Verify run on (batched) Blue Crab sample data."""
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 6.05e6,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_galveston',
                                     'Galveston_Subregion.shp'),
            u'beta': 4.14e-08,
            u'do_batch': True,
            u'harvest_units': 'Individuals',
            u'migr_cont': False,
            u'population_csv_dir': os.path.join(SAMPLE_DATA, 'input',
                                                'input_blue_crab'),
            u'population_type': 'Age-Based',
            u'recruitment_type': 'Ricker',
            u'sexsp': 'No',
            u'spawn_units': 'Individuals',
            u'total_init_recruits': 2e5,
            u'total_recur_recruits': 5.0,
            u'total_timesteps': 100,
            u'val_cont': False,
            u'workspace_dir': self.workspace_dir,
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
            u'alpha': 2e6,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_hood_canal',
                                     'DC_HoodCanal_Subregions.shp'),
            u'beta': 3.09e-7,
            u'do_batch': False,
            u'harvest_units': 'Individuals',
            u'migr_cont': False,
            u'population_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                'input_dungeness_crab',
                                                'population_params.csv'),
            u'population_type': 'Age-Based',
            u'recruitment_type': 'Ricker',
            u'sexsp': 'Yes',
            u'spawn_units': 'Individuals',
            u'total_init_recruits': 2e12,
            u'total_recur_recruits': 5.0,
            u'total_timesteps': 100,
            u'val_cont': False,
            u'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(self.workspace_dir)
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
            u'alpha': 5.77e6,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_belize',
                                     'Lob_Belize_Subregions.shp'),
            u'beta': 2.885e6,
            u'do_batch': False,
            u'harvest_units': 'Weight',
            u'migr_cont': False,
            u'population_csv_uri': os.path.join(TEST_DATA,
                                                'sample_fecundity_params.csv'),
            u'population_type': 'Age-Based',
            u'recruitment_type': 'Fecundity',
            u'sexsp': 'No',
            u'spawn_units': 'Weight',
            u'total_init_recruits': 1e5,
            u'total_recur_recruits': 2.16e11,
            u'total_timesteps': 100,
            u'val_cont': False,
            u'workspace_dir': workspace,
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
            u'alpha': 6050000.0,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_galveston',
                                     'Galveston_Subregion.shp'),
            u'beta': 4.14e-08,
            u'harvest_units': 'Individuals',
            u'spawn_units': 'Individuals',
            u'total_timesteps': 300,
        })
        return args

    def test_sampledata_fecundity(self):
        """Fisheries: Verify run with fecundity recruitment."""
        # Based on the lobster inputs, but need coverage for fecundity.
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.fecundity_args(self.workspace_dir)
        fisheries.execute(args)

        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 594922.52)
        self.assertEqual(final_timestep_data['harvest'], 205666.3)

    def test_sampledata_custom_function(self):
        """Fisheries: Verify results with custom function."""
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.galveston_args(self.workspace_dir)

        args.update({
            u'recruitment_type': 'Other',
            # This doesn't model anything real, but it will produce outputs as
            # expected.
            u'recruitment_func': lambda x: (numpy.ones((9,)),
                                             numpy.float64(100))
        })

        fisheries.execute(args)
        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(self.workspace_dir)
        self.assertEqual(final_timestep_data['spawners'], 100.0)
        self.assertEqual(final_timestep_data['harvest'], 1.83)

    def test_sampledata_invalid_custom_function(self):
        """Fisheries: Verify exception with invalid custom function."""
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.galveston_args(self.workspace_dir)

        args.update({
            u'recruitment_type': 'Other',
            # The custom function must return two values, so this should raise
            # an error.
            u'recruitment_func': lambda x: x,
        })
        with self.assertRaises(ValueError):
            fisheries.execute(args)

    def test_sampledata_invalid_recruitment(self):
        """Fisheries: Verify exception with invalid recruitment type."""
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.galveston_args(self.workspace_dir)

        args.update({
            u'recruitment_type': 'foo',
        })

        with self.assertRaises(ValueError):
            fisheries.execute(args)

    def test_sampledata_invalid_population_type(self):
        """Fisheries: Verify exception with invalid population type."""
        from natcap.invest.fisheries import fisheries
        args = FisheriesSampleDataTests.galveston_args(self.workspace_dir)

        args.update({
            u'population_type': 'INVALID TYPE',
        })

        with self.assertRaises(ValueError):
            fisheries.execute(args)

    def test_sampledata_shrimp_multiple_regions(self):
        """Fisheries: Verify shrimp run on multiple identical regions."""
        from natcap.invest.fisheries import fisheries

        args = {
            u'alpha': 6050000.0,
            u'beta': 4.14e-08,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_galveston',
                                     'Galveston_Subregion.shp'),
            u'do_batch': False,
            u'harvest_units': 'Weight',
            u'migr_cont': False,
            u'population_csv_uri': os.path.join(TEST_DATA,
                                                'shrimp_multiregion_pop_params.csv'),
            u'population_type': 'Stage-Based',
            u'recruitment_type': 'Fixed',
            u'sexsp': 'No',
            u'spawn_units': 'Individuals',
            u'total_init_recruits': 1e5,
            u'total_recur_recruits': 2.16e11,
            u'total_timesteps': 300,
            u'val_cont': False,
            u'results_suffix': 'foo',
            u'workspace_dir': self.workspace_dir,

        }
        fisheries.execute(args)
        final_timestep_data = FisheriesSampleDataTests.get_harvest_info(
            self.workspace_dir, 'results_table_foo.csv')
        self.assertEqual(final_timestep_data['spawners'], '(fixed recruitment)')
        self.assertEqual(final_timestep_data['harvest'], 3120557.88)


        # verify that two identical subregions were found.
        in_subregion = False
        subregions = {}
        harvest_table_path = os.path.join(self.workspace_dir, 'output',
                                          'results_table_foo.csv')
        with open(harvest_table_path) as harvest_table:
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
            u'gamma': 0.5,
            u'hab_cont': False,
            u'habitat_chg_csv_uri': os.path.join(HST_INPUTS,
                                                 'habitat_chg_params.csv'),
            u'habitat_dep_csv_uri': os.path.join(HST_INPUTS,
                                                 'habitat_dep_params.csv'),
            u'pop_cont': False,
            u'population_csv_uri': os.path.join(HST_INPUTS, 'pop_params.csv'),
            u'sexsp': 'No',
            u'workspace_dir': self.workspace_dir,
        }
        fisheries_hst.execute(args)

        natcap.invest.pygeoprocessing_0_3_3.testing.assert_csv_equal(
            os.path.join(TEST_DATA, 'pop_params_modified.csv'),
            os.path.join(args['workspace_dir'], 'output', 'pop_params_modified.csv'))

    def test_regression_sex_specific(self):
        """Fisheries-HST: Verify outputs of sex-specific run."""
        from natcap.invest.fisheries import fisheries_hst
        args = {
            u'gamma': 0.5,
            u'hab_cont': False,
            u'habitat_chg_csv_uri': os.path.join(HST_INPUTS,
                                                 'habitat_chg_params.csv'),
            u'habitat_dep_csv_uri': os.path.join(HST_INPUTS,
                                                 'habitat_dep_params.csv'),
            u'pop_cont': False,
            u'population_csv_uri': os.path.join(TEST_DATA,
                                                'hst_pop_params_sexsp.csv'),
            u'sexsp': 'Yes',
            u'workspace_dir': self.workspace_dir,
        }
        fisheries_hst.execute(args)

        natcap.invest.pygeoprocessing_0_3_3.testing.assert_csv_equal(
            os.path.join(TEST_DATA, 'hst_pop_params_sexsp_modified.csv'),
            os.path.join(args['workspace_dir'], 'output',
                         'hst_pop_params_sexsp_modified.csv'))
