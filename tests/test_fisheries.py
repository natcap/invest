import unittest
import tempfile
import shutil
import os

from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'Fisheries')
HST_INPUTS = os.path.join(SAMPLE_DATA, 'input', 'Habitat_Scenario_Tool')


class FisheriesSampleDataTests(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def get_harvest_info(workspace):
        """Extract final harvest info from the results CSV.

        Parameters:
            workspace (string): The path to the output workspace.  The file
                *workspace*/output/results_table.csv must exist.

        Returns:
            A dict with 4 attributes mapping to the 4 columns of the very last
            timestep in the fisheries calculation.  The 4 attributes are:
            'timestep', 'is_equilibrated', 'spawners', and 'harvest'.
        """
        filename = os.path.join(workspace, 'output', 'results_table.csv')
        with open(filename) as results_csv:
            last_line = results_csv.readlines()[-1].strip().split(',')
            timestep, is_equilibrated, spawners, harvest = last_line
            return {
                'timestep': int(timestep),
                'is_equilibrated': is_equilibrated,
                'spawners': float(spawners),
                'harvest': float(harvest),
            }

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_sampledata_shrimp(self):
        from natcap.invest.fisheries import fisheries
        # raise unittest.SkipTest('stage-based cycles not correctly implemented')
        args = {
            u'alpha': 6050000.0,  # TODO: supposedly ignored w/Fixed, keyerror
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_galveston',
                                     'Galveston_Subregion.shp'),
            u'beta': 4.14e-08,  # TODO: supposedly ignored w/Fixed, keyerror
            u'do_batch': False,
            u'harvest_units': 'Individuals',  # TODO: supposedly ignored w/Fixed, keyerror
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
            u'workspace_dir': self.workspace_dir,

        }
        fisheries.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_sampledata_lobster(self):
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


    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_sampledata_blue_crab(self):
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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_sampledata_dungeness_crab(self):
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


class FisheriesHSTTest(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_regression_sex_neutral(self):
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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_regression_sex_specific(self):
        raise unittest.SkipTest('Currently fails')
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
            u'sexsp': 'Yes',
            u'workspace_dir': self.workspace_dir,
        }
        fisheries_hst.execute(args)
