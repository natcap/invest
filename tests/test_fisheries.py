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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_sampledata_shrimp(self):
        raise unittest.SkipTest
        from natcap.invest.fisheries import fisheries
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
        raise unittest.SkipTest('Missing Maturity Vector')
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 5.77e6,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_belize',
                                     'Lob_Belize_subregion.shp'),
            u'beta': 2.885e6,
            u'do_batch': False,
            u'harvest_units': 'Weight',
            u'migr_cont': True,
            u'migration_dir': os.path.join(SAMPLE_DATA, 'input_lobster',
                                           'Migrations', 'migration_2.csv'),
            u'population_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                'input_shrimp',
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

class FisheriesTest(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_regression_with_valuation(self):
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 6050000.0,
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
            u'results_suffix': u'',
            u'sexsp': 'No',
            u'spawn_units': 'Individuals',
            u'total_init_recruits': 200000.0,
            u'total_recur_recruits': 5.0,
            u'total_timesteps': 100,
            u'val_cont': True,
            u'frac_post_process': 0.351487513,
            u'unit_price': 1.0,
            u'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)


    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_regression_sex_specific(self):
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 6050000.0,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_galveston',
                                     'Galveston_Subregion.shp'),
            u'beta': 4.14e-08,
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
            u'total_init_recruits': 200000.0,
            u'total_recur_recruits': 5.0,
            u'total_timesteps': 100,
            u'val_cont': False,
            u'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_regression_migrations(self):
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 6050000.0,
            u'aoi_uri': os.path.join(SAMPLE_DATA, 'input',
                                     'shapefile_galveston',
                                     'Galveston_Subregion.shp'),
            u'beta': 4.14e-08,
            u'do_batch': False,
            u'harvest_units': 'Individuals',
            u'migr_cont': True,
            u'migration_dir': os.path.join(SAMPLE_DATA, 'input',
                                           'input_lobster', 'Migrations'),
            u'population_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                'input_lobster',
                                                'population_params.csv'),
            u'population_type': 'Age-Based',
            u'recruitment_type': 'Ricker',
            u'sexsp': 'No',
            u'spawn_units': 'Individuals',
            u'total_init_recruits': 200000.0,
            u'total_recur_recruits': 5.0,
            u'total_timesteps': 100,
            u'val_cont': False,
            u'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_regression_batch(self):
        from natcap.invest.fisheries import fisheries
        args = {
            u'alpha': 6050000.0,
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
            u'results_suffix': u'',
            u'sexsp': 'No',
            u'spawn_units': 'Individuals',
            u'total_init_recruits': 200000.0,
            u'total_recur_recruits': 5.0,
            u'total_timesteps': 100,
            u'val_cont': False,
            u'workspace_dir': self.workspace_dir,
        }
        fisheries.execute(args)


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
        raise unittest.SkipTest('No sample data')
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
