import unittest
import tempfile
import shutil
import os

from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'Fisheries')


class FisheriesTest(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    def test_regression(self):
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
    def test_regression(self):
        from natcap.invest.fisheries import fisheries_hst
        args = {
            u'gamma': 0.5,
            u'hab_cont': False,
            u'habitat_chg_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                 'Habitat_Scenario_Tool',
                                                 'habitat_chg_params.csv'),
            u'habitat_dep_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                 'Habitat_Scenario_Tool',
                                                 'habitat_dep_params.csv'),
            u'pop_cont': False,
            u'population_csv_uri': os.path.join(SAMPLE_DATA, 'input',
                                                'Habitat_Scenario_Tool',
                                                'pop_params.csv'),
            u'sexsp': 'No',
            u'workspace_dir': self.workspace_dir,
        }
        fisheries_hst.execute(args)
