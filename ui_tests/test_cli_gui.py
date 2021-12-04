import unittest
import shutil
import tempfile
import os
import json


class CLIGUITests(unittest.TestCase):
    def setUp(self):
        """Use a temporary workspace for all tests in this class."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary workspace after a test run."""
        shutil.rmtree(self.workspace_dir)

    def test_run_model(self):
        """CLI-GUI: Run a model GUI through the cli."""
        from natcap.invest import cli

        # Choosing DelineatIt because there are only two required inputs,
        # we already have them in the test data repo, and it runs fast.
        input_data_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'delineateit', 'input')
        datastack_dict = {
            'model_name': 'natcap.invest.delineateit.delineateit',
            'invest_version': '3.10',
            'args': {
                'dem_path': os.path.join(
                    input_data_dir, 'dem.tif'),
                'outlet_vector_path': os.path.join(
                    input_data_dir, 'outlets.shp')
            }
        }
        parameter_set_path = os.path.join(
            self.workspace_dir, 'paramset.invs.json')
        with open(parameter_set_path, 'w') as parameter_set_file:
            parameter_set_file.write(
                json.dumps(datastack_dict, indent=4, sort_keys=True))

        # I tried patching the model import via mock, but GUI would hang.  I'd
        # rather have a reliable test that takes a few more seconds than a test
        # that hangs.
        cli.main([
            '--debug',
            'quickrun',
            'delineateit',
            parameter_set_path,
            '--workspace', os.path.join(
                self.workspace_dir, 'newdir'),  # avoids alert about overwrite
        ])
