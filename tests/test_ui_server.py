import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from natcap.invest import ui_server

TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data')

ROUTE_PREFIX = 'api'


class EndpointFunctionTests(unittest.TestCase):
    """Tests for UI server endpoint functions."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_get_vector_colnames(self):
        """UI server: get_vector_colnames endpoint."""
        test_client = ui_server.app.test_client()
        # an empty path
        response = test_client.post(
            f'{ROUTE_PREFIX}/colnames', json={'vector_path': ''})
        colnames = json.loads(response.get_data(as_text=True))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(colnames, [])
        # a vector with one column
        path = os.path.join(
            TEST_DATA_PATH, 'annual_water_yield', 'input',
            'watersheds.shp')
        response = test_client.post(
            f'{ROUTE_PREFIX}/colnames', json={'vector_path': path})
        colnames = json.loads(response.get_data(as_text=True))
        self.assertEqual(colnames, ['ws_id'])
        # a non-vector file
        path = os.path.join(TEST_DATA_PATH, 'ndr', 'input', 'dem.tif')
        response = test_client.post(
            f'{ROUTE_PREFIX}/colnames', json={'vector_path': path})
        colnames = json.loads(response.get_data(as_text=True))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(colnames, [])

    def test_get_invest_models(self):
        """UI server: get_invest_models endpoint."""
        test_client = ui_server.app.test_client()
        response = test_client.get(f'{ROUTE_PREFIX}/models')
        models_dict = json.loads(response.get_data(as_text=True))
        for model in models_dict.values():
            self.assertEqual(set(model), {'model_name', 'aliases'})

    def test_get_invest_spec(self):
        """UI server: get_invest_spec endpoint."""
        test_client = ui_server.app.test_client()
        response = test_client.post(f'{ROUTE_PREFIX}/getspec', json='sdr')
        spec = json.loads(response.get_data(as_text=True))
        self.assertEqual(
            set(spec),
            {'model_name', 'pyname', 'userguide',
             'args_with_spatial_overlap', 'args'})

    def test_get_invest_validate(self):
        """UI server: get_invest_validate endpoint."""
        from natcap.invest import carbon
        test_client = ui_server.app.test_client()
        args = {
            'workspace_dir': 'foo'
        }
        payload = {
            'model_module': carbon.ARGS_SPEC['pyname'],
            'args': json.dumps(args)
        }
        response = test_client.post(f'{ROUTE_PREFIX}/validate', json=payload)
        results = json.loads(response.get_data(as_text=True))
        expected = carbon.validate(args)
        # These differ only because a tuple was transformed to a list during
        # the json (de)serializing, so do the same with expected data
        self.assertEqual(results, json.loads(json.dumps(expected)))

    def test_post_datastack_file(self):
        """UI server: post_datastack_file endpoint."""
        test_client = ui_server.app.test_client()
        self.workspace_dir = tempfile.mkdtemp()
        expected_datastack = {
            'args': {
                'workspace_dir': 'foo'
            },
            'invest_version': '3.10.0',
            'model_name': 'natcap.invest.carbon'
        }
        filepath = os.path.join(self.workspace_dir, 'datastack.json')
        with open(filepath, 'w') as file:
            file.write(json.dumps(expected_datastack))
        response = test_client.post(
            f'{ROUTE_PREFIX}/post_datastack_file', json=filepath)
        response_data = json.loads(response.get_data(as_text=True))
        self.assertEqual(
            set(response_data),
            {'type', 'args', 'module_name', 'model_run_name',
             'model_human_name', 'invest_version'})

    def test_write_parameter_set_file(self):
        """UI server: write_parameter_set_file endpoint."""
        test_client = ui_server.app.test_client()
        self.workspace_dir = tempfile.mkdtemp()
        filepath = os.path.join(self.workspace_dir, 'datastack.json')
        payload = {
            'filepath': filepath,
            'moduleName': 'natcap.invest.carbon',
            'args': json.dumps({
                'workspace_dir': 'foo'
            }),
            'relativePaths': True,
        }
        _ = test_client.post(
            f'{ROUTE_PREFIX}/write_parameter_set_file', json=payload)
        with open(filepath, 'r') as file:
            actual_data = json.loads(file.read())
        self.assertEqual(
            set(actual_data),
            {'args', 'invest_version', 'model_name'})

    def test_save_to_python(self):
        """UI server: save_to_python endpoint."""
        test_client = ui_server.app.test_client()
        self.workspace_dir = tempfile.mkdtemp()
        filepath = os.path.join(self.workspace_dir, 'script.py')
        payload = {
            'filepath': filepath,
            'modelname': 'carbon',
            'args': json.dumps({
                'workspace_dir': 'foo'
            }),
        }
        _ = test_client.post(f'{ROUTE_PREFIX}/save_to_python', json=payload)
        # test_cli.py asserts the actual contents of the file
        self.assertTrue(os.path.exists(filepath))

    def test_build_datastack_archive(self):
        """UI server: build_datastack_archive endpoint."""
        test_client = ui_server.app.test_client()
        self.workspace_dir = tempfile.mkdtemp()
        target_filepath = os.path.join(self.workspace_dir, 'data.tgz')
        data_path = os.path.join(self.workspace_dir, 'data.csv')
        with open(data_path, 'w') as file:
            file.write('hello')

        payload = {
            'filepath': target_filepath,
            'moduleName': 'natcap.invest.carbon',
            'args': json.dumps({
                'workspace_dir': 'foo',
                'carbon_pools_path': data_path
            }),
        }
        _ = test_client.post(
            f'{ROUTE_PREFIX}/build_datastack_archive', json=payload)
        # test_datastack.py asserts the actual archiving functionality
        self.assertTrue(os.path.exists(target_filepath))

    @patch('natcap.invest.ui_server.usage.urlopen')
    def test_log_model_start(self, mock_urlopen):
        """UI server: log_model_start endpoint."""
        mock_response = Mock()
        mock_response.read.return_value = '{"START": "http://foo.org/bar.html"}'
        mock_urlopen.return_value = mock_response
        test_client = ui_server.app.test_client()
        payload = {
            'model_pyname': 'natcap.invest.carbon',
            'model_args': json.dumps({
                'workspace_dir': 'foo'
            }),
            'invest_interface': 'Workbench',
            'session_id': '12345'
        }
        response = test_client.post(
            f'{ROUTE_PREFIX}/log_model_start', json=payload)
        self.assertEqual(response.get_data(as_text=True), 'OK')

    @patch('natcap.invest.ui_server.usage.urlopen')
    def test_log_model_exit(self, mock_urlopen):
        """UI server: log_model_start endpoint."""
        mock_response = Mock()
        mock_response.read.return_value = '{"FINISH": "http://foo.org/bar.html"}'
        mock_urlopen.return_value = mock_response
        test_client = ui_server.app.test_client()
        payload = {
            'session_id': '12345',
            'status': ''
        }
        response = test_client.post(
            f'{ROUTE_PREFIX}/log_model_exit', json=payload)
        self.assertEqual(response.get_data(as_text=True), 'OK')
