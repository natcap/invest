import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from natcap.invest import ui_server, models
from osgeo import gdal

gdal.UseExceptions()
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

    def test_get_invest_models(self):
        """UI server: get_invest_models endpoint."""
        test_client = ui_server.app.test_client()
        response = test_client.get(f'{ROUTE_PREFIX}/models')
        self.assertEqual(response.status_code, 200)
        models_dict = json.loads(response.get_data(as_text=True))
        for model in models_dict.values():
            self.assertEqual(set(model), {'model_title', 'aliases'})

    def test_get_invest_spec(self):
        """UI server: get_invest_spec endpoint."""
        test_client = ui_server.app.test_client()
        response = test_client.post(f'{ROUTE_PREFIX}/getspec', json='carbon')
        spec = json.loads(response.get_data(as_text=True))
        self.assertEqual(
            set(spec),
            {'model_id', 'model_title', 'userguide', 'aliases',
             'input_field_order', 'different_projections_ok',
             'validate_spatial_overlap', 'args', 'outputs', 'module_name'})

    def test_get_invest_validate(self):
        """UI server: get_invest_validate endpoint."""
        from natcap.invest import carbon
        test_client = ui_server.app.test_client()
        args = {
            'workspace_dir': 'foo'
        }
        payload = {
            'model_id': carbon.MODEL_SPEC.model_id,
            'args': json.dumps(args)
        }
        response = test_client.post(f'{ROUTE_PREFIX}/validate', json=payload)
        self.assertEqual(response.status_code, 200)
        results = json.loads(response.get_data(as_text=True))
        expected = carbon.validate(args)
        # These differ only because a tuple was transformed to a list during
        # the json (de)serializing, so do the same with expected data
        self.assertEqual(results, json.loads(json.dumps(expected)))

    def test_post_datastack_file(self):
        """UI server: post_datastack_file endpoint."""
        test_client = ui_server.app.test_client()
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
            f'{ROUTE_PREFIX}/post_datastack_file', json={'filepath': filepath})
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.get_data(as_text=True))
        self.assertEqual(
            set(response_data),
            {'type', 'args', 'model_id'})

    def test_write_parameter_set_file(self):
        """UI server: write_parameter_set_file endpoint."""
        test_client = ui_server.app.test_client()
        filepath = os.path.join(self.workspace_dir, 'datastack.json')
        payload = {
            'filepath': filepath,
            'model_id': 'carbon',
            'args': json.dumps({
                'workspace_dir': 'foo'
            }),
            'relativePaths': True,
        }
        response = test_client.post(
            f'{ROUTE_PREFIX}/write_parameter_set_file', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json,
            {'message': 'Parameter set saved', 'error': False})
        with open(filepath, 'r') as file:
            actual_data = json.loads(file.read())
        self.assertEqual(
            set(actual_data),
            {'args', 'model_id'})

    def test_write_parameter_set_file_error_handling(self):
        """UI server: write_parameter_set_file endpoint
        should catch a ValueError and return an error message.
        """
        test_client = ui_server.app.test_client()
        filepath = os.path.join(self.workspace_dir, 'datastack.json')
        payload = {
            'filepath': filepath,
            'model_id': 'carbon',
            'args': json.dumps({
                'workspace_dir': 'foo'
            }),
            'relativePaths': True,
        }
        error_message = 'Error saving datastack'
        with patch('natcap.invest.datastack.build_parameter_set',
                   side_effect=ValueError(error_message)):
            response = test_client.post(
                f'{ROUTE_PREFIX}/write_parameter_set_file', json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json,
                {'message': error_message, 'error': True})

    def test_save_to_python(self):
        """UI server: save_to_python endpoint."""
        test_client = ui_server.app.test_client()
        filepath = os.path.join(self.workspace_dir, 'script.py')
        payload = {
            'filepath': filepath,
            'model_id': 'carbon',
            'args': json.dumps({
                'workspace_dir': 'foo'
            }),
        }
        response = test_client.post(f'{ROUTE_PREFIX}/save_to_python', json=payload)
        self.assertEqual(response.status_code, 200)
        # test_cli.py asserts the actual contents of the file
        self.assertTrue(os.path.exists(filepath))

    def test_build_datastack_archive(self):
        """UI server: build_datastack_archive endpoint."""
        test_client = ui_server.app.test_client()
        target_filepath = os.path.join(self.workspace_dir, 'data.tgz')
        data_path = os.path.join(self.workspace_dir, 'data.csv')
        with open(data_path, 'w') as file:
            file.write('hello')

        payload = {
            'filepath': target_filepath,
            'model_id': 'carbon',
            'args': json.dumps({
                'workspace_dir': 'foo',
                'carbon_pools_path': data_path
            }),
        }
        response = test_client.post(
            f'{ROUTE_PREFIX}/build_datastack_archive', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json,
            {'message': 'Datastack archive created', 'error': False})
        # test_datastack.py asserts the actual archiving functionality
        self.assertTrue(os.path.exists(target_filepath))

    def test_build_datastack_archive_error_handling(self):
        """UI server: build_datastack_archive endpoint
        should catch a ValueError and return an error message.
        """
        test_client = ui_server.app.test_client()
        target_filepath = os.path.join(self.workspace_dir, 'data.tgz')
        data_path = os.path.join(self.workspace_dir, 'data.csv')
        with open(data_path, 'w') as file:
            file.write('hello')

        payload = {
            'filepath': target_filepath,
            'model_id': 'carbon',
            'args': json.dumps({
                'workspace_dir': 'foo',
                'carbon_pools_path': data_path
            }),
        }
        error_message = 'Error saving datastack'
        with patch('natcap.invest.datastack.build_datastack_archive',
                   side_effect=ValueError(error_message)):
            response = test_client.post(
                f'{ROUTE_PREFIX}/build_datastack_archive', json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json,
                {'message': error_message, 'error': True})
    
    @patch('natcap.invest.ui_server.usage.requests.post')
    @patch('natcap.invest.ui_server.usage.requests.get')
    def test_log_model_start(self, mock_get, mock_post):
        """UI server: log_model_start endpoint."""
        mock_response = Mock()
        mock_url = 'http://foo.org/bar.html'
        mock_response.json.return_value = {'START': mock_url}
        mock_get.return_value = mock_response
        test_client = ui_server.app.test_client()
        payload = {
            'model_id': 'carbon',
            'model_args': json.dumps({
                'workspace_dir': 'foo'
            }),
            'invest_interface': 'Workbench',
            'session_id': '12345',
            'type': 'core'
        }
        response = test_client.post(
            f'{ROUTE_PREFIX}/log_model_start', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), 'OK')
        mock_get.assert_called_once()
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.args[0], mock_url)
        self.assertEqual(
            mock_post.call_args.kwargs['data']['model_name'],
            'natcap.invest.carbon')
        self.assertEqual(
            mock_post.call_args.kwargs['data']['invest_interface'],
            payload['invest_interface'])
        self.assertEqual(
            mock_post.call_args.kwargs['data']['session_id'],
            payload['session_id'])

    @patch('natcap.invest.ui_server.usage.requests.post')
    @patch('natcap.invest.ui_server.usage.requests.get')
    def test_log_model_exit(self, mock_get, mock_post):
        """UI server: log_model_start endpoint."""
        mock_response = Mock()
        mock_url = 'http://foo.org/bar.html'
        mock_response.json.return_value = {'FINISH': mock_url}
        mock_get.return_value = mock_response
        test_client = ui_server.app.test_client()
        payload = {
            'session_id': '12345',
            'status': ''
        }
        response = test_client.post(
            f'{ROUTE_PREFIX}/log_model_exit', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), 'OK')
        mock_get.assert_called_once()
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.args[0], mock_url)
        self.assertEqual(mock_post.call_args.kwargs['data'], payload)

    @patch('natcap.invest.ui_server.geometamaker.config.platformdirs.user_config_dir')
    def test_get_geometamaker_profile(self, mock_user_config_dir):
        """UI server: get_geometamaker_profile endpoint."""
        test_client = ui_server.app.test_client()
        response = test_client.get(f'{ROUTE_PREFIX}/get_geometamaker_profile')
        self.assertEqual(response.status_code, 200)
        profile_dict = json.loads(response.get_data(as_text=True))
        self.assertIn('contact', profile_dict)
        self.assertIn('license', profile_dict)

    @patch('natcap.invest.ui_server.geometamaker.config.platformdirs.user_config_dir')
    def test_set_geometamaker_profile(self, mock_user_config_dir):
        """UI server: set_geometamaker_profile endpoint."""
        mock_user_config_dir.return_value = self.workspace_dir
        test_client = ui_server.app.test_client()
        payload = {
            'contact': {
                'individual_name': 'Foo'
            },
            'license': {
                'title': 'Bar'
            },
        }
        response = test_client.post(
            f'{ROUTE_PREFIX}/set_geometamaker_profile', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json,
            {'message': 'Metadata profile saved', 'error': False})

    def test_model_specs_serialize(self):
        """MODEL_SPEC: test each arg spec can serialize to JSON."""
        for module in models.pyname_to_module.values():
            module.MODEL_SPEC.to_json()
