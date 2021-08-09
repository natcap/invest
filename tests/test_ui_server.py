import os
import json
import unittest

TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data')


class EndpointFunctionTests(unittest.TestCase):
    """Tests for UI server endpoint functions."""

    def test_get_vector_colnames(self):
        """UI server: get_vector_colnames endpoint."""
        from natcap.invest import ui_server
        test_client = ui_server.app.test_client()
        # an empty path
        response = test_client.post('/colnames', json={'vector_path': ''})
        colnames = json.loads(response.get_data(as_text=True))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(colnames, [])
        # a vector with one column
        path = os.path.join(
            TEST_DATA_PATH, 'aquaculture', 'Input', 'Finfish_Netpens.shp')
        response = test_client.post('/colnames', json={'vector_path': path})
        colnames = json.loads(response.get_data(as_text=True))
        self.assertEqual(colnames, ['FarmID'])
        # a non-vector file
        path = os.path.join(TEST_DATA_PATH, 'ndr', 'input', 'dem.tif')
        response = test_client.post('/colnames', json={'vector_path': path})
        colnames = json.loads(response.get_data(as_text=True))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(colnames, [])

    def test_get_invest_models(self):
        """UI server: get_invest_models endpoint."""
        from natcap.invest import ui_server
        test_client = ui_server.app.test_client()
        response = test_client.get('/models')
        models_dict = json.loads(response.get_data(as_text=True))
        for model in models_dict.values():
            self.assertEqual(list(model), ['internal_name', 'aliases'])

    def test_get_invest_spec(self):
        """UI server: get_invest_spec endpoint."""
        from natcap.invest import ui_server
        from natcap.invest.carbon import ARGS_SPEC
        test_client = ui_server.app.test_client()
        response = test_client.post('/getspec', json={'carbon'})
        spec = json.loads(response.get_data(as_text=True))
        self.assertEqual(spec, ARGS_SPEC)
        # self.assertEqual(
        #     list(spec), ['model_name', 'module', 'userguide_html', 'args'])

    def test_get_invest_validate(self):
        """UI server: get_invest_validate endpoint."""
        from natcap.invest import ui_server, carbon
        test_client = ui_server.app.test_client()
        args = {
            'workspace_dir': 'foo'
        }
        payload = {
            'model_module': carbon.ARGS_SPEC['module'],
            'args': json.dumps(args)
        }
        response = test_client.post('/validate', json=payload)
        results = json.loads(response.get_data(as_text=True))
        expected = carbon.validate(args)
        # These differ only because a tuple was transformed to a list during
        # the json (de)serializing, so do the same with expected data
        self.assertEqual(results, json.loads(json.dumps(expected)))
