import io
import json
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock

import babel
import babel.messages.mofile

TEST_LANG = 'en'
TEST_LOCALE_DIR = 'tests/test_translations/locales'
TEST_CATALOG = f'{TEST_LOCALE_DIR}/{TEST_LANG}/LC_MESSAGES/messages.mo'


# patch the LOCALE_DIR variable in natcap/invest/__init__.py
# this is where gettext will look for translations
@unittest.mock.patch('natcap.invest.LOCALE_DIR', TEST_LOCALE_DIR)
class TranslationTests(unittest.TestCase):
    """Tests for translation."""
    def setUp(self):
        """Use a temporary workspace for all tests in this class."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary workspace after a test run."""
        shutil.rmtree(self.workspace_dir)

    @classmethod
    def setUpClass(cls):
        """Compile .po test catalog to .mo."""
        subprocess.run(
            ['pybabel', 'compile', '-d', TEST_LOCALE_DIR, '-l', TEST_LANG])
        # read in the binary message catalog and look up expected value
        with open(TEST_CATALOG, 'r+b') as catalog_file:
            cls.catalog = babel.messages.mofile.read_mo(catalog_file)

    def test_invest_list(self):
        """Translation: test that CLI list output is translated."""
        from natcap.invest import cli

        # capture stdout
        with unittest.mock.patch('sys.stdout', new=io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                cli.main(['--language', TEST_LANG, 'list'])
        result = out.getvalue()

        self.assertTrue(self.catalog.get('Available models:').string in result)
        self.assertTrue(self.catalog.get(
            'Carbon Storage and Sequestration').string in result)

    def test_server_get_invest_models(self):
        """Translation: test that /models endpoint is translated."""
        from natcap.invest import ui_server
        test_client = ui_server.app.test_client()
        response = test_client.get('/models')
        result = json.loads(response.get_data(as_text=True))
        self.assertTrue(self.catalog.get(
            'Carbon Storage and Sequestration').string in result)

    def test_server_get_invest_getspec(self):
        """Translation: test that /getspec endpoint is translated."""
        from natcap.invest import ui_server
        test_client = ui_server.app.test_client()
        response = test_client.post('/getspec', json='carbon')
        spec = json.loads(response.get_data(as_text=True))
        self.assertEqual(
            spec['args']['lulc_cur_path']['name'],
            self.catalog.get('current LULC').string)

    def test_server_get_invest_validate(self):
        """Translation: test that /validate endpoint is translated."""
        from natcap.invest import ui_server, carbon
        test_client = ui_server.app.test_client()
        payload = {
            'model_module': carbon.ARGS_SPEC['module'],
            'args': json.dumps({})
        }
        response = test_client.post('/validate', json=payload)
        results = json.loads(response.get_data(as_text=True))
        messages = [item[1] for item in results]
        print(messages)
        self.assertTrue(self.catalog.get(
            'Key is missing from the args dict').string in messages)
