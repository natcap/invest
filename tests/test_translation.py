import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from babel.messages import Catalog, mofile, pofile
from natcap.invest import validation

TEST_LANG = 'en'

# assign to local variable so that it won't be changed by translation
missing_key_msg = validation.MESSAGES['MISSING_KEY']

# Fake translations for testing
TEST_MESSAGES = {
    "InVEST Carbon Model": "ιиνєѕт ςαявσи мσ∂єℓ",
    "Available models:": "αναιℓαвℓє мσ∂єℓѕ:",
    "Carbon Storage and Sequestration": "ςαявσи ѕтσяαgє αи∂ ѕєףυєѕтяαтισи",
    "current LULC": "ςυяяєит ℓυℓς",
    missing_key_msg: "кєу ιѕ мιѕѕιиg fяσм тнє αяgѕ ∂ιςт"
}

TEST_CATALOG = Catalog(locale=TEST_LANG)
for key, value in TEST_MESSAGES.items():
    TEST_CATALOG.add(key, value)


class TranslationTests(unittest.TestCase):
    """Tests for translation."""
    @classmethod
    def setUpClass(cls):
        """Create temporary workspace and write MO file to it."""
        cls.workspace_dir = tempfile.mkdtemp()
        cls.test_locale_dir = os.path.join(cls.workspace_dir, 'locales')
        mo_path = os.path.join(
            cls.test_locale_dir, TEST_LANG, 'LC_MESSAGES', 'messages.mo')
        os.makedirs(os.path.dirname(mo_path))
        # format the test catalog object into MO file format
        with open(mo_path, 'wb') as mo_file:
            mofile.write_mo(mo_file, TEST_CATALOG)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary workspace after a test run."""
        shutil.rmtree(cls.workspace_dir)

    def test_invest_list(self):
        """Translation: test that CLI list output is translated."""

        # patch the LOCALE_DIR variable in natcap/invest/__init__.py
        # this is where gettext will look for translations
        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            from natcap.invest import cli
            # capture stdout
            with patch('sys.stdout', new=io.StringIO()) as out:
                with self.assertRaises(SystemExit):
                    cli.main(['--language', TEST_LANG, 'list'])

        result = out.getvalue()
        self.assertTrue(TEST_MESSAGES['Available models:'] in result)
        self.assertTrue(
            TEST_MESSAGES['Carbon Storage and Sequestration'] in result)

    def test_invest_getspec(self):
        """Translation: test that CLI getspec output is translated."""

        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            from natcap.invest import cli
            # capture stdout
            with patch('sys.stdout', new=io.StringIO()) as out:
                with self.assertRaises(SystemExit):
                    cli.main(['--language', TEST_LANG, 'getspec', 'carbon'])

        result = out.getvalue()
        self.assertTrue(TEST_MESSAGES['current LULC'] in result)

    def test_invest_validate(self):
        """Translation: test that CLI validate output is translated."""
        from natcap.invest import carbon
        # write datastack to a JSON file
        datastack = {
            'model_name': carbon.ARGS_SPEC['pyname'],
            'invest_version': '0.0',
            'args': {}
        }
        datastack_path = os.path.join(self.workspace_dir, 'datastack.json')
        with open(datastack_path, 'w') as file:
            json.dump(datastack, file)

        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            from natcap.invest import cli
            # capture stdout
            with patch('sys.stdout', new=io.StringIO()) as out:
                with self.assertRaises(SystemExit):
                    cli.main(
                        ['--language', TEST_LANG, 'validate', datastack_path])

        result = out.getvalue()
        self.assertTrue(TEST_MESSAGES[missing_key_msg] in result)

    def test_server_get_invest_models(self):
        """Translation: test that /models endpoint is translated."""
        from natcap.invest import ui_server

        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            test_client = ui_server.app.test_client()
            response = test_client.get('/models')
            result = json.loads(response.get_data(as_text=True))
        self.assertTrue(
            TEST_MESSAGES['Carbon Storage and Sequestration'] in result)

    def test_server_get_invest_getspec(self):
        """Translation: test that /getspec endpoint is translated."""
        from natcap.invest import ui_server

        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            test_client = ui_server.app.test_client()
            response = test_client.post('/getspec', json='carbon')
            spec = json.loads(response.get_data(as_text=True))
        self.assertEqual(
            spec['args']['lulc_cur_path']['name'],
            TEST_MESSAGES['current LULC'])

    def test_server_get_invest_validate(self):
        """Translation: test that /validate endpoint is translated."""
        from natcap.invest import ui_server, carbon
        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            test_client = ui_server.app.test_client()
            payload = {
                'model_module': carbon.ARGS_SPEC['pyname'],
                'args': json.dumps({})
            }
            response = test_client.post('/validate', json=payload)
        results = json.loads(response.get_data(as_text=True))
        messages = [item[1] for item in results]
        self.assertTrue(
            TEST_MESSAGES[missing_key_msg] in messages)
