import importlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from babel.messages import Catalog, mofile
from natcap.invest import carbon, validation, install_language, ui_server

TEST_LANG = 'en'

# assign to local variable so that it won't be changed by translation
missing_key_msg = validation.MESSAGES['MISSING_KEY']
not_a_number_msg = validation.MESSAGES['NOT_A_NUMBER']

# Fake translations for testing
TEST_MESSAGES = {
    "InVEST Carbon Model": "ιиνєѕт ςαявσи мσ∂єℓ",
    "Available models:": "αναιℓαвℓє мσ∂єℓѕ:",
    "Carbon Storage and Sequestration": "ςαявσи ѕтσяαgє αи∂ ѕєףυєѕтяαтισи",
    "current LULC": "ςυяяєит ℓυℓς",
    missing_key_msg: "кєу ιѕ мιѕѕιиg fяσм тнє αяgѕ ∂ιςт",
    not_a_number_msg: 'ναℓυє "{value}" ςσυℓ∂ иσт вє ιитєяρяєтє∂ αѕ α иυмвєя'
}

TEST_CATALOG = Catalog(locale=TEST_LANG)
for key, value in TEST_MESSAGES.items():
    TEST_CATALOG.add(key, value)


def reset_global_context():
    """Reset affected parts of the global context."""
    def identity(x): return x
    __builtins__['_'] = identity

    # "unimport" the modules being translated
    # NOTE: it would be better to run each test in a new process,
    # but that's difficult on windows: https://stackoverflow.com/a/48310939
    importlib.reload(validation)
    importlib.reload(carbon)


class TranslationTests(unittest.TestCase):
    """Tests for translation."""

    def tearDown(self):
        reset_global_context()  # reset after each test case

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
        # reset so that these tests are unaffected by previous tests
        reset_global_context()

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
        self.assertIn(TEST_MESSAGES['Available models:'], result)
        self.assertIn(
            TEST_MESSAGES['Carbon Storage and Sequestration'], result)

    def test_invest_getspec(self):
        """Translation: test that CLI getspec output is translated."""
        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            from natcap.invest import cli
            # capture stdout
            with patch('sys.stdout', new=io.StringIO()) as out:
                with self.assertRaises(SystemExit):
                    cli.main(['--language', TEST_LANG, 'getspec', 'carbon'])

        result = out.getvalue()
        self.assertIn(TEST_MESSAGES['current LULC'], result)

    def test_invest_validate(self):
        """Translation: test that CLI validate output is translated."""
        # write datastack to a JSON file
        datastack = {
            'model_name': 'natcap.invest.carbon',
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
        self.assertIn(TEST_MESSAGES[missing_key_msg], result)

    def test_server_get_invest_models(self):
        """Translation: test that /models endpoint is translated."""
        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            test_client = ui_server.app.test_client()
            response = test_client.get('/models')
            result = json.loads(response.get_data(as_text=True))
        self.assertIn(
            TEST_MESSAGES['Carbon Storage and Sequestration'], result)

    def test_server_get_invest_getspec(self):
        """Translation: test that /getspec endpoint is translated."""
        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            test_client = ui_server.app.test_client()
            response = test_client.post('/getspec', json='carbon')
            spec = json.loads(response.get_data(as_text=True))
        self.assertEqual(
            spec['args']['lulc_cur_path']['name'],
            TEST_MESSAGES['current LULC'])

    def test_server_get_invest_validate(self):
        """Translation: test that /validate endpoint is translated."""
        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            test_client = ui_server.app.test_client()
            payload = {
                'model_module': carbon.ARGS_SPEC['pyname'],
                'args': json.dumps({})
            }
            response = test_client.post('/validate', json=payload)
        results = json.loads(response.get_data(as_text=True))
        messages = [item[1] for item in results]
        self.assertIn(TEST_MESSAGES[missing_key_msg], messages)

    def test_translate_formatted_string(self):
        with patch('natcap.invest.LOCALE_DIR', self.test_locale_dir):
            install_language('en')
        importlib.reload(validation)
        importlib.reload(carbon)
        args = {'n_workers': 'not a number'}
        validation_messages = carbon.validate(args)

        self.assertIn(
            TEST_MESSAGES[not_a_number_msg].format(value=args['n_workers']),
            str(validation_messages))
