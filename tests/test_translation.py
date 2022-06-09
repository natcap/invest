import importlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from babel.messages import Catalog, mofile
import natcap.invest
from natcap.invest import carbon, cli, validation, set_locale, ui_server

TEST_LANG = 'll'

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


def reset_locale():
    """Reset affected parts of the global context."""
    set_locale('en')

    # "unimport" the modules being translated
    # NOTE: it would be better to run each test in a new process,
    # but that's difficult on windows: https://stackoverflow.com/a/48310939
    importlib.reload(natcap.invest)
    importlib.reload(validation)
    importlib.reload(cli)
    importlib.reload(carbon)
    importlib.reload(ui_server)

class TranslationTests(unittest.TestCase):
    """Tests for translation."""

    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()
        test_locale_dir = os.path.join(self.workspace_dir, 'locales')
        mo_path = os.path.join(
            test_locale_dir, TEST_LANG, 'LC_MESSAGES', 'messages.mo')
        os.makedirs(os.path.dirname(mo_path))
        # format the test catalog object into MO file format
        with open(mo_path, 'wb') as mo_file:
            mofile.write_mo(mo_file, TEST_CATALOG)

        # patch the LOCALE_DIR variable in natcap/invest/__init__.py
        # this is where gettext will look for translations
        self.locale_dir_patcher = patch(
            'natcap.invest.LOCALE_DIR', test_locale_dir)
        # patch the LOCALES variable to allow a fake test locale
        self.locales_patcher = patch('natcap.invest.LOCALES', [TEST_LANG, 'en'])
        self.locale_dir_patcher.start()
        self.locales_patcher.start()

    def tearDown(self):
        reset_locale()  # reset after each test case
        self.locale_dir_patcher.stop()
        self.locales_patcher.stop()
        shutil.rmtree(self.workspace_dir)

    def test_invest_list(self):
        """Translation: test that CLI list output is translated."""
        from natcap.invest import cli
        with patch('sys.stdout', new=io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                cli.main(['--language', TEST_LANG, 'list'])
        result = out.getvalue()
        self.assertIn(TEST_MESSAGES['Available models:'], result)
        self.assertIn(
            TEST_MESSAGES['Carbon Storage and Sequestration'], result)

    def test_invest_getspec(self):
        """Translation: test that CLI getspec output is translated."""
        from natcap.invest import cli
        with patch('sys.stdout', new=io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                cli.main(['--language', TEST_LANG, 'getspec', 'carbon'])
        result = out.getvalue()
        self.assertIn(TEST_MESSAGES['current LULC'], result)

    def test_invest_validate(self):
        """Translation: test that CLI validate output is translated."""
        datastack = {  # write datastack to a JSON file
            'model_name': 'natcap.invest.carbon',
            'invest_version': '0.0',
            'args': {}
        }
        datastack_path = os.path.join(self.workspace_dir, 'datastack.json')
        with open(datastack_path, 'w') as file:
            json.dump(datastack, file)

        from natcap.invest import cli
        with patch('sys.stdout', new=io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                cli.main(
                    ['--language', TEST_LANG, 'validate', datastack_path])

        result = out.getvalue()
        self.assertIn(TEST_MESSAGES[missing_key_msg], result)

    def test_server_get_invest_models(self):
        """Translation: test that /models endpoint is translated."""
        test_client = ui_server.app.test_client()
        response = test_client.get(
            'api/models', query_string={'language': TEST_LANG})
        result = json.loads(response.get_data(as_text=True))
        self.assertIn(
            TEST_MESSAGES['Carbon Storage and Sequestration'], result)

    def test_server_get_invest_getspec(self):
        """Translation: test that /getspec endpoint is translated."""
        test_client = ui_server.app.test_client()
        response = test_client.post(
            'api/getspec', json='carbon', query_string={'language': TEST_LANG})
        spec = json.loads(response.get_data(as_text=True))
        self.assertEqual(
            spec['args']['lulc_cur_path']['name'],
            TEST_MESSAGES['current LULC'])

    def test_server_get_invest_validate(self):
        """Translation: test that /validate endpoint is translated."""
        test_client = ui_server.app.test_client()
        payload = {
            'model_module': carbon.ARGS_SPEC['pyname'],
            'args': json.dumps({})
        }
        response = test_client.post(
            'api/validate', json=payload,
            query_string={'language': TEST_LANG})
        results = json.loads(response.get_data(as_text=True))
        messages = [item[1] for item in results]
        self.assertIn(TEST_MESSAGES[missing_key_msg], messages)

    def test_translate_formatted_string(self):
        """Translation: test that f-string can be translated."""
        set_locale(TEST_LANG)
        importlib.reload(validation)
        importlib.reload(carbon)
        args = {'n_workers': 'not a number'}
        validation_messages = carbon.validate(args)
        self.assertIn(
            TEST_MESSAGES[not_a_number_msg].format(value=args['n_workers']),
            str(validation_messages))
