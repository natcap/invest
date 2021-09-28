import io
import shutil
import tempfile
import unittest
import unittest.mock

import babel
import babel.messages.mofile

TEST_LANG = 'es'
TEST_CATALOG = (
    f'src/natcap/invest/translations/locales/{TEST_LANG}/LC_MESSAGES/messages.mo')


class TranslationTests(unittest.TestCase):
    """Tests for translation."""
    def setUp(self):
        """Use a temporary workspace for all tests in this class."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary workspace after a test run."""
        shutil.rmtree(self.workspace_dir)

    def test_invest_list(self):
        from natcap.invest import cli
        with open(TEST_CATALOG, 'r+b') as catalog_file:
            catalog = babel.messages.mofile.read_mo(catalog_file)
        expected = catalog.get('Available models:').string

        with unittest.mock.patch('sys.stdout', new=io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                cli.main(['--language', TEST_LANG, 'list'])
        self.assertTrue(expected in out.getvalue())
