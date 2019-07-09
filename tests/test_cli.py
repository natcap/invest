import sys
import os
import shutil
import tempfile
import unittest


class CLIHeadlessTests(unittest.TestCase):
    def setUp(self):
        """Use a temporary workspace for all tests in this class."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary workspace after a test run."""
        shutil.rmtree(self.workspace_dir)

    def test_run_fisheries(self):
        """CLI: Run the fisheries model through the cli."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-sample-data',
            'spiny_lobster_belize.invs.json')

        cli.main([
            'fisheries',  # uses an exact modelname
            '--debug',  # set logging
            '--datastack', parameter_set_path,
            '--headless',
            '--workspace', self.workspace_dir,
            '--overwrite',
        ])

    def test_run_ambiguous_modelname(self):
        """CLI: Raise an error when an ambiguous model name used."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-sample-data',
            'spiny_lobster_belize.invs.json')

        with self.assertRaises(SystemExit) as exit_cm:
            cli.main([
                'fish',  # ambiguous substring
                '--datastack', parameter_set_path,
                '--headless',
                '--workspace', self.workspace_dir,
                '--overwrite',
            ])
        self.assertEqual(exit_cm.exception.code, 1)

    def test_model_alias(self):
        """CLI: Use a model alias through the CLI."""
        from natcap.invest import cli

        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-sample-data',
            'cbc_galveston_bay.invs.json')

        cli.main([
            'cbc',  # uses an alias
            '--datastack', parameter_set_path,
            '--headless',
            '--workspace', self.workspace_dir,
            '--overwrite',
        ])

    def test_no_model_given(self):
        """CLI: Raise an error when no model name given."""
        from natcap.invest import cli
        with self.assertRaises(SystemExit) as exit_cm:
            cli.main('')
        self.assertEqual(exit_cm.exception.code, 1)

    def test_no_model_matches(self):
        """CLI: raise an error when no model name matches what's given."""
        from natcap.invest import cli
        with self.assertRaises(SystemExit) as exit_cm:
            cli.main('qwerty')
        self.assertEqual(exit_cm.exception.code, 1)

    def test_list(self):
        """CLI: Verify no error when listing models."""
        from natcap.invest import cli
        with self.assertRaises(SystemExit) as exit_cm:
            cli.main(['--list'])
        self.assertEqual(exit_cm.exception.code, 0)
