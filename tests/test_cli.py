import sys
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import json

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


@contextlib.contextmanager
def redirect_stdout():
    """Redirect stdout to a stream, which is then yielded."""
    old_stdout = sys.stdout
    stdout_buffer = StringIO()
    sys.stdout = stdout_buffer
    yield stdout_buffer
    sys.stdout = old_stdout


class CLIHeadlessTests(unittest.TestCase):
    def setUp(self):
        """Use a temporary workspace for all tests in this class."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary workspace after a test run."""
        shutil.rmtree(self.workspace_dir)

    def test_run_fisheries_workspace_in_json(self):
        """CLI: Run the fisheries model with JSON-defined workspace."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'fisheries', 'spiny_lobster_belize.invs.json')

        datastack_dict = json.load(open(parameter_set_path))
        datastack_dict['args']['workspace_dir'] = self.workspace_dir
        new_parameter_set_path = os.path.join(
            self.workspace_dir, 'paramset.invs.json')
        with open(new_parameter_set_path, 'w') as parameter_set_file:
            parameter_set_file.write(
                json.dumps(datastack_dict, indent=4, sort_keys=True))

        with unittest.mock.patch(
                'natcap.invest.fisheries.fisheries.execute',
                return_value=None) as patched_model:
            cli.main([
                'run',
                'fisheries',  # uses an exact modelname
                '--datastack', new_parameter_set_path,
                '--headless',
            ])
        patched_model.assert_called_once()

    def test_run_fisheries(self):
        """CLI: Run the fisheries model through the cli."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'fisheries', 'spiny_lobster_belize.invs.json')

        with unittest.mock.patch(
                'natcap.invest.fisheries.fisheries.execute',
                return_value=None) as patched_model:
            cli.main([
                'run',
                'fisheries',  # uses an exact modelname
                '--datastack', parameter_set_path,
                '--headless',
                '--workspace', self.workspace_dir,
            ])
        patched_model.assert_called_once()

    def test_run_fisheries_no_workspace(self):
        """CLI: Run the fisheries model through the cli without a workspace."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'fisheries', 'spiny_lobster_belize.invs.json')

        with self.assertRaises(SystemExit) as exit_cm:
            cli.main([
                'run',
                'fisheries',  # uses an exact modelname
                '--datastack', parameter_set_path,
                '--headless',
            ])
        self.assertEqual(exit_cm.exception.code, 1)

    def test_run_fisheries_no_datastack(self):
        """CLI: Run the fisheries model through the cli without a datastack."""
        from natcap.invest import cli

        with self.assertRaises(SystemExit) as exit_cm:
            cli.main([
                'run',
                'fisheries',  # uses an exact modelname
                '--headless',
                '--workspace', self.workspace_dir,
            ])
        self.assertEqual(exit_cm.exception.code, 1)

    def test_run_fisheries_invalid_datastack(self):
        """CLI: Run the fisheries model through the cli invalid datastack."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            self.workspace_dir, 'bad-paramset.invs.json')

        with open(parameter_set_path, 'w') as paramset_file:
            paramset_file.write('not a json object')

        with self.assertRaises(SystemExit) as exit_cm:
            cli.main([
                'run',
                'fisheries',  # uses an exact modelname
                '--datastack', parameter_set_path,
                '--headless',
            ])
        self.assertEqual(exit_cm.exception.code, 1)

    def test_run_ambiguous_modelname(self):
        """CLI: Raise an error when an ambiguous model name used."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'fisheries', 'spiny_lobster_belize.invs.json')

        with self.assertRaises(SystemExit) as exit_cm:
            cli.main([
                'run',
                'fish',  # ambiguous substring
                '--datastack', parameter_set_path,
                '--headless',
                '--workspace', self.workspace_dir,
            ])
            self.assertEqual(exit_cm.exception.code, 1)

    def test_model_alias(self):
        """CLI: Use a model alias through the CLI."""
        from natcap.invest import cli

        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'coastal_blue_carbon', 'cbc_galveston_bay.invs.json')

        with unittest.mock.patch(
                'natcap.invest.coastal_blue_carbon.coastal_blue_carbon.execute',
                return_value=None) as patched_model:
            cli.main([
                'run',
                'cbc',  # uses an alias
                '--datastack', parameter_set_path,
                '--headless',
                '--workspace', self.workspace_dir,
            ])
        patched_model.assert_called_once()

    def test_no_model_given(self):
        """CLI: Raise an error when no model name given."""
        from natcap.invest import cli
        with self.assertRaises(SystemExit) as exit_cm:
            cli.main(['run'])
        self.assertEqual(exit_cm.exception.code, 2)

    def test_no_model_matches(self):
        """CLI: raise an error when no model name matches what's given."""
        from natcap.invest import cli
        with self.assertRaises(SystemExit) as exit_cm:
            cli.main(['run', 'qwerty'])
        self.assertEqual(exit_cm.exception.code, 1)

    def test_list(self):
        """CLI: Verify no error when listing models."""
        from natcap.invest import cli
        with self.assertRaises(SystemExit) as exit_cm:
            cli.main(['list'])
        self.assertEqual(exit_cm.exception.code, 0)

    def test_list_json(self):
        """CLI: Verify no error when listing models as JSON."""
        from natcap.invest import cli
        with redirect_stdout() as stdout_stream:
            with self.assertRaises(SystemExit) as exit_cm:
                cli.main(['list', '--json'])

        # Verify that we can load the JSON object without error
        stdout_value = stdout_stream.getvalue()
        loaded_list_object = json.loads(stdout_value)
        self.assertEqual(type(loaded_list_object), dict)

        self.assertEqual(exit_cm.exception.code, 0)

    def test_validate_fisheries(self):
        """CLI: Validate the fisheries model inputs through the cli."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'fisheries', 'spiny_lobster_belize.invs.json')

        # The InVEST sample data JSON arguments don't have a workspace, so I
        # need to add it in.
        datastack_dict = json.load(open(parameter_set_path))
        datastack_dict['args']['workspace_dir'] = self.workspace_dir
        new_parameter_set_path = os.path.join(
            self.workspace_dir, 'paramset.invs.json')
        with open(new_parameter_set_path, 'w') as parameter_set_file:
            parameter_set_file.write(
                json.dumps(datastack_dict, indent=4, sort_keys=True))

        with redirect_stdout() as stdout_stream:
            with self.assertRaises(SystemExit) as exit_cm:
                cli.main([
                    'validate',
                    new_parameter_set_path,
                ])
        self.assertTrue(len(stdout_stream.getvalue()) > 0)
        self.assertEqual(exit_cm.exception.code, 0)

    def test_validate_fisheries_missing_workspace(self):
        """CLI: Validate the fisheries model inputs through the cli."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'fisheries', 'spiny_lobster_belize.invs.json')

        # The InVEST sample data JSON arguments don't have a workspace.  In
        # this case, I want to leave it out and verify validation catches it.

        with redirect_stdout() as stdout_stream:
            with self.assertRaises(SystemExit) as exit_cm:
                cli.main([
                    'validate',
                    parameter_set_path,
                ])
        self.assertTrue(len(stdout_stream.getvalue()) > 0)

        # Validation failed, not the program.
        self.assertEqual(exit_cm.exception.code, 0)

    def test_validate_fisheries_missing_workspace_json(self):
        """CLI: Validate the fisheries model inputs through the cli."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'fisheries', 'spiny_lobster_belize.invs.json')

        # The InVEST sample data JSON arguments don't have a workspace.  In
        # this case, I want to leave it out and verify validation catches it.

        with redirect_stdout() as stdout_stream:
            with self.assertRaises(SystemExit) as exit_cm:
                cli.main([
                    'validate',
                    parameter_set_path,
                    '--json',
                ])
        stdout = stdout_stream.getvalue()
        self.assertTrue(len(stdout) > 0)
        self.assertEqual(len(json.loads(stdout)), 1)  # workspace_dir invalid

        # Validation failed, not the program.
        self.assertEqual(exit_cm.exception.code, 0)

    def test_validate_invalid_json(self):
        """CLI: Validate invalid json files set an error code."""
        from natcap.invest import cli

        paramset_path = os.path.join(self.workspace_dir, 'invalid.json')
        with open(paramset_path, 'w') as opened_file:
            opened_file.write('not a json object')

        with redirect_stdout() as stdout_stream:
            with self.assertRaises(SystemExit) as exit_cm:
                cli.main([
                    'validate',
                    paramset_path,
                    '--json',
                ])
        self.assertTrue(len(stdout_stream.getvalue()) == 0)
        self.assertEqual(exit_cm.exception.code, 1)

    def test_validate_fisheries_json(self):
        """CLI: Validate the fisheries model inputs as JSON through the cli."""
        from natcap.invest import cli
        parameter_set_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'invest-test-data',
            'fisheries', 'spiny_lobster_belize.invs.json')

        # The InVEST sample data JSON arguments don't have a workspace, so I
        # need to add it in.
        datastack_dict = json.load(open(parameter_set_path))
        datastack_dict['args']['workspace_dir'] = self.workspace_dir

        # In this case, I also want to set one of the inputs to an invalid path
        # to test the presentation of a validation error.
        datastack_dict['args']['aoi_vector_path'] = os.path.join(
            self.workspace_dir, 'not-a-vector.shp')

        new_parameter_set_path = os.path.join(
            self.workspace_dir, 'paramset.invs.json')
        with open(new_parameter_set_path, 'w') as parameter_set_file:
            parameter_set_file.write(
                json.dumps(datastack_dict, indent=4, sort_keys=True))

        with redirect_stdout() as stdout_stream:
            with self.assertRaises(SystemExit) as exit_cm:
                cli.main([
                    'validate',
                    new_parameter_set_path,
                    '--json',
                ])
        stdout = stdout_stream.getvalue()
        stdout_json = json.loads(stdout)
        self.assertEqual(len(stdout_json), 1)
        self.assertEqual(len(stdout_json['validation_results']), 4)

        # Validation returned successfully, so error code 0 even though there
        # are warnings.
        self.assertEqual(exit_cm.exception.code, 0)
