#!python

import argparse
import glob
import importlib
import logging
import os
import pprint
import shutil
import tempfile
import unittest

from natcap.invest import datastack
from natcap.invest import models

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('invest-autovalidate.py')


class ValidateExceptionTests(unittest.TestCase):
    """Tests for updating latest installer URLs."""

    def setUp(self):
        """Overriding setUp function to create temp file."""
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary dir."""
        shutil.rmtree(self.workspace)

    def test_exception_on_invalid_data(self):
        """Test ValueError is raised on invalid datastack."""
        datastack_path = os.path.join(
            self.workspace, 'dummy.invs.json')
        with open(datastack_path, 'w') as file:
            file.write('"args": {"something": "else"},')
            file.write('"model_id": "carbon"')
        with self.assertRaises(ValueError):
            main(self.workspace)

    def test_exception_on_workspace_dir(self):
        """Test ValueError is raised if workspace_dir is defined in datastack."""
        datastack_path = os.path.join(
            self.workspace, 'dummy.invs.json')
        with open(datastack_path, 'w') as file:
            file.write('"args": {"workspace_dir": "/home/foo"},')
            file.write('"model_id": "carbon"')
        with self.assertRaises(ValueError):
            main(self.workspace)


def main(sampledatadir):
    """Do validation for each datastack and store error messages.

    Parameters:
        sampledatadir (string): path to the invest-sample-data repository,
            where '*invs.json' datastack files are expected to be in the root.

    Returns:
        None

    Raises:
        ValueError if any module's `validate` function issued warnings.
    """
    validation_messages = ''
    datastacks = glob.glob(os.path.join(sampledatadir, '**/*.json'))
    if not datastacks:
        raise ValueError(f'no json files found in {sampledatadir}')

    for datastack_path in datastacks:
        paramset = datastack.extract_parameter_set(datastack_path)
        if 'workspace_dir' in paramset.args and \
                paramset.args['workspace_dir'] != '':
            msg = (
                f'{datastack_path} : workspace_dir should not be defined '
                'for sample datastacks' )
            validation_messages += os.linesep + msg
            LOGGER.error(msg)
        else:
            paramset.args['workspace_dir'] = tempfile.mkdtemp()
        model_module = importlib.import_module(
            name=models.model_id_to_pyname[paramset.model_id])

        model_warnings = []  # define here in case of uncaught exception.
        try:
            LOGGER.info(f'validating {os.path.abspath(datastack_path)}')
            model_warnings = getattr(
                model_module, 'validate')(paramset.args)
        except AttributeError as err:
            # If there was no validate function, don't crash but raise it later.
            model_warnings = err
        finally:
            if model_warnings:
                LOGGER.error(model_warnings)
                validation_messages += (
                    os.linesep + datastack_path + ': ' +
                    pprint.pformat(model_warnings))
            if os.path.exists(paramset.args['workspace_dir']):
                os.rmdir(paramset.args['workspace_dir'])

    if validation_messages:
        raise ValueError(validation_messages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Validate all sample datastacks "
                    "using InVEST modules' `validate`")
    parser.add_argument('sampledatadir', type=str)
    args = parser.parse_args()
    main(args.sampledatadir)
