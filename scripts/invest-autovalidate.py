#!python

import os
import tempfile
import logging
import argparse
import unittest
import glob
import importlib
import shutil
import pprint

from natcap.invest import datastack

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('invest-autovalidate.py')


class ValidateExceptionTests(unittest.TestCase):
    """Tests for updating latest installer URLs."""

    def setUp(self):
        """Overriding setUp function to create temp file."""
        self.workspace = tempfile.mkdtemp()
        datastack_path = os.path.join(
            self.workspace, 'dummy.invs.json')
        with open(datastack_path, 'wb') as file:
            file.write('"args": {"something":"else"}, "model_name": natcap.invest.carbon')

    def tearDown(self):
        """Overriding tearDown function to remove temporary file."""
        shutil.rmtree(self.workspace)

    def test_exception_on_invalid_data(self):
        """"""
        with self.assertRaises(ValueError):
            main(self.workspace)


def main(sampledatadir):
    """Do validation for each datastack and store error messages."""
    validation_messages = ''
    for datastack_path in glob.glob(os.path.join(sampledatadir, '*invs.json')):

        paramset = datastack.extract_parameter_set(datastack_path)
        paramset.args['workspace_dir'] = tempfile.mkdtemp()  # missing from some sample datastacks
        # module_name = paramset.model_name
        model_module = importlib.import_module(name=paramset.model_name)

        try:
            LOGGER.info('validating %s ', datastack_path)
            model_warnings = getattr(
                model_module, 'validate')(paramset.args)
        except AttributeError:
            LOGGER.warn(
                '%s does not have a defined validation function.',
                paramset.model_name)
        finally:
            if model_warnings:
                LOGGER.error(model_warnings)
                validation_messages += (
                    os.linesep + datastack_path + ': ' +
                    pprint.pformat(model_warnings))

    if validation_messages:
        raise ValueError(validation_messages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Validate all sample datastacks using InVEST modules' `validate`")
    parser.add_argument('sampledatadir', type=str)
    args = parser.parse_args()
    main(args.sampledatadir)
