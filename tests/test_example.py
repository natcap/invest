import importlib
import itertools
import logging
import os
import pkgutil
import shutil
import tempfile
import unittest

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    '_example_model')

LOGGER = logging.getLogger('test_example')

class ExampleTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        EXAMPLE: Regression test for basic functionality.
        """
        from natcap.invest import _example_model
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': os.path.join(SAMPLE_DATA, 'Base_Data',
                                         'Terrestrial', 'lulc_samp_cur'),
        }
        _example_model.execute(args)

        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(args['workspace_dir'], 'sum.tif'),
            os.path.join(REGRESSION_DATA, 'regression_sum.tif'),
            tolerance=1e-9)

        shutil.rmtree(args['workspace_dir'])


class InVESTImportTest(unittest.TestCase):
    def test_import_everything(self):
        """InVEST: Import everything for the sake of coverage."""
        import natcap.invest

        iteration_args = {
            'path': natcap.invest.__path__,
            'prefix': 'natcap.invest.',
        }
        for _loader, name, _is_pkg in itertools.chain(
                pkgutil.walk_packages(**iteration_args),  # catch packages
                pkgutil.iter_modules(**iteration_args)):  # catch modules
            try:
                importlib.import_module(name)
            except Exception:
                # If we encounter an exception when importing a module, log it
                # but continue.
                LOGGER.exception('Error importing %s', name)
