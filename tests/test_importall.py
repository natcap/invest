"""Test to import all InVEST modules for accurate nosetest coverage."""
import importlib
import itertools
import logging
import os
import pkgutil
import unittest

LOGGER = logging.getLogger('test_example')


class InVESTImportTest(unittest.TestCase):
    """A 'test' that imports all natcap.invest packages for coverage."""

    @unittest.skip(
        "skipping since it imports the ui which causes a Windows Exception")
    def test_import_everything(self):
        """InVEST: Import everything for the sake of coverage."""
        import natcap.invest

        iteration_args = {
            'path': natcap.invest.__path__,
            'prefix': 'natcap.invest.',
        }
        package_iterator = itertools.chain(
            pkgutil.walk_packages(**iteration_args),  # catch packages
            pkgutil.iter_modules(**iteration_args))
        while True:
            try:
                _, name, _ = package_iterator.next()
                importlib.import_module(name)
            except (ImportError, TypeError):
                # If we encounter an exception when importing a module, log it
                # but continue.
                LOGGER.exception('Error importing %s', name)
            except StopIteration:
                break
