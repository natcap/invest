"""Module for Regression Testing the InVEST Timber module."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'timber')


class TimberRegressionTests(unittest.TestCase):
    """Regression Tests for the Timber Model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across regression tests."""
        args = {
            'workspace_dir': workspace_dir,
            'timber_shape_uri': os.path.join(
                SAMPLE_DATA, 'timber', 'input', 'plantation.shp'),
            'attr_table_uri': os.path.join(
                SAMPLE_DATA, 'timber', 'input', 'plant_table.csv'),
            'market_disc_rate': 7
        }
        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_timber(self):
        """Timber: testing the Timber model."""
        from natcap.invest.timber import timber

        args = TimberRegressionTests.generate_base_args(self.workspace_dir)

        timber.execute(args)

        vector_result = 'timber.shp'
        pygeoprocessing.testing.assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_result),
            os.path.join(REGRESSION_DATA, vector_result),
            field_tolerance=1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_suffix(self):
        """Timber: testing that the suffix is handled correctly."""
        from natcap.invest.timber import timber

        args = TimberRegressionTests.generate_base_args(self.workspace_dir)
        args['results_suffix'] = 'test'

        timber.execute(args)

        self.assertTrue(os.path.exists(
            os.path.join(args['workspace_dir'], 'output', 'timber_test.shp')))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_suffix_underscore(self):
        """Timber: testing that a suffix w/ underscore is handled correctly."""
        from natcap.invest.timber import timber

        args = TimberRegressionTests.generate_base_args(self.workspace_dir)
        args['results_suffix'] = '_test'

        timber.execute(args)

        self.assertTrue(os.path.exists(
            os.path.join(args['workspace_dir'], 'output', 'timber_test.shp')))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_timber_remove_files(self):
        """Timber: testing the Timber model works if files already exist."""
        from natcap.invest.timber import timber

        args = TimberRegressionTests.generate_base_args(self.workspace_dir)

        timber.execute(args)
        # run one more time to make sure files are removed properly
        timber.execute(args)

        vector_result = 'timber.shp'
        pygeoprocessing.testing.assert_vectors_equal(
            os.path.join(args['workspace_dir'], 'output', vector_result),
            os.path.join(REGRESSION_DATA, vector_result),
            field_tolerance=1e-9)
