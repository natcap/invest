import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', '_example_model')


class ExampleTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_regression(self):
        """
        Regression test for basic functionality.
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
            os.path.join(REGRESSION_DATA, 'regression_sum.tif'))

        shutil.rmtree(args['workspace_dir'])

