"""Module for Regression Testing the InVEST Crop Production models."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

MODEL_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-crop-production-data',
    'global_dataset_20151210')
SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'crop_production20')


class CropProductionTests(unittest.TestCase):
    """Tests for the Crop Production model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(MODEL_DATA)
    def test_crop_production_percentile(self):
        """Crop Production: test crop production."""
        from natcap.invest import crop_production_percentile

        args = {
            'workspace_dir': self.workspace_dir,
        }
        crop_production_percentile.execute(args)
        self.fail("No test yet.")