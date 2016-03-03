"""Module for Regression Testing the InVEST Scenic Quality module."""
import unittest
import tempfile
import shutil
import os
import csv

from osgeo import ogr
import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'scenic_quality')


class ScenicQualityUnitTests(unittest.TestCase):
    """Unit tests for Scenic Quality Model."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)


class ScenicQualityRegressionTests(unittest.TestCase):
    """Regression Tests for Scenic Quality Model."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
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
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'ScenicQuality', 'Input', 'AOI_WCVI.shp'),
            'structure_path': os.path.join(
                SAMPLE_DATA, 'ScenicQuality', 'Input', 'AquaWEM_points.shp'),
            'keep_feat_viewsheds': 'No',
            'keep_val_viewsheds': 'No',
            'dem_path': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 'claybark_dem'),
            'refraction': 0.13,
            'max_valuation_radius': 8000
        }

        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_scenic_quality(self):
        """SQ: testing stuff."""
        from natcap.invest.hydropower import scenic_quality

        args = ScenicQualityTests.generate_base_args(self.workspace_dir)

        scenic_quality.execute(args)
