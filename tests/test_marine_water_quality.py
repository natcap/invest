"""Module for Regression Testing the InVEST Marine Water Quality model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from osgeo import ogr
import numpy

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'MarineWaterQuality', 'input')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'mwq')


class MarineWaterQualityTests(unittest.TestCase):
    """Tests for the Marine Water Quality model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_marine_water_quality_full(self):
        """MWQ: regression testing all functionality."""
        from natcap.invest.marine_water_quality import marine_water_quality_biophysical

        args = {
            'adv_uv_points_uri': os.path.join(
                SAMPLE_DATA, 'ADVuv_WGS1984_BCAlbers.shp'),
            'aoi_poly_uri': os.path.join(
                SAMPLE_DATA, 'AOI_clay_soundwideWQ.shp'),
            'kps': 0.001,
            'land_poly_uri': os.path.join(
                REGRESSION_DATA, 'simple_island.shp'),
            'layer_depth': 1.0,
            'pixel_size': 100.0,
            'source_point_data_uri': os.path.join(
                SAMPLE_DATA, 'WQM_PAR.csv'),
            'source_points_uri': os.path.join(
                SAMPLE_DATA, 'floathomes_centroids.shp'),
            'tide_e_points_uri': os.path.join(
                SAMPLE_DATA, 'TideE_WGS1984_BCAlbers.shp'),
            'workspace_dir': self.workspace_dir,
        }

        marine_water_quality_biophysical.execute(args)
        MarineWaterQualityTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'expected_file_list_regression.txt'),
            args['workspace_dir'])

        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(self.workspace_dir, 'output', 'concentration.tif'),
            os.path.join(
                REGRESSION_DATA, 'concentration_island.tif'), rel_tol=1e-4,
            abs_tol=1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_mwq_bad_cell_size(self):
        """MWQ: ensuring cells that are too large cause numerical failure."""
        from natcap.invest.marine_water_quality import marine_water_quality_biophysical

        args = {
            'adv_uv_points_uri': os.path.join(
                SAMPLE_DATA, 'ADVuv_WGS1984_BCAlbers.shp'),
            'aoi_poly_uri': os.path.join(
                SAMPLE_DATA, 'AOI_clay_soundwideWQ.shp'),
            'kps': 0.001,
            'land_poly_uri': os.path.join(
                REGRESSION_DATA, 'simple_island.shp'),
            'layer_depth': 1.0,
            'pixel_size': 100000.0,
            'source_point_data_uri': os.path.join(
                SAMPLE_DATA, 'WQM_PAR.csv'),
            'source_points_uri': os.path.join(
                SAMPLE_DATA, 'floathomes_centroids.shp'),
            'tide_e_points_uri': os.path.join(
                SAMPLE_DATA, 'TideE_WGS1984_BCAlbers.shp'),
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            marine_water_quality_biophysical.execute(args)

    @staticmethod
    def _test_same_files(base_list_path, directory_path):
        """Assert files in `base_list_path` are in `directory_path`.

        Parameters:
            base_list_path (string): a path to a file that has one relative
                file path per line.
            directory_path (string): a path to a directory whose contents will
                be checked against the files listed in `base_list_file`

        Returns:
            None

        Raises:
            AssertionError when there are files listed in `base_list_file`
                that don't exist in the directory indicated by `path`
        """
        missing_files = []
        with open(base_list_path, 'r') as file_list:
            for file_path in file_list:
                full_path = os.path.join(directory_path, file_path.rstrip())
                if full_path == '':
                    continue
                if not os.path.isfile(full_path):
                    missing_files.append(full_path)
        if len(missing_files) > 0:
            raise AssertionError(
                "The following files were expected but not found: " +
                '\n'.join(missing_files))
