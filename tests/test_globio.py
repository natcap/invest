"""Module for Regression Testing the InVEST GLOBIO model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from osgeo import ogr
from osgeo import gdal
import numpy

from natcap.invest import utils

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'globio',
    'Input')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'globio')


def _make_dummy_file(workspace_dir, file_name):
    """Within workspace, create a dummy output file to be overwritten.

    Parameters:
        workspace_dir: path to workspace for making the file
    """
    output_path = os.path.join(workspace_dir, file_name)
    output = open(output_path, 'wb')
    output.close()


class GLOBIOTests(unittest.TestCase):
    """Tests for the GLOBIO model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_globio_predefined_lulc(self):
        """GLOBIO: regression testing predefined LULC (mode b)."""
        from natcap.invest import globio

        args = {
            'aoi_path': '',
            'globio_lulc_path': os.path.join(
                SAMPLE_DATA, 'globio_lulc_small.tif'),
            'infrastructure_dir':  os.path.join(
                SAMPLE_DATA, 'infrastructure_dir'),
            'intensification_fraction': '0.46',
            'msa_parameters_path': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'predefined_globio': True,
            'workspace_dir': self.workspace_dir,
            'n_workers': '-1',
        }
        globio.execute(args)
        GLOBIOTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_lulc.txt'),
            args['workspace_dir'])

        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(args['workspace_dir'], 'msa.tif'),
            os.path.join(REGRESSION_DATA, 'msa_lulc_regression.tif'), 1e-6)

    def test_globio_empty_infra(self):
        """GLOBIO: testing that empty infra directory raises exception."""
        from natcap.invest import globio

        args = {
            'aoi_path': '',
            'globio_lulc_path': os.path.join(
                SAMPLE_DATA, 'globio_lulc_small.tif'),
            'infrastructure_dir':  os.path.join(
                SAMPLE_DATA, 'empty_dir'),
            'intensification_fraction': '0.46',
            'msa_parameters_path': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'predefined_globio': True,
            'workspace_dir': self.workspace_dir,
            'n_workers': '-1',
        }

        with self.assertRaises(ValueError):
            globio.execute(args)

    def test_globio_shape_infra(self):
        """GLOBIO: regression testing with shapefile infrastructure."""
        from natcap.invest import globio

        args = {
            'aoi_path': '',
            'globio_lulc_path': os.path.join(
                SAMPLE_DATA, 'globio_lulc_small.tif'),
            'infrastructure_dir':  os.path.join(
                SAMPLE_DATA, 'shape_infrastructure'),
            'intensification_fraction': '0.46',
            'msa_parameters_path': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'predefined_globio': True,
            'workspace_dir': self.workspace_dir,
            'n_workers': '-1',
        }
        globio.execute(args)
        GLOBIOTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_lulc.txt'),
            args['workspace_dir'])

        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(args['workspace_dir'], 'msa.tif'),
            os.path.join(REGRESSION_DATA, 'msa_shape_infra_regression.tif'),
            1e-6)

    def test_globio_full(self):
        """GLOBIO: regression testing all functionality (mode a)."""
        from natcap.invest import globio

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'sub_aoi.shp'),
            'globio_lulc_path': '',
            'infrastructure_dir': os.path.join(
                SAMPLE_DATA, 'infrastructure_dir'),
            'intensification_fraction': '0.46',
            'lulc_to_globio_table_path': os.path.join(
                SAMPLE_DATA, 'lulc_conversion_table.csv'),
            'lulc_path': os.path.join(SAMPLE_DATA, 'lulc_2008.tif'),
            'msa_parameters_path': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'pasture_threshold': '0.5',
            'pasture_path': os.path.join(SAMPLE_DATA, 'pasture.tif'),
            'potential_vegetation_path': os.path.join(
                SAMPLE_DATA, 'potential_vegetation.tif'),
            'predefined_globio': False,
            'primary_threshold': 0.66,
            'workspace_dir': self.workspace_dir,
            'n_workers': '-1',
        }

        # Test that overwriting output does not crash.
        _make_dummy_file(args['workspace_dir'], 'aoi_summary.shp')
        globio.execute(args)

        GLOBIOTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list.txt'),
            args['workspace_dir'])

        GLOBIOTests._assert_regression_results_eq(
            os.path.join(
                args['workspace_dir'], 'aoi_summary.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results.csv'))

        # Infer an explicit 'pass'
        self.assertTrue(True)

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

    @staticmethod
    def _assert_regression_results_eq(result_vector_path, agg_results_path):
        """Test output vector against expected aggregate results.

        Parameters:
            result_vector_path (string): path to the summary shapefile
                produced by GLOBIO model
            agg_results_path (string): path to a csv file that has the
                expected aoi_summary.shp table in the form of
                fid,msa_mean per line

        Returns:
            None

        Raises:
            AssertionError if results are out of range by `tolerance_places`
        """
        result_vector = gdal.OpenEx(result_vector_path, gdal.OF_VECTOR)
        result_layer = result_vector.GetLayer()

        # The tolerance of 3 digits after the decimal was determined by
        # experimentation on the application with the given range of numbers.
        # This is an apparently reasonable approach as described by ChrisF:
        # http://stackoverflow.com/a/3281371/42897
        # and even more reading about picking numerical tolerance (it's hard):
        # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        tolerance_places = 3
        expected_results = utils.build_lookup_from_csv(agg_results_path, 'fid')
        try:
            for feature in result_layer:
                fid = feature.GetFID()
                result_value = feature.GetField('msa_mean')
                if result_value is not None:
                    numpy.testing.assert_almost_equal(
                        result_value,
                        float(expected_results[fid]['msa_mean']),
                        decimal=tolerance_places)
                else:
                    # the out-of-bounds polygon will have no result_value
                    assert(expected_results[fid]['msa_mean'] == '')
        finally:
            feature = None
            result_layer = None
            gdal.Dataset.__swig_destroy__(result_vector)
            result_vector = None
