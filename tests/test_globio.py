"""Module for Regression Testing the InVEST GLOBIO model."""
import unittest
import tempfile
import shutil
import os

import natcap.invest.pygeoprocessing_0_3_3.testing
from natcap.invest.pygeoprocessing_0_3_3.testing import scm
from osgeo import ogr
import numpy

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'globio')
SAMPLE_DATA = r"C:\Users\Joanna Lin\Desktop\test_folder\globio\invest-data"
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'globio')
REGRESSION_DATA = r"C:\Users\Joanna Lin\Desktop\test_folder\globio\invest-test-data"

def _make_dummy_file(workspace_dir, file_name):
    """Within workspace, create a dummy output file.

    Parameters:
        workspace_dir: path to workspace for making the file
    """
    output_path = os.path.join(workspace_dir, file_name)
    output = open(output_path, 'wb')
    output.close()

tempdir = r"C:\Users\Joanna Lin\Desktop\test_folder\globio"

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

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_globio_predefined_lulc(self):
        """GLOBIO: regression testing predefined LULC."""
        from natcap.invest import globio

        args = {
            'aoi_uri': '',
            'globio_lulc_uri': os.path.join(
                SAMPLE_DATA, 'globio_lulc_small.tif'),
            'infrastructure_dir':  os.path.join(
                SAMPLE_DATA, 'infrastructure_dir'),
            'intensification_fraction': '0.46',
            'msa_parameters_uri': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'predefined_globio': True,
            'workspace_dir': self.workspace_dir,
        }
        globio.execute(args)
        GLOBIOTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_lulc.txt'),
            args['workspace_dir'])

        natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
            os.path.join(self.workspace_dir, 'msa.tif'),
            os.path.join(REGRESSION_DATA, 'msa_lulc_regression.tif'), 1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_globio_empty_infra(self):
        """GLOBIO: testing that empty infra directory raises exception."""
        from natcap.invest import globio

        args = {
            'aoi_uri': '',
            'globio_lulc_uri': os.path.join(
                REGRESSION_DATA, 'globio_lulc_small.tif'),
            'infrastructure_dir':  os.path.join(
                REGRESSION_DATA, 'empty_dir'),
            'intensification_fraction': '0.46',
            'msa_parameters_uri': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'predefined_globio': True,
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            globio.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_globio_shape_infra(self):
        """GLOBIO: regression testing with shapefile infrastructure."""
        from natcap.invest import globio

        args = {
            'aoi_uri': '',
            'globio_lulc_uri': os.path.join(
                SAMPLE_DATA, 'globio_lulc_small.tif'),
            'infrastructure_dir':  os.path.join(
                REGRESSION_DATA, 'small_infrastructure'),
            'intensification_fraction': '0.46',
            'msa_parameters_uri': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'predefined_globio': True,
            'workspace_dir': tempdir  # self.workspace_dir
        }
        globio.execute(args)
        GLOBIOTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_lulc.txt'),
            args['workspace_dir'])

        natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
            os.path.join(self.workspace_dir, 'msa.tif'),
            os.path.join(REGRESSION_DATA, 'msa_shape_infra_regression.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_globio_full(self):
        """GLOBIO: regression testing all functionality."""
        from natcap.invest import globio

        args = {
            'aoi_uri': os.path.join(SAMPLE_DATA, 'sub_aoi.shp'),
            'globio_lulc_uri': '',
            'infrastructure_dir': os.path.join(
                SAMPLE_DATA, 'infrastructure_dir'),
            'intensification_fraction': '0.46',
            'lulc_to_globio_table_uri': os.path.join(
                SAMPLE_DATA, 'lulc_conversion_table.csv'),
            'lulc_uri': os.path.join(SAMPLE_DATA, 'lulc_2008.tif'),
            'msa_parameters_uri': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'pasture_threshold': '0.5',
            'pasture_uri': os.path.join(SAMPLE_DATA, 'pasture.tif'),
            'potential_vegetation_uri': os.path.join(
                SAMPLE_DATA, 'potential_vegetation.tif'),
            'predefined_globio': False,
            'primary_threshold': 0.66,
            'workspace_dir': self.workspace_dir,
        }

        # Test that overwriting output does not crash.
        _make_dummy_file(args['workspace_dir'], 'aoi_summary.shp')
        globio.execute(args)

        GLOBIOTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list.txt'),
            args['workspace_dir'])

        GLOBIOTests._assert_regression_results_eq(
            args['workspace_dir'],
            os.path.join(
                args['workspace_dir'], 'aoi_summary.shp'),
            os.path.join(REGRESSION_DATA, 'globio_agg_results.csv'))

        # inferring an explicit 'pass'
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
    def _assert_regression_results_eq(
            workspace_dir, result_vector_path, agg_results_path):
        """Test workspace state against expected aggregate results.

        Parameters:
            workspace_dir (string): path to the completed model workspace
            result_vector_path (string): path to the summary shapefile
                produced by the Forest Carbon Edge model.
            agg_results_path (string): path to a csv file that has the
                expected aggregated_results.shp table in the form of
                c_sum,c_ha_mean per line

        Returns:
            None

        Raises:
            AssertionError if any files are missing or results are out of
            range by `tolerance_places`
        """
        result_vector = ogr.Open(result_vector_path)
        result_layer = result_vector.GetLayer()

        # The tolerance of 3 digits after the decimal was determined by
        # experimentation on the application with the given range of numbers.
        # This is an apparently reasonable approach as described by ChrisF:
        # http://stackoverflow.com/a/3281371/42897
        # and even more reading about picking numerical tolerance (it's hard):
        # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        tolerance_places = 3

        with open(agg_results_path, 'rb') as agg_result_file:
            for line in agg_result_file:
                fid, sp_rich, en_rich, msa_mean = [
                    float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                for field, value in [
                        ('sp_rich', sp_rich),
                        ('en_rich', en_rich),
                        ('msa_mean', msa_mean)]:
                    numpy.testing.assert_almost_equal(
                        feature.GetField(field), value,
                        decimal=tolerance_places)
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None
