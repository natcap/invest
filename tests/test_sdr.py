"""InVEST SDR model tests."""

import unittest
import tempfile
import shutil
import os

import numpy
from osgeo import ogr
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data',
    'Base_Data', 'Freshwater')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'sdr')


class SDRTests(unittest.TestCase):
    """Regression tests for InVEST SDR model."""

    def setUp(self):
        """Initalize SDRRegression tests."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate a base sample args dict for SDR."""
        args = {
            'biophysical_table_path': os.path.join(
                SAMPLE_DATA, 'biophysical_table.csv'),
            'dem_path': os.path.join(SAMPLE_DATA, 'dem'),
            'erodibility_path': os.path.join(SAMPLE_DATA, 'erodibility'),
            'erosivity_path': os.path.join(SAMPLE_DATA, 'erosivity'),
            'ic_0_param': '0.5',
            'k_param': '2',
            'lulc_path': os.path.join(SAMPLE_DATA, 'landuse_90'),
            'sdr_max': '0.8',
            'threshold_flow_accumulation': '1000',
            'watersheds_path': os.path.join(SAMPLE_DATA, 'watersheds.shp'),
            'workspace_dir': workspace_dir,
        }
        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_base_regression(self):
        """SDR base regression test on sample data.

        Execute SDR with sample data and checks that the output files are
        generated and that the aggregate shapefile fields are the same as the
        regression case.
        """
        from natcap.invest import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)
        # make args explicit that this is a base run of SWY
        sdr.execute(args)

        SDRTests._assert_regression_results_equal(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_base.txt'),
            os.path.join(args['workspace_dir'], 'watershed_results_sdr.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_base.csv'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_output_exists_regression(self):
        """SDR test case where an output shapefile already exists.

        Execute SDR with sample data but workspace already contains
        "watershed_results_sdr.shp".  Model should delete file and proceed
        with report.
        """
        from natcap.invest import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)

        # copy AOI on top of where the output shapefile should reside
        shutil.copy(
            args['watersheds_path'], os.path.join(
                self.workspace_dir, 'watershed_results_sdr.shp'))

        sdr.execute(args)

        SDRTests._assert_regression_results_equal(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_base.txt'),
            os.path.join(args['workspace_dir'], 'watershed_results_sdr.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_base.csv'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_drainage_regression(self):
        """SDR drainage layer regression test on sample data.

        Execute SDR with sample data and a drainage layer and checks that the
        output files are generated and that the aggregate shapefile fields
        are the same as the regression case.
        """
        from natcap.invest import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)
        args['drainage_path'] = os.path.join(
            REGRESSION_DATA, 'sample_drainage.tif')
        sdr.execute(args)

        SDRTests._assert_regression_results_equal(
            args['workspace_dir'],
            os.path.join(REGRESSION_DATA, 'file_list_drainage.txt'),
            os.path.join(args['workspace_dir'], 'watershed_results_sdr.shp'),
            os.path.join(REGRESSION_DATA, 'agg_results_drainage.csv'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_base_usle_c_too_large(self):
        """SDR test exepected exception for USLE_C > 1.0."""
        from natcap.invest import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)
        args['biophysical_table_path'] = os.path.join(
            REGRESSION_DATA, 'biophysical_table_too_large.csv')

        with self.assertRaises(ValueError):
            sdr.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_base_usle_p_nan(self):
        """SDR est expected exception for USLE_P not a number."""
        from natcap.invest import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)
        args['biophysical_table_path'] = os.path.join(
            REGRESSION_DATA, 'biophysical_table_invalid_value.csv')

        with self.assertRaises(ValueError):
            sdr.execute(args)

    @staticmethod
    def _assert_regression_results_equal(
            workspace_dir, file_list_path, result_vector_path,
            agg_results_path):
        """Test workspace state against expected aggregate results.

        Parameters:
            workspace_dir (string): path to the completed model workspace
            file_list_path (string): path to a file that has a list of all
                the expected files relative to the workspace base
            result_vector_path (string): path to the summary shapefile
                produced by the SWY model.
            agg_results_path (string): path to a csv file that has the
                expected aggregated_results.shp table in the form of
                fid,vri_sum,qb_val per line

        Returns:
            None

        Raises:
            AssertionError if any files are missing or results are out of
            range by `tolerance_places`
        """
        # test that the workspace has the same files as we expect
        SDRTests._test_same_files(
            file_list_path, workspace_dir)

        # we expect a file called 'aggregated_results.shp'
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
                fid, sed_retent, sed_export, usle_tot = [
                    float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                for field, value in [
                        ('sed_retent', sed_retent),
                        ('sed_export', sed_export),
                        ('usle_tot', usle_tot)]:
                    numpy.testing.assert_almost_equal(
                        feature.GetField(field), value,
                        decimal=tolerance_places)
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None

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
