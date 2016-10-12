"""Module for Regression Testing the InVEST Overlap Analysis model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'OverlapAnalysis',
    'Input')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'OverlapAnalysis')


class OverlapAnalysisTests(unittest.TestCase):
    """Tests for Overlap Analysis."""

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
    def test_overlap_analysis_mz(self):
        """Overlap Analysis: management zones."""
        import natcap.invest.overlap_analysis.overlap_analysis_mz

        args = {
            'overlap_data_dir_loc': os.path.join(
                SAMPLE_DATA, 'FisheriesLayers_RI'),
            'workspace_dir': self.workspace_dir,
            'zone_layer_loc': os.path.join(
                SAMPLE_DATA, 'ManagementZones_WCVI.shp'),
        }
        # invoke twice to cover the case where output already exists
        natcap.invest.overlap_analysis.overlap_analysis_mz.execute(args)
        natcap.invest.overlap_analysis.overlap_analysis_mz.execute(args)

        OverlapAnalysisTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_mz.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_vectors_equal(
            os.path.join(REGRESSION_DATA, 'mz_frequency.shp'),
            os.path.join(self.workspace_dir, 'output', 'mz_frequency.shp'),
            1e-6)


    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_overlap_analysis_full(self):
        """Overlap Analysis: regression test with all options enabled."""
        import natcap.invest.overlap_analysis.overlap_analysis

        args = {
            'decay_amt': '0.0001',
            'do_hubs': True,
            'do_inter': True,
            'do_intra': True,
            'grid_size': '1000',
            'hubs_uri': os.path.join(SAMPLE_DATA, 'PopulatedPlaces_WCVI.shx'),
            'intra_name': 'RI',
            'overlap_data_dir_uri': os.path.join(
                SAMPLE_DATA, 'FisheriesLayers_RI'),
            'overlap_layer_tbl': os.path.join(
                SAMPLE_DATA, 'Fisheries_Inputs.csv'),
            'workspace_dir': self.workspace_dir,
            'zone_layer_uri': os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp'),
        }
        natcap.invest.overlap_analysis.overlap_analysis.execute(args)
        OverlapAnalysisTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'hu_impscore.tif'),
            os.path.join(self.workspace_dir, 'output', 'hu_impscore.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_overlap_analysis_human_intra(self):
        """Overlap Analysis: regression test with hubs and intra."""
        import natcap.invest.overlap_analysis.overlap_analysis

        args = {
            'decay_amt': '0.0001',
            'do_hubs': True,
            'do_inter': False,
            'do_intra': True,
            'grid_size': '1000',
            'hubs_uri': os.path.join(SAMPLE_DATA, 'PopulatedPlaces_WCVI.shx'),
            'intra_name': 'RI',
            'overlap_data_dir_uri': os.path.join(
                SAMPLE_DATA, 'FisheriesLayers_RI'),
            'workspace_dir': self.workspace_dir,
            'zone_layer_uri': os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp'),
        }
        natcap.invest.overlap_analysis.overlap_analysis.execute(args)
        OverlapAnalysisTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'hu_impscore_human_intra.tif'),
            os.path.join(self.workspace_dir, 'output', 'hu_impscore.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_overlap_analysis_human_inter(self):
        """Overlap Analysis: regression test with hubs and inter."""
        import natcap.invest.overlap_analysis.overlap_analysis

        args = {
            'decay_amt': '0.0001',
            'do_hubs': True,
            'do_inter': True,
            'do_intra': False,
            'grid_size': '1000',
            'hubs_uri': os.path.join(SAMPLE_DATA, 'PopulatedPlaces_WCVI.shx'),
            'intra_name': 'RI',
            'overlap_layer_tbl': os.path.join(
                SAMPLE_DATA, 'Fisheries_Inputs.csv'),
            'overlap_data_dir_uri': os.path.join(
                SAMPLE_DATA, 'FisheriesLayers_RI'),
            'workspace_dir': self.workspace_dir,
            'zone_layer_uri': os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp'),
        }
        natcap.invest.overlap_analysis.overlap_analysis.execute(args)
        OverlapAnalysisTests._test_same_files(
            os.path.join(
                REGRESSION_DATA,
                'expected_file_list_inter_only.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'hu_impscore_human_inter.tif'),
            os.path.join(self.workspace_dir, 'output', 'hu_impscore.tif'),
            1e-6)


    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_overlap_analysis_intra_only(self):
        """Overlap Analysis: regression test with intra only enabled."""
        import natcap.invest.overlap_analysis.overlap_analysis

        args = {
            'do_hubs': False,
            'do_inter': False,
            'do_intra': True,
            'grid_size': '1000',
            'intra_name': 'RI',
            'overlap_data_dir_uri': os.path.join(
                SAMPLE_DATA, 'FisheriesLayers_RI'),
            'overlap_layer_tbl': os.path.join(
                SAMPLE_DATA, 'Fisheries_Inputs.csv'),
            'workspace_dir': self.workspace_dir,
            'zone_layer_uri': os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp'),
        }
        natcap.invest.overlap_analysis.overlap_analysis.execute(args)
        OverlapAnalysisTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_intra_only.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'hu_impscore_intra_only.tif'),
            os.path.join(self.workspace_dir, 'output', 'hu_impscore.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_overlap_analysis_inter_only(self):
        """Overlap Analysis: regression test with inter only enabled."""
        import natcap.invest.overlap_analysis.overlap_analysis

        args = {
            'do_hubs': False,
            'do_inter': True,
            'do_intra': False,
            'grid_size': '1000',
            'overlap_data_dir_uri': os.path.join(
                SAMPLE_DATA, 'FisheriesLayers_RI'),
            'overlap_layer_tbl': os.path.join(
                SAMPLE_DATA, 'Fisheries_Inputs.csv'),
            'workspace_dir': self.workspace_dir,
            'zone_layer_uri': os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp'),
        }
        natcap.invest.overlap_analysis.overlap_analysis.execute(args)
        OverlapAnalysisTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'expected_file_list_inter_only.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'hu_impscore_inter_only.tif'),
            os.path.join(self.workspace_dir, 'output', 'hu_impscore.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_overlap_analysis_all_disabled(self):
        """Overlap Analysis: regression test with all options disabled."""
        import natcap.invest.overlap_analysis.overlap_analysis

        args = {
            u'do_hubs': False,
            u'do_inter': False,
            u'do_intra': False,
            u'grid_size': u'1000',
            u'overlap_data_dir_uri': os.path.join(
                SAMPLE_DATA, 'FisheriesLayers_RI'),
            u'workspace_dir': self.workspace_dir,
            u'zone_layer_uri': os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp'),
        }
        natcap.invest.overlap_analysis.overlap_analysis.execute(args)
        OverlapAnalysisTests._test_same_files(
            os.path.join(
                REGRESSION_DATA, 'expected_file_list_all_disabled.txt'),
            args['workspace_dir'])
        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'hu_freq_all_disabled.tif'),
            os.path.join(self.workspace_dir, 'output', 'hu_freq.tif'),
            1e-6)

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
