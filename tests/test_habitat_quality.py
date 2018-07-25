"""Module for Regression Testing the InVEST Habitat Quality model."""
import unittest
import tempfile
import shutil
import os

import natcap.invest.pygeoprocessing_0_3_3.testing
from natcap.invest.pygeoprocessing_0_3_3.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'habitat_quality')


class HabitatQualityTests(unittest.TestCase):
    """Tests for the Habitat Quality model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_habitat_quality_regression(self):
        """Habitat Quality: base regression test."""
        from natcap.invest import habitat_quality

        args = {
            'access_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'access_samp.shp'),
            'half_saturation_constant': '0.5',
            'landuse_bas_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'lc_samp_bse_b.tif'),
            'landuse_cur_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'lc_samp_cur_b.tif'),
            'landuse_fut_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'lc_samp_fut_b.tif'),
            'sensitivity_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'sensitivity_samp.csv'),
            'suffix': 'regression',
            'threat_raster_folder': os.path.join(
                SAMPLE_DATA, 'HabitatQuality'),
            'threats_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'threats_samp.csv'),
            u'workspace_dir': self.workspace_dir,
        }

        habitat_quality.execute(args)
        HabitatQualityTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list_regression.txt'),
            args['workspace_dir'])

        for output_filename in [
                'rarity_f_regression.tif', 'deg_sum_out_c_regression.tif',
                'deg_sum_out_f_regression.tif',
                'quality_out_c_regression.tif',
                'quality_out_f_regression.tif', 'rarity_c_regression.tif']:
            natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
                os.path.join(REGRESSION_DATA, output_filename),
                os.path.join(self.workspace_dir, 'output', output_filename),
                1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_habitat_quality_missing_sensitivity_threat(self):
        """Habitat Quality: ValueError w/ missing threat in sensitivity."""
        from natcap.invest import habitat_quality

        args = {
            'access_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'access_samp.shp'),
            'half_saturation_constant': '0.5',
            'landuse_cur_uri': os.path.join(
                REGRESSION_DATA, 'small_lulc_base.tif'),
            'sensitivity_uri': os.path.join(
                REGRESSION_DATA, 'small_sensitivity_samp.csv'),
            'threat_raster_folder': os.path.join(REGRESSION_DATA),
            'threats_uri': os.path.join(
                REGRESSION_DATA, 'small_threats_samp_missing_threat.csv'),
            u'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_habitat_quality_missing_threat(self):
        """Habitat Quality: expected ValueError on missing threat raster."""
        from natcap.invest import habitat_quality

        args = {
            'access_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'access_samp.shp'),
            'half_saturation_constant': '0.5',
            'landuse_cur_uri': os.path.join(
                REGRESSION_DATA, 'small_lulc_base.tif'),
            'sensitivity_uri': os.path.join(
                REGRESSION_DATA, 'small_sensitivity_samp_missing_threat.csv'),
            'threat_raster_folder': os.path.join(REGRESSION_DATA),
            'threats_uri': os.path.join(
                REGRESSION_DATA, 'small_threats_samp_missing_threat.csv'),
            u'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_habitat_quality_invalid_decay_type(self):
        """Habitat Quality: expected ValueError on invalid decay type."""
        from natcap.invest import habitat_quality

        args = {
            'access_uri': os.path.join(
                SAMPLE_DATA, 'HabitatQuality', 'access_samp.shp'),
            'half_saturation_constant': '0.5',
            'landuse_cur_uri': os.path.join(
                REGRESSION_DATA, 'small_lulc_base.tif'),
            'sensitivity_uri': os.path.join(
                REGRESSION_DATA, 'small_sensitivity_samp.csv'),
            'threat_raster_folder': os.path.join(REGRESSION_DATA),
            'threats_uri': os.path.join(
                REGRESSION_DATA, 'small_threats_samp_invalid_decay.csv'),
            u'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_habitat_quality_bad_rasters(self):
        """Habitat Quality: on threats that aren't real rasters."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'landuse_cur_uri': os.path.join(
                REGRESSION_DATA, 'small_lulc_base.tif'),
            'sensitivity_uri': os.path.join(
                REGRESSION_DATA, 'small_sensitivity_samp.csv'),
            'threat_raster_folder': os.path.join(
                REGRESSION_DATA, 'bad_rasters'),
            'threats_uri': os.path.join(
                REGRESSION_DATA, 'small_threats_samp.csv'),
            u'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_habitat_quality_nodata_small(self):
        """Habitat Quality: on rasters that have nodata values."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'landuse_cur_uri': os.path.join(
                REGRESSION_DATA, 'small_lulc_base.tif'),
            'sensitivity_uri': os.path.join(
                REGRESSION_DATA, 'small_sensitivity_samp.csv'),
            'threat_raster_folder': os.path.join(REGRESSION_DATA),
            'threats_uri': os.path.join(
                REGRESSION_DATA, 'small_threats_samp.csv'),
            u'workspace_dir': self.workspace_dir,
        }

        habitat_quality.execute(args)
        HabitatQualityTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list_small_nodata.txt'),
            args['workspace_dir'])

        # reasonable to just check quality out in this case
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'small_quality_out_c.tif'),
            os.path.join(self.workspace_dir, 'output', 'quality_out_c.tif'),
            1e-6)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_habitat_quality_nodata_small_fut(self):
        """Habitat Quality: small test with future raster only."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'landuse_cur_uri': os.path.join(
                REGRESSION_DATA, 'small_lulc_base.tif'),
            'landuse_bas_uri': os.path.join(
                REGRESSION_DATA, 'small_lulc_base.tif'),
            'sensitivity_uri': os.path.join(
                REGRESSION_DATA, 'small_sensitivity_samp.csv'),
            'threat_raster_folder': os.path.join(REGRESSION_DATA),
            'threats_uri': os.path.join(
                REGRESSION_DATA, 'small_threats_samp.csv'),
            u'workspace_dir': self.workspace_dir,
        }

        habitat_quality.execute(args)
        HabitatQualityTests._test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list_small_nodata_fut.txt'),
            args['workspace_dir'])

        # reasonable to just check quality out in this case
        natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
            os.path.join(REGRESSION_DATA, 'small_quality_out_c.tif'),
            os.path.join(self.workspace_dir, 'output', 'quality_out_c.tif'),
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
