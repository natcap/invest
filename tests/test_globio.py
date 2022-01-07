"""Module for Regression Testing the InVEST GLOBIO model."""
import unittest
import tempfile
import shutil
import os

import pygeoprocessing
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

    Args:
        workspace_dir (string): path to workspace for making the file
        file_name (string): file path name
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

        model_array = pygeoprocessing.raster_to_numpy_array(
            os.path.join(args['workspace_dir'], 'msa.tif'))
        reg_array = pygeoprocessing.raster_to_numpy_array(
            os.path.join(REGRESSION_DATA, 'msa_lulc_regression.tif'))
        numpy.testing.assert_allclose(model_array, reg_array)

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

        model_array = pygeoprocessing.raster_to_numpy_array(
            os.path.join(args['workspace_dir'], 'msa.tif'))
        reg_array = pygeoprocessing.raster_to_numpy_array(
            os.path.join(REGRESSION_DATA, 'msa_shape_infra_regression.tif'))
        numpy.testing.assert_allclose(model_array, reg_array)

    def test_globio_single_infra(self):
        """GLOBIO: regression testing with single infrastructure raster."""
        from natcap.invest import globio

        # Use the projection and geostransform from sample data to build test
        roads_path = os.path.join(
            SAMPLE_DATA, 'infrastructure_dir', 'roads.tif')
        base_raster = gdal.OpenEx(roads_path, gdal.OF_RASTER)
        projection_wkt = base_raster.GetProjection()
        base_geotransform = base_raster.GetGeoTransform()
        base_raster = None

        # Create a temporary infrastructure directory with one raster
        tmp_infra_dir = os.path.join(self.workspace_dir, "single_infra")
        os.mkdir(tmp_infra_dir)
        tmp_roads_path = os.path.join(tmp_infra_dir, "roads.tif")

        tmp_roads_array = numpy.array([
            [0, 0, 0, 0], [0.5, 1, 1, 13.0], [1, 0, 1, 13.0], [1, 1, 0, 0]])
        tmp_roads_nodata = 13.0
        raster_driver = gdal.GetDriverByName('GTiff')
        ny, nx = tmp_roads_array.shape
        new_raster = raster_driver.Create(
            tmp_roads_path, nx, ny, 1, gdal.GDT_Float32)
        new_raster.SetProjection(projection_wkt)
        new_raster.SetGeoTransform(
            [base_geotransform[0], 10, 0.0, base_geotransform[3], 0.0, -10])
        new_band = new_raster.GetRasterBand(1)
        new_band.SetNoDataValue(tmp_roads_nodata)
        new_band.WriteArray(tmp_roads_array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        temp_dir = os.path.join(self.workspace_dir, "tmp_dir")
        os.mkdir(temp_dir)

        result_path = os.path.join(
            self.workspace_dir, 'combined_infrastructure.tif')

        # No need to run the whole model so call infrastructure combining
        # function directly
        globio._collapse_infrastructure_layers(
            tmp_infra_dir, tmp_roads_path, result_path, temp_dir)

        expected_result = numpy.array([
            [0, 0, 0, 0], [1, 1, 1, 255], [1, 0, 1, 255], [1, 1, 0, 0]])

        result_raster = gdal.OpenEx(result_path, gdal.OF_RASTER)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()

        numpy.testing.assert_allclose(result_array, expected_result)

        result_band = None
        result_raster = None

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

    def test_globio_missing_lulc_value(self):
        """GLOBIO: test error is raised when missing LULC value from table.

        This test is when an LULC value is not represented in the Land Cover
        to GLOBIO Land Cover table.
        """
        from natcap.invest import globio
        import pandas

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

        # remove a row from the lulc table so that lulc value is missing
        bad_lulc_to_globio_path = os.path.join(
            self.workspace_dir, 'bad_lulc_to_globio_table_path.csv')

        table_df = pandas.read_csv(args['lulc_to_globio_table_path'])
        table_df = table_df.loc[table_df['lucode'] != 2]
        table_df.to_csv(bad_lulc_to_globio_path)
        table_df = None

        args['lulc_to_globio_table_path'] = bad_lulc_to_globio_path
        with self.assertRaises(ValueError) as context:
            globio.execute(args)
        # Note the '2.' here, since the lulc_2008.tif is a Float32
        self.assertTrue(
            "The missing values found in the LULC raster but not the table"
            " are: [2.]" in str(context.exception))

    def test_globio_missing_globio_lulc_value(self):
        """GLOBIO: test error raised when missing GLOBIO LULC value from table.

        This test is when a GLOBIO LULC value is not represented in the
        MSA parameter table under the msa_lu rows.
        """
        from natcap.invest import globio
        import pandas

        args = {
            'aoi_path': os.path.join(SAMPLE_DATA, 'sub_aoi.shp'),
            'globio_lulc_path': os.path.join(
                SAMPLE_DATA, 'globio_lulc_small.tif'),
            'infrastructure_dir': os.path.join(
                SAMPLE_DATA, 'infrastructure_dir'),
            'intensification_fraction': '0.46',
            'lulc_to_globio_table_path': os.path.join(
                SAMPLE_DATA, 'lulc_conversion_table.csv'),
            'msa_parameters_path': os.path.join(
                SAMPLE_DATA, 'msa_parameters.csv'),
            'pasture_threshold': '0.5',
            'pasture_path': os.path.join(SAMPLE_DATA, 'pasture.tif'),
            'potential_vegetation_path': os.path.join(
                SAMPLE_DATA, 'potential_vegetation.tif'),
            'predefined_globio': True,
            'primary_threshold': 0.66,
            'workspace_dir': self.workspace_dir,
            'n_workers': '-1',
        }

        # remove a row from the msa table so that a msa_lu value is missing
        bad_msa_param_path = os.path.join(
            self.workspace_dir, 'bad_msa_param_table_path.csv')

        table_df = pandas.read_csv(args['msa_parameters_path'])
        # Using '3' here because Value column is of mix type and will be string
        table_df = table_df.loc[table_df['Value'] != '3']
        table_df.to_csv(bad_msa_param_path)
        table_df = None

        args['msa_parameters_path'] = bad_msa_param_path
        with self.assertRaises(ValueError) as context:
            globio.execute(args)
        self.assertTrue(
            "The missing values found in the GLOBIO LULC raster but not the"
            " table are: [3]" in str(context.exception))

    @staticmethod
    def _test_same_files(base_list_path, directory_path):
        """Assert files in `base_list_path` are in `directory_path`.

        Args:
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

        Args:
            result_vector_path (string): path to the summary shapefile
                produced by GLOBIO model
            agg_results_path (string): path to a csv file that has the
                expected aoi_summary.shp table in the form of
                fid,msa_mean per line

        Returns:
            None

        Raises:
            AssertionError if results are out of range by ``tolerance_places``
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
                    # The coefficient of 1.5 here derives from when
                    # `assert_almost_equal` was used, which had parameter
                    # `decimal`. In the numpy implementation, this meant an
                    # absolute tolerance of 1.5 * 10**-decimal.
                    # In other places we were able to round 1.5 down to 1,
                    # but here the slightly larger tolerance is needed.
                    numpy.testing.assert_allclose(
                        result_value,
                        float(expected_results[fid]['msa_mean']),
                        rtol=0, atol=1.5 * 10**-tolerance_places)
                else:
                    # the out-of-bounds polygon will have no result_value
                    assert(expected_results[fid]['msa_mean'] == '')
        finally:
            feature = None
            result_layer = None
            gdal.Dataset.__swig_destroy__(result_vector)
            result_vector = None


class GlobioValidationTests(unittest.TestCase):
    """Tests for the GLOBIO Model ARGS_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'primary_threshold',
            'pasture_path',
            'pasture_threshold',
            'lulc_path',
            'potential_vegetation_path',
            'msa_parameters_path',
            'lulc_to_globio_table_path',
            'workspace_dir',
            'intensification_fraction',
            'infrastructure_dir',
        ]

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys(self):
        """GLOBIO Validate: assert missing required keys."""
        from natcap.invest import globio
        from natcap.invest import validation

        validation_errors = globio.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_predefined_globio(self):
        """GLOBIO Validate: assert missing req. keys w/ predifined GLOBIO."""
        from natcap.invest import globio
        from natcap.invest import validation

        validation_errors = globio.validate({'predefined_globio': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            ['workspace_dir',
             'infrastructure_dir',
             'intensification_fraction',
             'msa_parameters_path',
             'globio_lulc_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_field_in_msa_parameters(self):
        """GLOBIO Validate: warning message on invalid fields."""
        from natcap.invest import globio, validation
        msa_parameters_path = os.path.join(self.workspace_dir, 'bad_table.csv')
        with open(msa_parameters_path, 'w') as file:
            file.write('foo,bar\n')
            file.write('1,2\n')
        validation_warnings = globio.validate(
            {'msa_parameters_path': msa_parameters_path})
        expected_warning = (
            ['msa_parameters_path'],
            validation.MESSAGES['MATCHED_NO_HEADERS'].format(
                header='column', header_name='msa_type'))
        self.assertTrue(expected_warning in validation_warnings)
