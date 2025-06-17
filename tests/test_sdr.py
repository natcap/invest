"""InVEST SDR model tests."""
import os
import shutil
import tempfile
import unittest

import numpy
import pygeoprocessing
from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'sdr')
SAMPLE_DATA = os.path.join(REGRESSION_DATA, 'input')


def assert_expected_results_in_vector(expected_results, vector_path):
    """Assert one feature vector maps to expected_results key/value pairs."""
    watershed_results_vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    watershed_results_layer = watershed_results_vector.GetLayer()
    watershed_results_feature = watershed_results_layer.GetFeature(0)
    actual_results = {}
    for key in expected_results:
        actual_results[key] = watershed_results_feature.GetField(key)
    watershed_results_vector = None
    watershed_results_layer = None
    watershed_results_feature = None
    incorrect_vals = {}
    for key in expected_results:
        # Using relative tolerance here because results with different
        # orders of magnitude are tested
        try:
            numpy.testing.assert_allclose(
                actual_results[key], expected_results[key],
                rtol=0.003, atol=0)
        except AssertionError:
            incorrect_vals[key] = (actual_results[key], expected_results[key])
    if incorrect_vals:
        raise AssertionError(
            f'these key (actual/expected) errors occured: {incorrect_vals}')


class SDRTests(unittest.TestCase):
    """Regression tests for InVEST SDR model."""

    def setUp(self):
        """Initialize SDRRegression tests."""
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
            'dem_path': os.path.join(SAMPLE_DATA, 'dem.tif'),
            'erodibility_path': os.path.join(
                SAMPLE_DATA, 'erodibility_SI_clip.tif'),
            'erosivity_path': os.path.join(SAMPLE_DATA, 'erosivity.tif'),
            'ic_0_param': '0.5',
            'k_param': '2',
            'lulc_path': os.path.join(SAMPLE_DATA, 'landuse_90.tif'),
            'sdr_max': '0.8',
            'l_max': '122',
            'threshold_flow_accumulation': '1000',
            'watersheds_path': os.path.join(SAMPLE_DATA, 'watersheds.shp'),
            'workspace_dir': workspace_dir,
            'n_workers': -1,
            'flow_dir_algorithm': 'MFD'
        }
        return args

    def test_sdr_validation(self):
        """SDR test regular validation."""
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(self.workspace_dir)
        args['drainage_path'] = os.path.join(
            REGRESSION_DATA, 'sample_drainage.tif')
        validate_result = sdr.validate(args, limit_to=None)
        self.assertFalse(
            validate_result,  # List should be empty if validation passes
            "expected no failed validations instead got %s" % str(
                validate_result))

    def test_sdr_validation_wrong_types(self):
        """SDR test validation for wrong GIS types."""
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(self.workspace_dir)
        # swap watershed and dem for different types
        args['dem_path'], args['watersheds_path'] = (
            args['watersheds_path'], args['dem_path'])
        validate_result = sdr.validate(args, limit_to=None)
        self.assertTrue(
            validate_result,
            "expected failed validations instead didn't get any")
        for (validation_keys, error_msg), phrase in zip(
                validate_result, ['GDAL raster', 'GDAL vector']):
            self.assertTrue(phrase in error_msg)

    def test_sdr_validation_key_no_value(self):
        """SDR test validation that's missing a value on a key."""
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)
        args['dem_path'] = ''
        validate_result = sdr.validate(args, limit_to=None)
        self.assertTrue(
            validate_result,
            'expected a validation error but didn\'t get one')

    def test_base_regression(self):
        """SDR base regression test on test data.

        Executes SDR with test data. Checks for accuracy of aggregate
        values in summary vector, presence of drainage raster in
        intermediate outputs, absence of negative (non-nodata) values
        in sed_deposition raster, and accuracy of raster outputs (as
        measured by the sum of their non-nodata pixel values).
        """
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(self.workspace_dir)

        sdr.execute(args)
        expected_watershed_totals = {
            'usle_tot': 2.62457418442,
            'sed_export': 0.09748090804,
            'sed_dep': 1.71672844887,
            'avoid_exp': 10199.46875,
            'avoid_eros': 274444.75,
        }

        vector_path = os.path.join(
            args['workspace_dir'], 'watershed_results_sdr.shp')
        assert_expected_results_in_vector(expected_watershed_totals,
                                          vector_path)

        # We only need to test that the drainage mask exists.  Functionality
        # for that raster is tested elsewhere
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    args['workspace_dir'], 'intermediate_outputs',
                    'what_drains_to_stream.tif')))

        # Check that sed_deposition does not have any negative, non-nodata
        # values, even if they are very small.
        sed_deposition_path = os.path.join(args['workspace_dir'],
                                           'sed_deposition.tif')
        sed_dep_nodata = pygeoprocessing.get_raster_info(
            sed_deposition_path)['nodata'][0]
        sed_dep_array = pygeoprocessing.raster_to_numpy_array(
            sed_deposition_path)
        negative_non_nodata_mask = (
            (~numpy.isclose(sed_dep_array, sed_dep_nodata)) &
            (sed_dep_array < 0))
        self.assertEqual(
            numpy.count_nonzero(sed_dep_array[negative_non_nodata_mask]), 0)
        
        # Check raster outputs to make sure values are in Mg/ha/yr.
        raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
        pixel_area = abs(numpy.prod(raster_info['pixel_size']))
        pixels_per_hectare = 10000 / pixel_area
        for (raster_name,
             attr_name) in [('usle.tif', 'usle_tot'),
                            ('sed_export.tif', 'sed_export'),
                            ('sed_deposition.tif', 'sed_dep'),
                            ('avoided_export.tif', 'avoid_exp'),
                            ('avoided_erosion.tif', 'avoid_eros')]:
            # Since pixel values are Mg/(ha•yr), raster sum is (Mg•px)/(ha•yr),
            # equal to the watershed total (Mg/yr) * (pixels_per_hectare px/ha).
            expected_sum = (expected_watershed_totals[attr_name]
                            * pixels_per_hectare)
            raster_path = os.path.join(args['workspace_dir'], raster_name)
            nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
            raster_sum = 0.0
            for _, block in pygeoprocessing.iterblocks((raster_path, 1)):
                raster_sum += numpy.sum(
                    block[~pygeoprocessing.array_equals_nodata(
                            block, nodata)], dtype=numpy.float64)
            numpy.testing.assert_allclose(raster_sum, expected_sum)

    def test_base_regression_d8(self):
        """SDR base regression test on sample data in D8 mode.

        Execute SDR with sample data and checks that the output files are
        generated and that the aggregate shapefile fields are the same as the
        regression case.
        """
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(self.workspace_dir)
        args['flow_dir_algorithm'] = 'D8'
        args['threshold_flow_accumulation'] = 100
        # make args explicit that this is a base run of SWY

        sdr.execute(args)
        expected_results = {
            'usle_tot': 2.520746,
            'sed_export': 0.187428,
            'sed_dep': 2.300645,
            'avoid_exp': 19283.767578,
            'avoid_eros': 263415,
        }

        vector_path = os.path.join(
            args['workspace_dir'], 'watershed_results_sdr.shp')
        assert_expected_results_in_vector(expected_results, vector_path)

        # We only need to test that the drainage mask exists.  Functionality
        # for that raster is tested elsewhere
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    args['workspace_dir'], 'intermediate_outputs',
                    'what_drains_to_stream.tif')))

        # Check that sed_deposition does not have any negative, non-nodata
        # values, even if they are very small.
        sed_deposition_path = os.path.join(args['workspace_dir'],
                                           'sed_deposition.tif')
        sed_dep_nodata = pygeoprocessing.get_raster_info(
            sed_deposition_path)['nodata'][0]
        sed_dep_array = pygeoprocessing.raster_to_numpy_array(
            sed_deposition_path)
        negative_non_nodata_mask = (
            (~numpy.isclose(sed_dep_array, sed_dep_nodata)) &
            (sed_dep_array < 0))
        self.assertEqual(
            numpy.count_nonzero(sed_dep_array[negative_non_nodata_mask]), 0)

    def test_regression_with_undefined_nodata(self):
        """SDR base regression test with undefined nodata values.

        Execute SDR with sample data with all rasters having undefined nodata
        values.
        """
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(self.workspace_dir)

        # set all input rasters to have undefined nodata values
        tmp_dir = os.path.join(args['workspace_dir'], 'nodata_raster_dir')
        os.makedirs(tmp_dir)
        for path_key in ['erodibility_path', 'erosivity_path', 'lulc_path']:
            target_path = os.path.join(
                tmp_dir, os.path.basename(args[path_key]))
            datatype = pygeoprocessing.get_raster_info(
                args[path_key])['datatype']
            pygeoprocessing.new_raster_from_base(
                args[path_key], target_path, datatype, [None])

            base_raster = gdal.OpenEx(args[path_key], gdal.OF_RASTER)
            base_band = base_raster.GetRasterBand(1)
            base_array = base_band.ReadAsArray()
            base_band = None
            base_raster = None

            target_raster = gdal.OpenEx(
                target_path, gdal.OF_RASTER | gdal.GA_Update)
            target_band = target_raster.GetRasterBand(1)
            target_band.WriteArray(base_array)

            target_band = None
            target_raster = None
            args[path_key] = target_path

        sdr.execute(args)
        expected_results = {
            'sed_export': 0.09748090804,
            'usle_tot': 2.62457418442,
            'avoid_exp': 10199.46875,
            'avoid_eros': 274444.75,
        }

        vector_path = os.path.join(
            args['workspace_dir'], 'watershed_results_sdr.shp')
        # make args explicit that this is a base run of SWY
        assert_expected_results_in_vector(expected_results, vector_path)

    def test_non_square_dem(self):
        """SDR non-square DEM pixels.

        Execute SDR with a non-square DEM and get a good result back.
        """
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(self.workspace_dir)
        args['dem_path'] = os.path.join(SAMPLE_DATA, 'dem_non_square.tif')
        # make args explicit that this is a base run of SWY
        sdr.execute(args)

        expected_results = {
            'sed_export': 0.08896198869,
            'usle_tot': 1.86480891705,
            'avoid_exp': 9203.955078125,
            'avoid_eros': 194212.28125,
        }

        vector_path = os.path.join(
            args['workspace_dir'], 'watershed_results_sdr.shp')
        assert_expected_results_in_vector(expected_results, vector_path)

    def test_drainage_regression(self):
        """SDR drainage layer regression test on sample data.

        Execute SDR with sample data and a drainage layer and checks that the
        output files are generated and that the aggregate shapefile fields
        are the same as the regression case.
        """
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(self.workspace_dir)
        args['drainage_path'] = os.path.join(
            REGRESSION_DATA, 'sample_drainage.tif')
        sdr.execute(args)

        expected_results = {
            'sed_export': 0.17336219549,
            'usle_tot': 2.56186032295,
            'avoid_exp': 17980.05859375,
            'avoid_eros': 267663.71875,
        }

        vector_path = os.path.join(
            args['workspace_dir'], 'watershed_results_sdr.shp')
        assert_expected_results_in_vector(expected_results, vector_path)

    def test_base_usle_c_too_large(self):
        """SDR test exepected exception for USLE_C > 1.0."""
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)
        args['biophysical_table_path'] = os.path.join(
            REGRESSION_DATA, 'biophysical_table_too_large.csv')

        with self.assertRaises(ValueError) as context:
            sdr.execute(args)
        self.assertIn(
            'A value in the biophysical table is not a number '
            'within range 0..1.', str(context.exception))

    def test_base_usle_p_nan(self):
        """SDR test expected exception for USLE_P not a number."""
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)
        args['biophysical_table_path'] = os.path.join(
            REGRESSION_DATA, 'biophysical_table_invalid_value.csv')

        with self.assertRaises(ValueError) as context:
            sdr.execute(args)
        self.assertIn(
            'could not be interpreted as RatioInput', str(context.exception))

    def test_lucode_not_a_number(self):
        """SDR test expected exception for invalid data in lucode column."""
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(
            self.workspace_dir)
        args['biophysical_table_path'] = os.path.join(
            self.workspace_dir, 'biophysical_table_invalid_lucode.csv')

        invalid_value = 'forest'
        with open(args['biophysical_table_path'], 'w') as file:
            file.write(
                f'desc,lucode,usle_p,usle_c\n'
                f'0,{invalid_value},0.5,0.5\n')

        with self.assertRaises(ValueError) as context:
            sdr.execute(args)
        self.assertIn(
            'could not be interpreted as IntegerInput', str(context.exception))

    def test_missing_lulc_value(self):
        """SDR test for ValueError when LULC value not found in table."""
        import pandas
        from natcap.invest.sdr import sdr

        # use predefined directory so test can clean up files during teardown
        args = SDRTests.generate_base_args(self.workspace_dir)

        # remove a row from the biophysical table so that lulc value is missing
        bad_biophysical_path = os.path.join(
            self.workspace_dir, 'bad_biophysical_table.csv')

        bio_df = pandas.read_csv(args['biophysical_table_path'])
        bio_df = bio_df[bio_df['lucode'] != 2]
        bio_df.to_csv(bad_biophysical_path)
        bio_df = None
        args['biophysical_table_path'] = bad_biophysical_path

        with self.assertRaises(ValueError) as context:
            sdr.execute(args)
        self.assertIn(
            "The missing values found in the LULC raster but not the table"
            " are: [2.]", str(context.exception))

    def test_what_drains_to_stream(self):
        """SDR test for what pixels drain to a stream."""
        from natcap.invest.sdr import sdr

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26910)  # NAD83 / UTM zone 11N
        srs_wkt = srs.ExportToWkt()
        origin = (463250, 4929700)
        pixel_size = (30, -30)

        flow_dir_mfd = numpy.array([
            [0, 1],
            [1, 1]], dtype=numpy.float64)
        flow_dir_mfd_nodata = 0  # Matches pygeoprocessing output
        flow_dir_mfd_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        pygeoprocessing.numpy_array_to_raster(
            flow_dir_mfd, flow_dir_mfd_nodata, pixel_size, origin, srs_wkt,
            flow_dir_mfd_path)

        dist_to_channel = numpy.array([
            [10, 5],
            [-1, 6]], dtype=numpy.float64)
        dist_to_channel_nodata = -1  # Matches pygeoprocessing output
        dist_to_channel_path = os.path.join(
            self.workspace_dir, 'dist_to_channel.tif')
        pygeoprocessing.numpy_array_to_raster(
            dist_to_channel, dist_to_channel_nodata, pixel_size, origin,
            srs_wkt, dist_to_channel_path)

        target_what_drains_path = os.path.join(
            self.workspace_dir, 'what_drains.tif')
        sdr._calculate_what_drains_to_stream(
            flow_dir_mfd_path, dist_to_channel_path, target_what_drains_path)

        # 255 is the byte nodata value assigned
        expected_drainage = numpy.array([
            [255, 1],
            [0, 1]], dtype=numpy.uint8)
        what_drains = pygeoprocessing.raster_to_numpy_array(
            target_what_drains_path)
        numpy.testing.assert_allclose(what_drains, expected_drainage)

    def test_ls_factor(self):
        """SDR test for our LS Factor function."""
        from natcap.invest.sdr import sdr

        nodata = -1

        # These varying percent slope values should cover all of the slope
        # factor and slope table cases.
        pct_slope_array = numpy.array(
            [[1.5, 4, 8, 10, 15, nodata]], dtype=numpy.float32)
        flow_accum_array = numpy.array(
            [[100, 100, 100, 100, 10000000, nodata]], dtype=numpy.float32)
        l_max = 25  # affects the last item in the array only

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26910)  # NAD83 / UTM zone 11N
        srs_wkt = srs.ExportToWkt()
        origin = (463250, 4929700)
        pixel_size = (30, -30)

        pct_slope_path = os.path.join(self.workspace_dir, 'pct_slope.tif')
        pygeoprocessing.numpy_array_to_raster(
            pct_slope_array, nodata, pixel_size, origin, srs_wkt,
            pct_slope_path)

        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')
        pygeoprocessing.numpy_array_to_raster(
            flow_accum_array, nodata, pixel_size, origin, srs_wkt,
            flow_accum_path)

        target_ls_factor_path = os.path.join(self.workspace_dir, 'ls.tif')
        sdr._calculate_ls_factor(flow_accum_path, pct_slope_path, l_max,
                                 target_ls_factor_path)

        ls = pygeoprocessing.raster_to_numpy_array(target_ls_factor_path)
        nodata = float(numpy.finfo(numpy.float32).max)
        expected_ls = numpy.array(
            [[0.253996, 0.657229, 1.345856, 1.776729, 49.802994, nodata]],
            dtype=numpy.float32)
        numpy.testing.assert_allclose(ls, expected_ls, rtol=1e-6)
