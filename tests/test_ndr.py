"""InVEST NDR model tests."""
import collections
import os
import shutil
import tempfile
import unittest

import numpy
import pygeoprocessing
from osgeo import gdal, ogr

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'ndr')


class NDRTests(unittest.TestCase):
    """Regression tests for InVEST SDR model."""

    def setUp(self):
        """Initalize SDRRegression tests."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate a base sample args dict for NDR."""
        args = {
            'biophysical_table_path':
            os.path.join(REGRESSION_DATA, 'input', 'biophysical_table.csv'),
            'calc_n': True,
            'calc_p': True,
            'dem_path': os.path.join(REGRESSION_DATA, 'input', 'dem.tif'),
            'k_param': 2.0,
            'lulc_path':
            os.path.join(REGRESSION_DATA, 'input', 'landuse_90.tif'),
            'runoff_proxy_path':
            os.path.join(REGRESSION_DATA, 'input', 'precip.tif'),
            'subsurface_critical_length_n': 150,
            'subsurface_eff_n': 0.4,
            'threshold_flow_accumulation': '1000',
            'watersheds_path':
            os.path.join(REGRESSION_DATA, 'input', 'watersheds.shp'),
            'workspace_dir': workspace_dir,
        }
        return args.copy()

    def test_normalize_raster_float64(self):
        """NDR _normalize_raster handle float64.

        Regression test for an issue raised on the forums when normalizing a
        Float64 raster that has a nodata value that exceeds Float32 space.  The
        output raster, in the buggy version, would have pixel values of -inf
        where they should have been nodata.

        https://community.naturalcapitalproject.org/t/ndr-null-values-in-watershed-results/914
        """
        from natcap.invest.ndr import ndr

        raster_xsize = 1124
        raster_ysize = 512
        float64_raster_path = os.path.join(
            self.workspace_dir, 'float64_raster.tif')
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create(
            float64_raster_path, raster_xsize, raster_ysize, 1,
            gdal.GDT_Float64)
        source_nodata = -1.797693e+308  # taken from user's data
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(source_nodata)
        source_array = numpy.empty(
            (raster_ysize, raster_xsize), dtype=numpy.float64)
        source_array[0:256][:] = 5.5  # Something, anything.
        source_array[256:][:] = source_nodata
        band.WriteArray(source_array)
        band = None
        raster = None
        driver = None

        normalized_raster_path = os.path.join(
            self.workspace_dir, 'normalized.tif')
        ndr._normalize_raster((float64_raster_path, 1), normalized_raster_path)

        normalized_raster_nodata = pygeoprocessing.get_raster_info(
            normalized_raster_path)['nodata'][0]

        normalized_array = gdal.OpenEx(normalized_raster_path).ReadAsArray()
        expected_array = numpy.empty(
            (raster_ysize, raster_xsize), dtype=numpy.float32)
        expected_array[0:256][:] = 1.
        expected_array[256:][:] = normalized_raster_nodata

        # Assert that the output values match the target nodata value
        self.assertEqual(
            287744,  # Nodata pixels
            numpy.count_nonzero(
                numpy.isclose(normalized_array, normalized_raster_nodata)))

        numpy.testing.assert_allclose(
            normalized_array, expected_array, rtol=0, atol=1e-6)

    def test_missing_headers(self):
        """NDR biphysical headers missing should raise a ValueError."""
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        # make args explicit that this is a base run of SWY
        args['biophysical_table_path'] = os.path.join(
            REGRESSION_DATA, 'input', 'biophysical_table_missing_headers.csv')
        with self.assertRaises(ValueError):
            ndr.execute(args)

    def test_crit_len_0(self):
        """NDR test case where crit len is 0 in biophysical table."""
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        new_table_path = os.path.join(self.workspace_dir, 'table_c_len_0.csv')
        with open(new_table_path, 'w') as target_file:
            with open(args['biophysical_table_path'], 'r') as table_file:
                target_file.write(table_file.readline())
                while True:
                    line = table_file.readline()
                    if not line:
                        break
                    line_list = line.split(',')
                    # replace the crit_len_p with 0 in this column
                    line = (
                        ','.join(line_list[0:12] + ['0.0'] + line_list[13::]))
                    target_file.write(line)

        args['biophysical_table_path'] = new_table_path
        ndr.execute(args)

        result_vector = ogr.Open(
            os.path.join(args['workspace_dir'], 'watershed_results_ndr.gpkg'))
        result_layer = result_vector.GetLayer()
        error_results = {}

        feature = result_layer.GetFeature(1)
        if not feature:
            raise AssertionError("No features were output.")
        for field, value in [
                ('p_surface_load', 41.921),
                ('p_surface_export', 5.59887886),
                ('n_surface_load', 2978.520),
                ('n_subsurface_load', 28.614),
                ('n_surface_export', 289.0498),
                ('n_subsurface_load', 28.614094),
                ('n_total_export', 304.66061401)]:
            if not numpy.isclose(feature.GetField(field), value, atol=1e-2):
                error_results[field] = (
                    'field', feature.GetField(field), value)
        ogr.Feature.__swig_destroy__(feature)
        feature = None
        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None

        if error_results:
            raise AssertionError(
                "The following values are not equal: %s" % error_results)

    def test_missing_lucode(self):
        """NDR missing lucode in biophysical table should raise a KeyError."""
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        # make args explicit that this is a base run of SWY
        args['biophysical_table_path'] = os.path.join(
            REGRESSION_DATA, 'input', 'biophysical_table_missing_lucode.csv')
        with self.assertRaises(KeyError) as cm:
            ndr.execute(args)
        actual_message = str(cm.exception)
        self.assertTrue(
            'present in the landuse raster but missing from the biophysical'
            in actual_message)

    def test_no_nutrient_selected(self):
        """NDR no nutrient selected should raise a ValueError."""
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        # make args explicit that this is a base run of SWY
        args['calc_n'] = False
        args['calc_p'] = False
        with self.assertRaises(ValueError):
            ndr.execute(args)

    def test_base_regression(self):
        """NDR base regression test on sample data.

        Execute NDR with sample data and checks that the output files are
        generated and that the aggregate shapefile fields are the same as the
        regression case.
        """
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        # make an empty output shapefile on top of where the new output
        # shapefile should reside to ensure the model overwrites it
        with open(
                os.path.join(self.workspace_dir, 'watershed_results_ndr.gpkg'),
                'wb') as f:
            f.write(b'')

        # make args explicit that this is a base run of SWY
        ndr.execute(args)

        result_vector = ogr.Open(os.path.join(
            args['workspace_dir'], 'watershed_results_ndr.gpkg'))
        result_layer = result_vector.GetLayer()
        result_feature = result_layer.GetFeature(1)
        result_layer = None
        result_vector = None
        mismatch_list = []
        # these values were generated by manual inspection of regression
        # results
        for field, expected_value in [
                ('p_surface_load', 41.921860),
                ('p_surface_export', 5.899117),
                ('n_surface_load', 2978.519775),
                ('n_surface_export', 289.0498),
                ('n_subsurface_load', 28.614094),
                ('n_subsurface_export', 15.61077),
                ('n_total_export', 304.660614)]:
            val = result_feature.GetField(field)
            if not numpy.isclose(val, expected_value):
                mismatch_list.append(
                    (field, 'expected: %f' % expected_value,
                     'actual: %f' % val))
        result_feature = None
        if mismatch_list:
            raise RuntimeError("results not expected: %s" % mismatch_list)

        # We only need to test that the drainage mask exists.  Functionality
        # for that raster is tested in SDR.
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    args['workspace_dir'], 'intermediate_outputs',
                    'what_drains_to_stream.tif')))

    def test_validation(self):
        """NDR test argument validation."""
        from natcap.invest import validation
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        # should not raise an exception
        validation_errors = ndr.validate(args)
        self.assertEqual(len(validation_errors), 0)

        del args['workspace_dir']
        validation_errors = ndr.validate(args)
        self.assertEqual(len(validation_errors), 1)

        args = NDRTests.generate_base_args(self.workspace_dir)
        args['workspace_dir'] = ''
        validation_error_list = ndr.validate(args)
        # we should have one warning that is an empty value
        self.assertEqual(len(validation_error_list), 1)

        # here the wrong GDAL type happens (vector instead of raster)
        args = NDRTests.generate_base_args(self.workspace_dir)
        args['lulc_path'] = args['watersheds_path']
        validation_error_list = ndr.validate(args)
        # we should have one warning that is an empty value
        self.assertEqual(len(validation_error_list), 1)

        # here the wrong GDAL type happens (raster instead of vector)
        args = NDRTests.generate_base_args(self.workspace_dir)
        args['watersheds_path'] = args['lulc_path']
        validation_error_list = ndr.validate(args)
        # we should have one warning that is an empty value
        self.assertEqual(len(validation_error_list), 1)

        # cover that there's no p and n calculation
        args = NDRTests.generate_base_args(self.workspace_dir)
        args['calc_p'] = False
        args['calc_n'] = False
        validation_error_list = ndr.validate(args)
        # we should have one warning that is an empty value
        self.assertEqual(len(validation_error_list), 1)
        self.assertTrue('calc_n' in validation_error_list[0][0] and
                        'calc_p' in validation_error_list[0][0])

        # cover that a file is missing
        args = NDRTests.generate_base_args(self.workspace_dir)
        args['lulc_path'] = 'this/path/does/not/exist.tif'
        validation_error_list = ndr.validate(args)
        # we should have one warning that is an empty value
        self.assertEqual(len(validation_error_list), 1)

        # cover that some args are conditionally required when
        # these args are present and true
        args = {'calc_p': True, 'calc_n': True}
        validation_error_list = ndr.validate(args)
        invalid_args = validation.get_invalid_keys(validation_error_list)
        expected_missing_args = [
            'biophysical_table_path',
            'threshold_flow_accumulation',
            'dem_path',
            'subsurface_critical_length_n',
            'runoff_proxy_path',
            'lulc_path',
            'workspace_dir',
            'k_param',
            'watersheds_path',
            'subsurface_eff_n',
        ]
        self.assertEqual(set(invalid_args), set(expected_missing_args))
