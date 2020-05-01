"""InVEST NDR model tests."""
import collections
import os
import shutil
import tempfile
import unittest

import numpy
import pygeoprocessing
from osgeo import gdal
from osgeo import ogr

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
            'subsurface_critical_length_p': '150',
            'subsurface_eff_n': 0.4,
            'subsurface_eff_p': '0.8',
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

        float64_raster_path = os.path.join(self.workspace_dir,
                                           'float64_raster.tif')
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create(float64_raster_path,
                               4,  # xsize
                               4,  # ysize
                               1,  # n_bands
                               gdal.GDT_Float64)
        source_nodata = -1.797693e+308  # taken from user's data
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(source_nodata)
        source_array = numpy.empty((4, 4), dtype=numpy.float64)
        source_array[0:2][:] = 5.5  # Something, anything.
        source_array[2:][:] = source_nodata
        band.WriteArray(source_array)
        band = None
        raster = None
        driver = None

        normalized_raster_path = os.path.join(self.workspace_dir,
                                              'normalized.tif')
        ndr._normalize_raster((float64_raster_path, 1), normalized_raster_path)

        normalized_raster_nodata = pygeoprocessing.get_raster_info(
            normalized_raster_path)['nodata'][0]

        normalized_array = gdal.OpenEx(normalized_raster_path).ReadAsArray()
        expected_array = numpy.array(
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [normalized_raster_nodata]*4,
             [normalized_raster_nodata]*4], dtype=numpy.float32)

        # Assert that the output values match the target nodata value
        self.assertEqual(
            8,  # Nodata pixels
            numpy.count_nonzero(numpy.isclose(normalized_array,
                                              normalized_raster_nodata)))

        numpy.testing.assert_array_almost_equal(
            normalized_array, expected_array)

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
                    line = ','.join(line_list[0:12] + ['0.0'] + line_list[13::])
                    target_file.write(line)

        args['biophysical_table_path'] = new_table_path
        ndr.execute(args)

        result_vector = ogr.Open(
            os.path.join(args['workspace_dir'], 'watershed_results_ndr.shp'))
        result_layer = result_vector.GetLayer()
        error_results = {}

        surf_p_ld = 41.921
        sub_p_ld = 0
        p_exp_tot = 7.666
        surf_n_ld = 2978.520
        sub_n_ld = 28.614
        n_exp_tot = 339.839
        feature = result_layer.GetFeature(0)
        if not feature:
            raise AssertionError("No features were output.")
        for field, value in [
                ('surf_p_ld', surf_p_ld),
                ('sub_p_ld', sub_p_ld),
                ('p_exp_tot', p_exp_tot),
                ('surf_n_ld', surf_n_ld),
                ('sub_n_ld', sub_n_ld),
                ('n_exp_tot', n_exp_tot)]:
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
                os.path.join(self.workspace_dir, 'watershed_results_ndr.shp'),
                'wb') as f:
            f.write(b'')

        # make args explicit that this is a base run of SWY
        ndr.execute(args)

        result_vector = ogr.Open(os.path.join(
            args['workspace_dir'], 'watershed_results_ndr.shp'))
        result_layer = result_vector.GetLayer()
        result_feature = result_layer.GetFeature(0)
        result_layer = None
        result_vector = None
        mismatch_list = []
        # these values were generated by manual inspection of regressino results
        for field, expected_value in [
                ('surf_p_ld', 41.921860),
                ('p_exp_tot', 8.598053),
                ('surf_n_ld', 2978.519775),
                ('sub_n_ld', 28.614094),
                ('n_exp_tot', 339.839386)]:
            val = result_feature.GetField(field)
            if not numpy.isclose(val, expected_value):
                mismatch_list.append(
                    (field, 'expected: %f' % expected_value, 'actual: %f' % val))
        result_feature = None
        if mismatch_list:
            raise RuntimeError("results not expected: %s" % mismatch_list)

    def test_validation(self):
        """NDR test argument validation."""
        from natcap.invest.ndr import ndr
        from natcap.invest import validation

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        # should not raise an exception
        ndr.validate(args)

        del args['workspace_dir']
        validation_errors = ndr.validate(args)
        self.assertEquals(len(validation_errors), 1)

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
            'subsurface_critical_length_p',
            'runoff_proxy_path',
            'lulc_path',
            'workspace_dir',
            'k_param',
            'watersheds_path',
            'subsurface_eff_p',
            'subsurface_eff_n',
        ]
        self.assertEqual(set(invalid_args), set(expected_missing_args))

    @staticmethod
    def _assert_regression_results_equal(
            workspace_dir, result_vector_path, agg_results_path):
        """Test workspace state against expected aggregate results.

        Parameters:
            workspace_dir (string): path to the completed model workspace
            result_vector_path (string): path to the summary shapefile
                produced by the SWY model.
            agg_results_path (string): path to a csv file that has the
                expected aggregated_results.shp table in the form of
                fid,p_load_tot,p_exp_tot,n_load_tot,n_exp_tot per line

        Returns:
            None

        Raises:
            AssertionError if any files are missing or results are out of
            range by `tolerance_places`
        """
        # we expect a file called 'aggregated_results.shp'
        result_vector = ogr.Open(result_vector_path)
        result_layer = result_vector.GetLayer()

        error_results = collections.defaultdict(dict)
        with open(agg_results_path, 'r') as agg_result_file:
            for line in agg_result_file:
                (fid, surf_p_ld, sub_p_ld, p_exp_tot,
                 surf_n_ld, sub_n_ld, n_exp_tot) = [
                    float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                if not feature:
                    raise AssertionError("The fid %s is missing." % fid)
                for field, value in [
                                    ('ws_id', fid),
                                    ('surf_p_ld', surf_p_ld),
                                    ('sub_p_ld', sub_p_ld),
                                    ('p_exp_tot', p_exp_tot),
                                    ('surf_n_ld', surf_n_ld),
                                    ('sub_n_ld', sub_n_ld),
                                    ('n_exp_tot', n_exp_tot)]:
                    if not numpy.isclose(feature.GetField(field), value):
                        error_results[fid][field] = (
                            feature.GetField(field), value)
                ogr.Feature.__swig_destroy__(feature)
                feature = None
        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None
        if error_results:
            raise AssertionError(
                "The following values are not equal: %s" % error_results)
