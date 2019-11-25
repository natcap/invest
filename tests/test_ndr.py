"""InVEST NDR model tests."""
import collections
import unittest
import tempfile
import shutil
import os

import numpy
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

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        # should not raise an exception
        ndr.validate(args)

        with self.assertRaises(KeyError) as context:
            del args['workspace_dir']
            ndr.validate(args)
        self.assertEquals(len(context.exception.args), 1)

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

        # cover that a file is missing
        args = NDRTests.generate_base_args(self.workspace_dir)
        args['lulc_path'] = 'this/path/does/not/exist.tif'
        validation_error_list = ndr.validate(args)
        # we should have one warning that is an empty value
        self.assertEqual(len(validation_error_list), 1)
