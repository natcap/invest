"""InVEST NDR model tests."""
import os
import shutil
import tempfile
import unittest

import numpy
import pandas
import pygeoprocessing
import shapely.geometry
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

gdal.UseExceptions()
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
            'flow_dir_algorithm': 'MFD'
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
        """NDR biophysical headers missing should return validation message."""
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        args['biophysical_table_path'] = os.path.join(
            REGRESSION_DATA, 'input', 'biophysical_table_missing_headers.csv')
        validation_messages = ndr.validate(args)
        self.assertEqual(len(validation_messages), 1)

    def test_crit_len_0(self):
        """NDR test case where crit len is 0 in biophysical table."""
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        new_table_path = os.path.join(self.workspace_dir, 'table_c_len_0.csv')

        bio_df = pandas.read_csv(args['biophysical_table_path'])
        # replace the crit_len_p with 0 in this column
        bio_df['crit_len_p'] = 0
        bio_df.to_csv(new_table_path)
        bio_df = None

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
                ('p_surface_load', 41.826904),
                ('p_surface_export', 5.566120),
                ('n_surface_load', 2977.551270),
                ('n_surface_export', 274.062129),
                ('n_subsurface_load', 28.558048),
                ('n_subsurface_export', 15.578484),
                ('n_total_export', 289.640609)]:
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
        args['biophysical_table_path'] = os.path.join(
            REGRESSION_DATA, 'input', 'biophysical_table_missing_lucode.csv')
        with self.assertRaises(KeyError) as cm:
            ndr.execute(args)
        actual_message = str(cm.exception)
        self.assertTrue(
            'present in the landuse raster but missing from the biophysical'
            in actual_message)

    def test_no_nutrient_selected(self):
        """NDR no nutrient selected should return a validation message."""
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        args['calc_n'] = False
        args['calc_p'] = False
        validation_messages = ndr.validate(args)
        self.assertEqual(len(validation_messages), 1)

    def test_base_regression(self):
        """NDR base regression test on test data.

        Executes NDR with test data. Checks for accuracy of aggregate
        values in summary vector, presence of drainage raster in
        intermediate outputs, and accuracy of raster outputs (as
        measured by the sum of their non-nodata pixel values).
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
        expected_watershed_totals = {
            'p_surface_load': 41.826904,
            'p_surface_export': 5.866880,
            'n_surface_load': 2977.551270,
            'n_surface_export': 274.062129,
            'n_subsurface_load': 28.558048,
            'n_subsurface_export': 15.578484,
            'n_total_export': 289.640609
        }

        for field in expected_watershed_totals:
            expected_value = expected_watershed_totals[field]
            val = result_feature.GetField(field)
            if not numpy.isclose(val, expected_value):
                mismatch_list.append(
                    (field, 'expected: %f' % expected_value,
                     'actual: %f' % val))
        result_feature = None
        if mismatch_list:
            raise AssertionError("results not expected: %s" % mismatch_list)

        # We only need to test that the drainage mask exists.  Functionality
        # for that raster is tested in SDR.
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    args['workspace_dir'], 'intermediate_outputs',
                    'what_drains_to_stream.tif')))
        
        # Check raster outputs to make sure values are in kg/ha/yr.
        raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
        pixel_area = abs(numpy.prod(raster_info['pixel_size']))
        pixels_per_hectare = 10000 / pixel_area
        for attr_name in ['p_surface_export',
                          'n_surface_export',
                          'n_subsurface_export',
                          'n_total_export']:
            # Since pixel values are kg/(ha•yr), raster sum is (kg•px)/(ha•yr),
            # equal to the watershed total (kg/yr) * (pixels_per_hectare px/ha).
            expected_sum = (expected_watershed_totals[attr_name]
                            * pixels_per_hectare)
            raster_name = attr_name + '.tif'
            raster_path = os.path.join(args['workspace_dir'], raster_name)
            nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
            raster_sum = 0.0
            for _, block in pygeoprocessing.iterblocks((raster_path, 1)):
                raster_sum += numpy.sum(
                    block[~pygeoprocessing.array_equals_nodata(
                            block, nodata)], dtype=numpy.float64)
            numpy.testing.assert_allclose(raster_sum, expected_sum, rtol=1e-6)

    def test_base_regression_d8(self):
        """NDR base regression test on sample data in D8 mode.

        Execute NDR with sample data and checks that the output files are
        generated and that the aggregate shapefile fields are the same as the
        regression case.
        """
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)
        args['flow_dir_algorithm'] = 'D8'
        # make an empty output shapefile on top of where the new output
        # shapefile should reside to ensure the model overwrites it
        with open(
                os.path.join(self.workspace_dir, 'watershed_results_ndr.gpkg'),
                'wb') as f:
            f.write(b'')
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
                ('p_surface_load', 41.826904),
                ('p_surface_export', 5.279964),
                ('n_surface_load', 2977.551914),
                ('n_surface_export', 318.641924),
                ('n_subsurface_load', 28.558048),
                ('n_subsurface_export', 12.609187),
                ('n_total_export', 330.571134)]:
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

    def test_regression_undefined_nodata(self):
        """NDR test when DEM, LULC and runoff proxy have undefined nodata."""
        from natcap.invest.ndr import ndr

        # use predefined directory so test can clean up files during teardown
        args = NDRTests.generate_base_args(self.workspace_dir)

        # unset nodata values for DEM, LULC, and runoff proxy
        # this is ok because the test data is 100% valid
        # regression test for https://github.com/natcap/invest/issues/1005
        for key in ['runoff_proxy_path', 'dem_path', 'lulc_path']:
            target_path = os.path.join(self.workspace_dir, f'{key}_no_nodata.tif')
            source = gdal.OpenEx(args[key], gdal.OF_RASTER)
            driver = gdal.GetDriverByName('GTIFF')
            target = driver.CreateCopy(target_path, source)
            target.GetRasterBand(1).DeleteNoDataValue()
            source, target = None, None
            args[key] = target_path

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
                ('p_surface_load', 41.826904),
                ('p_surface_export', 5.866880),
                ('n_surface_load', 2977.551270),
                ('n_surface_export', 274.062129),
                ('n_subsurface_load', 28.558048),
                ('n_subsurface_export', 15.578484),
                ('n_total_export', 289.640609)]:
            val = result_feature.GetField(field)
            if not numpy.isclose(val, expected_value):
                mismatch_list.append(
                    (field, 'expected: %f' % expected_value,
                     'actual: %f' % val))
        result_feature = None
        if mismatch_list:
            raise RuntimeError("results not expected: %s" % mismatch_list)

    def test_mask_raster_nodata_overflow(self):
        """NDR test when target nodata value overflows source dtype."""
        from natcap.invest.ndr import ndr

        source_raster_path = os.path.join(self.workspace_dir, 'source.tif')
        target_raster_path = os.path.join(
            self.workspace_dir, 'target.tif')
        source_dtype = numpy.int8
        target_dtype = gdal.GDT_Int32
        target_nodata = numpy.iinfo(numpy.int32).min

        pygeoprocessing.numpy_array_to_raster(
            base_array=numpy.full((4, 4), 1, dtype=source_dtype),
            target_nodata=None,
            pixel_size=(1, -1),
            origin=(0, 0),
            projection_wkt=None,
            target_path=source_raster_path)

        ndr._mask_raster(
            source_raster_path=source_raster_path,
            mask_raster_path=source_raster_path,  # mask=source for convenience
            target_masked_raster_path=target_raster_path,
            target_nodata=target_nodata,
            target_dtype=target_dtype)

        # Mostly we're testing that _mask_raster did not raise an OverflowError,
        # but we can assert the results anyway.
        array = pygeoprocessing.raster_to_numpy_array(target_raster_path)
        numpy.testing.assert_array_equal(
            array,
            numpy.full((4, 4), 1, dtype=numpy.int32))  # matches target_dtype

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
            'flow_dir_algorithm'
        ]
        self.assertEqual(set(invalid_args), set(expected_missing_args))

    def test_masking_invalid_geometry(self):
        """NDR test masking of invalid geometries.

        For more context, see https://github.com/natcap/invest/issues/1412.
        """
        from natcap.invest.ndr import ndr

        default_origin = (444720, 3751320)
        default_pixel_size = (30, -30)
        default_epsg = 3116
        default_srs = osr.SpatialReference()
        default_srs.ImportFromEPSG(default_epsg)

        # bowtie geometry is invalid; verify we can still create a mask.
        coordinates = []
        for pixel_x_offset, pixel_y_offset in [
                (0, 0), (0, 1), (1, 0.25), (1, 0.75), (0, 0)]:
            coordinates.append((
                default_origin[0] + default_pixel_size[0] * pixel_x_offset,
                default_origin[1] + default_pixel_size[1] * pixel_y_offset
            ))

        source_vector_path = os.path.join(self.workspace_dir, 'vector.geojson')
        pygeoprocessing.shapely_geometry_to_vector(
            [shapely.geometry.Polygon(coordinates)], source_vector_path,
            default_srs.ExportToWkt(), 'GeoJSON')

        source_raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        vector_info = pygeoprocessing.get_vector_info(source_vector_path)
        bbox_geom = shapely.geometry.box(*vector_info['bounding_box'])
        bbox_geom.buffer(50)  # expand around the vector
        pygeoprocessing.create_raster_from_bounding_box(
            bbox_geom.bounds, source_raster_path,
            default_pixel_size, gdal.GDT_Byte, default_srs.ExportToWkt(),
            target_nodata=255)

        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        ndr._create_mask_raster(source_raster_path, source_vector_path,
                                target_raster_path)

        expected_array = numpy.array([[1]])
        numpy.testing.assert_array_equal(
            expected_array,
            pygeoprocessing.raster_to_numpy_array(target_raster_path))

    def test_synthetic_runoff_proxy_av(self):
        """
        Test RPI given user-entered or auto-calculated runoff proxy average.

        Test that the runoff proxy index (RPI) is calculated correctly if
        (1) the user specifies a runoff proxy average value,
        (2) the user does not specify a value so the runoff proxy average
            is auto-calculated.
        """
        from natcap.invest.ndr import ndr

        # make simple raster
        runoff_proxy_path = os.path.join(self.workspace_dir, "ppt.tif")
        runoff_proxy_array = numpy.array(
            [[800, 799, 567, 234], [765, 867, 765, 654]], dtype=numpy.float32)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26910)
        projection_wkt = srs.ExportToWkt()
        origin = (461251, 4923445)
        pixel_size = (30, -30)
        no_data = -1
        pygeoprocessing.numpy_array_to_raster(
            runoff_proxy_array, no_data, pixel_size, origin, projection_wkt,
            runoff_proxy_path)
        target_rpi_path = os.path.join(self.workspace_dir, "out_raster.tif")

        # Calculate RPI with user-specified runoff proxy average
        runoff_proxy_av = 2
        ndr._normalize_raster((runoff_proxy_path, 1), target_rpi_path,
                              user_provided_mean=runoff_proxy_av)

        actual_rpi = pygeoprocessing.raster_to_numpy_array(target_rpi_path)
        expected_rpi = runoff_proxy_array/runoff_proxy_av

        numpy.testing.assert_allclose(actual_rpi, expected_rpi)

        # Now calculate RPI with auto-calculated RP average
        ndr._normalize_raster((runoff_proxy_path, 1), target_rpi_path,
                              user_provided_mean=None)

        actual_rpi = pygeoprocessing.raster_to_numpy_array(target_rpi_path)
        expected_rpi = runoff_proxy_array/numpy.mean(runoff_proxy_array)

        numpy.testing.assert_allclose(actual_rpi, expected_rpi)
    
    def test_calculate_load_type(self):
        """Test ``_calculate_load`` for both load_types."""
        from natcap.invest.ndr import ndr

        # make simple lulc raster
        lulc_path = os.path.join(self.workspace_dir, "lulc-load-type.tif")
        lulc_array = numpy.array(
            [[1, 2, 3, 4], [4, 3, 2, 1]], dtype=numpy.int16)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26910)
        projection_wkt = srs.ExportToWkt()
        origin = (461251, 4923445)
        pixel_size = (30, -30)
        no_data = -1
        pygeoprocessing.numpy_array_to_raster(
            lulc_array, no_data, pixel_size, origin, projection_wkt,
            lulc_path)

        target_load_path = os.path.join(self.workspace_dir, "load_raster.tif")

        # Calculate load
        lucode_to_params = {
            1: {'load_n': 10.0, 'eff_n': 0.5, 'load_type_n': 'measured-runoff'},
            2: {'load_n': 20.0, 'eff_n': 0.5, 'load_type_n': 'measured-runoff'},
            3: {'load_n': 10.0, 'eff_n': 0.5, 'load_type_n': 'application-rate'},
            4: {'load_n': 20.0, 'eff_n': 0.5, 'load_type_n': 'application-rate'}}
        ndr._calculate_load(lulc_path, lucode_to_params, 'n', target_load_path)

        expected_results = numpy.array(
            [[10.0, 20.0, 5.0, 10.0], [10.0, 5.0, 20.0, 10.0]])
        actual_results = pygeoprocessing.raster_to_numpy_array(target_load_path)

        numpy.testing.assert_allclose(actual_results, expected_results)
    
    def test_calculate_load_type_raises_error(self):
        """Test ``_calculate_load`` raises ValueError on bad load_type's."""
        from natcap.invest.ndr import ndr

        lulc_path = os.path.join(self.workspace_dir, "lulc-load-type.tif")
        target_load_path = os.path.join(self.workspace_dir, "load_raster.tif")

        # Calculate load
        lucode_to_params = {
            1: {'load_n': 10.0, 'eff_n': 0.5, 'load_type_n': 'measured-runoff'},
            2: {'load_n': 20.0, 'eff_n': 0.5, 'load_type_n': 'cheese'},
            3: {'load_n': 10.0, 'eff_n': 0.5, 'load_type_n': 'application-rate'},
            4: {'load_n': 20.0, 'eff_n': 0.5, 'load_type_n': 'application-rate'}}

        with self.assertRaises(ValueError) as cm:
            ndr._calculate_load(lulc_path, lucode_to_params, 'n', target_load_path)
        actual_message = str(cm.exception)
        self.assertTrue('found value of: "cheese"' in actual_message)
