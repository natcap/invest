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


def make_simple_raster(base_raster_path, array):
    """Create a raster on designated path with arbitrary values.
    Args:
        base_raster_path (str): the raster path for making the new raster.
    Returns:
        None.
    """
    # UTM Zone 10N
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    origin = (461251, 4923445)
    pixel_size = (30, -30)
    no_data = -1

    pygeoprocessing.numpy_array_to_raster(
        array, no_data, pixel_size, origin, projection_wkt,
        base_raster_path)


def make_simple_vector(path_to_shp):
    """
    Generate shapefile with two overlapping polygons
    Args:
        path_to_shp (str): path to store watershed results vector
    Outputs:
        None
    """
    # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
    shapely_geometry_list = [
        shapely.geometry.Polygon(
            [(461251, 4923195), (461501, 4923195),
             (461501, 4923445), (461251, 4923445),
             (461251, 4923195)])
    ]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    vector_format = "ESRI Shapefile"
    fields = {"id": ogr.OFTReal}
    attribute_list = [{"id": 0}]

    pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                               path_to_shp, projection_wkt,
                                               vector_format, fields,
                                               attribute_list)

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
                ('p_surface_load', 41.826904),
                ('p_surface_export', 5.566120),
                ('n_surface_load', 2977.551270),
                ('n_surface_export', 274.020844),
                ('n_subsurface_load', 28.558048),
                ('n_subsurface_export', 15.578484),
                ('n_total_export', 289.599314)]:
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
            'p_surface_export': 5.870544,
            'n_surface_load': 2977.551270,
            'n_surface_export': 274.020844,
            'n_subsurface_load': 28.558048,
            'n_subsurface_export': 15.578484,
            'n_total_export': 289.599314
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
                ('p_surface_export', 5.870544),
                ('n_surface_load', 2977.551270),
                ('n_surface_export', 274.020844),
                ('n_subsurface_load', 28.558048),
                ('n_subsurface_export', 15.578484),
                ('n_total_export', 289.599314)]:
            val = result_feature.GetField(field)
            if not numpy.isclose(val, expected_value):
                mismatch_list.append(
                    (field, 'expected: %f' % expected_value,
                     'actual: %f' % val))
        result_feature = None
        if mismatch_list:
            raise RuntimeError("results not expected: %s" % mismatch_list)

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
        Test that runoff proxy average is calculated correctly if
        (1) the user specified a runoff proxy average value,
        (2) the user does not specify a value so the runoff proxy average
            is auto-calculated
        """
        from natcap.invest.ndr.ndr import execute

        args = {
            'workspace_dir': self.workspace_dir,
            'runoff_proxy_av': 2,
            'biophysical_table_path': os.path.join(
                self.workspace_dir, "biophysical_table_gura.csv"),
            'calc_n': False,
            'calc_p': False,
            'dem_path': os.path.join(
                self.workspace_dir, "DEM_gura.tif"),
            'k_param': "2",
            'lulc_path': os.path.join(
                self.workspace_dir, "land_use_gura.tif"),
            'results_suffix': "_v1",
            'runoff_proxy_path': os.path.join(
                self.workspace_dir, "precipitation_gura.tif"),
            'threshold_flow_accumulation': "1000",
            'watersheds_path': os.path.join(
                self.workspace_dir, "watershed_gura.shp")
            }

        biophysical_table = pandas.DataFrame({
            "lucode": [1, 2, 3, 4],
            "description": ["water", "forest", "grass", "urban"],
            "load_p": [0, 1, 1, 0],
            "eff_p": [1, 1.1, .9, .3],
            "crit_len_p": [.05, .1, .2, .3],
            "load_n": [0, 1, 0, 0.2],
            'proportion_subsurface_n': [0, 0, 0, 0]
        })

        biophysical_csv_path = args['biophysical_table_path']
        biophysical_table.to_csv(biophysical_csv_path, index=False)

        runoff_proxy_ar = numpy.array([
            [800, 799, 567, 234, 422, 422, 555],
            [765, 867, 765, 654, 456, 677, 444],
            [556, 443, 456, 265, 876, 890, 333],
            [433, 266, 677, 776, 900, 687, 222],
            [456, 832, 234, 234, 234, 554, 345]
        ], dtype=numpy.float32)
        make_simple_raster(args['runoff_proxy_path'], runoff_proxy_ar)

        lulc_array = numpy.array([
            [2, 3, 1, 5, 5, 5, 5],
            [3, 3, 1, 1, 4, 5, 5],
            [5, 5, 4, 2, 3, 1, 5],
            [4, 1, 4, 2, 2, 1, 5],
            [1, 5, 4, 1, 1, 2, 5]
        ], dtype=numpy.float32)
        make_simple_raster(args['lulc_path'], lulc_array)

        dem = numpy.array([
            [800, 799, 567, 234, 422, 422, 555],
            [765, 867, 765, 654, 456, 677, 444],
            [556, 443, 456, 265, 876, 890, 333],
            [433, 266, 677, 776, 900, 687, 222],
            [456, 832, 234, 234, 234, 222, 300]
        ], dtype=numpy.float32)
        make_simple_raster(args['dem_path'], dem)

        make_simple_vector(args['watersheds_path'])

        execute(args)

        actual_output_path = os.path.join(
            args['workspace_dir'], "intermediate_outputs",
            f"runoff_proxy_index{args['results_suffix']}.tif")
        actual_output = gdal.Open(actual_output_path)
        band = actual_output.GetRasterBand(1)
        actual_rpi = band.ReadAsArray()

        expected_output = gdal.Open(args['runoff_proxy_path'])
        expected_band = expected_output.GetRasterBand(1)
        expected_rpi = expected_band.ReadAsArray()/args['runoff_proxy_av']

        numpy.testing.assert_allclose(actual_rpi, expected_rpi)

        expected_output = None
        expected_band = None
        expected_rpi = None

        # now run this without the a user average specified
        del args['runoff_proxy_av']

        execute(args)

        actual_output_path = os.path.join(
            args['workspace_dir'], "intermediate_outputs",
            f"runoff_proxy_index{args['results_suffix']}.tif")
        actual_output = gdal.Open(actual_output_path)
        band = actual_output.GetRasterBand(1)
        actual_rpi = band.ReadAsArray()

        # compare to rpi with automatically calculated mean
        expected_output = gdal.Open(args['runoff_proxy_path'])
        expected_band = expected_output.GetRasterBand(1)
        expected_rpi = expected_band.ReadAsArray()
        expected_rpi /= numpy.mean(expected_rpi)

        numpy.testing.assert_allclose(actual_rpi, expected_rpi)

        expected_output = None
        expected_band = None
        expected_rpi = None