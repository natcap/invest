"""Testing module for validation."""
# encoding=UTF-8
import tempfile
import unittest
from unittest.mock import Mock
import functools
import os
import shutil
import string
import time
import warnings

from osgeo import gdal, osr, ogr
import pandas


class SpatialOverlapTest(unittest.TestCase):
    """Test Spatial Overlap."""

    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir)

    def test_no_overlap(self):
        """Validation: verify lack of overlap."""
        from natcap.invest import validation
        import pygeoprocessing

        driver = gdal.GetDriverByName('GTiff')
        filepath_1 = os.path.join(self.workspace_dir, 'raster_1.tif')
        filepath_2 = os.path.join(self.workspace_dir, 'raster_2.tif')

        filepath_list = []
        bbox_list = []
        for filepath, geotransform in (
                (filepath_1, [1, 1, 0, 1, 0, 1]),
                (filepath_2, [100, 1, 0, 100, 0, 1])):
            raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            raster.SetProjection(wgs84_srs.ExportToWkt())
            raster.SetGeoTransform(geotransform)
            raster = None
            filepath_list.append(filepath)
            bbox_list.append(
                pygeoprocessing.get_raster_info(filepath)['bounding_box'])

        error_msg = validation.check_spatial_overlap([filepath_1, filepath_2])
        formatted_lists = validation._format_bbox_list(
            filepath_list, bbox_list)
        self.assertTrue(validation.MESSAGES['BBOX_NOT_INTERSECT'].format(
            bboxes=formatted_lists) in error_msg)

    def test_overlap(self):
        """Validation: verify overlap."""
        from natcap.invest import validation

        driver = gdal.GetDriverByName('GTiff')
        filepath_1 = os.path.join(self.workspace_dir, 'raster_1.tif')
        filepath_2 = os.path.join(self.workspace_dir, 'raster_2.tif')

        for filepath, geotransform in (
                (filepath_1, [1, 1, 0, 1, 0, 1]),
                (filepath_2, [2, 1, 0, 2, 0, 1])):
            raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            raster.SetProjection(wgs84_srs.ExportToWkt())
            raster.SetGeoTransform(geotransform)
            raster = None

        self.assertEqual(
            None, validation.check_spatial_overlap([filepath_1, filepath_2]))

    def test_check_overlap_undefined_projection(self):
        """Validation: check overlap of raster with an undefined projection."""
        from natcap.invest import validation

        driver = gdal.GetDriverByName('GTiff')
        filepath_1 = os.path.join(self.workspace_dir, 'raster_1.tif')
        filepath_2 = os.path.join(self.workspace_dir, 'raster_2.tif')

        raster_1 = driver.Create(filepath_1, 3, 3, 1, gdal.GDT_Int32)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        raster_1.SetProjection(wgs84_srs.ExportToWkt())
        raster_1.SetGeoTransform([1, 1, 0, 1, 0, 1])
        raster_1 = None

        # set up a raster with an undefined projection
        raster_2 = driver.Create(filepath_2, 3, 3, 1, gdal.GDT_Int32)
        raster_2.SetGeoTransform([2, 1, 0, 2, 0, 1])
        raster_2 = None

        error_msg = validation.check_spatial_overlap(
            [filepath_1, filepath_2], different_projections_ok=True)
        expected = validation.MESSAGES['NO_PROJECTION'].format(filepath=filepath_2)
        self.assertEqual(error_msg, expected)

    @unittest.skip("skipping due to unresolved projection comparison question")
    def test_different_projections_not_ok(self):
        """Validation: different projections not allowed by default.

        This test illustrates a bug we don't yet have a good solution for
        (natcap/invest#558)
        When ``different_projections_ok is False``, we don't check that the
        projections are actually the same, because there isn't a great way to
        do so. So there's the possibility that some bounding boxes overlap
        numerically, but have different projections, and thus pass validation
        when they shouldn't.
        """

        from natcap.invest import validation

        driver = gdal.GetDriverByName('GTiff')
        filepath_1 = os.path.join(self.workspace_dir, 'raster_1.tif')
        filepath_2 = os.path.join(self.workspace_dir, 'raster_2.tif')

        # bounding boxes overlap if we don't account for the projections
        for filepath, geotransform, epsg in (
                (filepath_1, [1, 1, 0, 1, 0, 1], 4326),
                (filepath_2, [2, 1, 0, 2, 0, 1], 2193)):
            raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(epsg)
            raster.SetProjection(wgs84_srs.ExportToWkt())
            raster.SetGeoTransform(geotransform)
            raster = None

        expected = (f'Spatial files {[filepath_1, filepath_2]} do not all '
                    'have the same projection')
        self.assertEqual(
            validation.check_spatial_overlap([filepath_1, filepath_2]),
            expected)


class ValidatorTest(unittest.TestCase):
    """Test Validator."""

    def test_args_wrong_type(self):
        """Validation: check for error when args is the wrong type."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args=123)

    def test_limit_to_wrong_type(self):
        """Validation: check for error when limit_to is the wrong type."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={}, limit_to=1234)

    def test_limit_to_not_in_args(self):
        """Validation: check for error when limit_to is not a key in args."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={}, limit_to='bar')

    def test_args_keys_must_be_strings(self):
        """Validation: check for error when args keys are not all strings."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={1: 'foo'})

    def test_return_keys_in_args(self):
        """Validation: check for error when return keys not all in args."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return [(('a',), 'error 1')]

        validation_errors = validate({})
        self.assertEqual(validation_errors,
                         [(('a',), 'error 1')])

    def test_wrong_parameter_names(self):
        """Validation: check for error when wrong function signature used."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(foo):
            pass

        with self.assertRaises(AssertionError):
            validate({})

    def test_return_value(self):
        """Validation: validation errors should be returned from decorator."""
        from natcap.invest import validation

        errors = [(('a', 'b'), 'Error!')]

        @validation.invest_validator
        def validate(args, limit_to=None):
            return errors

        validation_errors = validate({'a': 'foo', 'b': 'bar'})
        self.assertEqual(validation_errors, errors)

    def test_n_workers(self):
        """Validation: validation error returned on invalid n_workers."""
        from natcap.invest import spec_utils, validation

        args_spec = {
            'n_workers': spec_utils.N_WORKERS,
        }

        @validation.invest_validator
        def validate(args, limit_to=None):
            return validation.validate(args, args_spec)

        args = {'n_workers': 'not a number'}
        validation_errors = validate(args)
        expected = [(
            ['n_workers'],
            validation.MESSAGES['NOT_A_NUMBER'].format(value=args['n_workers']))]
        self.assertEqual(validation_errors, expected)

    def test_timeout_succeed(self):
        from natcap.invest import validation

        # both args and the kwarg should be passed to the function
        def func(arg1, arg2, kwarg=None):
            self.assertEqual(kwarg, 'kwarg')
            time.sleep(1)

        # this will raise an error if the timeout is exceeded
        # timeout defaults to 5 seconds so this should pass
        validation.timeout(func, 'arg1', 'arg2', kwarg='kwarg')

    def test_timeout_fail(self):
        from natcap.invest import validation

        # both args and the kwarg should be passed to the function
        def func(arg):
            time.sleep(6)

        # this will return a warning if the timeout is exceeded
        # timeout defaults to 5 seconds so this should fail
        with warnings.catch_warnings(record=True) as ws:
            # cause all warnings to always be triggered
            warnings.simplefilter("always")
            validation.timeout(func, 'arg')
            self.assertTrue(len(ws) == 1)
            self.assertTrue('timed out' in str(ws[0].message))


class DirectoryValidation(unittest.TestCase):
    """Test Directory Validation."""

    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir)

    def test_exists(self):
        """Validation: when a folder must exist and does."""
        from natcap.invest import validation

        self.assertEqual(None, validation.check_directory(self.workspace_dir))

    def test_not_exists(self):
        """Validation: when a folder must exist but does not."""
        from natcap.invest import validation

        dirpath = os.path.join(self.workspace_dir, 'nonexistent_dir')
        validation_warning = validation.check_directory(dirpath)
        self.assertEqual(validation_warning, validation.MESSAGES['DIR_NOT_FOUND'])

    def test_file(self):
        """Validation: when a file is given to folder validation."""
        from natcap.invest import validation

        filepath = os.path.join(self.workspace_dir, 'some_file.txt')
        with open(filepath, 'w') as opened_file:
            opened_file.write('the text itself does not matter.')

        validation_warning = validation.check_directory(filepath)
        self.assertEqual(validation_warning, validation.MESSAGES['NOT_A_DIR'])

    def test_valid_permissions(self):
        """Validation: folder permissions."""
        from natcap.invest import validation

        self.assertEqual(None, validation.check_directory(
            self.workspace_dir, permissions='rwx'))

    def test_workspace_not_exists(self):
        """Validation: when a folder's parent must exist with permissions."""
        from natcap.invest import validation

        dirpath = 'foo'
        new_dir = os.path.join(self.workspace_dir, dirpath)

        self.assertEqual(None, validation.check_directory(
            new_dir, must_exist=False, permissions='rwx'))


class FileValidation(unittest.TestCase):
    """Test File Validator."""

    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir)

    def test_file_exists(self):
        """Validation: test that a file exists."""
        from natcap.invest import validation
        filepath = os.path.join(self.workspace_dir, 'file.txt')
        with open(filepath, 'w') as new_file:
            new_file.write("Here's some text.")

        self.assertEqual(None, validation.check_file(filepath))

    def test_file_not_found(self):
        """Validation: test when a file is not found."""
        from natcap.invest import validation
        filepath = os.path.join(self.workspace_dir, 'file.txt')

        error_msg = validation.check_file(filepath)
        self.assertEqual(error_msg, validation.MESSAGES['FILE_NOT_FOUND'])


class RasterValidation(unittest.TestCase):
    """Test Raster Validation."""

    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir)

    def test_file_not_found(self):
        """Validation: test that a raster exists."""
        from natcap.invest import validation

        filepath = os.path.join(self.workspace_dir, 'file.txt')
        error_msg = validation.check_raster(filepath)
        self.assertEqual(error_msg, validation.MESSAGES['FILE_NOT_FOUND'])

    def test_invalid_raster(self):
        """Validation: test when a raster format is invalid."""
        from natcap.invest import validation

        filepath = os.path.join(self.workspace_dir, 'file.txt')
        with open(filepath, 'w') as bad_raster:
            bad_raster.write('not a raster')

        error_msg = validation.check_raster(filepath)
        self.assertEqual(error_msg, validation.MESSAGES['NOT_GDAL_RASTER'])

    def test_invalid_ovr_raster(self):
        """Validation: test when a .tif.ovr file is input as a raster."""
        from natcap.invest import validation

        # Use EPSG:32731  # WGS84 / UTM zone 31s
        driver = gdal.GetDriverByName('GTiff')
        filepath = os.path.join(self.workspace_dir, 'raster.tif')
        raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
        meters_srs = osr.SpatialReference()
        meters_srs.ImportFromEPSG(32731)
        raster.SetProjection(meters_srs.ExportToWkt())
        raster = None
        # I could only create overviews when opening the file, not on creation.
        # Build overviews taken from:
        # https://gis.stackexchange.com/questions/270498/compress-gtiff-external-overviews-with-gdal-api
        raster = gdal.OpenEx(filepath)
        gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
        raster.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32, 64, 128, 256])
        raster = None

        filepath_ovr = os.path.join(self.workspace_dir, 'raster.tif.ovr')
        error_msg = validation.check_raster(filepath_ovr)
        self.assertEqual(error_msg, validation.MESSAGES['OVR_FILE'])

    def test_raster_not_projected(self):
        """Validation: test when a raster is not linearly projected."""
        from natcap.invest import validation

        # use WGS84 as not linearly projected.
        driver = gdal.GetDriverByName('GTiff')
        filepath = os.path.join(self.workspace_dir, 'raster.tif')
        raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        raster.SetProjection(wgs84_srs.ExportToWkt())
        raster = None

        error_msg = validation.check_raster(filepath, projected=True)
        self.assertEqual(error_msg, validation.MESSAGES['NOT_PROJECTED'])

    def test_raster_incorrect_units(self):
        """Validation: test when a raster projection has wrong units."""
        from natcap.invest import spec_utils, validation

        # Use EPSG:32066  # NAD27 / BLM 16N (in US Survey Feet)
        driver = gdal.GetDriverByName('GTiff')
        filepath = os.path.join(self.workspace_dir, 'raster.tif')
        raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(32066)
        raster.SetProjection(wgs84_srs.ExportToWkt())
        raster = None

        error_msg = validation.check_raster(
            filepath, projected=True, projection_units=spec_utils.u.meter)
        expected_msg = validation.MESSAGES['WRONG_PROJECTION_UNIT'].format(
            unit_a='meter', unit_b='us_survey_foot')
        self.assertEqual(expected_msg, error_msg)


class VectorValidation(unittest.TestCase):
    """Test Vector Validation."""

    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir)

    def test_file_not_found(self):
        """Validation: test when a vector file is not found."""
        from natcap.invest import validation

        filepath = os.path.join(self.workspace_dir, 'file.txt')
        error_msg = validation.check_vector(filepath)
        self.assertEqual(error_msg, validation.MESSAGES['FILE_NOT_FOUND'])

    def test_invalid_vector(self):
        """Validation: test when a vector's format is invalid."""
        from natcap.invest import validation

        filepath = os.path.join(self.workspace_dir, 'file.txt')
        with open(filepath, 'w') as bad_vector:
            bad_vector.write('not a vector')

        error_msg = validation.check_vector(filepath)
        self.assertEqual(error_msg, validation.MESSAGES['NOT_GDAL_VECTOR'])

    def test_missing_fieldnames(self):
        """Validation: test when a vector is missing fields."""
        from natcap.invest import validation

        driver = gdal.GetDriverByName('GPKG')
        filepath = os.path.join(self.workspace_dir, 'vector.gpkg')
        vector = driver.Create(filepath, 0, 0, 0, gdal.GDT_Unknown)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('sample_layer', wgs84_srs, ogr.wkbUnknown)

        for field_name, field_type in (('COL_A', ogr.OFTInteger),
                                       ('col_b', ogr.OFTString)):
            layer.CreateField(ogr.FieldDefn(field_name, field_type))

        new_feature = ogr.Feature(layer.GetLayerDefn())
        new_feature.SetField('COL_A', 1)
        new_feature.SetField('col_b', 'hello')
        layer.CreateFeature(new_feature)

        new_feature = None
        layer = None
        vector = None

        error_msg = validation.check_vector(
            filepath, fields={'col_a': {}, 'col_b': {}, 'col_c': {}})
        expected = validation.MESSAGES['MATCHED_NO_HEADERS'].format(
            header='field', header_name='col_c')
        self.assertEqual(error_msg, expected)

    def test_vector_projected_in_m(self):
        """Validation: test that a vector's projection has expected units."""
        from natcap.invest import spec_utils, validation

        driver = gdal.GetDriverByName('GPKG')
        filepath = os.path.join(self.workspace_dir, 'vector.gpkg')
        vector = driver.Create(filepath, 0, 0, 0, gdal.GDT_Unknown)
        meters_srs = osr.SpatialReference()
        meters_srs.ImportFromEPSG(32731)
        layer = vector.CreateLayer('sample_layer', meters_srs, ogr.wkbUnknown)

        layer = None
        vector = None

        error_msg = validation.check_vector(
            filepath, projected=True, projection_units=spec_utils.u.foot)
        expected_msg = validation.MESSAGES['WRONG_PROJECTION_UNIT'].format(
            unit_a='foot', unit_b='metre')
        self.assertEqual(error_msg, expected_msg)

        self.assertEqual(None, validation.check_vector(
            filepath, projected=True, projection_units=spec_utils.u.meter))


class FreestyleStringValidation(unittest.TestCase):
    """Test Freestyle String Validation."""

    def test_int(self):
        """Validation: test that an int can be a valid string."""
        from natcap.invest import validation
        self.assertEqual(None, validation.check_freestyle_string(1234))

    def test_float(self):
        """Validation: test that a float can be a valid string."""
        from natcap.invest import validation
        self.assertEqual(None, validation.check_freestyle_string(1.234))

    def test_regexp(self):
        """Validation: test that we can check regex patterns on strings."""
        from natcap.invest import validation
        from natcap.invest.spec_utils import SUFFIX

        self.assertEqual(None, validation.check_freestyle_string(
            1.234, regexp='^1.[0-9]+$'))

        regexp = '^[a-zA-Z]+$'
        error_msg = validation.check_freestyle_string(
            'foobar12', regexp=regexp)
        self.assertEqual(
            error_msg, validation.MESSAGES['REGEXP_MISMATCH'].format(regexp=regexp))

        error_msg = validation.check_freestyle_string(
            '4/20', regexp=SUFFIX['regexp'])
        self.assertEqual(
            error_msg, validation.MESSAGES['REGEXP_MISMATCH'].format(regexp=SUFFIX['regexp']))


class OptionStringValidation(unittest.TestCase):
    """Test Option String Validation."""

    def test_valid_option_set(self):
        """Validation: test that a string is a valid option in a set."""
        from natcap.invest import validation
        self.assertEqual(None, validation.check_option_string(
            'foo', options={'foo', 'bar', 'Baz'}))

    def test_invalid_option_set(self):
        """Validation: test when a string is not a valid option in a set."""
        from natcap.invest import validation
        options = ['foo', 'bar', 'Baz']
        error_msg = validation.check_option_string('FOO', options=options)
        self.assertEqual(
            error_msg,
            validation.MESSAGES['INVALID_OPTION'].format(
                option_list=sorted(options)))

    def test_valid_option_dict(self):
        """Validation: test that a string is a valid option in a dict."""
        from natcap.invest import validation
        self.assertEqual(None, validation.check_option_string(
            'foo', options={'foo': 'desc', 'bar': 'desc', 'Baz': 'desc'}))

    def test_invalid_option_dict(self):
        """Validation: test when a string is not a valid option in a dict."""
        from natcap.invest import validation
        options = {'foo': 'desc', 'bar': 'desc', 'Baz': 'desc'}
        error_msg = validation.check_option_string(
            'FOO', options=options)
        self.assertEqual(
            error_msg,
            validation.MESSAGES['INVALID_OPTION'].format(
                option_list=sorted(options.keys())))


class NumberValidation(unittest.TestCase):
    """Test Number Validation."""

    def test_string(self):
        """Validation: test when a string is not a number."""
        from natcap.invest import validation
        value = 'this is a string'
        error_msg = validation.check_number(value)
        self.assertEqual(
            error_msg, validation.MESSAGES['NOT_A_NUMBER'].format(value=value))

    def test_expression(self):
        """Validation: test that we can use numeric expressions."""
        from natcap.invest import validation
        self.assertEqual(
            None, validation.check_number(
                35, '(value < 100) & (value > 4)'))

    def test_expression_missing_value(self):
        """Validation: test the expression string for the 'value' term."""
        from natcap.invest import validation
        with self.assertRaises(AssertionError):
            error_msg = validation.check_number(35, 'foo < 5')

    def test_expression_failure(self):
        """Validation: test when a number does not meet the expression."""
        from natcap.invest import validation
        value = 35
        condition = 'float(value) < 0'
        error_msg = validation.check_number(value, condition)
        self.assertEqual(error_msg, validation.MESSAGES['INVALID_VALUE'].format(
            value=value, condition=condition))

    def test_expression_failure_string(self):
        """Validation: test when string value does not meet the expression."""
        from natcap.invest import validation
        value = '35'
        condition = 'int(value) < 0'
        error_msg = validation.check_number(value, condition)
        self.assertEqual(error_msg, validation.MESSAGES['INVALID_VALUE'].format(
            value=value, condition=condition))


class BooleanValidation(unittest.TestCase):
    """Test Boolean Validation."""

    def test_actual_bool(self):
        """Validation: test when boolean type objects are passed."""
        from natcap.invest import validation
        self.assertEqual(None, validation.check_boolean(True))
        self.assertEqual(None, validation.check_boolean(False))

    def test_string_boolean(self):
        """Validation: an error should be raised when the type is wrong."""
        from natcap.invest import validation
        for non_boolean_value in ('true', 1, [], set()):
            self.assertTrue(
                isinstance(validation.check_boolean(non_boolean_value), str))

    def test_invalid_string(self):
        """Validation: test when invalid strings are passed."""
        from natcap.invest import validation
        value = 'not clear'
        error_msg = validation.check_boolean(value)
        self.assertEqual(
            error_msg, validation.MESSAGES['NOT_BOOLEAN'].format(value=value))


class CSVValidation(unittest.TestCase):
    """Test CSV Validation."""

    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def test_file_not_found(self):
        """Validation: test when a file is not found."""
        from natcap.invest import validation

        nonexistent_file = os.path.join(self.workspace_dir, 'nope.txt')
        error_msg = validation.check_csv(nonexistent_file)
        self.assertEqual(error_msg, validation.MESSAGES['FILE_NOT_FOUND'])

    def test_csv_fieldnames(self):
        """Validation: test that we can check fieldnames in a CSV."""
        from natcap.invest import validation

        df = pandas.DataFrame([
            {'foo': 1, 'bar': 2, 'baz': 3},
            {'foo': 2, 'bar': 3, 'baz': 4},
            {'foo': 3, 'bar': 4, 'baz': 5}])

        target_file = os.path.join(self.workspace_dir, 'test.csv')
        df.to_csv(target_file)

        self.assertEqual(None, validation.check_csv(
            target_file, columns={'foo': {}, 'bar': {}}))

    def test_csv_bom_fieldnames(self):
        """Validation: test that we can check fieldnames in a CSV with BOM."""
        from natcap.invest import validation

        df = pandas.DataFrame([
            {'foo': 1, 'bar': 2, 'baz': 3},
            {'foo': 2, 'bar': 3, 'baz': 4},
            {'foo': 3, 'bar': 4, 'baz': 5}])

        target_file = os.path.join(self.workspace_dir, 'test.csv')
        df.to_csv(target_file, encoding='utf-8-sig')

        self.assertEqual(None, validation.check_csv(
            target_file, columns={'foo': {}, 'bar': {}}))

    def test_csv_missing_fieldnames(self):
        """Validation: test that we can check missing fieldnames in a CSV."""
        from natcap.invest import validation

        df = pandas.DataFrame([
            {'foo': 1, 'bar': 2, 'baz': 3},
            {'foo': 2, 'bar': 3, 'baz': 4},
            {'foo': 3, 'bar': 4, 'baz': 5}])

        target_file = os.path.join(self.workspace_dir, 'test.csv')
        df.to_csv(target_file)

        error_msg = validation.check_csv(
            target_file, columns={'field_a': {}})
        expected_msg = validation.MESSAGES['MATCHED_NO_HEADERS'].format(
            header='column', header_name='field_a')
        self.assertEqual(error_msg, expected_msg)

    def test_excel_missing_fieldnames(self):
        """Validation: test that we can check missing fieldnames in excel."""
        from natcap.invest import validation

        df = pandas.DataFrame([
            {'foo': 1, 'bar': 2, 'baz': 3},
            {'foo': 2, 'bar': 3, 'baz': 4},
            {'foo': 3, 'bar': 4, 'baz': 5}])

        target_file = os.path.join(self.workspace_dir, 'test.xlsx')
        df.to_excel(target_file)

        error_msg = validation.check_csv(
            target_file, columns={'field_a': {}}, excel_ok=True)
        expected_msg = validation.MESSAGES['MATCHED_NO_HEADERS'].format(
            header='column', header_name='field_a')
        self.assertEqual(error_msg, expected_msg)

    def test_wrong_filetype(self):
        """Validation: verify CSV type does not open pickles."""
        from natcap.invest import validation

        df = pandas.DataFrame([
            {'foo': 1, 'bar': 2, 'baz': 3},
            {'foo': 2, 'bar': 3, 'baz': 4},
            {'foo': 3, 'bar': 4, 'baz': 5}])

        target_file = os.path.join(self.workspace_dir, 'test.pckl')
        df.to_pickle(target_file)

        error_msg = validation.check_csv(
            target_file, columns={'field_a': {}}, excel_ok=True)
        self.assertEqual(error_msg, validation.MESSAGES['NOT_CSV_OR_EXCEL'])

        error_msg = validation.check_csv(
            target_file, columns={'field_a': {}}, excel_ok=False)
        self.assertEqual(error_msg, validation.MESSAGES['NOT_CSV'])

    def test_slow_to_open(self):
        """Test timeout by mocking a CSV that is slow to open"""
        from natcap.invest import validation

        # make an actual file so that `check_file` will pass
        path = os.path.join(self.workspace_dir, 'slow.csv')
        with open(path, 'w') as file:
            file.write('1,2,3')

        spec = {
            "mock_csv_path": {
                "type": "csv",
                "required": True,
                "about": "A CSV that will be mocked.",
                "name": "CSV"
            }
        }

        # validate a mocked CSV that will take 6 seconds to return a value
        args = {"mock_csv_path": path}

        # define a side effect for the mock that will sleep
        # for longer than the allowed timeout
        def delay(*args, **kwargs):
            time.sleep(7)
            return []

        # make a copy of the real _VALIDATION_FUNCS and override the CSV function
        mock_validation_funcs = validation._VALIDATION_FUNCS.copy()
        mock_validation_funcs['csv'] = functools.partial(
            validation.timeout, delay)

        # replace the validation.check_csv with the mock function, and try to validate
        with unittest.mock.patch('natcap.invest.validation._VALIDATION_FUNCS',
                                 mock_validation_funcs):
            with warnings.catch_warnings(record=True) as ws:
                # cause all warnings to always be triggered
                warnings.simplefilter("always")
                validation.validate(args, spec)
                self.assertTrue(len(ws) == 1)
                self.assertTrue('timed out' in str(ws[0].message))

    def test_check_headers(self):
        """Validation: check that CSV header validation works."""
        from natcap.invest import validation
        expected_headers = ['hello', '1']
        actual = ['hello', '1', '2']
        result = validation.check_headers(expected_headers, actual)
        self.assertEqual(result, None)

        # each pattern should match at least one header
        actual = ['1', '2']
        result = validation.check_headers(expected_headers, actual)
        expected_msg = validation.MESSAGES['MATCHED_NO_HEADERS'].format(
            header='header', header_name='hello')
        self.assertEqual(result, expected_msg)

        # duplicate headers that match a pattern are not allowed
        actual = ['hello', '1', '1']
        result = validation.check_headers(expected_headers, actual, 'column')
        expected_msg = validation.MESSAGES['DUPLICATE_HEADER'].format(
            header='column', header_name='1', number=2)
        self.assertEqual(result, expected_msg)

        # duplicate headers that don't match a pattern are allowed
        actual = ['hello', '1', 'x', 'x']
        result = validation.check_headers(expected_headers, actual)
        self.assertEqual(result, None)


class TestValidationFromSpec(unittest.TestCase):
    """Test Validation From Spec."""

    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir)

    def test_conditional_requirement(self):
        """Validation: check that conditional requirements works."""
        from natcap.invest import validation

        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "number",
                "required": True,
            },
            "number_b": {
                "name": "The second parameter",
                "about": "About the second parameter",
                "type": "number",
                "required": False,
            },
            "number_c": {
                "name": "The third parameter",
                "about": "About the third parameter",
                "type": "number",
                "required": "number_b",
            },
            "number_d": {
                "name": "The fourth parameter",
                "about": "About the fourth parameter",
                "type": "number",
                "required": "number_b | number_c",
            },
            "number_e": {
                "name": "The fifth parameter",
                "about": "About the fifth parameter",
                "type": "number",
                "required": "number_b & number_d"
            },
            "number_f": {
                "name": "The sixth parameter",
                "about": "About the sixth parameter",
                "type": "number",
                "required": "not number_b"
            }
        }

        args = {
            "number_a": 123,
            "number_b": 456,
        }
        validation_warnings = validation.validate(args, spec)
        self.assertEqual(sorted(validation_warnings), [
            (['number_c'], validation.MESSAGES['MISSING_KEY']),
            (['number_d'], validation.MESSAGES['MISSING_KEY']),
        ])

        args = {
            "number_a": 123,
            "number_b": 456,
            "number_c": 1,
            "number_d": 3,
            "number_e": 4,
        }
        self.assertEqual([], validation.validate(args, spec))

        args = {
            "number_a": 123,
        }
        validation_warnings = validation.validate(args, spec)
        self.assertEqual(sorted(validation_warnings), [
            (['number_f'], validation.MESSAGES['MISSING_KEY'])
        ])

    def test_conditional_requirement_missing_var(self):
        """Validation: check AssertionError if expression is missing a var."""
        from natcap.invest import validation

        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "number",
                "required": True,
            },
            "number_b": {
                "name": "The second parameter",
                "about": "About the second parameter",
                "type": "number",
                "required": False,
            },
            "number_c": {
                "name": "The third parameter",
                "about": "About the third parameter",
                "type": "number",
                "required": "some_var_not_in_args",
            }
        }

        args = {
            "number_a": 123,
            "number_b": 456,
        }
        with self.assertRaises(AssertionError) as cm:
            validation_warnings = validation.validate(args, spec)
        self.assertTrue('some_var_not_in_args' in str(cm.exception))

    def test_conditional_requirement_not_required(self):
        """Validation: unrequired conditional requirement should always pass"""
        from natcap.invest import validation

        csv_a_path = os.path.join(self.workspace_dir, 'csv_a.csv')
        csv_b_path = os.path.join(self.workspace_dir, 'csv_b.csv')
        # initialize test CSV files
        with open(csv_a_path, 'w') as csv:
            csv.write('a,b,c')
        with open(csv_b_path, 'w') as csv:
            csv.write('1,2,3')

        spec = {
            "condition": {
                "name": "A condition that determines requirements",
                "about": "About the condition",
                "type": "boolean",
                "required": False,
            },
            "csv_a": {
                "name": "Conditionally required CSV A",
                "about": "About CSV A",
                "type": "csv",
                "required": "condition",
            },
            "csv_b": {
                "name": "Conditonally required CSV B",
                "about": "About CSV B",
                "type": "csv",
                "required": "not condition",
            }
        }

        # because condition = True, it shouldn't matter that the
        # csv_b parameter wouldn't pass validation
        args = {
            "condition": True,
            "csv_a": csv_a_path,
            "csv_b": 'x' + csv_b_path  # introduce a typo
        }

        validation_warnings = validation.validate(args, spec)
        self.assertEqual(validation_warnings, [])

    def test_requirement_missing(self):
        """Validation: verify absolute requirement on missing key."""
        from natcap.invest import validation
        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "number",
                "required": True,
            }
        }

        args = {}
        self.assertEqual(
            [(['number_a'], validation.MESSAGES['MISSING_KEY'])],
            validation.validate(args, spec))

    def test_requirement_no_value(self):
        """Validation: verify absolute requirement without value."""
        from natcap.invest import validation
        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "number",
                "required": True,
            }
        }

        args = {'number_a': ''}
        self.assertEqual(
            [(['number_a'], validation.MESSAGES['MISSING_VALUE'])],
            validation.validate(args, spec))

        args = {'number_a': None}
        self.assertEqual(
            [(['number_a'], validation.MESSAGES['MISSING_VALUE'])],
            validation.validate(args, spec))

    def test_invalid_value(self):
        """Validation: verify invalidity."""
        from natcap.invest import validation
        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "number",
                "required": True,
            }
        }

        args = {'number_a': 'not a number'}
        self.assertEqual(
            [(['number_a'], validation.MESSAGES['NOT_A_NUMBER'].format(
                value=args['number_a']))],
            validation.validate(args, spec))

    def test_conditionally_required_no_value(self):
        """Validation: verify conditional requirement when no value."""
        from natcap.invest import validation
        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "number",
                "required": True,
            },
            "string_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "freestyle_string",
                "required": "number_a",
            }
        }

        args = {'string_a': None, "number_a": 1}

        self.assertEqual(
            [(['string_a'], validation.MESSAGES['MISSING_VALUE'])],
            validation.validate(args, spec))

    def test_conditionally_required_invalid(self):
        """Validation: verify conditional validity behavior when invalid."""
        from natcap.invest import validation
        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "number",
                "required": True,
            },
            "string_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "option_string",
                "required": "number_a",
                "options": ['AAA', 'BBB']
            }
        }

        args = {'string_a': "ZZZ", "number_a": 1}

        self.assertEqual(
            [(['string_a'], validation.MESSAGES['INVALID_OPTION'].format(
                option_list=spec['string_a']['options']))],
            validation.validate(args, spec))

    def test_validation_exception(self):
        """Validation: Verify error when an unexpected exception occurs."""
        from natcap.invest import validation
        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "number",
                "required": True,
            },
        }

        args = {'number_a': 1}
        try:
            # Patch in a new function that raises an exception into the
            # validation functions dictionary.
            patched_function = Mock(side_effect=ValueError('foo'))
            validation._VALIDATION_FUNCS['number'] = patched_function

            validation_warnings = validation.validate(args, spec)
        finally:
            # No matter what happens with this test, always restore the state
            # of the validation functions dict.
            validation._VALIDATION_FUNCS['number'] = (
                validation.check_number)

        self.assertEqual(
            validation_warnings,
            [(['number_a'], validation.MESSAGES['UNEXPECTED_ERROR'])])

    def test_validation_other(self):
        """Validation: verify no error when 'other' type."""
        from natcap.invest import validation
        spec = {
            "number_a": {
                "name": "The first parameter",
                "about": "About the first parameter",
                "type": "other",
                "required": True,
            },
        }

        args = {'number_a': 1}
        self.assertEqual([], validation.validate(args, spec))

    def test_conditional_validity_recursive(self):
        """Validation: check that we can require from nested conditions."""
        from natcap.invest import validation

        spec = {}
        previous_key = None
        args = {}
        for letter in string.ascii_uppercase[:10]:
            key = 'arg_%s' % letter
            spec[key] = {
                'name': 'name ' + key,
                'about': 'about ' + key,
                'type': 'freestyle_string',
                'required': previous_key
            }
            previous_key = key
            args[key] = key

        del args[previous_key]  # delete the last addition to the dict.

        self.assertEqual(
            [(['arg_J'], validation.MESSAGES['MISSING_KEY'])],
            validation.validate(args, spec))

    def test_spatial_overlap_error(self):
        """Validation: check that we return an error on spatial mismatch."""
        from natcap.invest import validation

        spec = {
            'raster_a': {
                'type': 'raster',
                'name': 'raster 1',
                'about': 'raster 1',
                'required': True,
            },
            'raster_b': {
                'type': 'raster',
                'name': 'raster 2',
                'about': 'raster 2',
                'required': True,
            },
            'vector_a': {
                'type': 'vector',
                'name': 'vector 1',
                'about': 'vector 1',
                'required': True,
                'fields': {}
            }
        }

        driver = gdal.GetDriverByName('GTiff')
        filepath_1 = os.path.join(self.workspace_dir, 'raster_1.tif')
        filepath_2 = os.path.join(self.workspace_dir, 'raster_2.tif')
        reference_filepath = os.path.join(self.workspace_dir, 'reference.gpkg')

        # Filepaths 1 and 2 are obviously outside of UTM zone 31N.
        for filepath, geotransform, epsg_code in (
                (filepath_1, [1, 1, 0, 1, 0, 1], 4326),
                (filepath_2, [100, 1, 0, 100, 0, 1], 4326)):
            raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(epsg_code)
            raster.SetProjection(wgs84_srs.ExportToWkt())
            raster.SetGeoTransform(geotransform)
            raster = None

        gpkg_driver = gdal.GetDriverByName('GPKG')
        vector = gpkg_driver.Create(reference_filepath, 0, 0, 0,
                                    gdal.GDT_Unknown)
        vector_srs = osr.SpatialReference()
        vector_srs.ImportFromEPSG(32731)  # UTM 31N
        layer = vector.CreateLayer('layer', vector_srs, ogr.wkbPoint)
        new_feature = ogr.Feature(layer.GetLayerDefn())
        new_feature.SetGeometry(ogr.CreateGeometryFromWkt('POINT 1 1'))

        new_feature = None
        layer = None
        vector = None

        args = {
            'raster_a': filepath_1,
            'raster_b': filepath_2,
            'vector_a': reference_filepath,
        }

        validation_warnings = validation.validate(
            args, spec, {'spatial_keys': list(args.keys()),
                         'different_projections_ok': True})
        self.assertEqual(len(validation_warnings), 1)
        self.assertEqual(set(args.keys()), set(validation_warnings[0][0]))
        formatted_bbox_list = ''  # allows str matching w/o real bbox str
        self.assertTrue(
            validation.MESSAGES['BBOX_NOT_INTERSECT'].format(
                bboxes=formatted_bbox_list) in validation_warnings[0][1])

    def test_spatial_overlap_error_undefined_projection(self):
        """Validation: check spatial overlap message when no projection"""
        from natcap.invest import validation

        spec = {
            'raster_a': {
                'type': 'raster',
                'name': 'raster 1',
                'about': 'raster 1',
                'required': True,
            },
            'raster_b': {
                'type': 'raster',
                'name': 'raster 2',
                'about': 'raster 2',
                'required': True,
            }
        }

        driver = gdal.GetDriverByName('GTiff')
        filepath_1 = os.path.join(self.workspace_dir, 'raster_1.tif')
        filepath_2 = os.path.join(self.workspace_dir, 'raster_2.tif')

        raster_1 = driver.Create(filepath_1, 3, 3, 1, gdal.GDT_Int32)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        raster_1.SetProjection(wgs84_srs.ExportToWkt())
        raster_1.SetGeoTransform([1, 1, 0, 1, 0, 1])
        raster_1 = None

        # don't define a projection for the second raster
        driver.Create(filepath_2, 3, 3, 1, gdal.GDT_Int32)

        args = {
            'raster_a': filepath_1,
            'raster_b': filepath_2
        }

        validation_warnings = validation.validate(
            args, spec, {'spatial_keys': list(args.keys()),
                         'different_projections_ok': True})
        expected = [(['raster_b'], validation.MESSAGES['INVALID_PROJECTION'])]
        self.assertEqual(validation_warnings, expected)

    def test_spatial_overlap_error_optional_args(self):
        """Validation: check for spatial mismatch with insufficient args."""
        from natcap.invest import validation

        spec = {
            'raster_a': {
                'type': 'raster',
                'name': 'raster 1',
                'about': 'raster 1',
                'required': True,
            },
            'raster_b': {
                'type': 'raster',
                'name': 'raster 2',
                'about': 'raster 2',
                'required': False,
            },
            'vector_a': {
                'type': 'vector',
                'name': 'vector 1',
                'about': 'vector 1',
                'required': False,
            }
        }

        driver = gdal.GetDriverByName('GTiff')
        filepath_1 = os.path.join(self.workspace_dir, 'raster_1.tif')
        filepath_2 = os.path.join(self.workspace_dir, 'raster_2.tif')

        # Filepaths 1 and 2 do not overlap
        for filepath, geotransform, epsg_code in (
                (filepath_1, [1, 1, 0, 1, 0, 1], 4326),
                (filepath_2, [100, 1, 0, 100, 0, 1], 4326)):
            raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(epsg_code)
            raster.SetProjection(wgs84_srs.ExportToWkt())
            raster.SetGeoTransform(geotransform)
            raster = None

        args = {
            'raster_a': filepath_1,
        }
        # There should not be a spatial overlap check at all
        # when less than 2 of the spatial keys are sufficient.
        validation_warnings = validation.validate(
            args, spec, {'spatial_keys': list(spec.keys()),
                         'different_projections_ok': True})
        self.assertEqual(len(validation_warnings), 0)

        # And even though there are three spatial keys in the spec,
        # Only the ones checked should appear in the validation output
        args = {
            'raster_a': filepath_1,
            'raster_b': filepath_2,
        }
        validation_warnings = validation.validate(
            args, spec, {'spatial_keys': list(spec.keys()),
                         'different_projections_ok': True})
        self.assertEqual(len(validation_warnings), 1)
        formatted_bbox_list = ''  # allows str matching w/o real bbox str
        self.assertTrue(
            validation.MESSAGES['BBOX_NOT_INTERSECT'].format(
                bboxes=formatted_bbox_list) in validation_warnings[0][1])
        self.assertEqual(set(args.keys()), set(validation_warnings[0][0]))

    def test_allow_extra_keys(self):
        """Including extra keys in args that aren't in ARGS_SPEC should work"""
        from natcap.invest import validation

        args = {'a': 'a', 'b': 'b'}
        spec = {
            'a': {
                'type': 'freestyle_string',
                'name': 'a',
                'about': 'a freestyle string',
                'required': True
            }
        }
        message = 'DEBUG:natcap.invest.validation:Provided key b does not exist in ARGS_SPEC'

        with self.assertLogs('natcap.invest.validation', level='DEBUG') as cm:
            validation.validate(args, spec)
        self.assertTrue(message in cm.output)

    def test_check_ratio(self):
        """Validation: test ratio type validation."""
        from natcap.invest import validation
        args = {
            'a': 'xyz',  # not a number
            'b': '1.5',  # too large
            'c': '-1',   # too small
            'd': '0',    # lower bound
            'e': '0.5',  # middle
            'f': '1'     # upper bound
        }
        spec = {name: {'type': 'ratio'} for name in args}

        expected_warnings = [
            (['a'], validation.MESSAGES['NOT_A_NUMBER'].format(value=args['a'])),
            (['b'], validation.MESSAGES['NOT_WITHIN_RANGE'].format(
                value=args['b'], range='[0, 1]')),
            (['c'], validation.MESSAGES['NOT_WITHIN_RANGE'].format(
                value=float(args['c']), range='[0, 1]'))]
        actual_warnings = validation.validate(args, spec)
        for warning in actual_warnings:
            self.assertTrue(warning in expected_warnings)

    def test_check_percent(self):
        """Validation: test percent type validation."""
        from natcap.invest import validation
        args = {
            'a': 'xyz',    # not a number
            'b': '100.5',  # too large
            'c': '-1',     # too small
            'd': '0',      # lower bound
            'e': '55.5',   # middle
            'f': '100'     # upper bound
        }
        spec = {name: {'type': 'percent'} for name in args}

        expected_warnings = [
            (['a'], validation.MESSAGES['NOT_A_NUMBER'].format(value=args['a'])),
            (['b'], validation.MESSAGES['NOT_WITHIN_RANGE'].format(
                value=args['b'], range='[0, 100]')),
            (['c'], validation.MESSAGES['NOT_WITHIN_RANGE'].format(
                value=float(args['c']), range='[0, 100]'))]
        actual_warnings = validation.validate(args, spec)
        for warning in actual_warnings:
            self.assertTrue(warning in expected_warnings)

    def test_check_integer(self):
        """Validation: test integer type validation."""
        from natcap.invest import validation
        args = {
            'a': 'xyz',    # not a number
            'b': '1.5',    # not an integer
            'c': '-1',     # negative integers are ok
            'd': '0'
        }
        spec = {name: {'type': 'integer'} for name in args}

        expected_warnings = [
            (['a'], validation.MESSAGES['NOT_A_NUMBER'].format(value=args['a'])),
            (['b'], validation.MESSAGES['NOT_AN_INTEGER'].format(value=args['b']))]
        actual_warnings = validation.validate(args, spec)
        self.assertEqual(len(actual_warnings), len(expected_warnings))
        for warning in actual_warnings:
            self.assertTrue(warning in expected_warnings)

    def test_get_headers_to_validate(self):
        """Validation: test getting header patterns from a spec."""
        from natcap.invest import validation
        spec = {
            'a': {},
            'foo_[BAR]': {},
            'c': {'required': 'conditional statement'},
            'd': {'required': False}
        }
        patterns = validation.get_headers_to_validate(spec)
        # should only get the patterns that are static and always required
        self.assertEqual(sorted(patterns), ['a'])
