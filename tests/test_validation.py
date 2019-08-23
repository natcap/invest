import tempfile
import unittest
import os
import shutil

from osgeo import gdal, osr, ogr
import pandas


class ValidatorTest(unittest.TestCase):
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

    def test_invalid_return_value(self):
        """Validation: check for error when the return value type is wrong."""
        from natcap.invest import validation

        for invalid_value in (1, True, None):
            @validation.invest_validator
            def validate(args, limit_to=None):
                return invalid_value

            with self.assertRaises(AssertionError):
                validate({})

    def test_invalid_keys_iterable(self):
        """Validation: check for error when return keys not an iterable."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return [('a', 'error 1')]

        with self.assertRaises(AssertionError):
            validate({'a': 'foo'})

    def test_return_keys_in_args(self):
        """Validation: check for error when return keys not all in args."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return [(('a',), 'error 1')]

        with self.assertRaises(AssertionError):
            validate({})

    def test_error_string_wrong_type(self):
        """Validation: check for error when error message not a string."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return [(('a',), 1234)]

        with self.assertRaises(AssertionError):
            validate({'a': 'foo'})

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
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return []

        validation_errors = validate({'n_workers': 1.5})
        self.assertEqual(len(validation_errors), 1)
        self.assertTrue(validation_errors[0][0] == ['n_workers'])
        self.assertTrue('must be an integer' in validation_errors[0][1])

class DirectoryValidation(unittest.TestCase):
    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir)

    def test_exists(self):
        """Validation: when a folder must exist and does."""
        from natcap.invest import validation

        self.assertEqual(None, validation.check_directory(
            self.workspace_dir, exists=True))

    def test_not_exists(self):
        """Validation: when a folder must exist but does not."""
        from natcap.invest import validation

        dirpath = os.path.join(self.workspace_dir, 'nonexistent_dir')
        validation_warning = validation.check_directory(
            dirpath, exists=True)
        self.assertTrue('not found' in validation_warning)

    def test_file(self):
        """Validation: when a file is given to folder validation."""
        from natcap.invest import validation

        filepath = os.path.join(self.workspace_dir, 'some_file.txt')
        with open(filepath, 'w') as opened_file:
            opened_file.write('the text itself does not matter.')

        validation_warning = validation.check_directory(
            filepath, exists=True)

    def test_valid_permissions(self):
        """Validation: folder permissions."""
        from natcap.invest import validation

        self.assertEquals(None, validation.check_directory(
            self.workspace_dir, exists=True, permissions='rwx'))


class FileValidation(unittest.TestCase):
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

        self.assertTrue('File not found' in error_msg)


class RasterValidation(unittest.TestCase):
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
        self.assertTrue('not found' in error_msg)

    def test_invalid_raster(self):
        """Validation: test when a raster format is invalid."""
        from natcap.invest import validation

        filepath = os.path.join(self.workspace_dir, 'file.txt')
        with open(filepath, 'w') as bad_raster:
            bad_raster.write('not a raster')

        error_msg = validation.check_raster(filepath)
        self.assertTrue('could not be opened as a GDAL raster' in error_msg)

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
        self.assertTrue('must be projected in linear units' in error_msg)

    def test_raster_projected_in_m(self):
        """Validation: test when a raster is projected in meters."""
        from natcap.invest import validation

        # Use EPSG:32731  # WGS84 / UTM zone 31s
        driver = gdal.GetDriverByName('GTiff')
        filepath = os.path.join(self.workspace_dir, 'raster.tif')
        raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
        meters_srs = osr.SpatialReference()
        meters_srs.ImportFromEPSG(32731)
        raster.SetProjection(meters_srs.ExportToWkt())
        raster = None

        for unit in ('m', 'meter', 'metre', 'meters', 'metres'):
            error_msg = validation.check_raster(
                filepath, projected=True, projection_units=unit)
            self.assertEqual(error_msg, None)

        # Check error message when we validate that the raster should be
        # projected in feet.
        error_msg = validation.check_raster(
            filepath, projected=True, projection_units='feet')
        self.assertTrue('projected in feet' in error_msg)

    def test_raster_incorrect_units(self):
        """Validation: test when a raster projection has wrong units."""
        from natcap.invest import validation

        # Use EPSG:32066  # NAD27 / BLM 16N (in US Feet)
        driver = gdal.GetDriverByName('GTiff')
        filepath = os.path.join(self.workspace_dir, 'raster.tif')
        raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(32066)
        raster.SetProjection(wgs84_srs.ExportToWkt())
        raster = None

        error_msg = validation.check_raster(
            filepath, projected=True, projection_units='m')
        self.assertTrue('must be projected in meters' in error_msg)


class VectorValidation(unittest.TestCase):
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
        error_msg = validation.check_raster(filepath)
        self.assertTrue('not found' in error_msg)

    def test_invalid_vector(self):
        """Validation: test when a vector's format is invalid."""
        from natcap.invest import validation

        filepath = os.path.join(self.workspace_dir, 'file.txt')
        with open(filepath, 'w') as bad_vector:
            bad_vector.write('not a vector')

        error_msg = validation.check_vector(filepath)
        self.assertTrue('could not be opened as a GDAL vector' in error_msg)

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
            filepath, required_fields=['col_a', 'COL_B', 'col_c'])
        self.assertTrue('Fields are missing' in error_msg)
        self.assertTrue('col_c'.upper() in error_msg)

    def test_vector_projected_in_m(self):
        """Validation: test that a vector's projection has expected units."""
        from natcap.invest import validation

        driver = gdal.GetDriverByName('GPKG')
        filepath = os.path.join(self.workspace_dir, 'vector.gpkg')
        vector = driver.Create(filepath, 0, 0, 0, gdal.GDT_Unknown)
        meters_srs = osr.SpatialReference()
        meters_srs.ImportFromEPSG(32731)
        layer = vector.CreateLayer('sample_layer', meters_srs, ogr.wkbUnknown)

        layer = None
        vector = None

        error_msg = validation.check_vector(
            filepath, projected=True, projection_units='feet')
        self.assertTrue('projected in feet' in error_msg)

        self.assertEqual(None, validation.check_vector(
            filepath, projected=True, projection_units='m'))


class FreestyleStringValidation(unittest.TestCase):
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

        self.assertEqual(None, validation.check_freestyle_string(
            1.234, regexp={'pattern': '^1.[0-9]+$', 'case_sensitive': True}))

        error_msg = validation.check_freestyle_string(
            'foobar12', regexp={'pattern': '^[a-zA-Z]+$',
                                'case_sensitive':True})
        self.assertTrue('did not match expected pattern' in error_msg)


class OptionStringValidation(unittest.TestCase):
    def test_valid_option(self):
        """Validation: test that a string is a valid option."""
        from natcap.invest import validation
        self.assertEqual(None, validation.check_option_string(
            'foo', options=['foo', 'bar', 'Baz']))

    def test_invalid_option(self):
        """Validation: test when a string is not a valid option."""
        from natcap.invest import validation
        error_msg = validation.check_option_string(
            'FOO', options=['foo', 'bar', 'Baz'])
        self.assertTrue('must be one of' in error_msg)


class NumberValidation(unittest.TestCase):
    def test_string(self):
        """Validation: test when a string is not a number."""
        from natcap.invest import validation
        error_msg = validation.check_number('this is a string')
        self.assertTrue('could not be interpreted as a number' in error_msg)

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
        error_msg = validation.check_number(
            35, 'value < 0')
        self.assertTrue('does not meet condition' in error_msg)


class BooleanValidation(unittest.TestCase):
    def test_actual_bool(self):
        """Validation: test when boolean type objects are passed."""
        from natcap.invest import validation
        self.assertEqual(None, validation.check_boolean(True))
        self.assertEqual(None, validation.check_boolean(False))

    def test_string_boolean(self):
        """Validation: test when valid strings are passed."""
        from natcap.invest import validation
        self.assertEqual(None, validation.check_boolean('True'))
        self.assertEqual(None, validation.check_boolean('False'))
        self.assertEqual(None, validation.check_boolean('true'))
        self.assertEqual(None, validation.check_boolean('false'))
        self.assertEqual(None, validation.check_boolean('TRUE'))
        self.assertEqual(None, validation.check_boolean('FALSE'))

    def test_invalid_string(self):
        """Validation: test when invalid strings are passed."""
        from natcap.invest import validation
        error_msg = validation.check_boolean('not clear')
        self.assertTrue("must be one of 'True' or 'False'" in error_msg)


class CSVValidation(unittest.TestCase):
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
        self.assertTrue('not found' in error_msg)

    def test_csv_fieldnames(self):
        """Validation: test that we can check fieldnames in a CSV."""
        from natcap.invest import validation

        df = pandas.DataFrame([
            {'foo': 1, 'bar': 2, 'baz': 3},
            {'foo': 2, 'bar': 3, 'baz': 4},
            {'foo': 3, 'bar': 4, 'baz': 5}])

        target_file = os.path.join(self.workspace_dir, 'test.csv')
        df.to_csv(target_file)

        self.assertEquals(None, validation.check_csv(
            target_file, required_fields=['foo', 'bar']))

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
            target_file, required_fields=['field_a'])
        self.assertTrue('missing from this table' in error_msg)

    def test_excel_missing_fieldnames(self):
        """Validation: test that we can check missing fieldnames in excel."""
        from natcap.invest import validation

        df = pandas.DataFrame([
            {'foo': 1, 'bar': 2, 'baz': 3},
            {'foo': 2, 'bar': 3, 'baz': 4},
            {'foo': 3, 'bar': 4, 'baz': 5}])

        target_file = os.path.join(self.workspace_dir, 'test.xls')
        df.to_excel(target_file)

        error_msg = validation.check_csv(
            target_file, required_fields=['field_a'], excel_ok=True)
        self.assertTrue('missing from this table' in error_msg)


class TestValidationFromSpec(unittest.TestCase):
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
        }

        args = {
            "number_a": 123,
            "number_b": 456,
        }
        validation_warnings = validation.validate(args, spec)
        self.assertEquals(validation_warnings, [
            (['number_c'], 'Key is missing from the args dict'),
            (['number_d'], 'Key is missing from the args dict'),
        ])

        args = {
            "number_a": 123,
            "number_b": 456,
            "number_c": 1,
            "number_d": 3,
            "number_e": 4,
        }
        self.assertEquals([], validation.validate(args, spec))

    def test_requirement_missing(self):
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
        self.assertEquals(
            [(['number_a'], 'Key is missing from the args dict')],
            validation.validate(args, spec))

    def test_requirement_no_value(self):
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
        self.assertEquals(
            [(['number_a'], 'Key is required but has no value')],
            validation.validate(args, spec))

        args = {'number_a': None}
        self.assertEquals(
            [(['number_a'], 'Key is required but has no value')],
            validation.validate(args, spec))

    def test_invalid_value(self):
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
        self.assertEquals(
            [(['number_a'], 'Value could not be interpreted as a number')],
            validation.validate(args, spec))

    def test_conditionally_required_no_value(self):
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

        self.assertEquals(
            [(['string_a'], 'Key is required but has no value')],
            validation.validate(args, spec))
