# encoding=UTF-8
import tempfile
import unittest
from unittest.mock import Mock
import os
import shutil
import string

from osgeo import gdal, osr, ogr
import pandas


class SpatialOverlapTest(unittest.TestCase):
    def setUp(self):
        """Create a new workspace to use for each test."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace created for this test."""
        shutil.rmtree(self.workspace_dir)

    def test_no_overlap_no_reference(self):
        """Validation: verify lack of overlap without a reference."""
        from natcap.invest import validation

        driver = gdal.GetDriverByName('GTiff')
        filepath_1 = os.path.join(self.workspace_dir, 'raster_1.tif')
        filepath_2 = os.path.join(self.workspace_dir, 'raster_2.tif')

        for filepath, geotransform in (
                (filepath_1, [1, 1, 0, 1, 0, 1]),
                (filepath_2, [100, 1, 0, 100, 0, 1])):
            raster = driver.Create(filepath, 3, 3, 1, gdal.GDT_Int32)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            raster.SetProjection(wgs84_srs.ExportToWkt())
            raster.SetGeoTransform(geotransform)
            raster = None

        error_msg = validation.check_spatial_overlap([filepath_1, filepath_2])
        self.assertTrue('Bounding boxes do not intersect' in error_msg)

    def test_overlap_no_reference(self):
        """Validation: verify overlap without a reference."""
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

    def test_no_overlap_with_reference(self):
        """Validation: verify lack of overlap given reference projection."""
        from natcap.invest import validation

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

        error_msg = validation.check_spatial_overlap(
            [filepath_1, filepath_2, reference_filepath],
            vector_srs.ExportToWkt())
        self.assertTrue('Bounding boxes do not intersect' in error_msg)


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
        from natcap.invest import validation

        args_spec = {
            'n_workers': validation.N_WORKERS_SPEC,
        }

        @validation.invest_validator
        def validate(args, limit_to=None):
            return validation.validate(args, args_spec)

        validation_errors = validate({'n_workers': 'not a number'})
        self.assertEqual(len(validation_errors), 1)
        self.assertTrue(validation_errors[0][0] == ['n_workers'])
        self.assertTrue('could not be interpreted as a number'
                        in validation_errors[0][1])


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

        self.assertEqual(None, validation.check_directory(
            self.workspace_dir, exists=True, permissions='rwx'))

    def test_workspace_not_exists(self):
        """Validation: when a folder's parent must exist with permissions."""
        from natcap.invest import validation

        dirpath = 'foo'
        new_dir = os.path.join(self.workspace_dir, dirpath)

        self.assertEqual(None, validation.check_directory(
            new_dir, exists=False, permissions='rwx'))


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
        error_msg = validation.check_vector(filepath)
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
                                'case_sensitive': True})
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
            35, 'float(value) < 0')
        self.assertTrue('does not meet condition' in error_msg)

    def test_expression_failure_string(self):
        """Validation: test when string value does not meet the expression."""
        from natcap.invest import validation
        error_msg = validation.check_number(
            "35", 'int(value) < 0')
        self.assertTrue('does not meet condition' in error_msg)


class BooleanValidation(unittest.TestCase):
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
        error_msg = validation.check_boolean('not clear')
        self.assertTrue(isinstance(error_msg, str))
        self.assertTrue('must be either True or False' in error_msg)


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

        self.assertEqual(None, validation.check_csv(
            target_file, required_fields=['foo', 'bar']))

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

    def test_csv_not_utf_8(self):
        """Validation: test that non-UTF8 CSVs can validate."""
        from natcap.invest import validation

        df = pandas.DataFrame([
            {'fЮЮ': 1, 'bar': 2, 'baz': 3},  # special characters here.
            {'foo': 2, 'bar': 3, 'baz': 4},
            {'foo': 3, 'bar': 4, 'baz': 5}])

        target_file = os.path.join(self.workspace_dir, 'test.csv')

        # Save the CSV with the Windows Cyrillic codepage.
        # https://en.wikipedia.org/wiki/ISO/IEC_8859-5
        df.to_csv(target_file, encoding='iso8859_5')

        error_msg = validation.check_csv(target_file)
        self.assertEquals(error_msg, None)

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
            target_file, required_fields=['field_a'], excel_ok=True)
        self.assertTrue('could not be opened as a CSV or Excel file' in
                        error_msg)

        error_msg = validation.check_csv(
            target_file, required_fields=['field_a'], excel_ok=False)
        self.assertTrue('could not be opened as a CSV' in error_msg)


class TestValidationFromSpec(unittest.TestCase):
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
        self.assertEqual([], validation.validate(args, spec))

        args = {
            "number_a": 123,
        }
        validation_warnings = validation.validate(args, spec)
        self.assertEqual(sorted(validation_warnings), [
            (['number_f'], 'Key is missing from the args dict')
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
            [(['number_a'], 'Key is missing from the args dict')],
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
            [(['number_a'], 'Input is required but has no value')],
            validation.validate(args, spec))

        args = {'number_a': None}
        self.assertEqual(
            [(['number_a'], 'Input is required but has no value')],
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
            [(['number_a'], ("Value 'not a number' could not be interpreted "
                             "as a number"))],
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
            [(['string_a'], 'Key is required but has no value')],
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
                "validation_options": {
                    "options": ['AAA', 'BBB']
                }
            }
        }

        args = {'string_a': "ZZZ", "number_a": 1}

        self.assertEqual(
            [(['string_a'], "Value must be one of: ['AAA', 'BBB']")],
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
            [(['number_a'], 'An unexpected error occurred in validation')])

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
            [(['arg_J'], 'Key is missing from the args dict')],
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
        self.assertTrue('Bounding boxes do not intersect' in
                        validation_warnings[0][1])

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
        self.assertTrue('Bounding boxes do not intersect' in
                        validation_warnings[0][1])
        self.assertEqual(set(args.keys()), set(validation_warnings[0][0]))
