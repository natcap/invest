import importlib
import re
import subprocess
import unittest
import pytest

import pint
from natcap.invest.models import model_id_to_pyname
from natcap.invest import spec
from osgeo import gdal

PLUGIN_URL = 'git+https://github.com/emlys/demo-invest-plugin.git'
PLUGIN_NAME = 'foo-model'


gdal.UseExceptions()
valid_nested_input_types = {
    None: {  # if no parent type (arg is top-level), then all types are valid
        spec.BooleanInput,
        spec.CSVInput,
        spec.DirectoryInput,
        spec.FileInput,
        spec.IntegerInput,
        spec.NumberInput,
        spec.OptionStringInput,
        spec.PercentInput,
        spec.RasterOrVectorInput,
        spec.RatioInput,
        spec.SingleBandRasterInput,
        spec.StringInput,
        spec.VectorInput
    },
    spec.SingleBandRasterInput: {
        spec.IntegerInput,
        spec.NumberInput,
        spec.PercentInput,
        spec.RatioInput
    },
    spec.VectorInput: {
        spec.IntegerInput,
        spec.NumberInput,
        spec.OptionStringInput,
        spec.PercentInput,
        spec.RatioInput,
        spec.StringInput
    },
    spec.CSVInput: {
        spec.BooleanInput,
        spec.IntegerInput,
        spec.NumberInput,
        spec.OptionStringInput,
        spec.PercentInput,
        spec.RasterOrVectorInput,
        spec.RatioInput,
        spec.SingleBandRasterInput,
        spec.StringInput,
        spec.VectorInput
    },
    spec.DirectoryInput: {
        spec.CSVInput,
        spec.DirectoryInput,
        spec.FileInput,
        spec.RasterOrVectorInput,
        spec.SingleBandRasterInput,
        spec.VectorInput
    }
}

valid_nested_output_types = {
    None: {  # if no parent type (arg is top-level), then all types are valid
        spec.CSVOutput,
        spec.DirectoryOutput,
        spec.FileOutput,
        spec.IntegerOutput,
        spec.NumberOutput,
        spec.OptionStringOutput,
        spec.PercentOutput,
        spec.RatioOutput,
        spec.SingleBandRasterOutput,
        spec.StringOutput,
        spec.VectorOutput
    },
    spec.SingleBandRasterOutput: {
        spec.IntegerOutput,
        spec.NumberOutput,
        spec.PercentOutput,
        spec.RatioOutput
    },
    spec.VectorOutput: {
        spec.IntegerOutput,
        spec.NumberOutput,
        spec.OptionStringOutput,
        spec.PercentOutput,
        spec.RatioOutput,
        spec.StringOutput
    },
    spec.CSVOutput: {
        spec.IntegerOutput,
        spec.NumberOutput,
        spec.OptionStringOutput,
        spec.PercentOutput,
        spec.RatioOutput,
        spec.SingleBandRasterOutput,
        spec.StringOutput,
        spec.VectorOutput
    },
    spec.DirectoryOutput: {
        spec.CSVOutput,
        spec.DirectoryOutput,
        spec.FileOutput,
        spec.SingleBandRasterOutput,
        spec.VectorOutput
    }
}


class ValidateModelSpecs(unittest.TestCase):
    """Validate the contract for patterns and types in MODEL_SPEC."""

    def test_model_specs_are_valid(self):
        """MODEL_SPEC: test each spec meets the expected pattern."""

        required_keys = {'model_id', 'model_title', 'userguide',
                         'aliases', 'inputs', 'input_field_order', 'outputs'}
        for model_id, pyname in model_id_to_pyname.items():
            model = importlib.import_module(pyname)

            # Validate top-level keys are correct
            with self.subTest(pyname):
                self.assertTrue(
                    required_keys.issubset(set(dir(model.MODEL_SPEC))),
                    ("Required key(s) missing from MODEL_SPEC: "
                     f"{set(required_keys).difference(set(dir(model.MODEL_SPEC)))}"))

                self.assertIsInstance(model.MODEL_SPEC.input_field_order, list)
                found_keys = set()
                for group in model.MODEL_SPEC.input_field_order:
                    self.assertIsInstance(group, list)
                    for key in group:
                        self.assertIsInstance(key, str)
                        self.assertNotIn(key, found_keys)
                        found_keys.add(key)
                for arg_spec in model.MODEL_SPEC.inputs:
                    if arg_spec.hidden is True:
                        found_keys.add(arg_spec.id)
                self.assertEqual(found_keys, set([s.id for s in model.MODEL_SPEC.inputs]))

            # validate that each arg meets the expected pattern
            # save up errors to report at the end
            for arg_spec in model.MODEL_SPEC.inputs:
                # the top level should have 'name' and 'about' attrs
                # but they aren't required at nested levels
                self.validate_args(arg_spec, f'{model_id}.inputs.{arg_spec.id}')

            for output_spec in model.MODEL_SPEC.outputs:
                self.validate_output(output_spec, f'{model_id}.outputs.{output_spec.id}')

    def validate_output(self, output_spec, key, parent_type=None):
        """
        Recursively validate nested output specs against the output spec standard.

        Args:
            output_spec (dict): any nested output spec component of a MODEL_SPEC
            key (str): key to identify the spec by in error messages
            parent_type (str): the type of this output's parent output (or None if
                no parent).

        Returns:
            None

        Raises:
            AssertionError if the output spec violates the standard
        """
        with self.subTest(output=key):
            # if parent_type is None:  # all top-level args must have these attrs
            #     for attr in ['about']:
            #         self.assertIn(attr, spec)
            attrs = set(dir(output_spec)) - set(dir(object()))

            t = type(output_spec)
            self.assertIn(t, valid_nested_output_types[parent_type])

            if t is spec.NumberOutput:
                # number type should have a units property
                self.assertTrue(hasattr(output_spec, 'units'))
                # Undefined units should use the custom u.none unit
                self.assertIsInstance(output_spec.units, pint.Unit)

            elif t is spec.SingleBandRasterOutput:
                self.assertTrue(hasattr(output_spec, 'data_type'))
                self.assertTrue(hasattr(output_spec, 'units'))

            elif t is spec.VectorOutput:
                # vector type should have:
                # - a fields property that maps each field header to a nested
                #   type dictionary describing the data in that field
                # - a geometry_types property: the set of valid geometry types
                self.assertTrue(hasattr(output_spec, 'fields'))
                for field in output_spec.fields:
                    self.validate_output(
                        field,
                        f'{key}.fields.{field}',
                        parent_type=t)

                self.assertTrue(hasattr(output_spec, 'geometry_types'))
                self.assertIsInstance(output_spec.geometry_types, set)

            elif t is spec.CSVOutput:
                # csv type may have a columns property.
                # the columns property maps each expected column header
                # name/pattern to a nested type dictionary describing the data
                # in that column. may be absent if the table structure
                # is too complex to describe this way.
                self.assertTrue(hasattr(output_spec, 'columns'))
                for column in output_spec.columns:
                    self.validate_output(
                        column,
                        f'{key}.columns.{column}',
                        parent_type=t)
                if output_spec.index_col:
                    self.assertIn(output_spec.index_col, [s.id for s in output_spec.columns])

            elif t is spec.DirectoryOutput:
                # directory type should have a contents property that maps each
                # expected path name/pattern within the directory to a nested
                # type dictionary describing the data at that filepath
                self.assertTrue(hasattr(output_spec, 'contents'))
                for path in output_spec.contents:
                    self.validate_output(
                        path,
                        f'{key}.contents.{path}',
                        parent_type=t)

            elif t is spec.OptionStringOutput:
                    # option_string type should have an options property that
                    # describes the valid options
                    self.assertTrue(hasattr(output_spec, 'options'))
                    self.assertIsInstance(output_spec.options, dict)
                    for option, description in output_spec.options.items():
                        self.assertTrue(
                            isinstance(option, str) or
                            isinstance(option, int))

            elif t is spec.FileOutput:
                pass

            # iterate over the remaining attributes
            # type-specific ones have been removed by this point
            if output_spec.about:
                self.assertIsInstance(output_spec.about, str)
            if output_spec.created_if:
                # should be an arg key indicating that the output is
                # created if that arg is provided or checked
                self.assertTrue(
                    isinstance(output_spec.created_if, str) or
                    isinstance(output_spec.created_if, bool))

    def validate_args(self, arg, name, parent_type=None):
        """
        Recursively validate nested args against the arg spec standard.

        Args:
            arg (dict): any nested arg component of an MODEL_SPEC
            name (str): name to use in error messages to identify the arg
            parent_type (str): the type of this arg's parent arg (or None if
                no parent).

        Returns:
            None

        Raises:
            AssertionError if the arg violates the standard
        """
        with self.subTest(nested_arg_name=name):
            if parent_type is None:  # all top-level args must have these attrs
                for attr in ['name', 'about']:
                    self.assertTrue(hasattr(arg, attr))

            attrs = set(dir(arg))

            t = type(arg)
            self.assertIn(t, valid_nested_input_types[parent_type])

            if t is spec.OptionStringInput:
                # option_string type should have an options property that
                # describes the valid options
                self.assertTrue(hasattr(arg, 'options'))
                # May be a list or dict because some option sets are self
                # explanatory and others need a description
                self.assertIsInstance(arg.options, dict)
                for key, val in arg.options.items():
                    self.assertTrue(
                        isinstance(key, str) or
                        isinstance(key, int))
                    self.assertIsInstance(val, dict)
                    # top-level option_string args are shown as dropdowns
                    # so each option needs a display name
                    # an additional description is optional
                    if parent_type is None:
                        self.assertTrue(
                            set(val.keys()) == {'display_name'} or
                            set(val.keys()) == {
                                'display_name', 'description'})
                    # option_strings within a CSV or vector don't get a
                    # display name. the user has to enter the key.
                    else:
                        self.assertEqual(set(val.keys()), {'description'})

                    if 'display_name' in val:
                        self.assertIsInstance(val['display_name'], str)
                    if 'description' in val:
                        self.assertIsInstance(val['description'], str)

                attrs.remove('options')

            elif t is spec.StringInput:
                # freestyle_string may optionally have a regexp attribute
                # this is a regular expression that the string must match
                if arg.regexp:
                    self.assertIsInstance(arg.regexp, str)
                    re.compile(arg.regexp)  # should be regex compilable
                    attrs.remove('regexp')

            elif t is spec.NumberInput:
                # number type should have a units property
                self.assertTrue(hasattr(arg, 'units'))
                # Undefined units should use the custom u.none unit
                self.assertIsInstance(arg.units, pint.Unit)

                # number type may optionally have an 'expression' attribute
                # this is a string expression to be evaluated with the
                # intent of determining that the value is within a range.
                # The expression must contain the string ``value``, which
                # will represent the user-provided value (after it has been
                # cast to a float).  Example: "(value >= 0) & (value <= 1)"
                if arg.expression:
                    self.assertIsInstance(arg.expression, str)

            elif t is spec.SingleBandRasterInput:
                self.assertTrue(hasattr(arg, 'data_type'))
                self.assertTrue(hasattr(arg, 'units'))

                # may optionally have a 'projected' attribute that says
                # whether the raster must be linearly projected
                if arg.projected is not None:
                    self.assertIsInstance(arg.projected, bool)
                    attrs.remove('projected')
                # if 'projected' is True, may also have a 'projection_units'
                # attribute saying the expected linear projection unit
                if arg.projection_units:
                    # doesn't make sense to have projection units unless
                    # projected is True
                    self.assertTrue(arg.projected)
                    self.assertIsInstance(
                        arg.projection_units, pint.Unit)
                    attrs.remove('projection_units')

            elif t is spec.VectorInput:
                # vector type should have:
                # - a fields property that maps each field header to a nested
                #   type dictionary describing the data in that field
                # - a geometry_types property: the set of valid geometry types
                self.assertTrue(hasattr(arg, 'fields'))
                for field in arg.fields:
                    self.validate_args(
                        field,
                        f'{name}.fields.{field}',
                        parent_type=t)

                self.assertTrue(hasattr(arg, 'geometry_types'))
                self.assertIsInstance(arg.geometry_types, set)

                attrs.remove('fields')
                attrs.remove('geometry_types')

                # may optionally have a 'projected' attribute that says
                # whether the vector must be linearly projected
                if arg.projected is not None:
                    self.assertIsInstance(arg.projected, bool)
                    attrs.remove('projected')
                # if 'projected' is True, may also have a 'projection_units'
                # attribute saying the expected linear projection unit
                if arg.projection_units:
                    # doesn't make sense to have projection units unless
                    # projected is True
                    self.assertTrue(arg.projected)
                    self.assertIsInstance(
                        arg.projection_units, pint.Unit)
                    attrs.remove('projection_units')

            elif t is spec.CSVInput:
                # csv type should have a rows property, columns property, or
                # neither. rows or columns properties map each expected header
                # name/pattern to a nested type dictionary describing the data
                # in that row/column. may have neither if the table structure
                # is too complex to describe this way.
                has_rows = bool(arg.rows)
                has_cols = bool(arg.columns)
                # should not have both
                self.assertFalse(has_rows and has_cols)

                if has_cols or has_rows:
                    direction = 'rows' if has_rows else 'columns'
                    headers = arg.rows if has_rows else arg.columns

                    for header in headers:
                        self.validate_args(
                            header,
                            f'{name}.{direction}.{header}',
                            parent_type=t)

                if arg.index_col:
                    self.assertIn(arg.index_col, [s.id for s in arg.columns])

            elif t is spec.DirectoryInput:
                # directory type should have a contents property that maps each
                # expected path name/pattern within the directory to a nested
                # type dictionary describing the data at that filepath
                self.assertTrue(hasattr(arg, 'contents'))
                for path in arg.contents:
                    self.validate_args(
                        path,
                        f'{name}.contents.{path}',
                        parent_type=t)
                attrs.remove('contents')

                # may optionally have a 'permissions' attribute, which is a
                # string of the unix-style directory permissions e.g. 'rwx'
                if arg.permissions:
                    self.validate_permissions_value(arg.permissions)
                    attrs.remove('permissions')
                # may optionally have an 'must_exist' attribute, which says
                # whether the directory must already exist
                # this defaults to True
                if arg.must_exist is not None:
                    self.assertIsInstance(arg.must_exist, bool)
                    attrs.remove('must_exist')

            elif t is spec.FileInput:
                # file type may optionally have a 'permissions' attribute
                # this is a string listing the permissions e.g. 'rwx'
                if arg.permissions:
                    self.validate_permissions_value(arg.permissions)

            # iterate over the remaining attributes
            # type-specific ones have been removed by this point
            if arg.name:
                self.assertIsInstance(arg.name, str)
            if arg.about:
                self.assertIsInstance(arg.about, str)
            # required value may be True, False, or a string that can be
            # parsed as a python statement that evaluates to True or False
            self.assertTrue(isinstance(arg.required, bool) or
                            isinstance(arg.required, str))
            if arg.allowed:
                self.assertIn(type(arg.allowed), {str, bool})

    def validate_permissions_value(self, permissions):
        """
        Validate an rwx-style permissions string.

        Args:
            permissions (str): a string to validate as permissions

        Returns:
            None

        Raises:
            AssertionError if `permissions` isn't a string, if it's
            an empty string, if it has any letters besides 'r', 'w', 'x',
            or if it has any of those letters more than once
        """

        self.assertIsInstance(permissions, str)
        self.assertTrue(len(permissions) > 0)
        valid_letters = {'r', 'w', 'x'}
        for letter in permissions:
            self.assertIn(letter, valid_letters)
            # should only have a letter once
            valid_letters.remove(letter)

    def test_model_specs_serialize(self):
        """MODEL_SPEC: test each arg spec can serialize to JSON."""
        from natcap.invest import spec

        for pyname in model_id_to_pyname.values():
            model = importlib.import_module(pyname)
            model.MODEL_SPEC.to_json()


@pytest.mark.skip(reason="Possible race condition of plugin not being uninstalled before other tests are run.")
class PluginTests(unittest.TestCase):
    """Tests for natcap.invest plugins."""

    def tearDown(self):
        subprocess.run(['pip', 'uninstall', '--yes', PLUGIN_NAME])

    def test_plugin(self):
        """natcap.invest locates plugin as a namespace package."""
        from natcap.invest import models
        self.assertNotIn('foo', models.model_id_to_spec.keys())
        subprocess.run(['pip', 'install', '--no-deps', PLUGIN_URL])
        models = importlib.reload(models)
        self.assertIn('foo', models.model_id_to_spec.keys())


if __name__ == '__main__':
    unittest.main()
