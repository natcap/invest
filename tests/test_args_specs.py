import importlib
import re
import unittest

from natcap.invest.model_metadata import MODEL_METADATA
import pint


valid_nested_types = {
    None: {  # if no parent type (arg is top-level), then all types are valid
        'boolean',
        'integer',
        'csv',
        'directory',
        'file',
        'freestyle_string',
        'number',
        'option_string',
        'percent',
        'raster',
        'ratio',
        'vector',
    },
    'raster': {'integer', 'number', 'ratio'},
    'vector': {
        'integer',
        'freestyle_string',
        'number',
        'option_string',
        'percent',
        'ratio'},
    'csv': {
        'boolean',
        'integer',
        'freestyle_string',
        'number',
        'option_string',
        'percent',
        'raster',
        'ratio',
        'vector'},
    'directory': {'csv', 'directory', 'file', 'raster', 'vector'}
}


class ValidateArgsSpecs(unittest.TestCase):
    """Validate the contract for patterns and types in ARGS_SPEC."""

    def test_model_specs_are_valid(self):
        """ARGS_SPEC: test each spec meets the expected pattern."""

        required_keys = {'model_name', 'pyname', 'userguide', 'args'}
        optional_spatial_key = 'args_with_spatial_overlap'
        for model_name, metadata in MODEL_METADATA.items():
            # metadata is a collections.namedtuple, fields accessible by name
            model = importlib.import_module(metadata.pyname)

            # Validate top-level keys are correct
            with self.subTest(metadata.pyname):
                self.assertTrue(required_keys.issubset(model.ARGS_SPEC))
                extra_keys = set(model.ARGS_SPEC).difference(required_keys)
                if (extra_keys):
                    self.assertEqual(extra_keys, set([optional_spatial_key]))
                    self.assertTrue(
                        set(model.ARGS_SPEC[optional_spatial_key]).issubset(
                            {'spatial_keys', 'different_projections_ok'}))

            # validate that each arg meets the expected pattern
            # save up errors to report at the end
            for key, arg in model.ARGS_SPEC['args'].items():
                # the top level should have 'name' and 'about' attrs
                # but they aren't required at nested levels
                self.validate(arg, f'{model_name}.{key}')

    def validate(self, arg, name, parent_type=None, is_pattern=False):
        """
        Recursively validate nested args against the ARGS_SPEC standard.

        Args:
            arg (dict): any nested arg component of an ARGS_SPEC
            name (str): name to use in error messages to identify the arg
            parent_type (str): the type of this arg's parent arg (or None if
                no parent).
            is_pattern (bool): if True, the arg is validated as a pattern (such
                as for user-defined  CSV headers, vector fields, or directory
                paths).

        Returns:
            None

        Raises:
            AssertionError if the arg violates the standard
        """
        with self.subTest(nested_arg_name=name):
            if parent_type is None:  # all top-level args must have these attrs
                for attr in ['name', 'about']:
                    self.assertTrue(attr in arg)

            # arg['type'] can be either a string or a set of strings
            types = arg['type'] if isinstance(
                arg['type'], set) else [arg['type']]
            attrs = set(arg.keys())
            for t in types:
                self.assertTrue(t in valid_nested_types[parent_type])

                if t == 'option_string':
                    # option_string type should have an options property that
                    # describes the valid options
                    self.assertTrue('options' in arg)
                    # May be a list or dict because some option sets are self
                    # explanatory and others need a description
                    self.assertTrue(isinstance(arg['options'], dict))
                    for key, val in arg['options'].items():
                        self.assertTrue(
                            isinstance(key, str) or
                            isinstance(key, int))
                        self.assertTrue(isinstance(val, dict))
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
                            self.assertTrue(isinstance(val['display_name'], str))
                        if 'description' in val:
                            self.assertTrue(isinstance(val['description'], str))

                    attrs.remove('options')

                elif t == 'freestyle_string':
                    # freestyle_string may optionally have a regexp attribute
                    # this is a regular expression that the string must match
                    if 'regexp' in arg:
                        self.assertTrue(isinstance(arg['regexp'], str))
                        re.compile(arg['regexp'])  # should be regex compilable
                        attrs.remove('regexp')

                elif t == 'number':
                    # number type should have a units property
                    self.assertTrue('units' in arg)
                    # Undefined units should use the custom u.none unit
                    self.assertTrue(isinstance(arg['units'], pint.Unit))
                    attrs.remove('units')

                    # number type may optionally have an 'expression' attribute
                    # this is a string expression to be evaluated with the
                    # intent of determining that the value is within a range.
                    # The expression must contain the string ``value``, which
                    # will represent the user-provided value (after it has been
                    # cast to a float).  Example: "(value >= 0) & (value <= 1)"
                    if 'expression' in arg:
                        self.assertTrue(isinstance(arg['expression'], str))
                        attrs.remove('expression')

                elif t == 'raster':
                    # raster type should have a bands property that maps each band
                    # index to a nested type dictionary describing the band's data
                    self.assertTrue('bands' in arg)
                    self.assertTrue(isinstance(arg['bands'], dict))
                    for band in arg['bands']:
                        self.assertTrue(isinstance(band, int))
                        self.validate(
                            arg['bands'][band],
                            f'{name}.bands.{band}',
                            parent_type=t)
                    attrs.remove('bands')

                    # may optionally have a 'projected' attribute that says
                    # whether the raster must be linearly projected
                    if 'projected' in arg:
                        self.assertTrue(isinstance(arg['projected'], bool))
                        attrs.remove('projected')
                    # if 'projected' is True, may also have a 'projection_units'
                    # attribute saying the expected linear projection unit
                    if 'projection_units' in arg:
                        # doesn't make sense to have projection units unless
                        # projected is True
                        self.assertTrue(arg['projected'])
                        self.assertTrue(
                            isinstance(arg['projection_units'], pint.Unit))
                        attrs.remove('projection_units')

                elif t == 'vector':
                    # vector type should have:
                    # - a fields property that maps each field header to a nested
                    #   type dictionary describing the data in that field
                    # - a geometries property: the set of valid geometry types
                    self.assertTrue('fields' in arg)
                    self.assertTrue(isinstance(arg['fields'], dict))
                    for field in arg['fields']:
                        self.assertTrue(isinstance(field, str))
                        self.validate(
                            arg['fields'][field],
                            f'{name}.fields.{field}',
                            parent_type=t)

                    self.assertTrue('geometries' in arg)
                    self.assertTrue(isinstance(arg['geometries'], set))

                    attrs.remove('fields')
                    attrs.remove('geometries')

                    # may optionally have a 'projected' attribute that says
                    # whether the vector must be linearly projected
                    if 'projected' in arg:
                        self.assertTrue(isinstance(arg['projected'], bool))
                        attrs.remove('projected')
                    # if 'projected' is True, may also have a 'projection_units'
                    # attribute saying the expected linear projection unit
                    if 'projection_units' in arg:
                        # doesn't make sense to have projection units unless
                        # projected is True
                        self.assertTrue(arg['projected'])
                        self.assertTrue(
                            isinstance(arg['projection_units'], pint.Unit))
                        attrs.remove('projection_units')

                elif t == 'csv':
                    # csv type should have a rows property, columns property, or
                    # neither. rows or columns properties map each expected header
                    # name/pattern to a nested type dictionary describing the data
                    # in that row/column. may have neither if the table structure
                    # is too complex to describe this way.
                    has_rows = 'rows' in arg
                    has_cols = 'columns' in arg
                    # should not have both
                    self.assertTrue(not (has_rows and has_cols))

                    if has_cols or has_rows:
                        direction = 'rows' if has_rows else 'columns'
                        headers = arg[direction]
                        self.assertTrue(isinstance(headers, dict))

                        for header in headers:
                            self.assertTrue(isinstance(header, str))
                            self.validate(
                                headers[header],
                                f'{name}.{direction}.{header}',
                                parent_type=t)

                        attrs.discard('rows')
                        attrs.discard('columns')

                    # csv type may optionally have an 'excel_ok' attribute
                    if 'excel_ok' in arg:
                        self.assertTrue(isinstance(arg['excel_ok'], bool))
                        attrs.discard('excel_ok')

                elif t == 'directory':
                    # directory type should have a contents property that maps each
                    # expected path name/pattern within the directory to a nested
                    # type dictionary describing the data at that filepath
                    self.assertTrue('contents' in arg)
                    self.assertTrue(isinstance(arg['contents'], dict))
                    for path in arg['contents']:
                        self.assertTrue(isinstance(path, str))
                        self.validate(
                            arg['contents'][path],
                            f'{name}.contents.{path}',
                            parent_type=t)
                    attrs.remove('contents')

                    # may optionally have a 'permissions' attribute, which is a
                    # string of the unix-style directory permissions e.g. 'rwx'
                    if 'permissions' in arg:
                        self.validate_permissions_value(arg['permissions'])
                        attrs.remove('permissions')
                    # may optionally have an 'must_exist' attribute, which says
                    # whether the directory must already exist
                    # this defaults to True
                    if 'must_exist' in arg:
                        self.assertTrue(isinstance(arg['must_exist'], bool))
                        attrs.remove('must_exist')

                elif t == 'file':
                    # file type may optionally have a 'permissions' attribute
                    # this is a string listing the permissions e.g. 'rwx'
                    if 'permissions' in arg:
                        self.validate_permissions_value(arg['permissions'])

            # iterate over the remaining attributes
            # type-specific ones have been removed by this point
            if 'name' in attrs:
                self.assertTrue(isinstance(arg['name'], str))
                attrs.remove('name')
            if 'about' in attrs:
                self.assertTrue(isinstance(arg['about'], str))
                attrs.remove('about')
            if 'required' in attrs:
                # required value may be True, False, or a string that can be
                # parsed as a python statement that evaluates to True or False
                self.assertTrue(isinstance(arg['required'], bool) or
                                isinstance(arg['required'], str))
                attrs.remove('required')
            if 'type' in attrs:
                self.assertTrue(isinstance(arg['type'], str) or
                                isinstance(arg['type'], set))
                attrs.remove('type')

            # args should not have any unexpected properties
            # all attrs should have been removed by now
            if attrs:
                raise AssertionError(f'{name} has key(s) {attrs} that are not '
                                     'expected for its type')

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

        self.assertTrue(isinstance(permissions, str))
        self.assertTrue(len(permissions) > 0)
        valid_letters = {'r', 'w', 'x'}
        for letter in permissions:
            self.assertTrue(letter in valid_letters)
            # should only have a letter once
            valid_letters.remove(letter)

    def test_model_specs_serialize(self):
        """ARGS_SPEC: test each ARGS_SPEC can serialize to JSON."""
        from natcap.invest import spec_utils

        for model_name, metadata in MODEL_METADATA.items():
            model = importlib.import_module(metadata.pyname)
            try:
                _ = spec_utils.serialize_args_spec(model.ARGS_SPEC)
            except TypeError as error:
                self.fail(
                    f'Failed to avoid TypeError when serializing '
                    f'{metadata.pyname}.ARGS_SPEC: \n'
                    f'{error}')


class SpecUtilsTests(unittest.TestCase):
    """Tests for natcap.invest.spec_utils."""

    def test_format_unit(self):
        """spec_utils: test converting units to strings with format_unit."""
        from natcap.invest import spec_utils
        for unit_name, expected in [
                ('meter', 'm'),
                ('meter / second', 'm/s'),
                ('foot * mm', 'ft · mm'),
                ('t * hr * ha / ha / MJ / mm', 't · h · ha / (ha · MJ · mm)'),
                ('mm^3 / year', 'mm³/year')
        ]:
            unit = spec_utils.u.Unit(unit_name)
            actual = spec_utils.format_unit(unit)
            self.assertEqual(expected, actual)

    def test_format_unit_raises_error(self):
        """spec_utils: format_unit raises TypeError if not a pint.Unit."""
        from natcap.invest import spec_utils
        with self.assertRaises(TypeError):
            spec_utils.format_unit({})


if __name__ == '__main__':
    unittest.main()
