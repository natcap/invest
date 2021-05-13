import importlib
import re
import unittest

import pint


valid_nested_types = {
    'all': {
        'boolean',
        'code',
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
    'raster': {'code', 'number', 'ratio'},
    'vector': {
        'code',
        'freestyle_string',
        'number',
        'option_string',
        'percent',
        'ratio'},
    'csv': {
        'boolean',
        'code',
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

    def test_model_specs(self):

        model_names = [
            'carbon',
            'coastal_blue_carbon.coastal_blue_carbon',
            'coastal_blue_carbon.preprocessor',
            'coastal_vulnerability',
            'crop_production_regression',
            'crop_production_percentile',
            'delineateit.delineateit',
            'finfish_aquaculture.finfish_aquaculture',
            'fisheries.fisheries',
            'fisheries.fisheries_hst',
            'forest_carbon_edge_effect',
            'globio',
            'habitat_quality',
            'hra',
            'hydropower.hydropower_water_yield',
            'ndr.ndr',
            'pollination',
            'recreation.recmodel_client',
            'routedem',
            'scenic_quality.scenic_quality',
            'scenario_gen_proximity',
            'sdr.sdr',
            'seasonal_water_yield.seasonal_water_yield',
            'urban_cooling_model',
            'urban_flood_risk_mitigation',
            'wave_energy',
            # 'wind_energy'
        ]

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                model = importlib.import_module(f'natcap.invest.{model_name}')

                # validate that each arg meets the expected pattern
                # save up errors to report at the end
                for key, arg in model.ARGS_SPEC['args'].items():
                    with self.subTest(arg_name=f'{model_name}.{key}'):
                        # the top level should have 'name' and 'about' attrs
                        # but they aren't required at nested levels
                        self.validate(
                            arg,
                            f'{model_name}.{key}',
                            valid_types=valid_nested_types['all'],
                            required_attrs=['name', 'about'])

    def validate(self, arg, name, valid_types, is_pattern=False, required_attrs=[]):
        """
        Recursively validate nested args against the ARGS_SPEC standard.

        Args:
            arg (dict): any nested arg component of an ARGS_SPEC
            name (str): name to use in error messages to identify the arg
            valid_types (list[str]): a list of the arg types that are valid
                for this nested arg (due to its parent's type).
            is_pattern (bool): if True, the arg is validated as a pattern (such
                as for user-defined  CSV headers, vector fields, or directory
                paths).
            required_attrs (list[str]): a list of attributes that must be in
                the arg dictionary regardless of type

        Returns:
            None

        Raises:
            AssertionError if the arg violates the standard
        """
        with self.subTest(nested_arg_name=name):
            for attr in required_attrs:
                self.assertTrue(attr in arg)

            # arg['type'] can be either a string or a set of strings
            types = arg['type'] if isinstance(arg['type'], set) else [arg['type']]
            attrs = set(arg.keys())
            for t in types:
                self.assertTrue(t in valid_types)

                if arg['type'] == 'option_string':
                    # option_string type should have an options property that
                    # describes the valid options
                    self.assertTrue('options' in arg)
                    # May be a list or dict because some option sets are self
                    # explanatory and others need a description
                    if isinstance(arg['options'], list):
                        for item in arg['options']:
                            self.assertTrue(isinstance(item, str))
                    elif isinstance(arg['options'], dict):
                        for key, val in arg['options'].items():
                            self.assertTrue(isinstance(key, str))
                            self.assertTrue(isinstance(val, str))
                    attrs.remove('options')

                elif t == 'number':
                    # number type should have a units property
                    self.assertTrue('units' in arg)
                    # Undefined units should use the custom u.none unit
                    self.assertTrue(isinstance(arg['units'], pint.Unit))
                    attrs.remove('units')

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
                            valid_types=valid_nested_types['raster'])
                    attrs.remove('bands')

                elif t == 'vector':
                    # vector type should have:
                    # - a fields property that maps each field header to a nested
                    #   type dictionary describing the data in that field
                    # - a geometries property: the set of valid geometry types
                    self.assertTrue('fields' in arg)
                    self.assertTrue(isinstance(arg['fields'], dict))
                    for field in arg['fields']:
                        self.assertTrue(isinstance(field, str))
                        if '(' in field or '[' in field:
                            print(f"Warning: header {field} in {name}")
                        self.validate(
                            arg['fields'][field],
                            f'{name}.fields.{field}',
                            valid_types=valid_nested_types['vector'])

                    self.assertTrue('geometries' in arg)
                    self.assertTrue(isinstance(arg['geometries'], set))

                    attrs.remove('fields')
                    attrs.remove('geometries')

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
                            if '(' in header or '[' in header:
                                print(f"Warning: header {header} in {name}")
                            self.validate(
                                headers[header],
                                f'{name}.{direction}.{header}',
                                valid_types=valid_nested_types['csv'])

                        attrs.discard('rows')
                        attrs.discard('columns')

                elif t == 'directory':
                    # directory type should have a contents property that maps each
                    # expected path name/pattern within the directory to a nested
                    # type dictionary describing the data at that filepath
                    self.assertTrue('contents' in arg)
                    self.assertTrue(isinstance(arg['contents'], dict))
                    for path in arg['contents']:
                        self.assertTrue(isinstance(path, str))
                        if '(' in path or '[' in path:
                            print(f"Warning: header {path} in {name}")
                        self.validate(
                            arg['contents'][path],
                            f'{name}.contents.{path}',
                            valid_types=valid_nested_types['directory'])
                    attrs.remove('contents')

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
            if 'regexp' in attrs:
                # if it has 'regexp' it should have 'items' also
                self.assertTrue('items' in arg)
                self.assertTrue(isinstance(arg['regexp'], str))
                re.compile(arg['regexp'])  # should compile without errors
                attrs.remove('regexp')
            if 'items' in attrs:
                # if it has 'items' it should have 'regexp' also
                self.assertTrue('regexp' in arg)
                self.assertTrue(isinstance(arg['items'], str))
                attrs.remove('items')
            if 'validation_options' in attrs:
                # the allowed validation_options properties are type-specific
                opts = arg['validation_options']
                if arg['type'] == 'csv':
                    self.assertTrue(list(opts.keys()) == ['excel_ok'])
                    self.assertTrue(isinstance(opts['excel_ok'], bool))
                elif arg['type'] == 'number':
                    self.assertTrue(list(opts.keys()) == ['expression'])
                    self.assertTrue(isinstance(opts['expression'], str))
                elif arg['type'] == 'freestyle_string':
                    self.assertEqual(list(opts.keys()), ['regexp'])
                    self.assertTrue(isinstance(opts['regexp'], dict))
                    self.assertEqual(list(opts['regexp'].keys()), ['pattern'])
                    self.assertTrue(isinstance(opts['regexp']['pattern'], str))
                elif arg['type'] in {'raster', 'vector'}:
                    keys = set(opts.keys())
                    # should have at least one key; shouldn't have
                    # projection_units without projected
                    self.assertTrue(
                        (keys == {'projected'}) or
                        (keys == {'projected', 'projection_units'}))
                    self.assertTrue(isinstance(opts['projected'], bool))
                    if 'projection_units' in keys:
                        # doesn't make sense to have projection units unless
                        # projected is True
                        self.assertTrue(opts['projected'])
                        self.assertTrue(
                            isinstance(opts['projection_units'], pint.Unit))
                elif arg['type'] == 'file':
                    self.assertTrue(list(opts.keys()) == ['permissions'])
                    self.validate_permissions_value(opts['permissions'])
                elif arg['type'] == 'directory':
                    keys = set(opts.keys())
                    # should have at least one of 'permissions', 'exists'
                    self.assertTrue(len(keys) > 0)
                    self.assertTrue(keys.issubset({'permissions', 'exists'}))
                    if 'permissions' in keys:
                        self.validate_permissions_value(opts['permissions'])
                    if 'exists' in keys:
                        self.assertTrue(isinstance(opts['exists'], bool))

                # validation options should not exist for any other types
                else:
                    raise AssertionError(f"{name}'s type does not allow the "
                                         "validation_options attribute")
                attrs.remove('validation_options')

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


if __name__ == '__main__':
    unittest.main()
