import importlib
import pint
import unittest


valid_types = {
        'number',
        'ratio',
        'percent',
        'code',
        'freestyle_string',
        'option_string',
        'boolean',
        'raster',
        'vector',
        'csv',
        'file',
        'directory'
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
            'wind_energy'
        ]

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                model = importlib.import_module(f'natcap.invest.{model_name}')

                # validate that each arg meets the expected pattern
                for key, arg in model.ARGS_SPEC['args'].items():
                    with self.subTest(arg_name=key):

                        # attributes that are required at the top level but not
                        # necessarily in nested levels
                        required_attrs = ['name', 'about', 'type']
                        for attr in required_attrs:
                            self.assertTrue(
                                attr in arg,
                                f'Missing attribute "{attr}" at the top level')

                        self.validate(arg)

    def validate_permissions_value(self, permissions):
        self.assertTrue(isinstance(permissions, str))
        self.assertTrue(len(permissions) > 0)
        valid_letters = {'r', 'w', 'x'}
        for letter in permissions:
            self.assertTrue(letter in valid_letters)
            # should only have a letter once
            valid_letters.remove(letter)

    def validate(self, arg, valid_types=valid_types):

        valid_nested_types = {
            'raster': {'number', 'code', 'ratio'},
            'vector': {
                'freestyle_string',
                'number',
                'code',
                'option_string',
                'percent',
                'ratio'},
            'csv': {
                'number',
                'ratio',
                'percent',
                'code',
                'boolean',
                'freestyle_string',
                'option_string',
                'raster',
                'vector'},
            'directory': {'raster', 'vector', 'csv', 'file', 'directory'}
        }

        # arg['type'] can be either a string or a set of strings
        types = arg['type'] if isinstance(arg['type'], set) else [arg['type']]
        attrs = set(arg.keys())
        for t in types:
            self.assertTrue(t in valid_types, f'{t} is an invalid type\n{arg}')

            if t == 'number':
                self.assertTrue('units' in arg)
                # Undefined units should use the custom u.none unit
                self.assertTrue(isinstance(arg['units'], pint.Unit))
                attrs.remove('units')

            elif t == 'raster':
                self.assertTrue('bands' in arg)
                self.assertTrue(isinstance(arg['bands'], dict))
                for band in arg['bands']:
                    self.assertTrue(isinstance(band, int))
                    self.validate(
                        arg['bands'][band],
                        valid_types=valid_nested_types['raster'])
                attrs.remove('bands')

            elif t == 'vector':
                self.assertTrue('fields' in arg)
                self.assertTrue(isinstance(arg['fields'], dict))
                for field in arg['fields']:
                    self.assertTrue(isinstance(field, str))
                    self.validate(
                        arg['fields'][field],
                        valid_types=valid_nested_types['vector'])

                self.assertTrue('geometries' in arg)
                self.assertTrue(isinstance(arg['geometries'], set))

                attrs.remove('fields')
                attrs.remove('geometries')

            elif t == 'csv':
                has_rows = 'rows' in arg
                has_cols = 'columns' in arg
                # may have neither if table is too complex to define this way
                self.assertTrue(not (has_rows and has_cols), arg)

                if has_cols or has_rows:
                    headers = arg['columns'] if has_cols else arg['rows']
                    self.assertTrue(isinstance(headers, dict))

                    for header in headers:
                        self.assertTrue(isinstance(header, str))
                        self.validate(
                            headers[header],
                            valid_types=valid_nested_types['csv'])

                    attrs.discard('rows')
                    attrs.discard('columns')

            elif t == 'directory':
                print(arg)
                self.assertTrue('contents' in arg)
                self.assertTrue(isinstance(arg['contents'], dict))
                for path in arg['contents']:
                    self.assertTrue(isinstance(path, str))
                    self.validate(
                        arg['contents'][path],
                        valid_types=valid_nested_types['directory'])
                attrs.remove('contents')

        for attr in attrs:
            print(attr)
            if attr in {'name', 'about'}:
                self.assertTrue(isinstance(arg[attr], str))
            elif attr == 'required':
                # required value may be True, False, or a string that can be
                # parsed as a python statement that evaluates to True or False
                self.assertTrue(
                    isinstance(arg[attr], bool) or
                    isinstance(arg[attr], str))
            elif attr == 'type':
                self.assertTrue(
                    isinstance(arg[attr], str) or
                    isinstance(arg[attr], set))
            elif attr == 'validation_options':
                if arg['type'] == 'csv':
                    self.assertTrue(list(arg[attr].keys()) == ['excel_ok'])
                    self.assertTrue(isinstance(arg[attr]['excel_ok'], bool))
                elif arg['type'] == 'number':
                    self.assertTrue(list(arg[attr].keys()) == ['expression'])
                    self.assertTrue(isinstance(arg[attr]['expression'], str))
                elif arg['type'] == 'option_string':
                    self.assertTrue(list(arg[attr].keys()) == ['options'])
                    self.assertTrue(isinstance(arg[attr]['options'], list))
                    for item in arg[attr]['options']:
                        self.assertTrue(isinstance(item, str))
                elif arg['type'] == 'freestyle_string':
                    self.assertEqual(list(arg[attr].keys()), ['regexp'])
                    self.assertTrue(isinstance(arg[attr]['regexp'], dict))
                    self.assertEqual(
                        list(arg[attr]['regexp'].keys()), ['pattern'])
                    self.assertTrue(isinstance(
                        arg[attr]['regexp']['pattern'], str))


                elif arg['type'] in {'raster', 'vector'}:
                    keys = set(arg[attr].keys())
                    self.assertTrue(len(keys) > 0)
                    for key in keys:
                        if key == 'projected':
                            self.assertTrue(isinstance(
                                arg[attr]['projected'],
                                bool))
                        elif key == 'projection_units':
                            self.assertTrue(isinstance(
                                arg[attr]['projection_units'],
                                pint.Unit))
                        else:
                            raise ValueError(
                                f'Invalid key in validation_options: {key}')
                elif arg['type'] == 'file':
                    self.assertTrue(list(arg[attr].keys()) == ['permissions'])
                    self.validate_permissions_value(arg[attr]['permissions'])

                elif arg['type'] == 'directory':
                    keys = set(arg[attr].keys())
                    self.assertTrue(len(keys) > 0)
                    for key in keys:
                        if key == 'permissions':
                            self.validate_permissions_value(arg[attr]['permissions'])
                        elif key == 'exists':
                            self.assertTrue(isinstance(arg[attr]['exists'], bool))

                else:
                    raise ValueError(
                        f'{arg} has a validation_options key that is not '
                        'allowed for its type')


if __name__ == '__main__':
    unittest.main()
