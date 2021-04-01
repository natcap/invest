import importlib
import pint
import unittest


class ValidateArgsSpecs(unittest.TestCase):

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





    def validate(self, arg, valid_types=valid_types):

        valid_nested_types = {
            'raster': {'number', 'code', 'ratio'},
            'vector': {'freestyle_string', 'number', 'code', 'option_string',
                'percent', 'ratio'},
            'csv': {'number', 'ratio', 'percent', 'code', 'boolean',
                'freestyle_string', 'option_string', 'raster', 'vector'},
            'directory': {'raster', 'vector', 'csv', 'file', 'directory'}
        }

        valid_validation_options = {
            'csv': {
                # 'required_fields': list, 
                'excel_ok': bool
            },
            'number': {
                'expression': str
            },
            'boolean': {},
            'option_string': {
                'options': list[str]
            },
            'freestyle_string': ['regexp'],
            'vector': [
                # 'required_fields', 
                'projected', 'projection_units'],
            'raster': ['projected', 'projection_units'],
            'file': ['permissions'],
            'directory': ['exists', 'permissions']

        }


        allowed_attrs = ['name', 'about', 'required', 'type', 'validation_options']
        # arg['type'] can be either a string or a set of strings
        types = arg['type'] if isinstance(arg['type'], set) else [arg['type']]

        for t in types:
            self.assertTrue(t in valid_types, f'{t} is an invalid type\n{arg}')

            if t == 'number':
                self.assertTrue('units' in arg)
                if arg['units'] is not None:
                    self.assertEqual(type(arg['units']), pint.Unit)

            elif t == 'raster':
                self.assertTrue('bands' in arg)
                self.assertEqual(type(arg['bands']), dict)
                for band in arg['bands']:
                    self.assertTrue(isinstance(band, int))
                    self.validate(arg['bands'][band], valid_types=valid_nested_types['raster'])
                
            elif t == 'vector':
                self.assertTrue('fields' in arg)
                self.assertEqual(type(arg['fields']), dict)
                for field in arg['fields']:
                    self.assertTrue(isinstance(field, str))
                    self.validate(arg['fields'][field], valid_types=valid_nested_types['vector'])

                self.assertTrue('geometries' in arg)
                self.assertEqual(type(arg['geometries']), set)

            elif t == 'csv':
                has_rows = 'rows' in arg
                has_cols = 'columns' in arg
                self.assertTrue(has_rows or has_cols and not (has_rows and has_cols),
                    arg)
                headers = arg['columns'] if has_cols else arg['rows']

                # may be None if the table is too complex to define this way
                if headers is not None:
                    self.assertEqual(type(headers), dict)
                    for header in headers:
                        self.assertTrue(isinstance(header, str))
                        self.validate(headers[header], valid_types=valid_nested_types['csv'])

            elif t == 'directory':
                self.assertTrue('contents' in arg)
                self.assertEqual(type(arg['contents']), dict)
                for path in arg['contents']:
                    self.assertTrue(isinstance(path, str))
                    self.validate(arg['contents'][path], 
                        valid_types=valid_nested_types['directory'])


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
            model = importlib.import_module(f'natcap.invest.{model_name}')
            print(model_name)

            # validate that each arg meets the expected pattern
            for arg in model.ARGS_SPEC['args'].values():
                print(f'    {arg["name"]}')

                # attributes that are required at the top level but not 
                # necessarily in nested levels
                required_attrs = ['name', 'about', 'required', 'type']
                for attr in required_attrs:
                    self.assertTrue(attr in arg,
                    f'Missing attribute "{attr}" at the top level')

                self.validate(arg)




if __name__ == '__main__':
    unittest.main()
