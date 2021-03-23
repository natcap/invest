import importlib
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

        valid_raster_band_types = {'number', 'code'}
        valid_vector_field_types = {'freestyle_string'}
        valid_csv_data_types = {'number', 'ratio', 'percent', 'code', 'boolean',
            'freestyle_string', 'option_string', 'raster', 'vector'}
        valid_directory_path_types = {'raster', 'vector', 'csv', 'file'}

        # the arg should have a 'type' property
        self.assertTrue('type' in arg)

        self.assertTrue(arg['type'] in valid_types)

        if arg['type'] == 'number':
            self.assertTrue('units' in arg)
            if arg['units'] is not None:
                self.assertEqual(type(arg['units']), str)


        elif arg['type'] == 'raster':
            self.assertTrue('bands' in arg)
            self.assertEqual(type(arg['bands']), dict)
            for band in arg['bands']:
                self.assertTrue(isinstance(band, int))
                self.validate(arg['bands'][band], valid_types=valid_raster_band_types)

            
        elif arg['type'] == 'vector':
            self.assertTrue('fields' in arg)
            self.assertEqual(type(arg['fields']), dict)
            for field in arg['fields']:
                self.assertTrue(isinstance(field, str))
                self.validate(arg['fields'][field], valid_types=valid_vector_field_types)
            

        elif arg['type'] == 'csv':
            hasRows = 'rows' in arg
            hasCols = 'columns' in arg
            self.assertTrue(hasRows or hasCols and not (hasRows and hasCols))
            self.assertEqual(type(arg['columns']), dict)
            for column in arg['columns']:
                self.assertTrue(isinstance(column, str))
                self.validate(arg['columns'][column], valid_types=valid_csv_data_types)


        elif arg['type'] == 'directory':
            self.assertTrue('contents' in arg)
            self.assertEqual(type(arg['contents']), dict)
            for path in arg['contents']:
                self.assertTrue(isinstance(column, str))
                self.validate(arg['contents'][path], valid_types=valid_directory_path_types)


    def test_carbon(self):
        from natcap.invest import carbon

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
            'hydropower',
            'ndr.ndr',
            'pollination',
            'recreation.recmodel_client',
            'routedem',
            'scenic_quality.sceniq_quality',
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
            for arg in model.ARGS_SPEC['args'].values():
                print(f'    {arg["name"]}')
                self.validate(arg)




if __name__ == '__main__':
    unittest.main()
                






