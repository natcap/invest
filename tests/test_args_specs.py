import unittest


class ValidateArgsSpecs(unittest.TestCase):

    valid_types = {
        'number', 
        'ratio', 
        'percent', 
        'freestyle_string', 
        'option_string', 
        'boolean', 
        'raster', 
        'vector', 
        'csv', 
        'file', 
        'directory'
    }

    valid_raster_band_types = {'number'}
    valid_vector_field_types = {'freestyle_string'}
    valid_csv_data_types = {'number', 'freestyle_string', 'option_string', 'boolean', 'raster', 'vector'}
    valid_directory_path_types = {'raster', 'vector', 'csv', 'file'}

    models = [
        'carbon', 
        'coastal_vulnerability', 
        'crop_production_percentile', 
        'crop_production_regression']

    def validate(self, arg, valid_types=valid_types):

        # the arg should have a 'type' property
        self.assertTrue('type' in arg)

        self.assertTrue(arg['type'] in valid_types)

        if arg['type'] == 'number':
            self.assertTrue('units' in arg)
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

        for arg in carbon.ARGS_SPEC['args'].values():
            print(arg)
            self.validate(arg)


if __name__ == '__main__':
    unittest.main()
                






