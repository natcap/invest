ARGS_SPEC = {
    'args': {
        'blank': {'type': 'freestyle_string'},
        'a': {'type': 'integer'},
        'b': {'type': 'freestyle_string'},
        'c': {'type': 'freestyle_string'},
        'foo': {'type': 'file'},
        'bar': {'type': 'file'},
        'data_dir': {'type': 'directory'},
        'raster': {'type': 'raster'},
        'vector': {'type': 'vector'},
        'simple_table': {'type': 'csv'},
        'spatial_table': {
            'type': 'csv',
            'columns': {
                'ID': {'type': 'integer'},
                'path': {'type': {'raster', 'vector'}},
            }
        }
    }
}
