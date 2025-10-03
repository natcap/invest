from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(inputs=[
    spec.StringInput(id='blank'),
    spec.IntegerInput(id='a'),
    spec.StringInput(id='b'),
    spec.StringInput(id='c'),
    spec.FileInput(id='foo'),
    spec.FileInput(id='bar'),
    spec.DirectoryInput(id='data_dir', contents=[]),
    spec.SingleBandRasterInput(id='raster', units=None),
    spec.VectorInput(id='vector', fields=[], geometry_types=set()),
    spec.CSVInput(id='simple_table'),
    spec.CSVInput(
        id='spatial_table',
        columns=[
            spec.IntegerInput(id='ID'),
            spec.RasterOrVectorInput(
                id='path',
                fields=[],
                geometry_types={'POINT', 'POLYGON'},
                units=None
            )
        ]
    )],
    outputs=[],
    model_id='archive_extraction_model',
    model_title='',
    userguide='',
    module_name=__name__,
    input_field_order=[
        ['blank', 'a', 'b', 'c', 'foo', 'bar', 'data_dir',
         'raster', 'vector', 'simple_table', 'spatial_table']]
)
