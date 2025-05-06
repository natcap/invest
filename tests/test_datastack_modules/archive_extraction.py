from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(inputs=[
    spec.StringInput(id='blank'),
    spec.IntegerInput(id='a'),
    spec.StringInput(id='b'),
    spec.StringInput(id='c'),
    spec.FileInput(id='foo'),
    spec.FileInput(id='bar'),
    spec.DirectoryInput(id='data_dir', contents={}),
    spec.SingleBandRasterInput(id='raster', band=spec.Input()),
    spec.VectorInput(id='vector', fields={}, geometries={}),
    spec.CSVInput(id='simple_table'),
    spec.CSVInput(
        id='spatial_table',
        columns=[
            spec.IntegerInput(id='ID'),
            spec.RasterOrVectorInput(
                id='path',
                fields={},
                geometries={'POINT', 'POLYGON'},
                band=spec.NumberInput()
            )
        ]
    )],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec.UISpec(),
    args_with_spatial_overlap={}
)
