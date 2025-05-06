from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(
    inputs=[
        spec.FileInput(id='foo'),
        spec.FileInput(id='bar')
    ],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec.UISpec(),
    args_with_spatial_overlap={}
)
