from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(
    inputs=[
        spec.FileInput(id='foo'),
        spec.FileInput(id='bar')
    ],
    outputs=[],
    model_id='duplicate_filepaths_model',
    model_title='',
    userguide='',
    input_field_order=[['foo', 'bar']]
)
