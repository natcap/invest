from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(
    inputs=[spec.FileInput(id='some_file')],
    outputs=[],
    model_id='nonspatial_model',
    model_title='',
    userguide='',
    module_name=__name__,
    input_field_order=[['some_file']]
)
