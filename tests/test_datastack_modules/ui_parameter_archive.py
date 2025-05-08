from natcap.invest import spec

MODEL_SPEC = SimpleNamespace(inputs=[
    spec.StringInput(id='foo'),
    spec.StringInput(id='bar')],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    input_field_order=[],
    args_with_spatial_overlap={}
)
