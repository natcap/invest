from natcap.invest import spec

MODEL_SPEC = SimpleNamespace(inputs=[
    spec.StringInput(id='foo'),
    spec.StringInput(id='bar')],
    outputs=[],
    model_id='ui_parameters_model',
    model_title='',
    userguide='',
    module_name=__name__,
    input_field_order=[['foo', 'bar']]
)
