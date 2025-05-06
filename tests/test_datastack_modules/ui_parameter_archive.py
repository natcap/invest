from natcap.invest import spec

MODEL_SPEC = SimpleNamespace(inputs=[
    spec.StringInput(id='foo'),
    spec.StringInput(id='bar')],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec.UISpec(),
    args_with_spatial_overlap={}
)
