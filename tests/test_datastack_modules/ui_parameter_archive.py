from natcap.invest import spec_utils

MODEL_SPEC = SimpleNamespace(inputs=[
    spec_utils.StringInput(id='foo'),
    spec_utils.StringInput(id='bar')],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec_utils.UISpec(),
    args_with_spatial_overlap={}
)
