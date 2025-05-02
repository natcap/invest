from natcap.invest import spec_utils

MODEL_SPEC = spec_utils.ModelSpec(
    inputs=[
        spec_utils.FileInput(id='foo'),
        spec_utils.FileInput(id='bar')
    ],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec_utils.UISpec(),
    args_with_spatial_overlap={}
)
