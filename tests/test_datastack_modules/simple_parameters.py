from natcap.invest import spec_utils

MODEL_SPEC = spec_utils.ModelSpec(inputs=[
    spec_utils.IntegerInput(id='a'),
    spec_utils.StringInput(id='b'),
    spec_utils.StringInput(id='c'),
    spec_utils.StringInput(id='d'),
    spec_utils.DirectoryInput(
        id='workspace_dir',
        contents=spec_utils.Contents()
    )],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec_utils.UISpec(),
    args_with_spatial_overlap={}
)
