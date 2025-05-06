from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(inputs=[
    spec.IntegerInput(id='a'),
    spec.StringInput(id='b'),
    spec.StringInput(id='c'),
    spec.StringInput(id='d'),
    spec.DirectoryInput(
        id='workspace_dir',
        contents=spec.Contents()
    )],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec.UISpec(),
    args_with_spatial_overlap={}
)
