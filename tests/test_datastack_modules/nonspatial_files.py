from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(inputs=[
    spec.FileInput(id='some_file'),
    spec.DirectoryInput(
        id='data_dir',
        contents=[])],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec.UISpec(),
    args_with_spatial_overlap={}
)
