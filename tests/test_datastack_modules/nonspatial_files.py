from natcap.invest import spec_utils

MODEL_SPEC = spec_utils.ModelSpec(inputs=[
    spec_utils.FileInput(id='some_file'),
    spec_utils.DirectoryInput(
        id='data_dir',
        contents=spec_utils.Contents())],
    outputs={},
    model_id='',
    model_title='',
    userguide='',
    ui_spec=spec_utils.UISpec(),
    args_with_spatial_overlap={}
)
