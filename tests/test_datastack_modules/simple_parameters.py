from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(inputs=[
    spec.IntegerInput(id='a'),
    spec.StringInput(id='b'),
    spec.StringInput(id='c'),
    spec.StringInput(id='d'),
    spec.DirectoryInput(
        id='workspace_dir',
        contents=[]
    )],
    outputs=[],
    model_id='simple_model',
    model_title='',
    userguide='',
    module_name=__name__,
    input_field_order=[['a', 'b', 'c', 'd', 'workspace_dir']]
)
