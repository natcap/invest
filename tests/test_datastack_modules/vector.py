from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(inputs=[
    spec.VectorInput(
        id='vector', fields=[], geometry_types=set())],
    outputs=[],
    model_id='vector_model',
    model_title='',
    userguide='',
    module_name=__name__,
    input_field_order=[['vector']]
)
