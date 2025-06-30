from natcap.invest import spec

MODEL_SPEC = spec.ModelSpec(inputs=[
    spec.SingleBandRasterInput(id='raster', units=None)],
    outputs=[],
    model_id='raster_model',
    model_title='',
    userguide='',
    input_field_order=[['raster']]
)
