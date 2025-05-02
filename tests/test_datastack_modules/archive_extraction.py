from types import SimpleNamespace
from natcap.invest import spec_utils

MODEL_SPEC = SimpleNamespace(inputs=spec_utils.ModelInputs(
    spec_utils.StringInput(id='blank'),
    spec_utils.IntegerInput(id='a'),
    spec_utils.StringInput(id='b'),
    spec_utils.StringInput(id='c'),
    spec_utils.FileInput(id='foo'),
    spec_utils.FileInput(id='bar'),
    spec_utils.DirectoryInput(id='data_dir', contents={}),
    spec_utils.SingleBandRasterInput(id='raster', band=spec_utils.Input()),
    spec_utils.VectorInput(id='vector', fields={}, geometries={}),
    spec_utils.CSVInput(id='simple_table'),
    spec_utils.CSVInput(
        id='spatial_table',
        columns=spec_utils.Columns(
            spec_utils.IntegerInput(id='ID'),
            spec_utils.RasterOrVectorInput(
                id='path',
                fields={},
                geometries={'POINT', 'POLYGON'},
                band=spec_utils.NumberInput()
            )
        )
    )
))
