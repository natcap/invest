from types import SimpleNamespace
from natcap.invest import spec_utils

MODEL_SPEC = SimpleNamespace(inputs=spec_utils.ModelInputs(
    spec_utils.StringInputSpec(id='blank'),
    spec_utils.IntegerInputSpec(id='a'),
    spec_utils.StringInputSpec(id='b'),
    spec_utils.StringInputSpec(id='c'),
    spec_utils.FileInputSpec(id='foo'),
    spec_utils.FileInputSpec(id='bar'),
    spec_utils.DirectoryInputSpec(id='data_dir', contents={}),
    spec_utils.SingleBandRasterInputSpec(id='raster', band=spec_utils.InputSpec()),
    spec_utils.VectorInputSpec(id='vector', fields={}, geometries={}),
    spec_utils.CSVInputSpec(id='simple_table'),
    spec_utils.CSVInputSpec(
        id='spatial_table',
        columns=spec_utils.Columns(
            spec_utils.IntegerInputSpec(id='ID'),
            spec_utils.RasterOrVectorInputSpec(
                id='path',
                fields={},
                geometries={'POINT', 'POLYGON'},
                band=spec_utils.NumberInputSpec()
            )
        )
    )
))
