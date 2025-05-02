from types import SimpleNamespace
from natcap.invest import spec_utils

MODEL_SPEC = SimpleNamespace(inputs=spec_utils.ModelInputs(
    spec_utils.VectorInput(
        id='vector', fields={}, geometries={})
))
