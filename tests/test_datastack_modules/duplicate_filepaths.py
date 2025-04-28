from types import SimpleNamespace
from natcap.invest import spec_utils

MODEL_SPEC = SimpleNamespace(inputs=spec_utils.ModelInputs(
    spec_utils.FileInputSpec(id='foo'),
    spec_utils.FileInputSpec(id='bar'),
))
