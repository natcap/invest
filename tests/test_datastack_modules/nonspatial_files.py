from types import SimpleNamespace
from natcap.invest import spec_utils

MODEL_SPEC = SimpleNamespace(inputs=spec_utils.ModelInputs(
    spec_utils.FileInputSpec(id='some_file'),
    spec_utils.DirectoryInputSpec(
        id='data_dir',
        contents=spec_utils.Contents())
))
