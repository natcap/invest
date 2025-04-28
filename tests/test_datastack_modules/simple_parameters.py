from types import SimpleNamespace
from natcap.invest import spec_utils

MODEL_SPEC = SimpleNamespace(inputs=spec_utils.ModelInputs(
    spec_utils.IntegerInputSpec(id='a'),
    spec_utils.StringInputSpec(id='b'),
    spec_utils.StringInputSpec(id='c'),
    spec_utils.StringInputSpec(id='d'),
    spec_utils.DirectoryInputSpec(
        id='workspace_dir',
        contents=spec_utils.Contents()
    )
))
