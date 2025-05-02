from types import SimpleNamespace
from natcap.invest import spec_utils

MODEL_SPEC = SimpleNamespace(inputs=spec_utils.ModelInputs(
    spec_utils.IntegerInput(id='a'),
    spec_utils.StringInput(id='b'),
    spec_utils.StringInput(id='c'),
    spec_utils.StringInput(id='d'),
    spec_utils.DirectoryInput(
        id='workspace_dir',
        contents=spec_utils.Contents()
    )
))
