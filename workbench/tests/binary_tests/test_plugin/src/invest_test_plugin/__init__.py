# import the necessary model attributes from the submodule to the package level
# so that the attributes are exposed on the `invest_demo_plugin` package
from .foo import MODEL_SPEC
from .foo import execute
from .foo import validate
