import pint
import os

# the same unit registry instance should be shared across everything
# load from custom unit defintions file
# don't raise warnings when redefining units
u = pint.UnitRegistry(on_redefinition='ignore')
u.load_definitions(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'unit_definitions.txt'))
