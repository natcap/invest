"""__init__ module for pygeprocessing, imports all the geoprocessing functions
    into the pygeoprocessing namespace"""

__version__ = '0.3.3'

import logging
import types

from geoprocessing import *

__all__ = []
for attrname in dir(geoprocessing):
    if type(getattr(geoprocessing, attrname)) is types.FunctionType:
        __all__.append(attrname)

LOGGER = logging.getLogger('pygeoprocessing_0_3_3')
LOGGER.setLevel(logging.DEBUG)
