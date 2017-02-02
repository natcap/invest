
import collections

_MODELMETA = collections.namedtuple('ModelMeta', 'id ui_module classname')

#    modelname, ui modulename, classname
_MODEL_UIS = (
    _MODELMETA(id='pollination', ui_module='pollination', classname='Pollination'),
)
