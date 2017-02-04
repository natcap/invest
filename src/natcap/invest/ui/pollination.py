from __future__ import absolute_import
import logging

from .model import Model
from ..pollination import pollination

LOGGER = logging.getLogger(__name__)
_validate = lambda args, limit_to: []


class Pollination(Model):
    label = pollination.LABEL
    target = staticmethod(pollination.execute)
    validator = _validate
    localdoc = 'croppollination.html'

    def __init__(self):
        Model.__init__(self)

    def assemble_args(self):
        return {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value()
        }
