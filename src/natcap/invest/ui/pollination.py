from __future__ import absolute_import

from .model import Model
from ..pollination import pollination


class Pollination(Model):
    label = pollination.LABEL
    target = pollination.execute
    localdoc = 'croppollination.html'

    def __init__(self):
        Model.__init__(self)

    def assemble_args(self):
        return {
            self.workspace_dir.args_key: self.workspace_dir.value(),
            self.suffix.args_key: self.suffix.value()
        }


if __name__ == '__main__':
    model = Pollination()
    model.exec_()
