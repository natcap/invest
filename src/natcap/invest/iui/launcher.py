from __future__ import absolute_import
import logging
import sys
import subprocess

from PyQt4 import QtGui
from PyQt4 import QtCore

try:
    from . import cli
except ImportError:
    import cli

import natcap.invest

LOGGER = logging.getLogger(__name__)

APP = QtGui.QApplication.instance()
if APP is None:
    APP = QtGui.QApplication(sys.argv)


class ModelLaunchButton(QtGui.QPushButton):
    def __init__(self, text, model):
        QtGui.QPushButton.__init__(self, text)
        self._model = model
        self.clicked.connect(self.launch)

    def launch(self, attr):
        LOGGER.info('Launching %s', self._model)
        subprocess.Popen('invest %s' % self._model, shell=True)


def main():
    launcher_window = QtGui.QMainWindow()
    launcher_window.setWindowTitle('InVEST Launcher')
    scroll_area = QtGui.QScrollArea()

    layout = QtGui.QGridLayout()
    layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
    main_widget = QtGui.QWidget()
    main_widget.setLayout(layout)

    launcher_window.setCentralWidget(scroll_area)
    scroll_area.setWidget(main_widget)

    labels_and_buttons = []
    for model in cli.list_models():
        row = layout.rowCount()
        label = QtGui.QLabel()
        button = ModelLaunchButton('Launch', model)
        labels_and_buttons.append((label, button))

        layout.addWidget(QtGui.QLabel(model), row, 0, QtCore.Qt.AlignRight)
        layout.addWidget(button, row, 1)

    version_label = QtGui.QLabel(
        '<em>InVEST %s</em>' % natcap.invest.__version__)
    version_label.setStyleSheet('QLabel {color: gray;}')
    layout.addWidget(version_label, layout.rowCount(), 0)


    scroll_area.setMinimumWidth(layout.sizeHint().width() + 25)
    scroll_area.setMinimumHeight(400)
    launcher_window.show()
    APP.exec_()

if __name__ == '__main__':
    main()
