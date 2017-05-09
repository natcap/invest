from __future__ import absolute_import
import logging
import sys
import subprocess

from PyQt4 import QtGui

import cli

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
    layout = QtGui.QGridLayout()
    main_widget = QtGui.QWidget()
    main_widget.setLayout(layout)
    launcher_window.setCentralWidget(main_widget)

    labels_and_buttons = []
    for model in cli.list_models():
        row = layout.rowCount()
        label = QtGui.QLabel()
        button = ModelLaunchButton('Launch', model)
        labels_and_buttons.append((label, button))

        layout.addWidget(QtGui.QLabel(model), row, 0)
        layout.addWidget(button, row, 1)

    launcher_window.show()
    APP.exec_()

if __name__ == '__main__':
    main()
