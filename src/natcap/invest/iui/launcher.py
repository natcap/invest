from __future__ import absolute_import
import sys

from PyQt4 import QtGui

import cli

APP = QtGui.QApplication.instance()
if APP is None:
    APP = QtGui.QApplication(sys.argv)


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
        button = QtGui.QPushButton()
        labels_and_buttons.append((label, button))

        layout.addWidget(QtGui.QLabel(model), row, 0)
        layout.addWidget(QtGui.QPushButton('Launch'), row, 1)

    launcher_window.show()
    APP.exec_()

if __name__ == '__main__':
    main()
