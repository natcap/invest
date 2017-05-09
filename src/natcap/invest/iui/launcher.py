from __future__ import absolute_import
import sys

from PyQt4 import QtGui

from . import cli

APP = QtGui.QApplication.instance()
if APP is None:
    APP = QtGui.QApplication(sys.argv)


def main():
    launcher_window = QtGui.QMainWindow()
    launcher_window.setLayout(QtGui.QGridLayout())

    labels_and_buttons = []
    for model in cli.list_models():
        row = launcher_window.layout().rowCount()
        label = QtGui.QLabel()
        button = QtGui.QPushButton()
        labels_and_buttons.append((label, button))


        launcher_window.layout().addWidget(
            row, 0, QtGui.QLabel(model))
        launcher_window.layout().addWidget(
            row, 1, QtGui.QPushButton('foo'))

    launcher_window.show()
    launcher_window.exec_()

if __name__ == '__main__':
    #main()
    pass
