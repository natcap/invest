import logging
import sys
import subprocess
import os
import PySide2

from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui


import natcap.invest
from natcap.invest.model_metadata import MODEL_METADATA

LOGGER = logging.getLogger(__name__)

try:
    QApplication = QtGui.QApplication
except AttributeError:
    QApplication = QtWidgets.QApplication

APP = QApplication.instance()
if APP is None:
    APP = QApplication(sys.argv)  # pragma: no cover


class ModelLaunchButton(QtWidgets.QPushButton):
    def __init__(self, text, model):
        QtWidgets.QPushButton.__init__(self, text)
        self._model = model
        self.clicked.connect(self.launch)

    def launch(self, attr):
        # If we're in a pyinstaller build, run this command from the location
        # of the application, wherever that is.  Otherwise, launch from CWD by
        # looking through PATH.
        if getattr(sys, '_MEIPASS', False):
            cwd = os.path.dirname(sys.executable)
            command = './invest'
        else:
            cwd = None  # subprocess.Popen default value for cwd.
            command = 'invest'
        LOGGER.info('Launching %s from CWD %s', self._model, cwd)
        subprocess.Popen('%s run %s' % (command, self._model), shell=True, cwd=cwd)


def main():
    launcher_window = QtWidgets.QMainWindow()
    launcher_window.setWindowTitle('InVEST Launcher')
    scroll_area = QtWidgets.QScrollArea()

    layout = QtWidgets.QGridLayout()
    layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
    main_widget = QtWidgets.QWidget()
    main_widget.setLayout(layout)

    launcher_window.setCentralWidget(scroll_area)
    scroll_area.setWidget(main_widget)

    labels_and_buttons = []
    for model_name, model_data in sorted(MODEL_METADATA.items(),
            # sort alphabetically by display name
            key=lambda item: item[1].model_title):
        row = layout.rowCount()
        label = QtWidgets.QLabel()
        button = ModelLaunchButton('Launch', model_name)
        labels_and_buttons.append((label, button))

        layout.addWidget(
            QtWidgets.QLabel(model_data.model_title), row, 0,
            QtCore.Qt.AlignRight)
        layout.addWidget(button, row, 1)

    version_label = QtWidgets.QLabel(
        '<em>InVEST %s</em>' % natcap.invest.__version__)
    version_label.setStyleSheet('QLabel {color: gray;}')
    layout.addWidget(version_label, layout.rowCount(), 0)

    scroll_area.setMinimumWidth(layout.sizeHint().width() + 25)
    scroll_area.setMinimumHeight(400)
    launcher_window.show()
    APP.exec_()


if __name__ == '__main__':
    main()
