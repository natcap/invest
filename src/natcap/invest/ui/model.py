from __future__ import absolute_import

import logging
import os
from datetime import datetime

from qtpy import QtWidgets
from qtpy import QtCore
import natcap.invest
from natcap.ui import inputs

from .. import utils

LOG_FMT = "%(asctime)s %(name)-18s %(levelname)-8s %(message)s"
DATE_FMT = "%m/%d/%Y %H:%M:%S "
LOGGER = logging.getLogger(__name__)


class Model(object):
    label = None
    target = None
    validator = None
    localdoc = None

    def __init__(self):
        self.window = QtWidgets.QWidget()
        self.window.setLayout(QtWidgets.QVBoxLayout())
        if self.label:
            self.window.setWindowTitle(self.label)

        self.links = QtWidgets.QLabel()
        self._make_links(self.links)
        self.window.layout().addWidget(self.links)

        self.form = inputs.Form()
        self.window.layout().addWidget(self.form)
        self.run_dialog = inputs.FileSystemRunDialog()

        # set up a system tray icon.
        self.systray_icon = QtWidgets.QSystemTrayIcon()

        # start with workspace and suffix inputs
        self.workspace = inputs.Folder(args_key='workspace_dir',
                                       label='Workspace',
                                       required=True)
        self.suffix = inputs.Text(args_key='suffix',
                                  label='Results suffix',
                                  required=False)
        self.suffix.textfield.setMaximumWidth(150)
        self.add_input(self.workspace)
        self.add_input(self.suffix)

        self.form.submitted.connect(self.execute_model)
        self.form.run_finished.connect(self._show_alert)

    def _show_alert(self):
        self.systray_icon.showMessage(
            'InVEST', 'Model run finished')

    def _make_links(self, qlabel):
        qlabel.setAlignment(QtCore.Qt.AlignRight)
        qlabel.setOpenExternalLinks(True)
        links = ['InVEST version ' + natcap.invest.__version__]

        try:
            doc_uri = 'file://' + os.path.abspath(self.localdoc)
            links.append('<a href=\"%s\">Model documentation</a>' % doc_uri)
        except AttributeError:
            # When self.localdoc is None, documentation is undefined.
            LOGGER.info('Skipping docs link; undefined.')

        feedback_uri = 'http://forums.naturalcapitalproject.org/'
        links.append('<a href=\"%s\">Report an issue</a>' % feedback_uri)

        qlabel.setText(' | '.join(links))

    def add_input(self, input):
        # Add the model's validator if it hasn't already been set.
        if hasattr(input, 'validator') and input.validator is None:
            input.validator = self.validator
        self.form.add_input(input)

    def execute_model(self, logfile=None, tempdir=None):
        args = self.assemble_args()

        if not os.path.exists(args['workspace_dir']):
            os.makedirs(args['workspace_dir'])

        if not logfile:
            logfile = os.path.join(
                args['workspace_dir'],
                'InVEST-{modelname}-log-{timestamp}.txt'.format(
                    modelname='-'.join(self.label.split(' ')),
                    timestamp=datetime.now().strftime("%Y-%m-%d--%H_%M_%S")))

        if not tempdir:
            tempdir = os.path.join(args['workspace_dir'], 'tmp')
            if not os.path.exists(tempdir):
                os.makedirs(tempdir)

        def _logged_target():
            with utils.log_to_file(logfile), utils.sandbox_tempdir(tempdir):
                return self.target(args=args)

        self.form.run(target=_logged_target,
                      window_title='Running %s' % self.label,
                      out_folder=args['workspace_dir'])

    def assemble_args(self):
        raise NotImplementedError

    def run(self):
        self.window.show()
        self.window.raise_()  # raise window to top of stack.
        inputs.QT_APP.exec_()
