from __future__ import absolute_import

import logging
import os
import pprint
import warnings

from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui
import natcap.invest
from natcap.ui import inputs

from .. import utils
from .. import scenarios

LOG_FMT = "%(asctime)s %(name)-18s %(levelname)-8s %(message)s"
DATE_FMT = "%m/%d/%Y %H:%M:%S "
LOGGER = logging.getLogger(__name__)


class Model(object):
    label = None
    target = None
    validator = None
    localdoc = None

    def __init__(self):
        self._quickrun = False

        # Main operational widgets for the form
        self.window = QtWidgets.QWidget()
        self.window.setLayout(QtWidgets.QVBoxLayout())
        if self.label:
            self.window.setWindowTitle(self.label)

        for attr in ('label', 'target', 'validator', 'localdoc'):
            if not getattr(self, attr):
                warnings.warn('Class attribute %s.%s is not defined' % (
                    self.__class__.__name__, attr))

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

        # Menu items.
        self.file_menu = QtWidgets.QMenu('&File')
        self.save_to_scenario = self.file_menu.addAction(
            'Save scenario as ...', self._save_scenario_as,
            QtGui.QKeySequence(QtGui.QKeySequence.SaveAs))

    def _save_scenario_as(self):
        file_dialog = inputs.FileDialog()
        save_filepath = file_dialog.save_file(
            title='Save current parameters as scenario',
            start_dir=None,  # might change later, last dir is fine
            savefile='%s_scenario.invs.json' % (
                '.'.join(self.target.__name__.split('.')[2:-1])))
        LOGGER.info('Saved current parameters to scenario file %s',
                    save_filepath)

    def _show_alert(self):
        self.systray_icon.showMessage(
            'InVEST', 'Model run finished')

    def _close_model(self):
        # exit with an error code that matches exception status of run.
        exit_code = self.form.run_dialog.messageArea.error
        inputs.QT_APP.exit(int(exit_code))

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
            LOGGER.info('Setting validator of %s to %s',
                        input, self.validator)
            input.validator = self.validator
        elif not hasattr(input, 'validator'):
            LOGGER.info('Input does not have a validator at all: %s',
                        input)
        else:
            LOGGER.info('Validator already set for %s: %s',
                        input, input.validator)

        self.form.add_input(input)

    def execute_model(self):
        args = self.assemble_args()

        def _logged_target():
            name = self.target.__name__
            with utils.prepare_workspace(args['workspace_dir'], name):
                return self.target(args=args)

        self.form.run(target=_logged_target,
                      window_title='Running %s' % self.label,
                      out_folder=args['workspace_dir'])

    def load_scenario(self, scenario_path):
        LOGGER.info('Loading scenario from %s', scenario_path)
        paramset = scenarios.read_parameter_set(scenario_path)
        self.load_args(paramset.args)

    def load_args(self, scenario_args):
        _inputs = dict((attr.args_key, attr) for attr in
                       self.__dict__.itervalues()
                       if isinstance(attr, inputs.Input))
        LOGGER.debug(pprint.pformat(_inputs))

        for args_key, args_value in scenario_args.iteritems():
            try:
                _inputs[args_key].set_value(args_value)
            except KeyError:
                LOGGER.warning(('Scenario args_key %s not associated with '
                                'any inputs'), args_key)

    def assemble_args(self):
        raise NotImplementedError

    def run(self, quickrun=False):
        if quickrun:
            self.form.run_finished.connect(self._close_model)
            QtCore.QTimer.singleShot(50, self.execute_model)

        self.window.show()
        self.window.raise_()  # raise window to top of stack.

        return inputs.QT_APP.exec_()
