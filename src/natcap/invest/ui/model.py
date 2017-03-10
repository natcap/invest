from __future__ import absolute_import

import logging
import os
import pprint
import warnings
import collections
import json
import textwrap

from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui
import natcap.invest
from natcap.ui import inputs
import qtawesome

from .. import cli
from .. import utils
from .. import scenarios

LOG_FMT = "%(asctime)s %(name)-18s %(levelname)-8s %(message)s"
DATE_FMT = "%m/%d/%Y %H:%M:%S "
LOGGER = logging.getLogger(__name__)

_SCENARIO_BASE_FILENAME = 'scenario.invs.%s'
_SCENARIO_DIALOG_TITLE = 'Select where to save the parameter %s'
_SCENARIO_PARAMETER_SET = 'Parameter set'
_SCENARIO_DATA_ARCHIVE = 'Data archive'
_SCENARIO_SAVE_OPTS = {
    _SCENARIO_PARAMETER_SET: {
        'title': _SCENARIO_DIALOG_TITLE % 'set',
        'savefile': _SCENARIO_BASE_FILENAME % 'json',
    },
    _SCENARIO_DATA_ARCHIVE: {
        'title': _SCENARIO_DIALOG_TITLE % 'archive',
        'savefile': _SCENARIO_BASE_FILENAME % 'tar.gz',
    }
}


def try_cast(value, target_type):
    try:
        return target_type(value)
    except ValueError:
        return value


class WindowTitle(QtCore.QObject):

    title_changed = QtCore.Signal(unicode)
    format_string = "{modelname}: {filename}{modified}"

    def __init__(self, modelname=None, filename=None, modified=False):
        QtCore.QObject.__init__(self)
        self.modelname = modelname
        self.filename = filename
        self.modified = modified

    def __setattr__(self, name, value):
        LOGGER.info('__setattr__: %s, %s', name, value)
        old_attr = getattr(self, name, 'None')
        QtCore.QObject.__setattr__(self, name, value)
        if old_attr != value:
            new_value = repr(self)
            LOGGER.info('Emitting new title %s', new_value)
            self.title_changed.emit(new_value)

    def __repr__(self):
        try:
            return self.format_string.format(
                modelname=self.modelname if self.modelname else 'InVEST',
                filename=self.filename if self.filename else 'Scenario1',
                modified='*' if self.modified else '')
        except AttributeError:
            return ''

ScenarioSaveOpts = collections.namedtuple(
    'ScenarioSaveOpts', 'scenario_type use_relpaths include_workspace')


def _prompt_for_scenario_options():
    dialog = QtWidgets.QDialog()
    dialog.setLayout(QtWidgets.QVBoxLayout())
    dialog.setWindowModality(QtCore.Qt.WindowModal)

    prompt = inputs.Container(label='Scenario options')
    dialog.layout().addWidget(prompt)

    scenario_type = inputs.Dropdown(
        label='Scenario type',
        options=_SCENARIO_SAVE_OPTS.keys())
    scenario_type.set_value(_SCENARIO_PARAMETER_SET)  # default selection
    prompt.add_input(scenario_type)
    use_relative_paths = inputs.Checkbox(
        label='Use relative paths')
    include_workspace = inputs.Checkbox(
        label='Include workspace in scenario')
    include_workspace.set_value(False)
    prompt.add_input(use_relative_paths)
    prompt.add_input(include_workspace)

    @QtCore.Slot(unicode)
    def _optionally_disable(value):
        use_relative_paths.set_interactive(value == _SCENARIO_PARAMETER_SET)
    scenario_type.value_changed.connect(_optionally_disable)

    buttonbox = QtWidgets.QDialogButtonBox()
    ok_button = QtWidgets.QPushButton(' Continue')
    ok_button.setIcon(inputs.ICON_ENTER)
    ok_button.pressed.connect(dialog.accept)
    buttonbox.addButton(ok_button, QtWidgets.QDialogButtonBox.AcceptRole)
    cancel_button = QtWidgets.QPushButton(' Cancel')
    cancel_button.setIcon(qtawesome.icon('fa.times',
                                         color='grey'))
    cancel_button.pressed.connect(dialog.reject)
    buttonbox.addButton(cancel_button, QtWidgets.QDialogButtonBox.RejectRole)
    dialog.layout().addWidget(buttonbox)

    dialog.raise_()
    dialog.show()
    result = dialog.exec_()
    if result == QtWidgets.QDialog.Accepted:
        return ScenarioSaveOpts(
            scenario_type.value(), use_relative_paths.value(),
            include_workspace.value())
    return None


class Model(QtWidgets.QMainWindow):
    label = None
    target = None
    validator = None
    localdoc = None

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self._quickrun = False
        self._validator = inputs.Validator(parent=self)
        self._validator.finished.connect(self._validation_finished)

        # These attributes should be defined in subclass
        for attr in ('label', 'target', 'validator', 'localdoc'):
            if not getattr(self, attr):  # None unless overridden in subclass
                warnings.warn('Class attribute %s.%s is not defined' % (
                    self.__class__.__name__, attr))

        # Main operational widgets for the form
        self._central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self._central_widget)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        self._central_widget.setLayout(QtWidgets.QVBoxLayout())
        self.status_bar = QtWidgets.QStatusBar()
        self.validation_warning = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.validation_warning)
        self.setStatusBar(self.status_bar)
        self.menuBar().setNativeMenuBar(True)
        self._central_widget.layout().setSizeConstraint(
            QtWidgets.QLayout.SetMinimumSize)

        self.window_title = WindowTitle()
        self.window_title.title_changed.connect(self.setWindowTitle)
        self.window_title.modelname = self.label

        # Format the text links at the top of the window.
        self.links = QtWidgets.QLabel()
        self.links.setAlignment(QtCore.Qt.AlignRight)
        self.links.setOpenExternalLinks(True)
        self.links.setText(' | '.join((
            'InVEST version %s' % natcap.invest.__version__,
            '<a href="file://%s">Model documentation</a>' % self.localdoc,
            ('<a href="http://forums.naturalcapitalproject.org">'
             'Report an issue</a>'))))
        self._central_widget.layout().addWidget(self.links)

        self.form = inputs.Form()
        self._central_widget.layout().addWidget(self.form)
        self.run_dialog = inputs.FileSystemRunDialog()

        # start with workspace and suffix inputs
        self.workspace = inputs.Folder(args_key='workspace_dir',
                                       label='Workspace',
                                       validator=self.validator,
                                       required=True)

        # natcap.invest.pollination.pollination --> pollination
        self.workspace.set_value(os.path.normpath(
            os.path.expanduser('~/Documents/{model}_workspace').format(
                model=self.target.__module__.split('.')[-1])))

        self.suffix = inputs.Text(args_key='suffix',
                                  label='Results suffix',
                                  validator=self.validator,
                                  required=False)
        self.suffix.textfield.setMaximumWidth(150)
        self.add_input(self.workspace)
        self.add_input(self.suffix)

        self.form.submitted.connect(self.execute_model)

        # Menu items.
        self.file_menu = QtWidgets.QMenu('&File')
        self.file_menu.addAction(
            'Save as ...', self._save_scenario_as,
            QtGui.QKeySequence(QtGui.QKeySequence.SaveAs))
        self.file_menu.addAction(
            'Open ...', self.load_scenario,
            QtGui.QKeySequence(QtGui.QKeySequence.Open))
        self.file_menu.addAction(
            'Quit', self.close,
            QtGui.QKeySequence(QtGui.QKeySequence.Quit))
        self.file_menu.addAction(
            'About', self.about)
        self.menuBar().addMenu(self.file_menu)

        # Settings files
        self.settings = QtCore.QSettings(
            QtCore.QSettings.IniFormat,
            QtCore.QSettings.UserScope,
            'Natural Capital Project',
            self.label)

    def about(self):
        about_dialog = QtWidgets.QDialog()
        about_dialog.setLayout(QtWidgets.QVBoxLayout())
        label_text = textwrap.dedent(
            """
            <h1>InVEST</h1>
            <b>Version {version}</b> <br/> <br/>

            Documentation:<br/>
            Homepage: http://naturalcapitalproject.org<br/>
            Copyright 2017, The Natural Capital Project<br/>
            License: BSD 3-clause<br/>
            Project page: https://bitbucket.org/natcap/invest<br/>

            <h2>Open-Source Licenses</h2>
            """.format(
                version=natcap.invest.__version__))

        for lib_name, lib_license, lib_homepage in [
                ('PyInstaller', 'GPL', 'http://pyinstaller.org'),
                ('GDAL', 'MIT and others', 'http://gdal.org'),
                ('matplotlib', 'BSD', 'http://matplotlib.org'),
                ('natcap.versioner', 'BSD',
                 'http://bitbucket.org/jdouglass/versioner'),
                ('numpy', 'BSD', 'http://numpy.org'),
                ('pyamg', 'BSD', 'http://github.com/pyamg/pyamg'),
                ('pygeoprocessing', 'BSD',
                 'http://bitbucket.org/richpsharp/pygeoprocessing'),
                ('PyQt', 'GPL',
                 'http://riverbankcomputing.com/software/pyqt/intro'),
                ('rtree', 'LGPL', 'http://toblerity.org/rtree/'),
                ('scipy', 'BSD', 'http://www.scipy.org/'),
                ('shapely', 'BSD', 'http://github.com/Toblerity/Shapely')]:
            label_text += '{project} {license} {homepage} <br/>'.format(
                project=lib_name,
                license=('<a href="licenses/{project}_license.txt">{license}'
                         '</a>').format(project=lib_name,
                                        license=lib_license),
                homepage='<a href="{0}">{0}</a>'.format(lib_homepage))

        label = QtWidgets.QLabel(label_text)
        label.setTextFormat(QtCore.Qt.RichText)
        label.setOpenExternalLinks(True)
        about_dialog.layout().addWidget(label)

        button_box = QtWidgets.QDialogButtonBox()
        accept_button = QtWidgets.QPushButton('OK')
        button_box.addButton(accept_button,
                             QtWidgets.QDialogButtonBox.AcceptRole)
        about_dialog.layout().addWidget(button_box)
        accept_button.pressed.connect(about_dialog.close)

        about_dialog.exec_()

    def _save_scenario_as(self):
        scenario_opts = _prompt_for_scenario_options()
        if not scenario_opts:  # user pressed cancel
            return

        file_dialog = inputs.FileDialog()
        save_filepath, last_filter = file_dialog.save_file(
            title=_SCENARIO_SAVE_OPTS[scenario_opts.scenario_type]['title'],
            start_dir=None,  # might change later, last dir is fine
            savefile='{model}_{file_base}'.format(
                model='.'.join(self.target.__module__.split('.')[2:-1]),
                file_base=_SCENARIO_SAVE_OPTS[
                    scenario_opts.scenario_type]['savefile']))

        if not save_filepath:
            # The user pressed cancel.
            return

        current_args = self.assemble_args()
        if (not scenario_opts.include_workspace or
                scenario_opts.scenario_type == _SCENARIO_DATA_ARCHIVE):
            del current_args['workspace_dir']

        LOGGER.info('Current parameters:\n%s', pprint.pformat(current_args))

        if scenario_opts.scenario_type == _SCENARIO_DATA_ARCHIVE:
            scenarios.build_scenario_archive(
                args=current_args,
                scenario_path=save_filepath
            )
        else:
            scenarios.write_parameter_set(
                filepath=save_filepath,
                args=current_args,
                name=self.target.__module__,
                relative=scenario_opts.use_relpaths
            )

        alert_message = (
            'Saved current parameters to %s' % save_filepath)
        LOGGER.info(alert_message)
        self.status_bar.showMessage(alert_message, 10000)
        self.window_title.filename = os.path.basename(save_filepath)

    def add_input(self, input):
        self.form.add_input(input)

    def execute_model(self):
        args = self.assemble_args()

        # If the workspace exists, confirm the overwrite.
        if os.path.exists(args['workspace_dir']):
            dialog = QtWidgets.QMessageBox()
            dialog.setWindowFlags(QtCore.Qt.Dialog)
            dialog.setText('Workspace exists!')
            dialog.setInformativeText(
                'Overwrite files from a previous run?')
            dialog.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            dialog.setDefaultButton(QtWidgets.QMessageBox.No)
            dialog.setIconPixmap(
                qtawesome.icon(
                    'fa.exclamation-triangle',
                    color='orange').pixmap(100, 100))

            button_pressed = dialog.exec_()
            if button_pressed != QtWidgets.QMessageBox.Yes:
                return

        def _logged_target():
            name = self.target.__name__
            with utils.prepare_workspace(args['workspace_dir'], name):
                LOGGER.info('Starting model with parameters: \n%s',
                            cli._format_args(args))
                return self.target(args=args)

        self.form.run(target=_logged_target,
                      window_title='Running %s' % self.label,
                      out_folder=args['workspace_dir'])

    @QtCore.Slot()
    def load_scenario(self, scenario_path=None):
        if not scenario_path:
            file_dialog = inputs.FileDialog()
            scenario_path = file_dialog.open_file(
                title='Select scenario')

        LOGGER.info('Loading scenario from %s', scenario_path)
        try:
            paramset = scenarios.read_parameter_set(scenario_path)
            args = paramset.args
        except ValueError:
            # when a JSON object cannot be decoded, assume it's a logfile.
            args = scenarios.read_parameters_from_logfile(scenario_path)
        self.load_args(args)
        self.status_bar.showMessage(
            'Loaded scenario from %s' % os.path.abspath(scenario_path), 10000)

        self.window_title.filename = os.path.basename(scenario_path)

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
            except Exception:
                LOGGER.exception('Error setting %s to %s', args_key,
                                 args_value)

    def assemble_args(self):
        raise NotImplementedError

    def _validation_finished(self, validation_warnings):
        inputs.QT_APP.processEvents()
        LOGGER.info('Whole-model validation finished with: %s',
                    validation_warnings)
        # Double-check that there aren't any required inputs that aren't
        # satisfied.
        if (validation_warnings or any(
                [input_.required and not input_.valid()
                 for input_ in self.inputs()])):
            self.validation_warning.setText('Warnings!')
            icon = qtawesome.icon('fa.times', color='red')
        else:
            self.validation_warning.setText('All good!')
            icon = qtawesome.icon('fa.check', color='green')
        self.validation_warning.setPixmap(icon.pixmap(16, 16))

    def inputs(self):
        return [ref for ref in self.__dict__.values()
                if isinstance(ref, inputs.Input)]

    def run(self, quickrun=False):
        # recurse through attributes of self.form.  If the attribute is an
        # instance of inputs.Input, then link its value_changed signal to the
        # model-wide validation slot.
        def _validate_all_inputs(new_value):
            self._validator.validate(
                target=self.validator,
                args=self.assemble_args(),
                limit_to=None)

        for input_obj in self.inputs():
            input_obj.value_changed.connect(_validate_all_inputs)
            try:
                input_obj.validity_changed.connect(_validate_all_inputs)
            except AttributeError:
                # Not all inputs can have validity (e.g. Container, dropdown)
                pass

        # Set up quickrun options if we're doing a quickrun
        if quickrun:
            @QtCore.Slot()
            def _quickrun_close_model():
                # exit with an error code that matches exception status of run.
                exit_code = self.form.run_dialog.messageArea.error
                inputs.QT_APP.exit(int(exit_code))

            self.form.run_finished.connect(_quickrun_close_model)
            QtCore.QTimer.singleShot(50, self.execute_model)

        # The scrollArea defaults to a size that is too small to actually view
        # the contents of the enclosed widget appropriately.  By adjusting the
        # size here, we ensure that the widgets are an appropriate height.
        # Note that self.resize() does take the window size into account, so
        # all parts of the application window will still be visible, even if
        # the minimumSize().height() would have it extend over the edge of the
        # screen.
        self.resize(
            self.form.scroll_area.widget().minimumSize().width()+100,
            self.form.scroll_area.widget().minimumSize().height()+150)

        inputs.center_window(self)

        # if we're not working off a scenario file, load the last run.
        if not self.window_title.filename:
            self.load_lastrun()

        self.show()
        self.raise_()  # raise window to top of stack.

        return inputs.QT_APP.exec_()

    def closeEvent(self, event):
        dialog = QtWidgets.QMessageBox()
        dialog.setWindowFlags(QtCore.Qt.Dialog)
        dialog.setText('Are you sure you want to quit?')
        dialog.setInformativeText(
            'Any unsaved changes to your parameters will be lost.')
        dialog.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        dialog.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        dialog.setIconPixmap(
            qtawesome.icon(
                'fa.question').pixmap(100, 100))

        button_pressed = dialog.exec_()
        if button_pressed != QtWidgets.QMessageBox.Yes:
            event.ignore()
        else:
            self.save_lastrun()

    def save_lastrun(self, lastrun_args=None):
        if not lastrun_args:
            lastrun_args = self.assemble_args()
        LOGGER.debug('Saving lastrun args %s', lastrun_args)
        self.settings.setValue("lastrun", json.dumps(lastrun_args))

    def load_lastrun(self):
        lastrun_args = self.settings.value("lastrun")
        # If no lastrun args saved, None is returned.
        if not lastrun_args:
            return
        self.load_args(json.loads(lastrun_args))
        self.status_bar.showMessage('Loaded parameters from previous run.',
                                    10000)
