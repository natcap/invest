# coding=UTF-8
from __future__ import absolute_import

import logging
import os
import pprint
import warnings
import collections
import json
import textwrap
import cgi
import tarfile
import contextlib
import functools

from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui
import natcap.invest
import qtawesome

from . import inputs
from . import usage
from .. import cli
from .. import utils
from .. import scenarios

LOG_FMT = "%(asctime)s %(name)-18s %(levelname)-8s %(message)s"
DATE_FMT = "%m/%d/%Y %H:%M:%S "
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
QT_APP = inputs.QT_APP

ICON_BACK = qtawesome.icon('fa.arrow-circle-o-left',
                           color='grey')
ICON_ALERT = qtawesome.icon('fa.exclamation-triangle',
                            color='orange')
_ONLINE_DOCS_LINK = (
    'http://data.naturalcapitalproject.org/nightly-build/'
    'invest-users-guide/html/')
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


@contextlib.contextmanager
def wait_on_signal(signal, timeout=250):
    """Block loop until signal emitted, or timeout (ms) elapses."""
    global QT_APP
    loop = QtCore.QEventLoop()
    signal.connect(loop.quit)

    try:
        yield
        if QT_APP.hasPendingEvents():
            QT_APP.processEvents()
    except Exception as error:
        LOGGER.exception('Error encountered while witing for signal %s',
                         signal)
        raise error
    finally:
        if timeout is not None:
            QtCore.QTimer.singleShot(timeout, loop.quit)
        loop.exec_()
    loop = None


def about():
    """Show a dialog describing InVEST.

    In reasonable accordance with licensing and distribution requirements,
    this dialog not only has information about InVEST and the Natural
    Capital Project, but it also has details about the software used to
    develop and run InVEST and contains links to the licenses for each of
    these other projects.

    Returns:
        None."""
    about_dialog = QtWidgets.QDialog()
    about_dialog.setLayout(QtWidgets.QVBoxLayout())
    label_text = textwrap.dedent(
        """
        <h1>InVEST</h1>
        <b>Version {version}</b> <br/> <br/>

        Documentation: <a href="http://data.naturalcapitalproject.org/nightly-
        build/invest-users-guide/html/">online</a><br/>
        Homepage: <a href="http://naturalcapitalproject.org">
                    naturalcapitalproject.org</a><br/>
        Copyright 2017, The Natural Capital Project<br/>
        License: BSD 3-clause<br/>
        Project page: <a href="https://bitbucket.org/natcap/invest">
                        bitbucket.org/natcap/invest</a><br/>

        <h2>Open-Source Licenses</h2>
        """.format(
            version=natcap.invest.__version__))

    label_text += "<table>"
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
        label_text += (
            '<tr>'
            '<td>{project}  </td>'
            '<td>{license}  </td>'
            '<td>{homepage}  </td></tr/>').format(
                project=lib_name,
                license=(
                    '<a href="licenses/{project}_license.txt">'
                    '{license}</a>').format(project=lib_name,
                                            license=lib_license),
                homepage='<a href="{0}">{0}</a>'.format(lib_homepage))

    label_text += "</table>"

    label = QtWidgets.QLabel(label_text)
    label.setTextFormat(QtCore.Qt.RichText)
    label.setOpenExternalLinks(True)
    about_dialog.layout().addWidget(label)

    button_box = QtWidgets.QDialogButtonBox()
    accept_button = QtWidgets.QPushButton('OK')
    button_box.addButton(accept_button,
                         QtWidgets.QDialogButtonBox.AcceptRole)
    about_dialog.layout().addWidget(button_box)
    accept_button.clicked.connect(about_dialog.close)

    about_dialog.exec_()


def _check_docs_link_exists(link):
    if link.startswith('file://'):
        local_link_path = link.replace('file://', '')

        if os.path.exists(local_link_path):
            QtCore.QDesktopServices.OpenUrl(link)
        else:
            LOGGER.info("Can't find docs link %s", local_link_path)
            dialog = QtWidgets.QMessageBox()
            dialog.setWindowFlags(QtCore.Qt.Dialog)
            dialog.setText("<h2>Local docs not found<h2>")
            remote_link = _ONLINE_DOCS_LINK + os.path.basename(local_link_path)
            dialog.setInformativeText(
                'Online docs: [<a href="%s">documentation</a>]'
                '<br/><br/>Local documentation link could not be found: %s' %
                (remote_link, local_link_path))
            dialog.setStandardButtons(
                QtWidgets.QMessageBox.Ok)
            dialog.setIconPixmap(
                qtawesome.icon(
                    'fa.exclamation-triangle',
                    color='orange').pixmap(100, 100))
            dialog.show()
            dialog.exec_()
    elif link.startswith(('http', 'ftp')):
        QtCore.QDesktopServices.OpenUrl(link)
    else:
        LOGGER.warning("Don't know how to open link %s", link)


class WindowTitle(QtCore.QObject):
    """Object to manage the title string for a model window.

    The title string is dependent on several characteristics:

        * The name of the model currently being run.
        * The filename (basename) of the current scenario file
        * Whether the scenario has been modified from the time it was last
            saved.

    The window's title is updated based on the state of three attributes.
    These attributes may be initialized by using the parameters to
    ``__init__()``, or by updating the appropriate attribute after object
    creation:

    When any attributes are changed, this object emits the ``title_changed``
    signal with the new title string.

    Attributes:
        modelname (string or None): If a string, this is assumed to be the
            name of the model.  If ``None``, the string ``"InVEST"`` is
            assumed in the window title.
        filename (string or None): If a string, the filename to be displayed
            to the user in the title bar.  No manipulations are performed on
            this filename; it will be used verbatim.  If ``None``,
            ``"new scenario"`` is assumed.
        modified (bool): Whether the scenario file has been modified.  If so,
            a ``'*'`` is displayed next to the scenario filename.
    """

    title_changed = QtCore.Signal(unicode)
    format_string = "{modelname}: {filename}{modified}"

    def __init__(self, modelname=None, filename=None, modified=False):
        """Initialize the WindowTitle.

        Parameters:
            modelname (string or None): The modelname to use.
            filename (string or None): The filename to use.
            modified (bool): Whether the scenario file has been modified.
        """
        QtCore.QObject.__init__(self)
        self.modelname = modelname
        self.filename = filename
        self.modified = modified

    def __setattr__(self, name, value):
        """Attribute setter.

        Set the given attribute and emit the ``title_changed`` signal with
        the new window title if the rendered title is different from the
        previous title.

        Parameters:
            name (string): the name of the attribute to set.
            value: The new value for the attribute.
        """
        LOGGER.info('__setattr__: %s, %s', name, value)
        old_attr = getattr(self, name, 'None')
        QtCore.QObject.__setattr__(self, name, value)
        if old_attr != value:
            new_value = repr(self)
            LOGGER.info('Emitting new title %s', new_value)
            self.title_changed.emit(new_value)

    def __repr__(self):
        """Produce a string representation of the window title.

        Returns:
            The string wundow title."""
        try:
            return self.format_string.format(
                modelname=self.modelname if self.modelname else 'InVEST',
                filename=self.filename if self.filename else 'new scenario',
                modified='*' if self.modified else '')
        except AttributeError:
            return ''

ScenarioSaveOpts = collections.namedtuple(
    'ScenarioSaveOpts', 'scenario_type use_relpaths include_workspace')


def _prompt_for_scenario_options():
    """Provide a GUI model dialog with options for saving a scenario.

    There are two types of scenarios:

        * Parameter sets (a file with the values of the current inputs)
        * Data archives (all-inclusive archive of current inputs)

    This dialog provides a couple of options to the user depending on which
    type of scenario is desired.  If a parameter set is selected, paths may
    be stored relative to the location of the scenario file.  Both types of
    scenarios may optionally include the value of the workspace input.

    Returns:
        An instance of :ref:ScenarioSaveOpts namedtuple.
    """
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
        label='Include workspace path in scenario')
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


def _prompt_for_scenario_archive_extraction(archive_path):
    dialog = QtWidgets.QDialog()
    dialog.setLayout(QtWidgets.QVBoxLayout())
    dialog.setWindowModality(QtCore.Qt.WindowModal)

    container = inputs.Container(label='Scenario extraction parameters')
    dialog.layout().addWidget(container)

    extraction_point = inputs.Folder(
        label='Where should this archive be extracted?',
    )

    container.add_input(extraction_point)

    buttonbox = QtWidgets.QDialogButtonBox()
    ok_button = QtWidgets.QPushButton(' Extract')
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
        extract_to_dir = extraction_point.value()
        args = scenarios.extract_scenario_archive(
            archive_path, extract_to_dir)
        return (args, extract_to_dir)


class WholeModelValidationErrorDialog(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        self.warnings = []
        self.setLayout(QtWidgets.QVBoxLayout())

        self.title_icon = QtWidgets.QLabel()
        self.title_icon.setPixmap(ICON_ALERT.pixmap(75, 75))
        self.title_icon.setAlignment(QtCore.Qt.AlignCenter)
        self.title = QtWidgets.QWidget()
        self.title.setLayout(QtWidgets.QHBoxLayout())
        self.title.layout().addWidget(self.title_icon)

        self.title_label = QtWidgets.QLabel('<h2>Validating inputs ...</h2>')
        self.title.layout().addWidget(self.title_label)
        self.layout().addWidget(self.title)

        self.scroll_widget = QtWidgets.QScrollArea()
        self.scroll_widget.setWidgetResizable(True)
        self.scroll_widget_container = QtWidgets.QWidget()
        self.scroll_widget_container.setLayout(QtWidgets.QVBoxLayout())
        self.scroll_widget.setWidget(self.scroll_widget_container)
        self.scroll_widget.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.layout().addWidget(self.scroll_widget)

        self.label = QtWidgets.QLabel('')
        self.label.setWordWrap(True)
        self.scroll_widget.widget().layout().addWidget(self.label)
        self.scroll_widget.widget().layout().insertStretch(-1)

        self.buttonbox = QtWidgets.QDialogButtonBox()
        self.back_button = QtWidgets.QPushButton(' Back')
        self.back_button.setIcon(ICON_BACK)
        self.back_button.clicked.connect(self.close)
        self.buttonbox.addButton(self.back_button,
                                 QtWidgets.QDialogButtonBox.RejectRole)
        self.layout().addWidget(self.buttonbox)

    def validation_started(self):
        self.label.setText('<h2>Validating inputs ...</h2>')

    def validation_finished(self, validation_warnings):
        LOGGER.info('Posting validation warnings to WMV dialog: %s',
                    validation_warnings)
        self.warnings = validation_warnings

        if validation_warnings:
            # cgi.escape handles escaping of characters <, >, &, " for HTML.
            self.title_label.setText(
                '<h2>Validation warnings found</h2>'
                '<h4>To ensure the model works as expected, please fix these '
                'erorrs:</h4>')
            self.label.setText(
                '<ul>%s</ul>' % ''.join(
                    ['<li><b>%s</b>: %s</li>' % (
                        ', '.join(labels), cgi.escape(warning_, quote=True))
                     for labels, warning_ in validation_warnings]))
            self.label.repaint()
            self.label.setVisible(True)


class Model(QtWidgets.QMainWindow):

    """An InVEST model window.

    This class represents an abstraction of a variety of Qt widgets that
    together comprise an InVEST model window.  This class is designed to be
    subclassed for each invdividual model.  Subclasses must, at a minimum,
    override these four attributes at the class level:

        * ``label`` (string): The model label.
        * ``target`` (function reference): The reference to the target function.
            For InVEST, this will always be the ``execute`` function of the
            target model.
        * ``validator`` (function reference): The reference to the target
            validator function.  For InVEST, this will always be the
            ``validate`` function of the target model.
        * ``localdoc`` (string): The filename of the user's guide chapter for
            this model.

    If any of these attributes are not overridden, a warning will be raised.
    """

    label = None
    target = None
    validator = None
    localdoc = None

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAcceptDrops(True)
        self._quickrun = False
        self._validator = inputs.Validator(parent=self)
        self._validator.finished.connect(self._validation_finished)
        self._validation_report_dialog = WholeModelValidationErrorDialog()

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
        self.links.setText(' | '.join((
            'InVEST version %s' % natcap.invest.__version__,
            '<a href="localdocs">Model documentation</a>',
            ('<a href="http://forums.naturalcapitalproject.org">'
             'Report an issue</a>'))))
        self._central_widget.layout().addWidget(self.links)
        self.links.linkActivated.connect(self._check_local_docs)

        self.form = inputs.Form()
        self._central_widget.layout().addWidget(self.form)
        self.run_dialog = inputs.FileSystemRunDialog()

        # start with workspace and suffix inputs
        self.workspace = inputs.Folder(args_key='workspace_dir',
                                       label='Workspace',
                                       validator=self.validator)

        # natcap.invest.pollination.pollination --> pollination
        self.workspace.set_value(os.path.normpath(
            os.path.expanduser('~/Documents/{model}_workspace').format(
                model=self.target.__module__.split('.')[-1])))

        self.suffix = inputs.Text(args_key='suffix',
                                  label='Results suffix',
                                  validator=self.validator)
        self.suffix.textfield.setMaximumWidth(150)
        self.add_input(self.workspace)
        self.add_input(self.suffix)

        self.form.submitted.connect(self.execute_model)

        # Menu items.
        self.file_menu = QtWidgets.QMenu('&File')
        self.file_menu.addAction(
            qtawesome.icon('fa.floppy-o'),
            'Save as ...', self._save_scenario_as,
            QtGui.QKeySequence(QtGui.QKeySequence.SaveAs))
        self.file_menu.addAction(
            qtawesome.icon('fa.arrow-circle-o-up'),
            'Open parameter file ...', self.load_scenario,
            QtGui.QKeySequence(QtGui.QKeySequence.Open))
        self.file_menu.addAction(
            'Quit', self.close,
            QtGui.QKeySequence('Ctrl+Q'))
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help')
        self.help_menu.addAction(
            qtawesome.icon('fa.info'),
            'About InVEST', about)
        self.help_menu.addAction(
            qtawesome.icon('fa.external-link'),
            'View documentation', self._check_local_docs)
        self.menuBar().addMenu(self.help_menu)

        # Settings files
        self.settings = QtCore.QSettings(
            QtCore.QSettings.IniFormat,
            QtCore.QSettings.UserScope,
            'Natural Capital Project',
            self.label)

    def _check_local_docs(self, link=None):
        path = 'file://' + os.path.abspath(self.localdoc)
        _check_docs_link_exists(path)

    def dragEnterEvent(self, event):
        # Determine whether to accept or reject a drop
        # Drag/drop must be a single file and must have a discernable scenario
        # format.
        inputs._handle_drag_enter_event(self, event)

    def dropEvent(self, event):
        # When a file is dropped onto the window
        scenario_path = inputs._handle_drop_enter_event(self, event)
        self.load_scenario(scenario_path)

    def _save_scenario_as(self):
        """Save the current set of inputs as a scenario.

        Presents a dialog to the user for input on how to save the scenario,
        and then makes it happen.  A status message is displayed to the
        satus bar when the operation is complete.

        Returns:
           ``None``."""
        scenario_opts = _prompt_for_scenario_options()
        if not scenario_opts:  # user pressed cancel
            return

        file_dialog = inputs.FileDialog()
        save_filepath, last_filter = file_dialog.save_file(
            title=_SCENARIO_SAVE_OPTS[scenario_opts.scenario_type]['title'],
            start_dir=None,  # might change later, last dir is fine
            savefile='{model}_{file_base}'.format(
                model=self.target.__module__.split('.')[-1],
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
                name=self.target.__module__,
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
        """Add an input to the model.

        Parameters:
            input (natcap.invest.ui.inputs.Input): An Input instance to add to
                the model.

        Returns:
            ``None``"""
        self.form.add_input(input)

    def is_valid(self):
        if self._validation_report_dialog.warnings:
            return False
        return True

    def show_validation_window(self):
        self._validation_report_dialog.show()
        return self._validation_report_dialog.exec_()

    def execute_model(self):
        """Run the target model.

        Executing the target model is the objective of the UI.  Once this
        method is triggered, the following steps are taken:

            * Collect all of the inputs into an ``args`` dictionary.
            * Verify that all of the ``args`` passes validation.  If not,
              the model cannot be run, and the user must correct the errors
              before running it.
            * If the workspace directory exists, prompt the user to confirm
              overwriting the files in the workspace.  Return to the inputs
              if the dialog is cancelled.
            * Run the model, capturing all GDAL log messages as python logging
              messages, writing log messages to a logfile within the workspace,
              and finally executing the model.

        Returns:
            ``None``
        """
        args = self.assemble_args()

        # If we have validation warnings, show them and return to inputs.
        if self._validation_report_dialog.warnings:
            self._validation_report_dialog.show()
            self._validation_report_dialog.exec_()
            return

        # If the workspace exists, confirm the overwrite.
        if os.path.exists(args['workspace_dir']):
            dialog = QtWidgets.QMessageBox()
            dialog.setWindowFlags(QtCore.Qt.Dialog)
            dialog.setText('<h2>Workspace exists!<h2>')
            dialog.setInformativeText(
                'Overwrite files from a previous run?')
            dialog.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            dialog.setDefaultButton(QtWidgets.QMessageBox.No)
            dialog.setIconPixmap(
                ICON_ALERT.pixmap(100, 100))

            button_pressed = dialog.exec_()
            if button_pressed != QtWidgets.QMessageBox.Yes:
                return

        def _logged_target():
            name = self.target.__name__
            with utils.prepare_workspace(args['workspace_dir'], name):
                with usage.log_run(name.replace('.execute', ''), args):
                    LOGGER.info('Starting model with parameters: \n%s',
                                cli._format_args(args))
                    return self.target(args=args)

        self.form.run(target=_logged_target,
                      window_title='Running %s' % self.label,
                      out_folder=args['workspace_dir'])

    @QtCore.Slot()
    def load_scenario(self, scenario_path=None):
        file_dialog = inputs.FileDialog()
        if not scenario_path:
            filters = (
                'Any file (*.*)',
                'Parameter set (*.invs.json)',
                'Parameter archive (*.invs.tar.gz)',
                'Logfile (*.txt)')
            scenario_path = file_dialog.open_file(
                title='Select scenario', filters=filters)

        # When the user pressed cancel, scenario_path == ''
        if not scenario_path:
            return

        LOGGER.info('Loading scenario from "%s"', scenario_path)
        if tarfile.is_tarfile(scenario_path):  # it's a scenario archive!
            # Where should the tarfile be extracted to?
            args, extract_dir = _prompt_for_scenario_archive_extraction(
                scenario_path)
            if args is None:
                return
            window_title_filename = os.path.basename(extract_dir)
        else:
            try:
                paramset = scenarios.read_parameter_set(scenario_path)
                args = paramset.args
            except ValueError:
                # when a JSON object cannot be decoded, assume it's a logfile.
                args = scenarios.read_parameters_from_logfile(scenario_path)
            window_title_filename = os.path.basename(scenario_path)

        self.load_args(args)
        self.window_title.filename = window_title_filename
        self.status_bar.showMessage(
            'Loaded scenario from %s' % os.path.abspath(scenario_path), 10000)

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
        LOGGER.info('Whole-model validation returned: %s',
                    validation_warnings)
        if validation_warnings:
            icon = qtawesome.icon('fa.times', color='red')
        else:
            icon = inputs.ICON_ENTER
        self.form.run_button.setIcon(icon)

        # post warnings to the WMV dialog
        args_to_inputs = dict((input_.args_key, input_) for input_ in
                              self.inputs())
        warnings_ = []
        for keys, warning in validation_warnings:
            warnings_.append(
                ((args_to_inputs[key].label for key in keys), warning))
        self._validation_report_dialog.validation_finished(warnings_)

    def inputs(self):
        return [ref for ref in self.__dict__.values()
                if isinstance(ref, inputs.Input)]

    def validate(self, block=True):
        validate_callable = functools.partial(
            self._validator.validate,
            target=self.validator,
            args=self.assemble_args(),
            limit_to=None)
        if block:
            with wait_on_signal(self._validator.finished):
                validate_callable()
        else:
            validate_callable()

    def run(self, quickrun=False):
        # iterate through attributes of self.form.  If the attribute is an
        # instance of inputs.Input, then link its value_changed signal to the
        # model-wide validation slot.
        for input_obj in self.inputs():
            input_obj.value_changed.connect(self.validate)
            try:
                input_obj.validity_changed.connect(self.validate)
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

    def closeEvent(self, event):
        dialog = QtWidgets.QMessageBox()
        dialog.setWindowFlags(QtCore.Qt.Dialog)
        dialog.setText('<h2>Are you sure you want to quit?</h2>')
        dialog.setInformativeText(
            'Any unsaved changes to your parameters will be lost.')
        dialog.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        dialog.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        dialog.setIconPixmap(
            qtawesome.icon(
                'fa.question').pixmap(100, 100))
        checkbox = QtWidgets.QCheckBox('Remember inputs')
        checkbox.setChecked(
            self.settings.value('remember_lastrun', True, bool))
        dialog.layout().addWidget(checkbox, dialog.layout().rowCount()-1,
                                  0, 1, 1)

        button_pressed = dialog.exec_()
        if button_pressed != QtWidgets.QMessageBox.Yes:
            event.ignore()
        elif checkbox.isChecked():
            self.save_lastrun()
        self.settings.setValue('remember_lastrun', checkbox.isChecked())

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
        self.window_title.filename = 'loaded from autosave'
