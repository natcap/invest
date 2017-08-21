# coding=UTF-8
"""Classes related to the InVEST Model class."""

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
from .. import validation

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
_SCENARIO_DIALOG_TITLE = 'Select where to save the scenario'
_SCENARIO_PARAMETER_SET = 'Parameter set (archive parameters)'
_SCENARIO_DATA_ARCHIVE = 'Data archive (archive parameters and files)'
_SCENARIO_SAVE_OPTS = {
    _SCENARIO_PARAMETER_SET: {
        'title': _SCENARIO_DIALOG_TITLE,
        'savefile': _SCENARIO_BASE_FILENAME % 'json',
    },
    _SCENARIO_DATA_ARCHIVE: {
        'title': _SCENARIO_DIALOG_TITLE,
        'savefile': _SCENARIO_BASE_FILENAME % 'tar.gz',
    }
}


@contextlib.contextmanager
def wait_on_signal(signal, timeout=250):
    """Block loop until signal emitted, or timeout (ms) elapses."""
    # TODO: need a global here?  You aren't setting the value...
    global QT_APP
    loop = QtCore.QEventLoop()
    signal.connect(loop.quit)

    try:
        yield
        if QT_APP.hasPendingEvents():
            QT_APP.processEvents()
    finally:
        if timeout is not None:
            QtCore.QTimer.singleShot(timeout, loop.quit)
        loop.exec_()
    loop = None


class OptionsDialog(QtWidgets.QDialog):
    """A common dialog class for Options-style functionality.

    Subclasses are required to implement a ``postprocess`` method that handles
    how to save the options in the dialog.  ``postprocess`` must take a single
    int argument with the name ``exitcode``.  The exit code of the dialog will
    be passed to the method when it is called.

    The buttons installed in this dialog are:

        * ``self.ok_button``: A button with the 'accept' role. The text of the
          button can be set via the ``accept_text`` parameter.
        * ``self.cancel_button``: A button with the 'reject' role. The text of
          the button can be set via the ``reject_text`` parameter.
    """

    def __init__(self, title=None, modal=False, accept_text='save',
                 reject_text='cancel'):
        """Initialize the OptionsDialog.

        Parameters:
            title=None (string): The title of the dialog.  If ``None``, the
                dialog title will not be set.
            modal=False (bool): The dialog's modality. If ``True``, the dialog
                will be modal.
            accept_text='save' (string): The text of the dialog-acceptance
                button.
            reject_text='cancel' (string): The text of the dialog-rejection
                button.

        Returns:
            ``None``
        """
        QtWidgets.QDialog.__init__(self)
        self._accept_text = ' ' + accept_text.strip()
        self._reject_text = ' ' + reject_text.strip()
        if title:
            self.setWindowTitle(title)

        self.setModal(modal)
        self.setLayout(QtWidgets.QVBoxLayout())

        self._buttonbox = None
        self.ok_button = QtWidgets.QPushButton(self._accept_text)
        self.ok_button.setIcon(inputs.ICON_ENTER)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QtWidgets.QPushButton(self._reject_text)
        self.cancel_button.setIcon(qtawesome.icon('fa.times',
                                                  color='grey'))
        self.cancel_button.clicked.connect(self.reject)

        self.finished.connect(self._call_postprocess)

    @QtCore.Slot(int)
    def _call_postprocess(self, exitcode):
        """A slot to call ``self.postprocess`` when the dialog is closed."""
        # need to have this bound method registered with the signal,
        # but then we'll call the subclass's postprocess method.
        try:
            self.postprocess(exitcode)
        except NotImplementedError:
            LOGGER.info('postprocess method not implemented for object '
                        '%s' % repr(self))

    def postprocess(self, exitcode=0):
        """Save the options in the dialog.

        Subclasses of ``OptionsDialog`` must reimplement this method.

        Parameters:
            exitcode=0 (int): The exit code of the dialog.

        Raises:
            NotImplementedError: This method must be reimplemented.

        Returns:
            ``None``
        """
        raise NotImplementedError

    def showEvent(self, showEvent):
        """Create the buttonbox at the end of the dialog when it's shown.

        Reimplemented to QDialog.showEvent.

        Parameters:
            showEvent (QEvent): The current showEvent.

        Returns:
            ``None``
        """
        # last thing: add the buttonbox if it hasn't been created yet.
        if not self._buttonbox:
            self._buttonbox = QtWidgets.QDialogButtonBox()
            self._buttonbox.addButton(self.ok_button,
                                      QtWidgets.QDialogButtonBox.AcceptRole)
            self._buttonbox.addButton(self.cancel_button,
                                      QtWidgets.QDialogButtonBox.RejectRole)
            self.layout().addWidget(self._buttonbox)

        QtWidgets.QDialog.show(self)


class QuitConfirmDialog(QtWidgets.QMessageBox):
    """A dialog for confirming that the user would like to quit.

    In addition to having accept and reject buttons, an icon, and some
    informative text, there is also a checkbox to indicate whether the
    form's current values should be remembered for the next run.

    The state of the checkbox can be accessed via the ``checkbox``
    attribute.
    """

    def __init__(self):
        """Initialize the QuitConfirmDialog."""
        QtWidgets.QMessageBox.__init__(self)
        self.setWindowFlags(QtCore.Qt.Dialog)
        self.setText('<h2>Are you sure you want to quit?</h2>')
        self.setInformativeText(
            'Any unsaved changes to your parameters will be lost.')
        self.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        self.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        self.setIconPixmap(
            qtawesome.icon(
                'fa.question').pixmap(100, 100))
        self.checkbox = QtWidgets.QCheckBox('Remember inputs')
        self.layout().addWidget(self.checkbox,
                                self.layout().rowCount()-1,
                                0, 1, 1)

    def exec_(self, starting_checkstate):
        """Execute the dialog.

        Parameters:
            starting_checkstate (bool): Whether the "Remember inputs" checkbox
                should be checked when the dialog is shown.

        Returns:
            The int return code from the QMessageBox's ``exec_()`` method.
        """
        self.checkbox.setChecked(starting_checkstate)
        return QtWidgets.QMessageBox.exec_(self)


class WorkspaceOverwriteConfirmDialog(QtWidgets.QMessageBox):
    """A message box to confirm that the workspace should be overwritten."""

    def __init__(self):
        """Initialize the dialog.

        Returns:
            ``None``
        """
        QtWidgets.QMessageBox.__init__(self)
        self.setWindowFlags(QtCore.Qt.Dialog)
        self.setText('<h2>Workspace exists!<h2>')
        self.setInformativeText(
            'Overwrite files from a previous run?')
        self.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        self.setDefaultButton(QtWidgets.QMessageBox.Yes)
        self.setIconPixmap(
            ICON_ALERT.pixmap(100, 100))


class SettingsDialog(OptionsDialog):
    """A dialog for global InVEST settings."""

    def __init__(self):
        """Initialize the SettingsDialog.

        Returns:
            ``None``
        """
        OptionsDialog.__init__(self, title='InVEST Settings',
                               modal=True)

        self._container = inputs.Container(label='Global options')
        self.layout().addWidget(self._container)

        try:
            # Qt4
            cache_dir = QtGui.QDesktopServices.storageLocation(
                QtGui.QDesktopServices.CacheLocation)
        except AttributeError:
            # Package location changed in Qt5
            cache_dir = QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.CacheLocation)
        self.cache_directory = inputs.Folder(
            label='Cache directory',
            helptext=('Where local files will be stored.'
                      'Default value: %s') % cache_dir)
        self.cache_directory.set_value(inputs.INVEST_SETTINGS.value(
            'cache_dir', cache_dir, unicode))
        self._container.add_input(self.cache_directory)

    #TODO: the base postprocess has a default of 0 for `exitcode` should this also?  Or remove parent's default?
    def postprocess(self, exitcode):
        """Save the settings from the dialog.

        Returns:
            ``None``
        """
        if exitcode == QtWidgets.QDialog.Accepted:
            inputs.INVEST_SETTINGS.setValue(
                'cache_dir', self.cache_directory.value())


class AboutDialog(QtWidgets.QDialog):
    """Show a dialog describing InVEST.

    In reasonable accordance with licensing and distribution requirements,
    this dialog not only has information about InVEST and the Natural
    Capital Project, but it also has details about the software used to
    develop and run InVEST and contains links to the licenses for each of
    these other projects.

    Returns:
        None.
    """

    def __init__(self):
        """Initialize the AboutDialog."""
        QtWidgets.QDialog.__init__(self)
        self.setWindowTitle('About InVEST')
        self.setLayout(QtWidgets.QVBoxLayout())
        label_text = textwrap.dedent(
            """
            <h1>InVEST</h1>
            <b>Version {version}</b> <br/> <br/>

            Documentation: <a href="http://data.naturalcapitalproject.org/nightly-
            build/invest-users-guide/html/">online</a><br/>
            Homepage: <a href="http://naturalcapitalproject.org">
                        naturalcapitalproject.org</a><br/>
            Copyright 2017, The Natural Capital Project<br/>
            License:
            <a href="https://bitbucket.org/natcap/invest/src/tip/LICENSE.txt">
                        BSD 3-clause</a><br/>
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

        self.label = QtWidgets.QLabel(label_text)
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setOpenExternalLinks(True)
        self.layout().addWidget(self.label)

        self.button_box = QtWidgets.QDialogButtonBox()
        self.accept_button = QtWidgets.QPushButton('OK')
        self.button_box.addButton(
            self.accept_button,
            QtWidgets.QDialogButtonBox.AcceptRole)
        self.layout().addWidget(self.button_box)
        self.accept_button.clicked.connect(self.close)


class LocalDocsMissingDialog(QtWidgets.QMessageBox):
    """A dialog to explain that local documentation can't be found."""

    def __init__(self, local_docs_link):
        """Initialize the LocalDocsMissingDialog.

        Parameters:
            local_docs_link (string): The local path to the local HTML
                documentation.

        Returns:
            ``None``
        """
        QtWidgets.QMessageBox.__init__(self)
        self.setWindowFlags(QtCore.Qt.Dialog)
        self.setText("<h2>Local docs not found<h2>")
        if not local_docs_link:
            local_docs_link = 'None'
        else:
            local_docs_link = os.path.basename(local_docs_link)

        remote_link = _ONLINE_DOCS_LINK + local_docs_link
        self.setInformativeText(
            'Online docs: [<a href="%s">documentation</a>]'
            '<br/><br/>Local documentation link could not be found: %s' %
            (remote_link, local_docs_link))
        self.setStandardButtons(QtWidgets.QMessageBox.Ok)
        self.setIconPixmap(
            qtawesome.icon(
                'fa.exclamation-triangle',
                color='orange').pixmap(100, 100))


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
    # TODO: just want to check that it makes sense to have these be Class attributes other than object attributes?  Looks like it's referred to later as an object attribute, and I could see an issue if there were multiple WindowTitles in one process?
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
            The string wundow title.
        """
        try:
            return self.format_string.format(
                modelname=self.modelname if self.modelname else 'InVEST',
                filename=self.filename if self.filename else 'new scenario',
                modified='*' if self.modified else '')
        except AttributeError:
            return ''

ScenarioSaveOpts = collections.namedtuple(
    'ScenarioSaveOpts', 'scenario_type use_relpaths include_workspace archive_path')


class ScenarioOptionsDialog(OptionsDialog):
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

    def __init__(self, paramset_basename):
        """Initialize the ScenarioOptionsDialog.

        Parameters:
            paramset_basename (string): The basename of the new parameter set file.

        Returns:
            ``None``
        """
        OptionsDialog.__init__(self,
                               title='Scenario options',
                               modal=True,
                               accept_text='Save scenario',
                               reject_text='Cancel')
        self._container = inputs.Container(label='Scenario options')
        self.layout().addWidget(self._container)
        self.paramset_basename = paramset_basename

        self.scenario_type = inputs.Dropdown(
            label='Scenario type',
            options=sorted(_SCENARIO_SAVE_OPTS.keys()))
        self.scenario_type.set_value(_SCENARIO_PARAMETER_SET)
        self.use_relative_paths = inputs.Checkbox(
            label='Use relative paths')
        self.include_workspace = inputs.Checkbox(
            label='Include workspace path in scenario')
        self.include_workspace.set_value(False)

        @validation.validator
        # TODO: `limit_to` is unused in the body of this function.  Remove?
        def _validate_parameter_file(args, limit_to=None):
            warnings = []
            archive_dir = os.path.dirname(args['archive_path'])

            if not os.path.exists(archive_dir):
                warnings.append((('archive_path',),
                                 'The parent folder of the specified path '
                                 'does not exist.'))

            return warnings

        self.save_parameters = inputs.SaveFile(
            label=_SCENARIO_SAVE_OPTS[_SCENARIO_PARAMETER_SET]['title'],
            args_key='archive_path',
            validator=_validate_parameter_file,
            default_savefile='{model}_{file_base}'.format(
                model=self.paramset_basename,
                file_base=_SCENARIO_SAVE_OPTS[
                    _SCENARIO_PARAMETER_SET]['savefile']))

        self._container.add_input(self.scenario_type)
        self._container.add_input(self.use_relative_paths)
        self._container.add_input(self.include_workspace)
        self._container.add_input(self.save_parameters)
        self.ok_button.setEnabled(False)  # initially disable the ok button

        @QtCore.Slot(unicode)
        def _optionally_disable(value):
            """A slot to optionally disable inputs based on scenario type.

            Parameters:
                value (string): The scenario type, one of the strings in the
                    scenario type dropdown menu.

            Returns:
                ``None``
            """
            # If the options have not yet been populated.
            if not value:
                return

            self.use_relative_paths.set_interactive(
                value == _SCENARIO_PARAMETER_SET)

            self.save_parameters.path_select_button.set_dialog_options(
                title=_SCENARIO_SAVE_OPTS[value]['title'],
                savefile='{model}_{file_base}'.format(
                    model=self.paramset_basename,
                    file_base=_SCENARIO_SAVE_OPTS[value]['savefile']))

        @QtCore.Slot(bool)
        def _enable_continue_button(new_validity):
            """A slot to enable the continue button when inputs are valid.

            Parameters:
                new_validity (bool): The validity of the form.

            Returns:
                ``None``
            """
            self.ok_button.setEnabled(new_validity)

        self.scenario_type.value_changed.connect(_optionally_disable)
        self.save_parameters.validity_changed.connect(_enable_continue_button)

    def exec_(self):
        """Execute the dialog.

        Returns:
            If the dialog is rejected, ``None`` is returned.
            If the dialog is accepted, a ``ScenarioSaveOpts`` instance is
                returned.
        """
        result = OptionsDialog.exec_(self)
        if result == QtWidgets.QDialog.Accepted:
            return ScenarioSaveOpts(
                self.scenario_type.value(),
                self.use_relative_paths.value(),
                self.include_workspace.value(),
                self.save_parameters.value()
            )
        return None


class ScenarioArchiveExtractionDialog(OptionsDialog):
    """A dialog for extracting a scenario archive."""

    def __init__(self):
        """Initialize the ScenarioArchiveExtractionDialog."""
        OptionsDialog.__init__(self,
                               title='Extract scenario',
                               modal=True,
                               accept_text='Extract',
                               reject_text='Cancel')
        self._container = inputs.Container(
            label='Scenario extraction parameters')
        self.layout().addWidget(self._container)

        self.extraction_point = inputs.Folder(
            label='Where should this archive be extracted?',
        )
        self._container.add_input(self.extraction_point)

    def exec_(self, archive_path):
        """Execute the dialog.

        Parameters:
            archive_path (string): The path to the archive that needs to be
                extracted.

        Returns:
            A 2-tuple.

            If the dialog was accepted, the return value is a 2-tuple of the
            extracted scenario args and the extracion directory.

            If the value was rejected, ``(None, None)`` is returned.
        """
        result = OptionsDialog.exec_(self)

        if result == QtWidgets.QDialog.Accepted:
            extract_to_dir = self.extraction_point.value()
            args = scenarios.extract_scenario_archive(
                archive_path, extract_to_dir)
            return (args, extract_to_dir)
        return (None, None)


class WholeModelValidationErrorDialog(QtWidgets.QDialog):
    """A dialog for presenting errors from whole-model validation."""

    def __init__(self):
        """Initialize the WholeModelValidationErrorDialog."""
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

    def post_warnings(self, validation_warnings):
        """Post validation warnings to the dialog.

        Parameters:
            validation_warnings (list): A list of validation warnings from
                whole-model validation.

        Returns:
            ``None``
        """
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
        * ``target`` (function reference): The reference to the target
            function. For InVEST, this will always be the ``execute`` function
            of the target model.
        * ``validator`` (function reference): The reference to the target
            validator function.  For InVEST, this will always be the
            ``validate`` function of the target model.
        * ``localdoc`` (string): The filename of the user's guide chapter for
            this model.

    If any of these attributes are not overridden, a warning will be raised.
    """
    # TODO: does it make sense to have these be class attributes instead of object parameters?
    label = None
    target = None
    validator = None
    localdoc = None

    def __init__(self):
        """Initialize the Model."""
        QtWidgets.QMainWindow.__init__(self)
        self.setAcceptDrops(True)
        self._quickrun = False
        self._validator = inputs.Validator(parent=self)
        self._validator.finished.connect(self._validation_finished)
        self.prompt_on_close = True
        self.exit_code = None

        # dialogs
        self.about_dialog = AboutDialog()
        self.settings_dialog = SettingsDialog()
        self.file_dialog = inputs.FileDialog()

        try:
            paramset_basename = self.target.__module__.split('.')[-1]
        except AttributeError:
            paramset_basename = 'scenario'
        self.scenario_options_dialog = ScenarioOptionsDialog(
            paramset_basename=paramset_basename)

        self.scenario_archive_extract_dialog = (
            ScenarioArchiveExtractionDialog())
        self.quit_confirm_dialog = QuitConfirmDialog()
        self.validation_report_dialog = WholeModelValidationErrorDialog()
        self.workspace_overwrite_confirm_dialog = (
            WorkspaceOverwriteConfirmDialog())
        self.local_docs_missing_dialog = LocalDocsMissingDialog(self.localdoc)

        def _settings_saved_message():
            # TODO: what's the 10000 mean?
            self.statusBar().showMessage('Settings saved', 10000)
        self.settings_dialog.accepted.connect(_settings_saved_message)

        # These attributes should be defined in subclass
        for attr in ('label', 'target', 'validator', 'localdoc'):
            if not getattr(self, attr):  # None unless overridden in subclass
                #TODO: this seems to be the only use of the `warnings` package.  Can it be replaced with a logger call?
                warnings.warn('Class attribute %s.%s is not defined' % (
                    self.__class__.__name__, attr))

        # Main operational widgets for the form
        self._central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self._central_widget)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        self._central_widget.setLayout(QtWidgets.QVBoxLayout())
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
        try:
            modelname = self.target.__module__.split('.')[-1]
        except AttributeError:
            modelname = 'model'
        self.workspace.set_value(os.path.normpath(
            os.path.expanduser('~/Documents/{model}_workspace').format(
                model=modelname)))

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
            qtawesome.icon('fa.cog'),
            'Settings ...', self.settings_dialog.exec_,
            QtGui.QKeySequence(QtGui.QKeySequence.Preferences))
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
            'About InVEST', self.about_dialog.exec_)
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
        # TODO: are you using `not link` here to test for None?  If so, can you change to link is not None.  Otherwise, this is some fun functionality for the doc link.
        if not link or link == 'localdocs':
            link = 'file://' + os.path.abspath(self.localdoc)

        if link.startswith(('http', 'ftp', 'file')):
            if os.path.exists(link.replace('file://', '')):
                try:
                    QtCore.QDesktopServices.openUrl(link)
                except AttributeError:
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl(link))
            else:
                self.local_docs_missing_dialog.exec_()

    def _save_scenario_as(self):
        """Save the current set of inputs as a scenario.

        Presents a dialog to the user for input on how to save the scenario,
        and then makes it happen.  A status message is displayed to the
        satus bar when the operation is complete.

        Returns:
           ``None``.
        """
        scenario_opts = self.scenario_options_dialog.exec_()
        if not scenario_opts:  # user pressed cancel
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
                scenario_path=scenario_opts.archive_path
            )
        else:
            scenarios.write_parameter_set(
                filepath=scenario_opts.archive_path,
                args=current_args,
                name=self.target.__module__,
                relative=scenario_opts.use_relpaths
            )

        save_filepath = os.path.basename(scenario_opts.archive_path)
        alert_message = (
            'Saved current parameters to %s' % save_filepath)
        LOGGER.info(alert_message)
        # TODO: what's the 10000 mean?
        self.statusBar().showMessage(alert_message, 10000)
        self.window_title.filename = os.path.basename(save_filepath)

    # TODO: we talked about this, but wondering if there's a better name than `input` and `Input`.  For one, `input` is a built in Python function already.  And maybe UIInput and ui_input might be a better alternative?  Just to consider...
    def add_input(self, input):
        """Add an input to the model.

        Parameters:
            input (natcap.invest.ui.inputs.Input): An Input instance to add to
                the model.

        Returns:
            ``None``
        """
        self.form.add_input(input)

    def is_valid(self):
        """Check whether the form is valid.

        The form is considered valid when there are no validation warnings.

        Returns:
            A boolean of whether there are known validation warnings.
        """
        if self.validation_report_dialog.warnings:
            return False
        return True

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
        if self.validation_report_dialog.warnings:
            self.validation_report_dialog.show()
            self.validation_report_dialog.exec_()
            return

        # If the workspace exists, confirm the overwrite.
        if os.path.exists(args['workspace_dir']):
            button_pressed = self.workspace_overwrite_confirm_dialog.exec_()
            if button_pressed != QtWidgets.QMessageBox.Yes:
                return

        def _logged_target():
            name = getattr(self, 'label', self.target.__module__)
            with utils.prepare_workspace(args['workspace_dir'], name):
                with usage.log_run(self.target.__module__, args):
                    LOGGER.info('Starting model with parameters: \n%s',
                                cli._format_args(args))
                    try:
                        return self.target(args=args)
                    except:
                        LOGGER.exception('Exception while executing %s',
                                         self.target)
                        raise
                    finally:
                        LOGGER.info('Execution finished')

        self.form.run(target=_logged_target,
                      window_title='Running %s' % self.label,
                      out_folder=args['workspace_dir'])

    @QtCore.Slot()
    def load_scenario(self, scenario_path=None):
        """Load a scenario.

        This method is also a slot that accepts no arguments.

        A scenario could be any one of:

            * A logfile from a previous model run.
            * A parameter set (*.invs.json)
            * A parameter archive (*.invs.tar.gz)

        Scenarios may be saved and loaded through the Model UI. For API access
        to scenarios, look at :ref:natcap.invest.scenarios.

        Parameters:
            scenario_path=None (string): The path to the scenario file to
                load.  If ``None``, the user will be prompted for a file
                with a file dialog.

        Returns:
            ``None``
        """
        if not scenario_path:
            scenario_path = self.file_dialog.open_file(
                title='Select scenario', filters=(
                    'Any file (*.*)',
                    'Parameter set (*.invs.json)',
                    'Parameter archive (*.invs.tar.gz)',
                    'Logfile (*.txt)'))

            # When the user pressed cancel, scenario_path == ''
            if not scenario_path:
                return

        LOGGER.info('Loading scenario from "%s"', scenario_path)
        if tarfile.is_tarfile(scenario_path):  # it's a scenario archive!
            # Where should the tarfile be extracted to?
            args, extract_dir = self.scenario_archive_extract_dialog.exec_(
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
        # TODO: what's the 10000 mean?
        self.statusBar().showMessage(
            'Loaded scenario from %s' % os.path.abspath(scenario_path), 10000)

    def load_args(self, scenario_args):
        """Load arguments from an args dict.

        Parameters:
            scenario_args (dict): The arguments dictionary from which model
                parameters will be loaded.

        Returns:
            ``None``
        """
        _inputs = dict((input.args_key, input) for input in
                       self.inputs())
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
        """Collect arguments from the UI and assemble them into a dictionary.

        This method must be reimplemented by subclasses.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @QtCore.Slot(list)
    def _validation_finished(self, validation_warnings):
        """A slot to handle whole-model validation errors.

        Parameters:
            validation_warnings (list): A list of string validation warnings.

        Returns:
            ``None``
        """
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
        self.validation_report_dialog.post_warnings(warnings_)

    def inputs(self):
        """Fetch a list of all model inputs.

        Returns:
            A list of all objects known to this ``Model`` instance that are
            subclasses of ``Input``.
        """
        # TODO: we'd talked about an alternative to __dict__ such as a dictionary that maps arg key to the Input object in the `add_input` function.  When I tried to do this myself, it looked promising but I broke a bunch of test cases that assume "input()" and others are stlil legit functions.
        return [ref for ref in self.__dict__.values()
                if isinstance(ref, inputs.Input)]

    def validate(self, block=False):
        """Trigger validation for the whole model.

        Parameters:
            block=False (bool): Whether to block on validation.

        Returns:
            ``None``
        """
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
        """Run the model.

        Parameters:
            quickrun=False (bool): If True, the model will close when the
                model finishes.

        Returns:
            ``None``
        """
        # iterate through attributes of self.form.  If the attribute is an
        # instance of inputs.Input, then link its value_changed signal to the
        # model-wide validation slot.
        def _validate(new_value):
            # We want to validate the whole form; discard the individual value
            self.validate(block=False)

        for input_obj in self.inputs():
            input_obj.value_changed.connect(_validate)
            try:
                input_obj.validity_changed.connect(_validate)
            except AttributeError:
                # Not all inputs can have validity (e.g. Container, dropdown)
                pass

        # Set up quickrun options if we're doing a quickrun
        if quickrun:
            @QtCore.Slot()
            def _quickrun_close_model():
                # exit with an error code that matches exception status of run.
                self.form.run_dialog.close_window()
                self.close(prompt=False)

            self.form.run_finished.connect(_quickrun_close_model)
            # TODO: Can you comment on the 50?
            QtCore.QTimer.singleShot(50, self.execute_model)

        # The scrollArea defaults to a size that is too small to actually view
        # the contents of the enclosed widget appropriately.  By adjusting the
        # size here, we ensure that the widgets are an appropriate height.
        # Note that self.resize() does take the window size into account, so
        # all parts of the application window will still be visible, even if
        # the minimumSize().height() would have it extend over the edge of the
        # screen.
        # TODO: can you comment where the 100 and 150 numbers come from? Or paramaterize them globally or in settings or something?
        self.resize(
            self.form.scroll_area.widget().minimumSize().width()+100,
            self.form.scroll_area.widget().minimumSize().height()+150)

        inputs.center_window(self)

        # if we're not working off a scenario file, load the last run.
        if not self.window_title.filename:
            self.load_lastrun()

        self.show()
        self.raise_()  # raise window to top of stack.

    def close(self, prompt=True):
        """Close the window.

        Overridden from QMainWindow.close to allow for an optional ``prompt``
        argument.

        Parameters:
            prompt=True (bool): Whether to prompt for the user to confirm the
                window's closure.  If ``False``, the model window will be
                closed without confirmation.

        Returns:
            ``None``
        """
        self.prompt_on_close = prompt
        QtWidgets.QMainWindow.close(self)

    def closeEvent(self, event):
        """Handle close events for the QMainWindow.

        The user will be prompted to confirm that the window should be closed
        unless ``self.prompt_on_close`` is ``True``.

        Reimplemented from QMainWindow.closeEvent.

        Parameters:
            event (QEvent): The current event.

        Returns:
            ``None``
        """
        if self.prompt_on_close:
            starting_checkstate = self.settings.value(
                'remember_lastrun', True, bool)
            button_pressed = self.quit_confirm_dialog.exec_(
                starting_checkstate)
            if button_pressed != QtWidgets.QMessageBox.Yes:
                event.ignore()
            elif self.quit_confirm_dialog.checkbox.isChecked():
                self.save_lastrun()
            self.settings.setValue(
                'remember_lastrun',
                self.quit_confirm_dialog.checkbox.isChecked())
        self.prompt_on_close = True

    def save_lastrun(self):
        """Save lastrun args to the model's settings.

        Returns:
            ``None``
        """
        lastrun_args = self.assemble_args()
        LOGGER.debug('Saving lastrun args %s', lastrun_args)
        self.settings.setValue("lastrun", json.dumps(lastrun_args))

    def load_lastrun(self):
        """Load lastrun settings from the model's settings.

        Returns:
            ``None``
        """
        # If no lastrun args saved, "{}" (empty json object) is returned
        lastrun_args = self.settings.value("lastrun", "{}")
        self.load_args(json.loads(lastrun_args))
        # TODO: what's the 10000 mean?
        self.statusBar().showMessage('Loaded parameters from previous run.',
                                     10000)
        self.window_title.filename = 'loaded from autosave'
