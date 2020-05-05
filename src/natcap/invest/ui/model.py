# coding=UTF-8
"""Classes related to the InVEST Model class."""

import logging
import os
import pprint
from pkg_resources import parse_version
import collections
import json
import requests
import textwrap
import cgi
import tarfile
import contextlib
import functools
import datetime
import codecs
import multiprocessing
import threading
import PySide2

from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui
import qtawesome
import natcap.invest

from . import inputs
from . import usage
from . import execution
from .. import utils
from .. import datastack
from .. import validation

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
QT_APP = inputs.QT_APP

# How long satus bar messages should be visible, in milliseconds.
STATUSBAR_MSG_DURATION = 10000
ICON_BACK = qtawesome.icon('fa.arrow-circle-o-left', color='grey')
ICON_ALERT = qtawesome.icon('fa.exclamation-triangle', color='orange')
ICON_UPDATE = qtawesome.icon('fa.refresh', color='orange')

_ONLINE_DOCS_LINK = (
    'http://releases.naturalcapitalproject.org/invest-userguide/latest/')
_DATASTACK_BASE_FILENAME = 'datastack.invest.%s'
_DATASTACK_DIALOG_TITLE = 'Select where to save the datastack'
_DATASTACK_PARAMETER_SET = 'Parameter set (saves parameter values only)'
_DATASTACK_DATA_ARCHIVE = 'Data archive (archives parameters and files)'
_DATASTACK_SAVE_OPTS = {
    _DATASTACK_PARAMETER_SET: {
        'title': _DATASTACK_DIALOG_TITLE,
        'savefile': _DATASTACK_BASE_FILENAME % 'json',
    },
    _DATASTACK_DATA_ARCHIVE: {
        'title': _DATASTACK_DIALOG_TITLE,
        'savefile': _DATASTACK_BASE_FILENAME % 'tar.gz',
    }
}
# To create a QSettings object, call this with the model label as the only
# argument.  Example:  settings = SETTINGS_TEMPLATE('My Model')
SETTINGS_TEMPLATE = functools.partial(
    QtCore.QSettings, QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
    'Natural Capital Project')


@contextlib.contextmanager
def wait_on_signal(signal, timeout=250):
    """Block loop until signal emitted, or timeout (ms) elapses."""
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


def is_probably_datastack(filepath):
    """Check to see if the path provided is probably a datastack.

    Parameters:
        filepath (string): A path to a location on disk.

    Returns:
        True if the path is likely to be a datastack.  False otherwise.
    """
    # If the filepath provided is a directory, it's not a datastack.
    if os.path.isdir(filepath):
        return False

    # Does the extension indicate that it's probably a datastack?
    if filepath.endswith((datastack.DATASTACK_EXTENSION,
                          datastack.PARAMETER_SET_EXTENSION)):
        return True

    # Is it a datastack parameter set?
    with codecs.open(filepath, encoding='UTF-8') as opened_file:
        # Valid JSON starts with '{'
        if opened_file.read(1) == '{':
            return True

        # Is it a logfile?
        # "Arguments" might be at the very beginning of the file.
        opened_file.seek(0)
        try:
            if 'Arguments' in ' '.join(opened_file.readlines(200)):
                return True
        except UnicodeDecodeError:
            # When ``filepath`` is a .tar.gz, the text read in probably won't
            # be valid UTF-8, which will raise a UnicodeDecodeError when python
            # tries to decode it.
            pass

    try:
        # If we can open it as a .tar.gz, assume it's a datastack
        tarfile.open(filepath, mode='r|gz', bufsize=1024)
        return True
    except tarfile.ReadError:
        # tarfile.ReadError raised when the file is not formatted as expected.
        pass

    return False


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
                 reject_text='cancel', parent=None):
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
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.

        Returns:
            ``None``
        """
        QtWidgets.QDialog.__init__(self, parent=parent)
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

    def postprocess(self, exitcode):
        """Save the options in the dialog.

        Subclasses of ``OptionsDialog`` must reimplement this method.

        Parameters:
            exitcode (int): The exit code of the dialog.

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

    def __init__(self, parent=None):
        """Initialize the QuitConfirmDialog.

        Paremeters:
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.
        """
        QtWidgets.QMessageBox.__init__(self, parent=parent)
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
        self.checkbox.setChecked(bool(starting_checkstate))
        return QtWidgets.QMessageBox.exec_(self)


class ConfirmDialog(QtWidgets.QMessageBox):
    """A message box for confirming something with the user."""

    def __init__(self, title_text, body_text, parent=None):
        """Initialize the dialog.

        Parameters:
            title_text (string): The title of the dialog.
            body_text (string): The body text of the dialog.
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.

        Returns:
            None.
        """
        QtWidgets.QMessageBox.__init__(self, parent=parent)
        self.setWindowFlags(QtCore.Qt.Dialog)
        self.setText('<h2>%s<h2>' % title_text)
        self.setInformativeText(body_text)
        self.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        self.setDefaultButton(QtWidgets.QMessageBox.Yes)
        self.setIconPixmap(
            ICON_ALERT.pixmap(100, 100))


class ModelMismatchConfirmDialog(ConfirmDialog):
    """Confirm datastack load when it looks like the wrong model."""

    def __init__(self, current_modelname, parent=None):
        """Initialize the dialog.

        Parameters:
            current_modelname (string): The modelname of the current
                InVESTModel target.
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.

        Returns:
            None.
        """
        self._current_modelname = current_modelname

        self._body_text = (
            "This datastack was created for the model \"{datastack_model}\", "
            "which looks different from this model (\"{current_model}\").\n\n "
            "Load these parameters anyway?"
        )

        ConfirmDialog.__init__(
            self,
            title_text='Are you sure this is the right model?',
            body_text=self._body_text,
            parent=parent)

    def exec_(self, datastack_modelname):
        """Show the dialog and enter its event loop.

        Also updates the informative text based on the provided datastack
        modelname.

        Parameters:
            datastack_modelname (string): The modelname of the datastack.

        Returns:
            The result code of ``ConfirmDialog.exec_()``.
        """
        new_text = self._body_text.format(
            datastack_model=datastack_modelname,
            current_model=self._current_modelname)
        self.setInformativeText(new_text)

        return ConfirmDialog.exec_(self)


class SettingsDialog(OptionsDialog):
    """A dialog for global InVEST settings."""

    def __init__(self, parent=None):
        """Initialize the SettingsDialog.

        Parameters:
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.

        Returns:
            ``None``
        """
        OptionsDialog.__init__(self, title='InVEST Settings',
                               modal=True, parent=parent)
        self.resize(600, 200)

        self.global_label = QtWidgets.QLabel(
            'Note: these settings affect all InVEST models.')
        self.global_label.setStyleSheet(inputs.QLABEL_STYLE_INFO)
        self.layout().addWidget(self.global_label)

        self._global_opts_container = inputs.Container(label='Global options')
        self.layout().addWidget(self._global_opts_container)

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
            'cache_dir', cache_dir))
        self._global_opts_container.add_input(self.cache_directory)

        logging_options = (
            'CRITICAL',
            'ERROR',
            'WARNING',
            'INFO',
            'DEBUG',
            'NOTSET')
        self.dialog_logging_level = inputs.Dropdown(
            label='Dialog logging threshold',
            helptext=('The minimum logging level for messages to be '
                      'displayed in the run dialog.  Log messages with '
                      'a level lower than this will not be displayed in the '
                      'run dialog. Default: INFO'),
            options=logging_options)
        self.dialog_logging_level.set_value(inputs.INVEST_SETTINGS.value(
            'logging/run_dialog', 'INFO'))
        self._global_opts_container.add_input(self.dialog_logging_level)

        self.logfile_logging_level = inputs.Dropdown(
            label='Logfile logging threshold',
            helptext=('The minimum logging level for messages to be '
                      'displayed in the logfile for a run.  Log messages with '
                      'a level lower than this will not be written to the '
                      'logfile. Default: NOTSET'),
            options=logging_options)
        self.logfile_logging_level.set_value(inputs.INVEST_SETTINGS.value(
            'logging/logfile', 'NOTSET'))
        self._global_opts_container.add_input(self.logfile_logging_level)

        self.taskgraph_logging_level = inputs.Dropdown(
            label='Taskgraph logging threshold',
            helptext=('The minimum logging level for taskgraph messages to be '
                      'displayed in either the logfile or the UI.  Log '
                      'messages with a level lower than this will not be '
                      'written to the logfile. Default: ERROR'),
            options=logging_options)
        self.taskgraph_logging_level.set_value(inputs.INVEST_SETTINGS.value(
            'logging/taskgraph', 'ERROR'))
        self._global_opts_container.add_input(self.taskgraph_logging_level)

        # Taskgraph n_workers settings.
        # Using a dropdown to avoid the need to validate.
        n_workers_values = {
            'Synchronous (-1)': '-1',
            'Threaded task management (0)': '0'}
        n_workers_values.update(dict(('%s CPUs' % n, str(n)) for n in range(
            1, multiprocessing.cpu_count()*2)))
        self.taskgraph_n_workers = inputs.Dropdown(
            label='Taskgraph n_workers parameter',
            helptext=('For models that are implemented with taskgraph, this '
                      'is provided to the graph at creation.  The default '
                      'value of -1 is best for most users, as this will '
                      'eliminate the risk of deadlocks and improve the '
                      'coherency of the logfile. Allowed values are<ul> '
                      '<li>-1: Synchronous task execution (most reliable) </li>'
                      '<li>0: Tasks execute in the main process, but use '
                      'threaded task management. </li>'
                      '<li><em>n</em>: Where <em>n</em> is a positive integer, '
                      'taskgraph will execute tasks in <em>n</em> processes. '
                      'This can yield a nice speedup, but incurs a risk of '
                      'deadlock.</li>'
                      '</ul>Regardless of this value, all models that are '
                      'taskgraph-enabled take advantage of '
                      'avoided recomputation. To see if a model uses '
                      "taskgraph, take a look at the User's Guide chapter "
                      'for the model, or inspect the source code.'),
            options=[pair[0] for pair in sorted(
                n_workers_values.items(), key=lambda x: int(x[1]))],
            return_value_map=n_workers_values)
        self.taskgraph_n_workers.set_value(inputs.INVEST_SETTINGS.value(
            'taskgraph/n_workers', '-1'))
        self._global_opts_container.add_input(self.taskgraph_n_workers)

    def postprocess(self, exitcode):
        """Save the settings from the dialog.

        Parameters:
            exitcode (int): The exit code of the dialog.

        Returns:
            ``None``
        """
        if exitcode == QtWidgets.QDialog.Accepted:
            inputs.INVEST_SETTINGS.setValue(
                'cache_dir', self.cache_directory.value())
            inputs.INVEST_SETTINGS.setValue(
                'logging/run_dialog',
                self.dialog_logging_level.value())
            inputs.INVEST_SETTINGS.setValue(
                'logging/logfile',
                self.logfile_logging_level.value())
            inputs.INVEST_SETTINGS.setValue(
                'logging/taskgraph',
                self.taskgraph_logging_level.value())
            inputs.INVEST_SETTINGS.setValue(
                'taskgraph/n_workers',
                self.taskgraph_n_workers.value())


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

    def __init__(self, parent=None):
        """Initialize the AboutDialog.

        Parameters:
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.
        """
        QtWidgets.QDialog.__init__(self, parent=parent)
        self.setWindowTitle('About InVEST')
        self.setLayout(QtWidgets.QVBoxLayout())
        label_text = textwrap.dedent(
            """
            <h1>InVEST</h1>
            <b>Version {version}</b> <br/> <br/>

            Documentation: <a href="http://releases.naturalcapitalproject.org/
            invest-userguide/latest/">online</a><br/>
            Homepage: <a href="http://naturalcapitalproject.org">
                        naturalcapitalproject.org</a><br/>
            Copyright 2017, The Natural Capital Project<br/>
            License:
            <a href="https://github.com/natcap/invest/blob/master/LICENSE.txt">
                        BSD 3-clause</a><br/>
            Project page: <a href="https://github.com/natcap/invest">
                        github.com/natcap/invest</a><br/>

            <h2>Open-Source Licenses</h2>
            """.format(
                version=natcap.invest.__version__))

        label_text += "<table>"
        for lib_name, lib_license, lib_homepage in [
                ('PyInstaller', 'GPL', 'http://pyinstaller.org'),
                ('GDAL', 'MIT and others', 'http://gdal.org'),
                ('numpy', 'BSD', 'http://numpy.org'),
                ('pyamg', 'BSD', 'http://github.com/pyamg/pyamg'),
                ('pygeoprocessing', 'BSD',
                 'https://github.com/natcap/pygeoprocessing'),
                ('PyQt', 'GPL',
                 'https://riverbankcomputing.com/software/pyqt/intro'),
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
        label_text += textwrap.dedent(
            """
            <br/>
            <p>
            The source code for GPL'd components are included as an extra
            component on your <br/> installation medium.
            </p>
            """)

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

    def __init__(self, local_docs_link, parent=None):
        """Initialize the LocalDocsMissingDialog.

        Parameters:
            local_docs_link (string): The local path to the local HTML
                documentation.
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.

        Returns:
            ``None``
        """
        QtWidgets.QMessageBox.__init__(self)
        self.setWindowFlags(QtCore.Qt.Dialog)
        self.setText("<h2>Local docs not found<h2>")
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
        * The filename (basename) of the current datastack file
        * Whether the datastack has been modified from the time it was last
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
            ``"new datastack"`` is assumed.
        modified (bool): Whether the datastack file has been modified.  If so,
            a ``'*'`` is displayed next to the datastack filename.
    """

    # Signals must be defined as class attributes, and are transformed into
    # instance attributes on object initialization.
    title_changed = QtCore.Signal(str)

    def __init__(self, modelname='', filename='', modified=''):
        """Initialize the WindowTitle.

        Parameters:
            modelname (string or None): The modelname to use.
            filename (string or None): The filename to use.
            modified (bool): Whether the datastack file has been modified.
        """
        super(WindowTitle, self).__init__()

        self.modelname = modelname
        self.filename = filename
        self.modified = modified

        # Python strings are immutable; this can be accessed like an instance
        # variable.
        self._format_string = "{modelname}: {filename}{modified}"

    def set_title_attr(self, name, value):
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
        object.__setattr__(self, name, value)
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
            return self._format_string.format(
                modelname=self.modelname if self.modelname else 'InVEST',
                filename=self.filename if self.filename else 'new datastack',
                modified='*' if self.modified else '')
        except AttributeError:
            return ''


DatastackSaveOpts = collections.namedtuple(
    'DatastackSaveOpts',
    'datastack_type use_relpaths include_workspace archive_path')


class DatastackOptionsDialog(OptionsDialog):
    """Provide a GUI model dialog with options for saving a datastack.

    There are two types of datastacks:

        * Parameter sets (a file with the values of the current inputs)
        * Data archives (all-inclusive archive of current inputs)

    This dialog provides a couple of options to the user depending on which
    type of datastack is desired.  If a parameter set is selected, paths may
    be stored relative to the location of the datastack file.  Both types of
    datastacks may optionally include the value of the workspace input.

    Directories in the save file's path are created if they do not exist.

    Returns:
        An instance of :ref:DatastackSaveOpts namedtuple.
    """

    def __init__(self, paramset_basename, parent=None):
        """Initialize the DatastackOptionsDialog.

        Parameters:
            paramset_basename (string): The basename of the new parameter set
                file.
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.

        Returns:
            ``None``
        """
        OptionsDialog.__init__(self,
                               title='Datastack options',
                               modal=True,
                               accept_text='Save datastack',
                               reject_text='Cancel',
                               parent=parent)
        self._container = inputs.Container(label='Datastack options')
        self.layout().addWidget(self._container)
        self.paramset_basename = paramset_basename

        self.datastack_type = inputs.Dropdown(
            label='Datastack type',
            options=sorted(_DATASTACK_SAVE_OPTS.keys()))
        self.datastack_type.set_value(_DATASTACK_PARAMETER_SET)
        self.use_relative_paths = inputs.Checkbox(
            label='Use relative paths')
        self.include_workspace = inputs.Checkbox(
            label='Include workspace path in datastack')
        self.include_workspace.set_value(False)

        self.save_parameters = inputs.SaveFile(
            label=_DATASTACK_SAVE_OPTS[_DATASTACK_PARAMETER_SET]['title'],
            args_key='archive_path',
            default_savefile='{model}_{file_base}'.format(
                model=self.paramset_basename,
                file_base=_DATASTACK_SAVE_OPTS[
                    _DATASTACK_PARAMETER_SET]['savefile']))

        self._container.add_input(self.datastack_type)
        self._container.add_input(self.use_relative_paths)
        self._container.add_input(self.include_workspace)
        self._container.add_input(self.save_parameters)
        self.ok_button.setEnabled(False)  # disabled until a value is entered

        @QtCore.Slot(str)
        def _optionally_disable(value):
            """A slot to optionally disable inputs based on datastack type.

            Parameters:
                value (string): The datastack type, one of the strings in the
                    datastack type dropdown menu.

            Returns:
                ``None``
            """
            # If the options have not yet been populated.
            if not value:
                return

            self.use_relative_paths.set_interactive(
                value == _DATASTACK_PARAMETER_SET)

            self.save_parameters.path_select_button.set_dialog_options(
                title=_DATASTACK_SAVE_OPTS[value]['title'],
                savefile='{model}_{file_base}'.format(
                    model=self.paramset_basename,
                    file_base=_DATASTACK_SAVE_OPTS[value]['savefile']))

        @QtCore.Slot(bool)
        def _enable_continue_button(new_value):
            """A slot to enable the continue button when input is filled.

            Parameters:
                new_value (string): The value of the form.

            Returns:
                ``None``
            """
            self.ok_button.setEnabled(
                (new_value is not None) and (new_value.strip() not in ''))

        self.datastack_type.value_changed.connect(_optionally_disable)
        self.save_parameters.value_changed.connect(_enable_continue_button)

    def exec_(self):
        """Execute the dialog.

        Returns:
            If the dialog is rejected, ``None`` is returned.
            If the dialog is accepted, a ``DatastackSaveOpts`` instance is
                returned.
        """
        result = OptionsDialog.exec_(self)
        if result == QtWidgets.QDialog.Accepted:
            return_value = DatastackSaveOpts(
                self.datastack_type.value(),
                self.use_relative_paths.value(),
                self.include_workspace.value(),
                self.save_parameters.value()
            )
        else:
            return_value = None

        # Clear the inputs now that we have what we need.
        for input_obj in (self.datastack_type, self.use_relative_paths,
                          self.include_workspace, self.save_parameters):
            input_obj.clear()

        return return_value


class DatastackArchiveExtractionDialog(OptionsDialog):
    """A dialog for extracting a datastack archive."""

    def __init__(self, parent=None):
        """Initialize the DatastackArchiveExtractionDialog.

        Parameters:
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.
        """
        OptionsDialog.__init__(self,
                               title='Extract datastack',
                               modal=True,
                               accept_text='Extract',
                               reject_text='Cancel',
                               parent=parent)
        self._container = inputs.Container(
            label='Datastack extraction parameters')
        self.layout().addWidget(self._container)

        self.extraction_point = inputs.Folder(
            label='Where should this archive be extracted?',
        )
        self._container.add_input(self.extraction_point)

    def exec_(self):
        """Execute the dialog.

        Returns:
            The string path to the extraction directory.  None if the dialog
            was cancelled.
        """
        result = OptionsDialog.exec_(self)

        if result == QtWidgets.QDialog.Accepted:
            return self.extraction_point.value()
        return None


class DatastackProgressDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        """Initialize the DatastackProgressDialog.

        Parameters:
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.
        """
        QtWidgets.QDialog.__init__(self, parent=parent)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.executor = None
        self.title = QtWidgets.QLabel('', parent=self)
        self.layout().addWidget(self.title)

        self.progressbar = QtWidgets.QProgressBar(parent=self)  # indeterminate!
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(0)
        self.layout().addWidget(self.progressbar)

        self.checkbox = QtWidgets.QCheckBox(
            'Automatically close when finished.',
            parent=self)
        self.layout().addWidget(self.checkbox)

        self.buttonbox = QtWidgets.QDialogButtonBox(parent=self)
        self.close_button = QtWidgets.QPushButton('Close', parent=self)
        self.close_button.setEnabled(False)  # disable until executor finishes.
        self.close_button.clicked.connect(self.close)
        self.buttonbox.addButton(self.close_button,
                                 QtWidgets.QDialogButtonBox.AcceptRole)
        self.layout().addWidget(self.buttonbox)

    def exec_build(self, args, model_name, datastack_path):
        """Build a datastack archive.

        Presents a dialog with an indeterminate progress bar while the archive
        is being built.  The actual processing happens in a separate thread of
        control.

        Parameters:
            args (dict): The model arguments to archive.
            model_name (string): The python-importable model identifier.
            datastack_path (string): The path to the file on disk where the
                datastack archive should be saved

        Returns:
            The exit code of the underlying implementation of
            ``QDialog.exec_()``.
        """
        self.setWindowTitle('Creating archive')
        self.title.setText('<h2>Creating archive</h2>')
        self.executor = execution.Executor(
            target=datastack.build_datastack_archive,
            kwargs={'args': args,
                    'model_name': model_name,
                    'datastack_path': datastack_path})
        return self.exec_()

    def exec_extract(self, datastack_path, dest_dir_path):
        """Extract a datastack archive.

        Presents a dialog with an indeterminate progress bar while the archive
        is being extracted.  The actual processing happens in a separate thread
        of control.

        Parameters:
            datastack_path (string): The path to the datastack archive on disk
                that should be extracted.
            dest_dir_path (string): The path to the directory on disk where the
                archive should be extracted to.

        Returns:
            The exit code of the underlying implementation of
            ``QDialog.exec_()``.

        """
        self.setWindowTitle('Extracting archive')
        self.title.setText('<h2>Extracting archive</h2>')
        self.executor = execution.Executor(
            target=datastack.extract_datastack_archive,
            kwargs={'datastack_path': datastack_path,
                    'dest_dir_path': dest_dir_path})
        return self.exec_()

    def _thread_finished(self):
        """Slot for updating the UI when the processing thread finishes."""
        self.close_button.setEnabled(True)
        self.title.setText('<h2>Complete.</h2>')
        self.progressbar.setMaximum(1)  # stop the progress bar.

        if self.checkbox.isChecked():
            self.checkbox.setChecked(False)  # reset to unchecked for next run
            self.close()

    def exec_(self):
        """Enter the dialog's event loop.

        Overridden from ``QtWidgets.QDialog.exec_()``.  This method is not
        intended to be used directly.  Use ``exec_build`` or ``exec_extract``
        instead.

        Raises:
            RuntimeError when called directly, without the appropriate setup.

        Returns:
            The exit code of the underlying QDialog.
        """
        self.close_button.setEnabled(False)

        if self.executor is None:
            raise RuntimeError(
                'Call exec_build or exec_extract instead of exec_()')

        self.executor.finished.connect(self._thread_finished)
        self.executor.start()

        # Enter the dialog's event loop
        return_code = QtWidgets.QDialog.exec_(self)
        self.executor = None
        return return_code


class WholeModelValidationErrorDialog(QtWidgets.QDialog):
    """A dialog for presenting errors from whole-model validation."""

    def __init__(self, parent=None):
        """Initialize the WholeModelValidationErrorDialog.

        Parameters:
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.
        """
        QtWidgets.QDialog.__init__(self, parent=parent)
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
        self.scroll_widget.setWidgetResizable(True)

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


class InvestVersionUpdateDialog(QtWidgets.QDialog):
    """A dialog for notifying users of the link to a new InVEST version."""
    def __init__(self, parent=None, latest_version=None):
        """Initialize the InvestVersionUpdateDialog.

        Parameters:
            parent=None (QWidget or None): The parent of the dialog. None if
                no parent.

            latest_version (str or None): A string representing the latest
                version of InVEST.

        """
        QtWidgets.QDialog.__init__(self, parent=parent)
        self.latest_version = latest_version if latest_version else ''
        self.setLayout(QtWidgets.QVBoxLayout())
        self.setWindowTitle('InVEST Version Update')

        self.title_icon = QtWidgets.QLabel()
        self.title_icon.setPixmap(ICON_UPDATE.pixmap(50, 50))
        self.title_icon.setAlignment(QtCore.Qt.AlignCenter)
        self.title = QtWidgets.QWidget()
        self.title.setLayout(QtWidgets.QHBoxLayout())
        self.title.layout().addWidget(self.title_icon)

        self.download_link = "https://naturalcapitalproject.stanford.edu/invest/"
        self.download_qurl = QtCore.QUrl(self.download_link)
        self.title_label = QtWidgets.QLabel(
            '<h2>A new InVEST version %s is available</h2>'
            '<h4>To install a new version, please go to the download page:</h4>'
            '<a href="%s">%s' % (
                self.latest_version, self.download_link, self.download_link))

        self.title_label.linkActivated.connect(functools.partial(
            InVESTModel._activate_link, self.download_qurl))
        self.title.layout().addWidget(self.title_label)
        self.layout().addWidget(self.title)


class InVESTModel(QtWidgets.QMainWindow):
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

    def __init__(self, label, target, validator, localdoc,
                 suffix_args_key='results_suffix'):
        """Initialize the Model.

        Parameters:
            label (string): The model label.
            target (callable): The reference to the target ``execute``
                function.
            validator (callable): The reference to the target ``validate``
                function.
            localdoc (string): The filename of the user's guide chapter for
                this model.
            suffix_args_key='results_suffix' (string): The args key to use for
                suffix input.  Defaults to ``results_suffix``.
        """
        QtWidgets.QMainWindow.__init__(self)
        self.label = label
        self.target = target
        self.validator = validator
        self.localdoc = localdoc

        self.inputs = set([])

        self.setAcceptDrops(True)
        self._quickrun = False
        self._validator = inputs.Validator(parent=self)
        self._validator.finished.connect(self._validation_finished)
        self.prompt_on_close = True
        self.exit_code = None

        # dialogs
        self.about_dialog = AboutDialog(parent=self)
        self.settings_dialog = SettingsDialog(parent=self)
        self.file_dialog = inputs.FileDialog(parent=self)
        self.datastack_progress_dialog = DatastackProgressDialog(parent=self)

        paramset_basename = self.target.__module__.split('.')[-1]
        self.datastack_options_dialog = DatastackOptionsDialog(
            paramset_basename=paramset_basename, parent=self)
        self.datastack_archive_extract_dialog = (
            DatastackArchiveExtractionDialog(parent=self))

        self.quit_confirm_dialog = QuitConfirmDialog(self)
        self.validation_report_dialog = WholeModelValidationErrorDialog(self)
        self.local_docs_missing_dialog = LocalDocsMissingDialog(self.localdoc,
                                                                parent=self)
        self.input_overwrite_confirm_dialog = ConfirmDialog(
            title_text='Overwrite parameters?',
            body_text=('Loading a datastack will overwrite any unsaved '
                       'parameters. Are you sure you want to continue?'),
            parent=self)
        self.workspace_overwrite_confirm_dialog = ConfirmDialog(
            title_text='Workspace exists!',
            body_text='Overwrite files from a previous run?',
            parent=self)
        self.model_mismatch_confirm_dialog = ModelMismatchConfirmDialog(
            self.target.__module__, parent=self)

        def _settings_saved_message():
            self.statusBar().showMessage('Settings saved',
                                         STATUSBAR_MSG_DURATION)
        self.settings_dialog.accepted.connect(_settings_saved_message)

        # Main operational widgets for the form
        self._central_widget = QtWidgets.QWidget(parent=self)
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
        self.window_title.set_title_attr('modelname', self.label)

        # Add InVEST version update button and links at the top of the window.
        self.links_layout = QtWidgets.QHBoxLayout()
        self.links_layout.setAlignment(QtCore.Qt.AlignRight)

        # Add update button to the left of text links, if a more recent version
        # is found
        self.latest_version = self._get_latest_version()
        self.needs_update = self._needs_update(self.latest_version)
        if self.needs_update:
            self.update_button = QtWidgets.QPushButton('')
            self.update_button.setIcon(ICON_UPDATE)
            self.update_button.setFixedWidth(25)
            self.update_button.setToolTip('New InVEST Version Available')
            self.version_update_dialog = InvestVersionUpdateDialog(
                self, self.latest_version)
            self.update_button.clicked.connect(
                self.version_update_dialog.show)
            self.links_layout.addWidget(self.update_button)

        # Format the text links
        self.links = QtWidgets.QLabel(parent=self)
        self.links.setText(' | '.join((
            'InVEST version %s' % natcap.invest.__version__,
            '<a href="localdocs">Model documentation</a>',
            ('<a href="https://community.naturalcapitalproject.org">'
             'Report an issue</a>'))))
        self.links.linkActivated.connect(self._check_local_docs)
        self.links_layout.addWidget(self.links)
        self.links_widget = QtWidgets.QWidget()
        self.links_widget.setLayout(self.links_layout)
        self._central_widget.layout().addWidget(self.links_widget)

        self.form = inputs.Form(parent=self)
        self._central_widget.layout().addWidget(self.form)

        # start with workspace and suffix inputs
        self.workspace = inputs.Folder(args_key='workspace_dir',
                                       label='Workspace',
                                       validator=self.validator)

        # natcap.invest.pollination.pollination --> pollination
        modelname = self.target.__module__.split('.')[-1]
        self.workspace.set_value(os.path.normpath(
            os.path.expanduser('~/Documents/{model}_workspace').format(
                model=modelname)))

        self.suffix = inputs.Text(
            args_key=suffix_args_key,
            helptext=(
                'A string that will be added to the end of the output file '
                'paths.'),
            label='Results suffix (optional)',
            validator=self.validator)
        self.suffix.textfield.setMaximumWidth(150)

        self.add_input(self.workspace)
        self.add_input(self.suffix)
        self.form.submitted.connect(self.execute_model)

        # Settings files
        self.settings = SETTINGS_TEMPLATE(self.label)
        LOGGER.info('Model settings stored in %s', self.settings.fileName())

        # Menu items.
        self.file_menu = QtWidgets.QMenu('&File', parent=self)
        self.file_menu.addAction(
            qtawesome.icon('fa.cog'),
            'Settings ...', self.settings_dialog.exec_,
            QtGui.QKeySequence(QtGui.QKeySequence.Preferences))
        self.file_menu.addAction(
            qtawesome.icon('fa.floppy-o'),
            'Save as ...', self._save_datastack_as,
            QtGui.QKeySequence(QtGui.QKeySequence.SaveAs))
        self.open_menu = QtWidgets.QMenu('Load parameters', parent=self)
        self.open_menu.setIcon(qtawesome.icon('fa.folder-open-o'))
        self.build_open_menu()
        self.file_menu.addMenu(self.open_menu)

        self.file_menu.addAction(
            'Quit', self.close,
            QtGui.QKeySequence('Ctrl+Q'))
        self.menuBar().addMenu(self.file_menu)

        self.edit_menu = QtWidgets.QMenu('&Edit', parent=self)
        self.edit_menu.addAction(
            qtawesome.icon('fa.undo', color='red'),
            'Clear inputs', self.clear_inputs)
        self.edit_menu.addAction(
            qtawesome.icon('fa.trash-o'),
            'Clear parameter cache for %s' % self.label,
            self.clear_local_settings)
        self.menuBar().addMenu(self.edit_menu)

        self.dev_menu = QtWidgets.QMenu('&Development', parent=self)
        self.dev_menu.addAction(
            qtawesome.icon('fa.file-code-o'),
            'Save to python script ...', self.save_to_python)
        self.menuBar().addMenu(self.dev_menu)

        self.help_menu = QtWidgets.QMenu('&Help', parent=self)
        self.help_menu.addAction(
            qtawesome.icon('fa.info'),
            'About InVEST', self.about_dialog.exec_)
        self.help_menu.addAction(
            qtawesome.icon('fa.external-link'),
            'View documentation', self._check_local_docs,
            QtGui.QKeySequence(QtGui.QKeySequence.HelpContents))
        self.menuBar().addMenu(self.help_menu)

    def build_open_menu(self):
        """(Re-)Build the "Open datastack" menu.

        This menu consists of:

            * An option to select a new datastack file
            * A separator
            * A dynamically-generated list of the 10 most recently-accessed
              datastack files.

        Returns:
            None.
        """
        self.open_menu.clear()
        self.open_file_action = self.open_menu.addAction(
            qtawesome.icon('fa.arrow-circle-o-up'),
            'L&oad datastack, parameter set or logfile...',
            self.load_datastack,
            QtGui.QKeySequence(QtGui.QKeySequence.Open))
        self.open_menu.addSeparator()

        recently_opened_datastacks = json.loads(
            self.settings.value('recent_datastacks', '{}'))

        for datastack_filepath, timestamp in sorted(
                recently_opened_datastacks.items(), key=lambda x: x[1]):

            time_obj = datetime.datetime.strptime(timestamp,
                                                  '%Y-%m-%dT%H:%M:%S.%f')
            if time_obj.date() == datetime.date.today():
                date_label = 'Today at %s' % time_obj.strftime('%H:%M')
            else:
                date_label = time_obj.strftime('%Y-%m-%d at %H:%m')

            # Shorten the path label to only show the topmost directory and the
            # datastack filename.
            datastack_path_directories = datastack_filepath.split(os.sep)
            if len(datastack_path_directories) <= 2:
                # path should be short, show the whole path in the menu.
                path_label = datastack_filepath
            else:
                # show the filename and its parent directory in the menu.
                path_label = os.sep.join(
                    ['...'] + datastack_path_directories[-2:])

            datastack_action = QtWidgets.QAction('%s (Loaded %s)' % (
                path_label, date_label), self.open_menu)
            datastack_action.setData(datastack_filepath)
            datastack_action.triggered.connect(
                self._load_recent_datastack_from_action)
            self.open_menu.addAction(datastack_action)

    def _load_recent_datastack_from_action(self):
        """Load a recent datastack when an action is triggered.

        This slot is assumed to be called when an appropriate QAction is
        triggered.  The ``data()`` set on the QAction must be the filename of
        the datastack selected.  The datastack will be loaded in the model
        interface.

        Returns:
            None.
        """
        # self.sender() is set when this is called as a slot
        self.load_datastack(self.sender().data(), confirm=True)

    def _add_to_open_menu(self, datastack_path):
        """Add a datastack file to the Open-Recent menu.

        This will also store the datastack path in the model's settings object
        and will cause the Open-Recent menu to be rebuilt, limiting the number
        of items in the menu to the 10 most recently-loaded datastack files.

        Parameters:
            datastack_path (string): The path to the datastack file.

        Returns:
            None.
        """
        # load the {path: timestamp} map as a dict from self.settings
        # set the {path: timestamp} tuple
        # store the new value.
        recently_opened_datastacks = json.loads(
            self.settings.value('recent_datastacks', '{}'))
        timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')

        recently_opened_datastacks[datastack_path] = timestamp

        most_recent_datastack_tuples = sorted(
            recently_opened_datastacks.items(), key=lambda x: x[1],
            reverse=True)[:10]

        self.settings.setValue('recent_datastacks',
                               json.dumps(dict(most_recent_datastack_tuples)))
        self.build_open_menu()

    def clear_local_settings(self):
        """Clear all parameters saved for this model.

        Returns:
            None.
        """
        self.settings.clear()
        self.statusBar().showMessage('Cached parameters have been cleared.',
                                     STATUSBAR_MSG_DURATION)

    def clear_inputs(self):
        """Clear the values from any inputs except the workspace.

        This is done for each input object by calling its clear() method.

        Returns:
            None
        """
        for input_obj in self.inputs:
            if input_obj is self.workspace:
                continue
            input_obj.clear()

    def __setattr__(self, name, value):
        """Track Input instances in self.inputs.

        All local attributes will be set, but instances of ``inputs.Input``
        will have their reference added to ``self.inputs``.

        Parameters:
            name (string): The string name of the local attribute being set.
            value (object): The value of the local attribute being set.

        Returns:
            None.
        """
        if isinstance(value, inputs.InVESTModelInput):
            self.inputs.add(value)
        object.__setattr__(self, name, value)

    def _check_local_docs(self, link=None):
        if link in (None, 'localdocs'):
            local_path = os.path.abspath(
                os.path.join('./documentation/userguide/', self.localdoc))
            if not os.path.exists(local_path):
                LOGGER.warning('Could not find local docs path %s',
                               local_path)
                self.local_docs_missing_dialog.exec_()
                return
            link = QtCore.QUrl.fromLocalFile(local_path)
        else:
            link = QtCore.QUrl(link)

        InVESTModel._activate_link(link)

    def _save_datastack_as(self):
        """Save the current set of inputs as a datastack.

        Presents a dialog to the user for input on how to save the datastack,
        and then makes it happen.  A status message is displayed to the
        satus bar when the operation is complete.

        Returns:
           ``None``.
        """
        datastack_opts = self.datastack_options_dialog.exec_()
        if not datastack_opts:  # user pressed cancel
            return

        current_args = self.assemble_args()
        if (not datastack_opts.include_workspace or
                datastack_opts.datastack_type == _DATASTACK_DATA_ARCHIVE):
            del current_args['workspace_dir']

        LOGGER.info('Current parameters:\n%s', pprint.pformat(current_args))

        # if parent dir of archive_path does not exist, create it.
        archive_dir = os.path.dirname(
            os.path.abspath(datastack_opts.archive_path))
        if not os.path.exists(archive_dir):
            try:
                os.makedirs(archive_dir)
            except FileNotFoundError as error:
                alert_message = error.strerror + ': ' + error.filename
                self.statusBar().showMessage(
                    alert_message, STATUSBAR_MSG_DURATION)
                LOGGER.exception(error)
                return

        if datastack_opts.datastack_type == _DATASTACK_DATA_ARCHIVE:
            self.datastack_progress_dialog.exec_build(
                args=current_args,
                model_name=self.target.__module__,
                datastack_path=datastack_opts.archive_path
            )
        else:
            datastack.build_parameter_set(
                args=current_args,
                model_name=self.target.__module__,
                paramset_path=datastack_opts.archive_path,
                relative=datastack_opts.use_relpaths
            )

        save_filepath = os.path.basename(datastack_opts.archive_path)
        alert_message = (
            'Saved current parameters to %s' % save_filepath)
        LOGGER.info(alert_message)
        self.statusBar().showMessage(alert_message, STATUSBAR_MSG_DURATION)
        self.window_title.set_title_attr('filename',
                                         os.path.basename(save_filepath))

    def add_input(self, input_obj):
        """Add an input to the model.

        Parameters:
            input_obj (natcap.invest.ui.inputs.InVESTModelInput): An
                InVESTModelInput instance to add to the model.

        Returns:
            ``None``
        """
        self.form.add_input(input_obj)

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

        # If the workspace exists and contains files, confirm the overwrite.
        if os.path.exists(args['workspace_dir']):
            if len(os.listdir(args['workspace_dir'])) > 0:
                button_pressed = (
                    self.workspace_overwrite_confirm_dialog.exec_())
                if button_pressed != QtWidgets.QMessageBox.Yes:
                    return

        # This is the thread that the UI is executing within.
        ui_thread_name = threading.current_thread().name

        def _logged_target():
            if 'n_workers' in args:
                raise RuntimeError(
                    'n_workers defined in args. It should not be defined.')

            args['n_workers'] = inputs.INVEST_SETTINGS.value(
                'taskgraph/n_workers', '-1')

            name = getattr(self, 'label', self.target.__module__)
            logfile_log_level = getattr(logging, inputs.INVEST_SETTINGS.value(
                'logging/logfile', 'NOTSET'))

            taskgraph_log_level = getattr(
                logging, inputs.INVEST_SETTINGS.value('logging/taskgraph', 'ERROR'))
            logging.getLogger('taskgraph').setLevel(taskgraph_log_level)

            threads_to_exclude = [usage._USAGE_LOGGING_THREAD_NAME]

            with utils.prepare_workspace(args['workspace_dir'],
                                         name,
                                         logging_level=logfile_log_level,
                                         exclude_threads=threads_to_exclude):
                with usage.log_run(self.target.__module__, args):
                    LOGGER.log(datastack.ARGS_LOG_LEVEL,
                               'Starting model with parameters: \n%s',
                               datastack.format_args_dict(
                                   args, self.target.__module__))

                    try:
                        return self.target(args=args)
                    except Exception:
                        LOGGER.exception('Exception while executing %s',
                                         self.target)
                        raise
                    finally:
                        LOGGER.info('Execution finished')

        self.form.run(target=_logged_target,
                      window_title='Running %s' % self.label,
                      out_folder=args['workspace_dir'])

    @QtCore.Slot()
    def load_datastack(self, datastack_path=None, confirm=False):
        """Load a datastack.

        This method is also a slot that accepts no arguments.

        A datastack could be any one of:

            * A logfile from a previous model run.
            * A parameter set (*.invest.json)
            * A parameter archive (*.invest.tar.gz)

        Datastacks may be saved and loaded through the Model UI. For API access
        to datastacks, look at :ref:natcap.invest.datastack.

        Parameters:
            datastack_path=None (string): The path to the datastack file to
                load.  If ``None``, the user will be prompted for a file
                with a file dialog.
            confirm=False (boolean): If True, confirm that values will be
                overwritten by the new datastack.

        Returns:
            ``None``
        """
        if confirm:
            confirm_response = self.input_overwrite_confirm_dialog.exec_()
            if confirm_response != QtWidgets.QMessageBox.Yes:
                return

        if not datastack_path:
            datastack_path = self.file_dialog.open_file(
                title='Select datastack, parameter set or logfile', filters=(
                    'Any file (*.*)',
                    'Parameter set (*.invest.json)',
                    'Parameter archive (*.invest.tar.gz)',
                    'Logfile (*.txt)'))

            # When the user pressed cancel, datastack_path == ''
            if not datastack_path:
                return

        LOGGER.info('Loading datastack from "%s"', datastack_path)
        try:
            stack_type, stack_info = datastack.get_datastack_info(
                datastack_path)
        except Exception:
            fail_message = 'Could not load datastack %s' % datastack_path
            self.statusBar().showMessage(fail_message, STATUSBAR_MSG_DURATION)
            LOGGER.exception(fail_message)
            return

        if stack_info.model_name != self.target.__module__:
            confirm_response = self.model_mismatch_confirm_dialog.exec_(
                stack_info.model_name)
            if confirm_response != QtWidgets.QMessageBox.Yes:
                return

        if stack_type == 'archive':
            extract_dir = self.datastack_archive_extract_dialog.exec_()
            if extract_dir is None:
                return

            self.datastack_progress_dialog.exec_extract(datastack_path,
                                                        extract_dir)

            paramset = datastack.extract_parameter_set(
                os.path.join(extract_dir,
                             datastack.DATASTACK_PARAMETER_FILENAME))
            args = paramset.args

            window_title_filename = os.path.basename(extract_dir)
        elif stack_type in ('json', 'logfile'):
            args = stack_info.args
            window_title_filename = os.path.basename(datastack_path)
        else:
            raise ValueError('Unknown stack type "%s"' % stack_type)

        self.load_args(args)
        self.window_title.set_title_attr('filename', window_title_filename)

        self._add_to_open_menu(datastack_path)
        self.statusBar().showMessage(
            'Loaded datastack from %s' % os.path.abspath(datastack_path),
            STATUSBAR_MSG_DURATION)

    def load_args(self, datastack_args):
        """Load arguments from an args dict.

        Parameters:
            datastack_args (dict): The arguments dictionary from which model
                parameters will be loaded.

        Returns:
            ``None``
        """
        _inputs = dict((ui_input.args_key, ui_input) for ui_input in
                       self.inputs)
        LOGGER.debug(pprint.pformat(_inputs))

        for args_key, args_value in datastack_args.items():
            try:
                _inputs[args_key].set_value(args_value)
            except KeyError:
                LOGGER.warning(('Datastack args_key %s not associated with '
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

        # Post warnings to the WMV dialog and to the inputs themselves if the
        # error only affects that one input.
        # We can only post validation warnings if the input supports it (is a
        # GriddedInput instance).
        args_to_inputs = {}
        for input_ in self.inputs:
            if isinstance(input_, inputs.GriddedInput):
                args_to_inputs[input_.args_key] = input_

        warnings_ = []
        for keys, warning in validation_warnings:
            warnings_.append(
                (list(args_to_inputs[key].label for key in keys), warning))

        for key in args_to_inputs:
            args_to_inputs[key]._validation_finished(validation_warnings)

        self.validation_report_dialog.post_warnings(warnings_)

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

        self.validate(block=False)
        for input_obj in self.inputs:
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
                self.form.run_dialog.close()
                self.close(prompt=False)

            self.form.run_finished.connect(_quickrun_close_model)
            # Start the model immediately (after 0 ms).
            QtCore.QTimer.singleShot(0, self.execute_model)

        # The scrollArea defaults to a size that is too small to actually view
        # the contents of the enclosed widget appropriately.  By adjusting the
        # size here, we ensure that the widgets are an appropriate height.
        # Note that self.resize() does take the window size into account, so
        # all parts of the application window will still be visible, even if
        # the minimumSize().height() would have it extend over the edge of the
        # screen.
        #
        # The 100, 150 additions to the width and height hints come from trial
        # and error, trying to find a decent starting height and width for the
        # window.  I'd prefer to have the window resized according to some
        # internal properties, but the scroll area (self.form.scroll_area)
        # makes that difficult.
        # Adding 200, 250 to the dimensions allows for the window to be wide
        # enough to not hide any widgets in the form, and tall enough to
        # usually show most of the rows in the layout.
        ideal_width = self.form.scroll_area.widget().minimumSize().width() + 200
        ideal_height = max(
            self.sizeHint().height() + 100,
            self.form.scroll_area.widget().minimumSize().height() + 250)
        screen_geometry = QtWidgets.QDesktopWidget().availableGeometry()
        self.resize(
            min(screen_geometry.width(), ideal_width),
            min(screen_geometry.height(), ideal_height))

        inputs.center_window(self)

        # if we're not working off a datastack file, load the last run.
        if not self.window_title.filename:
            self.load_lastrun()

        self.show()
        self.raise_()  # raise window to top of stack.
        self.validate(block=False)  # initial validation for the model

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
                'remember_lastrun', True)
            button_pressed = self.quit_confirm_dialog.exec_(
                starting_checkstate)
            if button_pressed != QtWidgets.QMessageBox.Yes:
                event.ignore()
            elif self.quit_confirm_dialog.checkbox.isChecked():
                self.save_lastrun()
                super(QtWidgets.QMainWindow, self).closeEvent(event)
            self.settings.setValue(
                'remember_lastrun',
                self.quit_confirm_dialog.checkbox.isChecked())
        self.prompt_on_close = True

    def save_to_python(self, filepath=None):
        """Save the current arguments to a python script.

        Parameters:
            filepath (string or None): If a string, this is the path to the
                file that will be written.  If ``None``, a dialog will pop up
                prompting the user to provide a filepath.

        Returns:
            ``None``.
        """
        if filepath is None:
            save_filepath = self.file_dialog.save_file(
                'Save parameters as a python script',
                savefile='python_script.py')
            if save_filepath == '':
                return
        else:
            save_filepath = filepath

        script_template = textwrap.dedent("""\
        # coding=UTF-8
        # -----------------------------------------------
        # Generated by InVEST {invest_version} on {today}
        # Model: {modelname}

        import {py_model}

        args = {model_args}

        if __name__ == '__main__':
            {py_model}.execute(args)
        """)

        with codecs.open(save_filepath, 'w', encoding='utf-8') as py_file:
            cast_args = dict((str(key), value) for (key, value)
                             in self.assemble_args().items())
            args = pprint.pformat(cast_args,
                                  indent=4)  # 4 spaces

            # Tweak formatting from pprint:
            # * Bump parameter inline with starting { to next line
            # * add trailing comma to last item item pair
            # * add extra space to spacing before first item
            args = args.replace('{', '{\n ')
            args = args.replace('}', ',\n}')
            py_file.write(script_template.format(
                invest_version=natcap.invest.__version__,
                today=datetime.datetime.now().strftime('%c'),
                modelname=self.label,
                py_model=self.target.__module__,
                model_args=args))

        self.statusBar().showMessage(
            'Saved run to python script %s' % save_filepath,
            STATUSBAR_MSG_DURATION)

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

        self.statusBar().showMessage('Loaded parameters from previous run.',
                                     STATUSBAR_MSG_DURATION)
        self.window_title.set_title_attr('filename', 'loaded from autosave')

    def dragEnterEvent(self, event):
        """Handle the event where something has been dragged into the window.

        If the thing dragged into the window meets all the following rules:

            * It has text data
            * It has exactly 1 URL
            * The filepath passed via the URL is probably a datastack (as
                determined by ``model.is_probably_datastack()``)

        Then a visual change is made to the model window (text color changes
        and the background color of the window changes) and we accept the
        event.

        Parameters:
            event (QDragEnterEvent): The event to handle.

        Returns:
            None.
        """
        if (len(event.mimeData().urls()) == 1 and
                is_probably_datastack(
                    event.mimeData().urls()[0].toLocalFile())):
            LOGGER.info('Accepting drag enter event for "%s"',
                        event.mimeData().text())
            self.setStyleSheet(
                'QWidget {background-color: rgb(255, 255, 255); '
                'color: rgb(200, 200, 200)}')
            event.accept()
        else:
            LOGGER.info('Rejecting drag enter event for "%s"',
                        event.mimeData().text())
            self.setStyleSheet('')
            event.ignore()

    def dragLeaveEvent(self, event):
        """If the user drags something out of the model, reset the stylesheet.

        This is triggered when something dragged into the window is dragged
        back out.

        Parameters:
            event (QDragLeaveEvent): The event to handle.

        Returns:
            None.
        """
        self.setStyleSheet('')

    def dropEvent(self, event):
        """When something is dropped onto the window.

        Called after it's been dragged into the winodw via a QDragEnterEvent.
        When something is dropped, we assume that it has 1 URL and that its
        path should be loaded as a datastack.

        Parameters:
            event (QDropEvent): The event to handle.

        Returns:
            None.
        """
        path = event.mimeData().urls()[0].toLocalFile()
        self.setStyleSheet('')
        self.load_datastack(path)

    @staticmethod
    def _activate_link(link):
        """Activate a QUrl.

        link (QUrl): a QUrl object that is constructed either from a local file
            path or a URI.

        Returns:
            None.

        """
        LOGGER.debug('Activating link: %s', link)
        # Qt4 and Qt5 have QDesktopServices located in different places.
        try:
            QtCore.QDesktopServices.openUrl(link)
        except AttributeError:
            QtGui.QDesktopServices.openUrl(link)

    @staticmethod
    def _get_latest_version():
        """Get the latest InVEST version string from PyPI page.

        Returns:
            latest_version (str): if the HTTP request is successfully, or
                None if not.

        """
        # Make an HTTP call to InVEST's PyPI page, set timeout of 10s
        try:
            response = requests.get(
                'https://pypi.python.org/pypi/natcap.invest/json', timeout=10)
            # Get the latest version string
            latest_version = json.loads(response.text)['info']['version']
            return latest_version

        # If any ConnectionError, HTTPError, Timeout, or TooManyRedirects
        # exception happens
        except requests.exceptions.RequestException as err:
            LOGGER.exception('Exception while requesting PyPI page: %s' % err)

        # If the text doesn't have the 'info' or 'version' keys
        except KeyError as err:
            LOGGER.exception('Version string could not be found from PyPI page.'
                             'Exception raised: %s' % err)
        return None

    def _needs_update(self, latest_version):
        """Compare the latest version with current version.

        Returns:
            True if the latest version is later than the current version, False
                if the latest version is the same as the current version, or
                the request to the PyPI page wasn't successful.

        """
        latest_version = self._get_latest_version()
        if latest_version:
            return parse_version(latest_version) > parse_version(
                natcap.invest.__version__)
        return False
