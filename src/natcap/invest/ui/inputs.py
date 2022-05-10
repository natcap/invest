# coding=UTF-8
"""Input classes for the InVEST UI, a Qt-based UI abstraction layer."""

import functools
import os
import logging
import platform
import subprocess
import warnings
import sys
import atexit
import itertools
import PySide2

import qtpy
from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui
import qtawesome
import chardet
from .. import utils

try:
    import faulthandler
    faulthandler.enable()
except (ImportError, AttributeError):
    # ImportError when faulthandler not installed
    # AttributeError happens all the time on jenkins.
    pass

from . import execution

try:
    QApplication = QtGui.QApplication
except AttributeError:
    QApplication = QtWidgets.QApplication

QT_APP = QApplication.instance()
if QT_APP is None:
    QT_APP = QApplication(sys.argv)  # pragma: no cover

# If we're running PyQt4, we need to instruct Qt to use UTF-8 strings
# internally.
if qtpy.API in ('pyqt', 'pyqt4'):
    QtCore.QTextCodec.setCodecForCStrings(
        QtCore.QTextCodec.codecForName('UTF-8'))

LOGGER = logging.getLogger(__name__)
ICON_FOLDER = qtawesome.icon('fa5s.folder')
ICON_FILE = qtawesome.icon('fa5s.file')
ICON_ENTER = qtawesome.icon('fa5s.arrow-alt-circle-right',
                            color='green')
ICON_MINUS = qtawesome.icon('fa5s.minus')
_QLABEL_STYLE_TEMPLATE = ('QLabel {{padding:{padding};'
                          'background-color:{bg_color};'
                          'border:{border};}}')
QLABEL_STYLE_INFO = _QLABEL_STYLE_TEMPLATE.format(
    padding='15px', bg_color='#d4efcc', border='2px solid #3e895b')
QLABEL_STYLE_ERROR = _QLABEL_STYLE_TEMPLATE.format(
    padding='15px', bg_color='#ebabb6', border='2px solid #a23332')
INVEST_SETTINGS = QtCore.QSettings(
    QtCore.QSettings.IniFormat,
    QtCore.QSettings.UserScope,
    'Natural Capital Project',
    'InVEST')
DEFAULT_LASTDIR = ''


def _cleanup():
    QT_APP.deleteLater()  # pragma: no cover
atexit.register(_cleanup)


def _apply_sizehint(widget):
    size_hint = widget.sizeHint()
    if size_hint.isValid():
        widget.setMinimumSize(size_hint)


def open_workspace(dirname):
    """Call the correct program to open a folder on disk.

    The program called will depend on the operating system:

        * On mac: ``open``
        * On Windows: ``explorer``
        * On Linux: ``xdg-open``

    Args:
        dirname (string): The folder to open.
    """
    LOGGER.debug("Opening dirname %s", dirname)
    # Try opening up a file explorer to see the results.
    try:
        LOGGER.info('Opening file explorer to workspace directory')
        if platform.system() == 'Windows':
            # Try to launch a windows file explorer to visit the workspace
            # directory now that the operation has finished executing.
            LOGGER.info('Using windows explorer to view files')
            subprocess.Popen('explorer "%s"' % os.path.normpath(dirname))
        elif platform.system() == 'Darwin':
            LOGGER.info('Using mac finder to view files')
            subprocess.Popen(
                'open %s' % os.path.normpath(dirname), shell=True)
        else:
            # Assume we're on linux.  No biggie, just use xdg-open to use
            # default file opening scheme.
            LOGGER.info('Not on windows or mac, using default file browser')
            subprocess.Popen(['xdg-open', dirname])
    except OSError as error:
        # OSError is thrown if the given file browser program (whether
        # explorer or xdg-open) cannot be found.  No biggie, just pass.
        LOGGER.error(error)
        LOGGER.error(
            ('Cannot find default file browser. Platform: %s |'
             ' folder: %s'), platform.system(), dirname)


def center_window(window_ptr):
    """Center the provided window on the current screen.

    Args:
        window_ptr (QtWidgets.QWidget): a reference to a Qt window.
    """
    geometry = window_ptr.frameGeometry()
    center = QtWidgets.QDesktopWidget().availableGeometry().center()
    geometry.moveCenter(center)
    window_ptr.move(geometry.topLeft())


class Validator(QtCore.QObject):
    """A class to manage validating in a separate Qt thread."""

    started = QtCore.Signal()
    finished = QtCore.Signal(list)

    def __init__(self, parent):
        """Initialize the Validator instance.

        Args:
            parent (QtWidgets.QWidget): The parent qwidget.  This will be the
                parent of the validation thread.
        """
        # TODO: remove parent here?
        QtCore.QObject.__init__(self, parent)
        self._validation_worker = None

    def in_progress(self):
        """Whether the validation thread is running.

        Returns:
            is_running (bool): Whether the validation thread is running.
        """
        return False
        #return self._validation_thread.isRunning()

    def validate(self, target, args, limit_to=None):
        """Validate the provided args with the provided target.

        Args:
            target (callable): The validation callable.  Must adhere to the
                InVEST validation API.
            args (dict): The arguments dictionary to validate.
            limit_to=None (string or None): Optional. If provided, this is the
                validation key that should be validated.  All other keys will
                be excluded.  Part of the InVEST Validation API.

        Returns:
            ``None``
        """
        self.started.emit()
        self._validation_worker = ValidationWorker(
            target=target,
            args=args,
            limit_to=limit_to)
        self._validation_worker.run()
        warnings_ = [w for w in self._validation_worker.warnings]
        self.finished.emit(warnings_)


class MessageArea(QtWidgets.QLabel):
    """An object to represent the status box in the model progress dialog.

    Example:
        area = MessageArea()
        area.setText('some success text')
        area.set_error(False)  # sets the stylesheet for non-error messages.
    """

    def __init__(self):
        """Initialize the MessageArea.

        From a Qt perspective, this is little more than calling
        ``QLabel.__init__`` and ensuring that the qlabel has word wrapping and
        rich text enabled.
        """
        QtWidgets.QLabel.__init__(self)
        self.setWordWrap(True)
        self.setTextFormat(QtCore.Qt.RichText)
        self.error = False

    def set_error(self, is_error):
        """Set the label stylesheet for error or success.

        The label is shown when the error status is set.

        Args:
            is_error (bool): If ``True``, a green success style will be used.
                If ``False``, a red failure style will be used instead.

        Returns:
            ``None``
        """
        self.error = is_error
        if is_error:
            self.setStyleSheet(QLABEL_STYLE_ERROR)
        else:
            self.setStyleSheet(QLABEL_STYLE_INFO)
        self.show()


class QLogHandler(logging.StreamHandler):
    """A ``logging.StreamHandler`` subclass for writing to a stream widget."""

    def __init__(self, stream_widget):
        """Initialize the logging handler.

        Args:
            stream_widget (QtWidgets.QWidget): A QWidget that supports the
                python streams API.
        """
        logging.StreamHandler.__init__(self, stream=stream_widget)
        self._stream = stream_widget
        self.setLevel(logging.NOTSET)  # capture everything

        self.formatter = logging.Formatter(
            fmt=utils.LOG_FMT)
        self.setFormatter(self.formatter)


class LogMessagePane(QtWidgets.QPlainTextEdit):
    """A subclass of ``QtWidgets.QPlainTextEdit`` to support write().

    Uses the signals/slots framework to support writing text to the
    QPlainTextEdit from different threads.
    """

    message_received = QtCore.Signal(str)

    def __init__(self, parent):
        """Initialize the LogMessagePane instance.

        Sets the stylesheet for the QPlainTextEdit, and sets it to read-only.
        """
        QtWidgets.QPlainTextEdit.__init__(self, parent=parent)

        self.setReadOnly(True)
        self.setStyleSheet("QWidget { background-color: White; "
                           'font-family: monospace;}')
        self.message_received.connect(self._write)

    def write(self, message):
        """'Write' the message to the message pane.

        In actuality, this emits the ``message_received`` signal, with the
        message as the value.  This allows this ``write`` method  to be
        called from any thread, and the signal/slot framework will cause the
        message to actually be rendered on an iteration of the event loop.

        Args:
            message (string): The message to be written to the message pane.

        Returns:
            ``None``
        """
        try:
            self.message_received.emit(message)
        except RuntimeError:
            pass

    def _write(self, message):
        """Write the message provided to the message pane.

        Calling this method from a thread other than the main thread will
        cause an error from within Qt.

        Args:
            message (string): The message to be appended to the end of the
                QMessagePane.

        Returns:
            ``None``
        """
        self.insertPlainText(message)
        self.textCursor().movePosition(QtGui.QTextCursor.End)
        self.setTextCursor(self.textCursor())


class FileSystemRunDialog(QtWidgets.QDialog):
    """A dialog to display messages to the user while a process is running.

    Messages are displayed to a message pane that scrolls continuously as new
    messages are added, and an indeterminate progress bar is visible to
    offer a visual queue that something is happening.

    While the process is running, there is a checkbox that may be selected.
    When ths process finishes (and the checkbox is selected), the workspace
    folder is opened in the OS's default file explorer.  When the process
    finishes, the checkbox is converted to a button that, when pressed, wil
    open the workspace with the OS's default file explorer.
    """

    def __init__(self):
        """Initialize the dialog."""
        QtWidgets.QDialog.__init__(self)

        self.is_executing = False
        self.cancel = False
        self.out_folder = None

        self.setLayout(QtWidgets.QVBoxLayout())
        self.resize(700, 500)
        center_window(self)
        self.setModal(True)

        # create statusArea-related widgets for the window.
        self.statusAreaLabel = QtWidgets.QLabel(
            FileSystemRunDialog._build_status_area_label())

        self.log_messages_pane = LogMessagePane(parent=self)
        self.loghandler = QLogHandler(self.log_messages_pane)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.NOTSET)
        self.logger.addHandler(self.loghandler)

        # create an indeterminate progress bar.  According to the Qt
        # documentation, an indeterminate progress bar is created when a
        # QProgressBar's minimum and maximum are both set to 0.
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)
        self.progressBar.setTextVisible(False)
        progress_sizehint = self.progressBar.sizeHint()
        if progress_sizehint.isValid():
            self.progressBar.setMinimumSize(progress_sizehint)

        self.openWorkspaceCB = QtWidgets.QCheckBox(
            'Open workspace after success')
        self.openWorkspaceButton = QtWidgets.QPushButton('Open workspace')
        self.openWorkspaceButton.pressed.connect(self._request_workspace)
        self.openWorkspaceButton.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.openWorkspaceButton.setMaximumWidth(150)
        self.openWorkspaceButton.setVisible(False)
        self.messageArea = MessageArea()
        self.messageArea.clear()

        # Add the new widgets to the window
        self.layout().addWidget(self.statusAreaLabel)
        self.layout().addWidget(self.log_messages_pane)
        self.layout().addWidget(self.messageArea)
        self.layout().addWidget(self.progressBar)
        self.layout().addWidget(self.openWorkspaceCB)
        self.layout().addWidget(self.openWorkspaceButton)

        self.backButton = QtWidgets.QPushButton(' Back')
        self.backButton.setToolTip('Return to parameter list')

        # add button icons
        self.backButton.setIcon(QtGui.QIcon(ICON_ENTER))

        # disable the 'Back' button by default
        self.backButton.setDisabled(True)

        # create the buttonBox (a container for buttons) and add the buttons to
        # the buttonBox.
        self.buttonBox = QtWidgets.QDialogButtonBox()
        self.buttonBox.addButton(
            self.backButton, QtWidgets.QDialogButtonBox.AcceptRole)

        # connect the buttons to their callback functions.
        self.backButton.clicked.connect(self.close)

        # add the buttonBox to the window.
        self.layout().addWidget(self.buttonBox)

        # Indicate that this window should be styled like a dialog.
        self.setWindowFlags(QtCore.Qt.Dialog)

    @staticmethod
    def _build_status_area_label():
        """Build the status area label.

        This is a static method that checks the value of the
        ``logging/run_dialog`` QSetting and returns a formatted string for
        use as the log message area label.

        Returns:
            A string.
        """
        return 'Messages (%s and higher):' % (
            INVEST_SETTINGS.value('logging/run_dialog', 'INFO'))

    def __del__(self):
        """Delete/deregister required objects."""
        self.logger.removeHandler(self.loghandler)
        try:
            self.deleteLater()
        except RuntimeError:
            # When this dialog has already been deleted.
            LOGGER.debug('This FileSystemRunDialog has already been deleted.')

    def start(self, window_title, out_folder):
        """Set the state of the dialog to indicate processing has started."""
        logging_level = INVEST_SETTINGS.value(
            'logging/run_dialog', 'INFO')
        self.loghandler.setLevel(getattr(logging, logging_level))

        # set the label atop the messages pane to include the currently-set
        # logging level for the run dialog.
        self.statusAreaLabel.setText(
            FileSystemRunDialog._build_status_area_label())

        if not window_title:
            window_title = "Running ..."
        self.setWindowTitle(window_title)
        self.out_folder = out_folder

        self.is_executing = True
        self.log_messages_pane.clear()
        self.progressBar.setMaximum(0)  # start the progressbar.
        self.backButton.setDisabled(True)

        self.log_messages_pane.write('Initializing...\n')
        self.log_messages_pane.write(
            'Showing messages with level %s and higher\n' % logging_level)

    def finish(self, exception):
        """Notify the user that model processing has finished.

        Args:
            exception (Exception or None): The exception object
                If the error encountered.  None if no error found.

        Returns:
            ``None``
        """
        self.is_executing = False
        self.progressBar.setMaximum(1)  # stops the progressbar.
        self.backButton.setDisabled(False)

        if exception:
            self.messageArea.set_error(True)
            self.messageArea.setText(
                ('<b>%s</b> encountered: <em>%s</em> <br/>'
                 'See the log for details.') % (
                    exception.__class__.__name__,
                    exception))
            self.messageArea.setStyleSheet(
                'QLabel { padding: 15px;'
                'background-color: #ebabb6; border: 2px solid #a23332;}')
        else:
            self.messageArea.set_error(False)
            self.messageArea.setText('Model completed successfully.')
            self.messageArea.setStyleSheet(
                'QLabel { padding: 15px;'
                'background-color: #d4efcc; border: 2px solid #3e895b;}')

        # Change the open workspace presentation.
        if self.openWorkspaceCB.isChecked():
            self._request_workspace()
        self.openWorkspaceCB.setVisible(False)
        self.openWorkspaceButton.setVisible(True)

    def _request_workspace(self):
        """Slot for attempting to open a workspace.

        This slot may be called by signals that do not pass a parameter value.
        """
        open_workspace(self.out_folder)

    def reject(self):
        """Reject the dialog.

        Triggered when the user presses ESC.  Overridden from Qt.
        """
        # Called when the user presses ESC.
        if self.is_executing:
            # Don't allow the window to close if we're executing.
            return
        QtWidgets.QDialog.reject(self)

    def closeEvent(self, event):
        """CloseEvent handler, overridden from QWidget.closeEvent.

        Overridden to prevent the user from closing the modal dialog if the
        thread is executing.

        Returns:
            ``None``.
        """
        if self.is_executing:
            event.ignore()
        else:
            self.openWorkspaceCB.setVisible(True)
            self.openWorkspaceButton.setVisible(False)
            self.messageArea.clear()
            self.cancel = False

            QtWidgets.QDialog.closeEvent(self, event)


class InfoButton(QtWidgets.QPushButton):
    """An informational button that shows helpful text when clicked."""

    def __init__(self, default_message=None):
        """Initialize an instance of InfoButton.

        Args:
            default_message=None (string or None).  If not None, the message
                that the button should show by default when clicked.

        Returns:
            ``None``
        """
        QtWidgets.QPushButton.__init__(self)
        self.setFlat(True)
        if default_message:
            self.setWhatsThis(default_message)
        self.clicked.connect(self._show_popup)

    def _show_popup(self, clicked=False):
        """Slot for QPushButton.clicked() signal.

        Args:
            clicked=False: This parameter will always be false, so long as the
                InfoButton instance isn't checkable.  The parameter still has
                to be here to match the signature the clicked signal expects.
        """
        QtWidgets.QWhatsThis.enterWhatsThisMode()

        # QtCore.QPoint(0, 0) maps to the top-left corner of this widget.
        # mapToGlobal() turns that coordinate into a global coordinate.
        QtWidgets.QWhatsThis.showText(self.mapToGlobal(QtCore.QPoint(0, 0)),
                                      self.whatsThis(), self)


class ValidButton(InfoButton):
    """An informational button, styled for validation success or errors."""

    def __init__(self, *args, **kwargs):
        """Initialize the ValidButton.

        Any parameters provided are passed directly through to the underlying
        instance of InfoButton.
        """
        InfoButton.__init__(self, *args, **kwargs)
        self.successful = True

    def clear(self):
        """Clear the icon, WhatsThis text and ToolTip text.

        Returns:
            None.
        """
        self.setIcon(QtGui.QIcon())  # clear the icon
        self.setWhatsThis('')
        self.setToolTip('')

    def set_errors(self, errors):
        """Set the error message and style based on the provided errors.

        Args:
            errors (list): A list of strings.  If this list is empty, the
                style of the button is set to green, indicating validation
                success.  If this list is not empty, the strings in this list
                will be formatted and set as the error text, and the button
                style will be set to a red.

        Returns:
            ``None``
        """
        if errors:
            self.setIcon(qtawesome.icon('fa5s.times',
                                        color='red'))
            error_string = '<br/>'.join(errors)
            self.successful = False
        else:
            self.setIcon(qtawesome.icon('fa5s.check',
                                        color='green'))
            error_string = 'Validation successful'
            self.successful = True

        self.setWhatsThis(error_string)
        self.setToolTip(error_string)


class HelpButton(InfoButton):
    """An InfoButton with an informational help icon."""

    def __init__(self, default_message=None):
        """Initialize the HelpButton.

        Args:
            default_message=None (string): The default message of this button.
                See InfoButton.__init__ for more information.

        Returns:
            ``None``
        """
        InfoButton.__init__(self, default_message)
        self.setIcon(qtawesome.icon('fa5s.info-circle',
                                    color='blue'))


class ValidationWorker(QtCore.QObject):
    """A worker object for executing validation.

    This object is implemented for use with a QThread, and is not started
    until the start() method is called.
    """

    started = QtCore.Signal()
    finished = QtCore.Signal()

    def __init__(self, target, args, limit_to=None):
        """Initialize the ValidationWorker.

        Args:
            target (callable): The validation function.  Must adhere to the
                InVEST validation API.
            args (dict): The arguments dictionary to validate.
            limit_to=None (string): The string key that will limit validation.
                ``None`` if all keys should be validated.

        Returns:
            ``None``
        """
        QtCore.QObject.__init__(self)
        self.target = target
        self.args = args
        self.limit_to = limit_to
        self.warnings = []
        self.error = None
        self.started.connect(self.run)
        self._finished = False

    def isFinished(self):
        """Check whether the validation callable has finished executing.

        Returns:
            finished (bool): Whether validation has finished.
        """
        return self._finished

    def start(self):
        """Begin execution of the validation callable.

        This method is non-blocking.

        Returns:
            ``None``
        """
        self.started.emit()

    def run(self):
        """Execute the validation callable.

        Warnings are saved to ``self.warnings``.  The signal ``self.finished``
        is emitted when processing finishes.  If an exception is encountered,
        the exception object is saved to ``self.error`` and the exception is
        logged.

        Returns:
            ``None``
        """
        # Target must adhere to InVEST validation API.
        LOGGER.info(
            'Starting validation thread with target=%s, args=%s, limit_to=%s',
            self.target, self.args, self.limit_to)
        try:
            self.warnings = self.target(
                self.args, limit_to=self.limit_to)
            LOGGER.info(
                'Validation thread returned warnings: %s', self.warnings)
        except Exception as error:
            self.error = str(error)
            LOGGER.exception(
                'Validation: Error when validating %s:', self.target)
        self._finished = True
        self.finished.emit()


class FileDialog(object):
    """A convenience wrapper for QtWidgets.QFileDialog."""

    def __init__(self, parent=None):
        """Initialize the FileDialog instance.

        Returns:
            ``None``
        """
        object.__init__(self)
        self.file_dialog = QtWidgets.QFileDialog(parent=parent)

    def __del__(self):
        """Destructor for the FileDialog instance."""
        try:
            self.file_dialog.deleteLater()
        except RuntimeError:
            # Raised when the file dialog has already been deleted.
            pass

    def save_file(self, title, start_dir=None, savefile=None):
        """Prompt the user to save a file.

        Args:
            title (string): The title of the save file dialog.
            start_dir=None (string): The starting directory.  If ``None``,
                the last-accessed directory will be fetched from
                the invest settings.
            savefile=None (string): The filename to use by default.
                If ``None``, no default filename will be provided in the
                dialog, and the user will need to provide a filename.

        Returns:
            The absolute path to the filename selected by the user.
        """
        if not start_dir:
            start_dir = os.path.expanduser(
                INVEST_SETTINGS.value('last_dir', DEFAULT_LASTDIR))

        # Allow us to open folders with spaces in them.
        os.path.normpath(start_dir)

        if savefile:
            default_path = os.path.join(start_dir, savefile)
        else:
            # If we pass a folder, the dialog will open to the folder
            default_path = start_dir

        result = self.file_dialog.getSaveFileName(self.file_dialog, title,
                                                  default_path)
        # Different versions of PyQt5 variously return a single filename or a
        # tuple of (filename, last_filter).  I haven't been able to figure out
        # where this break is as of yet, so just catching the ValueError when
        # there's only one return value should be good enough.
        try:
            filename, last_filter = result
        except ValueError:
            filename = result

        INVEST_SETTINGS.setValue('last_dir',
                                 os.path.dirname(str(filename)))
        return filename

    def open_file(self, title, start_dir=None, filters=()):
        """Prompt the user for a file to open.

        Args:
            title (string): The title of the dialog.
            start_dir=None (string): The starting directory.  If ``None``,
                the last-accessed directory will be fetched from the invest
                settings.
            filters=() (iterable): an iterable of filter strings to use in the
                dialog.  An example iterable would have the format::

                    filters = (
                        'Images (*.png *.xpm *.jpg)',
                        'GeoTiffs (*.tif)'
                    )

        Returns:
            The absolute path to the selected file to open.
        """
        if not start_dir:
            start_dir = os.path.expanduser(
                INVEST_SETTINGS.value('last_dir', DEFAULT_LASTDIR))

        # Allow us to open folders with spaces in them.
        os.path.normpath(start_dir)

        filters = ';;'.join(filters)
        LOGGER.info('Using filters "%s"', filters)

        result = self.file_dialog.getOpenFileName(self.file_dialog, title,
                                                  start_dir, filters)
        # Different versions of PyQt5 variously return a single filename or a
        # tuple of (filename, last_filter).  I haven't been able to figure out
        # where this break is as of yet, so just catching the ValueError when
        # there's only one return value should be good enough.
        try:
            filename, last_filter = result
        except ValueError:
            filename = result

        INVEST_SETTINGS.setValue('last_dir',
                                 os.path.dirname(str(filename)))
        return filename

    def open_folder(self, title, start_dir=None):
        """Prompt the user for a directory to open.

        Args:
            title (string): The title of the dialog.
            start_dir=None (string): The starting directory.  If ``None``,
                the last-accessed directory will be fetched from the invest
                settings.

        Returns:
            The absolute path to the directory selected.
        """
        if not start_dir:
            start_dir = os.path.expanduser(
                INVEST_SETTINGS.value('last_dir', DEFAULT_LASTDIR))
        dialog_title = 'Select folder: ' + title

        dirname = self.file_dialog.getExistingDirectory(
            self.file_dialog, dialog_title, start_dir)
        dirname = str(dirname)
        INVEST_SETTINGS.setValue('last_dir', dirname)
        return dirname


class AbstractFileSystemButton(QtWidgets.QPushButton):
    """Shared base class for buttons that prompt for a path when pressed.

    Subclasses are expected to set the local attribute ``self.open_method``
    with a callable that takes no parameters.  This method should prompt
    the user with an appropriate dialog.  If the dialog needs to take some
    parameters as input, these may be set via
    ``self.set_dialog_options``.

    Example:
        class SomeSubclass(AbstractFileSystemButton):
            def __init__(self):
                AbstractFileSystemButton.__init__(self, 'title')
                self.open_method = self.dialog.open_file

        button = SomeSubclass()

        # Set options for the dialog class
        button.set_dialog_options(
            start_dir=os.path.getcwd(),
            filters=())
    """

    path_selected = QtCore.Signal(str)

    def __init__(self, dialog_title):
        """Initialize the AbstractFileSystemButton.

        Args:
            dialog_title (string): The title of the filesystem dialog owned
                by this object.

        Returns:
            ``None``
        """
        QtWidgets.QPushButton.__init__(self)
        if not hasattr(self, '_icon'):
            self._icon = ICON_FOLDER
        self.setIcon(self._icon)
        self.dialog_title = dialog_title
        self.dialog = FileDialog()
        self.open_method = None  # This should be overridden
        self.clicked.connect(self._get_path)
        self._dialog_kwargs = {
            'title': self.dialog_title,
            'start_dir': INVEST_SETTINGS.value('last_dir',
                                               DEFAULT_LASTDIR),
        }

    def _get_path(self):
        """Use ``self.open_method`` to present a dialog to the user.

        ``self.open_method`` is called with any dialog kwargs that happen to
        be set.

        When a path is selected, the ``path_selected`` signal is emitted with
        the path selected by the user.
        """
        selected_path = self.open_method(**self._dialog_kwargs)
        self.path_selected.emit(selected_path)

    def set_dialog_options(self, **kwargs):
        """Set the dialog keyword arguments from args passed to this method.

        Any keyword arguments may be passed to this method.
        """
        self._dialog_kwargs = kwargs


class FileButton(AbstractFileSystemButton):
    """A filesystem button that prompts to open a file."""

    def __init__(self, dialog_title):
        """Initialize the FileButton.

        Args:
            dialog_title (string): The title of the file selection dialog.

        Returns:
            ``None``
        """
        self._icon = ICON_FILE
        AbstractFileSystemButton.__init__(self, dialog_title)
        self.open_method = self.dialog.open_file


class SaveFileButton(AbstractFileSystemButton):
    """A filesystem button that prompts to save a file."""

    def __init__(self, dialog_title, default_savefile):
        """Initialize the SaveFileButton.

        Args:
            dialog_title (string): The title of the file selection dialog.
            default_savefile (string): The file basename to use by default.
                The user may override this filename within the dialog.

        Returns:
            ``None``
        """
        self._icon = ICON_FILE
        AbstractFileSystemButton.__init__(self, dialog_title)
        self.open_method = functools.partial(
            self.dialog.save_file,
            savefile=default_savefile)


class FolderButton(AbstractFileSystemButton):
    """A filesystem button that prompts to select a folder."""

    def __init__(self, dialog_title):
        """Initialize the FolderButton.

        Args:
            dialog_title (string): The title of the folder selection dialog.

        Returns:
            ``None``
        """
        AbstractFileSystemButton.__init__(self, dialog_title)
        self.open_method = self.dialog.open_folder


class InVESTModelInput(QtCore.QObject):
    """Base class for InVEST inputs.

    Key concepts for the input class include:

        * Sufficiency: Whether an input has value and is interactive.  When
          sufficiency changes, the ``sufficiency_changed`` signal is emitted
          with the new sufficiency. The current sufficiency may be accessed
          with the ``self.sufficient`` attribute.
        * Interactivity: Whether the component widgets may be interacted with
          by the user.  When this changes, the ``interactivity_changed``
          signal is emitted with the new interactivity.  The current
          interactivity may be accessed with the ``self.interactive``
          attribute.
        * Value: Every input has a value that can be set by interacting with
          the InVESTModelInput's component widgets. How the value is changed by
          interacting with these widgets depends on the subclass. The current
          value can be fetched with ``self.value()``.  When the value changes,
          the ``value_changed`` signal is emitted.
        * Visibility: With Qt, visibility is actually controlled by
          containers and the parent window, among other things.  Visibility
          here indicates whether the widgets should be considered by the
          package as being visible.

    Subclasses of InVESTModelInput must implement these methods:

        * value(self)
        * set_value(self, value)

    Signals used by this class:

        * ``value_changed`` (string): Emitted when the value of the InVESTModelInput
            instance changes.  Slots are called with the string value of the
            input as the one and only parameter.
        * ``interactivity_changed`` (bool): Emitted when an element's
            interactivity changes, as when set by ``set_interactive``.  The
            parameter passed to slots is the new interactivity of the input.
            So, if the input is becoming interactive, the parameter passed from
            interactivity_changed will be ``True``.
        * ``sufficiency_changed`` (bool).  Emitted when the input's sufficiency
            changes.  See note above on sufficiency.  The parameter passed to
            slots indicates the new sufficiency.
    """

    value_changed = QtCore.Signal(str)
    interactivity_changed = QtCore.Signal(bool)
    sufficiency_changed = QtCore.Signal(bool)

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None):
        """Initialize the InVESTModelInput instance.

        Args:
            label (string): The string label of the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.

        Returns:
            ``None``
        """
        try:
            QtCore.QObject.__init__(self)
        except RuntimeError:
            # Happens when we initialize the object more than once.
            # This is known to happen when initializing the Container class.
            # I'm not currently sure how to work around this other than
            # catching this exception at the moment.  This wasn't an issue
            # with PyQt4.
            pass

        self.label = label
        self.widgets = []
        self.dirty = False
        self.interactive = interactive
        self.args_key = args_key
        self.helptext = helptext
        self.sufficient = False
        self._visible_hint = True

        self.value_changed.connect(self._check_sufficiency)
        self.interactivity_changed.connect(self._check_sufficiency)

    def _check_sufficiency(self, value_or_interactivity):
        """Check the sufficiency of the input.

        Emits the signal ``self.sufficiency_changed`` if sufficiency has
        changed.

        Args:
            value_or_interactivity: The value passed from a signal that
                has triggered this slot.

        Returns:
            ``None``
        """
        # We're using self.value() instead of ``value_or_interactivity``
        # parameter because the parameter could be either a string or a
        # boolean representing either the value or the interactivity.
        # Therefore, we need to check the local methods and variables to
        # determine sufficiency.
        try:
            value_valid = len(self.value()) > 0
        except TypeError:
            # Some InVESTModelInputs (containers, and checkboxes, most notably) return
            # True or False based on whether they are checked.
            value_valid = self.value()
        new_sufficiency = value_valid and self.interactive

        LOGGER.debug('Sufficiency for %s %s --> %s', self,
                     self.sufficient, new_sufficiency)

        if self.sufficient != new_sufficiency:
            self.sufficient = new_sufficiency
            self.sufficiency_changed.emit(new_sufficiency)

    def clear(self):
        """Reset the input to an initial, 'blank' state.

        This method must be reimplemented for each subclass.

        Returns:
            None.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def visible(self):
        """Whether the input is supposed to be visible.

        Note that the visibility of a Qt widget is dependent on many things,
        such as the visibility of the widget that contains this widget (as when
        there is a collapsed QGroupBox that contains this widget).

        Returns:
            A bool of whether the input should be visible.
        """
        return self._visible_hint

    def set_visible(self, visible_hint):
        """Set the visibility hint for the input.

        Qt visibility is actually controlled by containers and the parent
        window.  Visibility here indicates whether the widgets
        should be considered by the package as being visible.

        Args:
            visible_hint (bool): Whether the InVESTModelInput instance should be
                considered to be visible.

        Returns:
            ``None``
        """
        self._visible_hint = visible_hint
        if any([widget.parent().isVisible() for widget in self.widgets
                if widget is not None and widget.parent() is not None]):
            for widget in self.widgets:
                if not widget:
                    continue
                widget.setVisible(self._visible_hint)

    def value(self):
        """Fetch the value of this InVESTModelInput.

        Note:
            This method must be reimplemented by subclasses.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def set_value(self, value):
        """Set the value of this input.

        Note:
            This method must be reimplemented by subclasses.

        Args:
            value: The new value of the InVESTModelInput.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def set_interactive(self, enabled):
        """Set the interactivity of the component widgets.

        Emits the ``interactivity_changed`` signal if the interactivity
        changes.

        Args:
            enabled (bool): Whether inputs should be interactive.

        Returns:
            ``None``
        """
        self.interactive = enabled
        for widget in self.widgets:
            if widget is None:  # widgets to be skipped are None
                continue
            widget.setEnabled(enabled)
        self.interactivity_changed.emit(self.interactive)

    def add_to(self, layout):
        """Add the component widgets of this InVESTModelInput to a QGridLayout.

        Args:
            layout (QtWidgets.QGridLayout): A QGridLayout to which all
                component widgets should be added.

        Returns:
            ``None``
        """
        self.setParent(layout.parent().window())  # all widgets belong to Form
        current_row = layout.rowCount()
        for widget_index, widget in enumerate(self.widgets):
            if widget is None:
                continue

            # set the default interactivity based on self.interactive
            widget.setEnabled(self.interactive)

            _apply_sizehint(widget)
            layout.addWidget(
                widget,  # widget
                current_row,  # row
                widget_index)  # column


class GriddedInput(InVESTModelInput):
    """A subclass of InVESTModelInput that assumes it's using a QGridLayout.

    In addition to the core concepts of InVESTModelInput, GriddedInput adds a few:

        * Validity: A GriddedInput has a value that is either valid or
          invalid. The current validity may be accessed through
          ``self.valid()``.  If the input has never been validated, validity
          will be ``None``.  Otherwise, a bool will be returned. Validity
          is typically checked when the input's value changes, but subclasses
          must manage how and when validation is triggered.  When
          the validity of the input changes, ``validity_changed`` is emitted
          with the new validity.
        * Hidden: A GriddedInput may be initialized with the ``hideable``
          parameter set to ``True``.  If this is the case, most of the
          component widgets are hidden from view until a checkbox is triggered
          to make the widgets visible again.  This is useful in contexts where
          a single checkbox is needed to control whether an input should be
          interactive, and this approach reduces the code needed to implement
          this behavior within a UI window. When the state of this checkbox
          changes, the ``hidden_changed`` signal is emitted.
    """

    hidden_changed = QtCore.Signal(bool)
    validity_changed = QtCore.Signal(bool)

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        """Initialize this GriddedInput instance.

        Args:
            label (string): The string label to use for the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.
            hideable=False (bool): If ``True``, the input will have a
                checkbox that, when triggered, will show/hide the other
                component widgets in this InVESTModelInput.
            validator=None (callable): The validator callable to use for
                validation.  This callable must adhere to the InVEST
                Validation API.

        Returns:
            ``None``
        """
        InVESTModelInput.__init__(
            self, label=label, helptext=helptext, interactive=interactive,
            args_key=args_key)

        self._valid = None
        self.validator_ref = validator
        self._validator = Validator(self)
        self._validator.finished.connect(self._validation_finished)

        self.label_widget = QtWidgets.QLabel(self.label)
        self.hideable = hideable
        self.sufficient = False  # False until value set and interactive
        self.valid_button = ValidButton()
        if helptext:
            self.help_button = HelpButton(helptext)
        else:
            self.help_button = QtWidgets.QWidget()  # empty widget!

        # Within a GriddedInput, a single Input instance occupies a whole row
        # of the grid layout.  If the input should occupy only some of the
        # columns in the grid, represent a grid cell being blank here with a
        # None value.
        self.widgets = [
            self.valid_button,
            self.label_widget,
            None,
            None,
            self.help_button,
        ]

        if self.hideable:
            self.label_widget = QtWidgets.QCheckBox(self.label_widget.text())
            self.widgets[1] = self.label_widget
            self.label_widget.stateChanged.connect(self._hideability_changed)
            self._hideability_changed(True)

        # initialize visibility, as we've changed the input's widgets
        self.set_visible(self._visible_hint)

    def _validate(self):
        """Validate the input using its current value.

        Validation is intended to be triggered by events in the UI and not by
        the user, hence the private function signature.
        """
        if self.validator_ref is not None:
            LOGGER.info(
                'Validation: validator taken from self.validator_ref: %s',
                self.validator_ref)
            validator_ref = self.validator_ref
        else:
            if self.args_key is None:
                LOGGER.info(
                    ('Validation: No validator and no args_id defined; '
                     'skipping.  Input assumed to be valid. %s'),
                    self)
                self._validation_finished(validation_warnings=[])
                return
            else:
                # args key defined, but a validator is not; input assumed
                # to be valid.
                warnings.warn(('Validation: args_key defined, but no '
                               'validator defined.  Input assumed to be '
                               'valid. %s') % self)
                self._validation_finished(validation_warnings=[])
                return

        try:
            args = self.parent().assemble_args()
        except AttributeError:
            # When self.parent() is not set, as in testing.
            # self.parent() is only set when the InVESTModelInput is added to a
            # layout.
            args = {self.args_key: self.value()}

        LOGGER.info(
            ('Starting validation thread for %s with target:%s, args:%s, '
             'limit_to:%s'),
            self, validator_ref, args, self.args_key)

        self._validator.validate(
            target=validator_ref,
            args=args)

    def _validation_finished(self, validation_warnings):
        """Interpret any validation errors and format them for the UI.

        This is signaled whenever the validataion for this object is complete.
        Either through an error or through a callback from the threadded call
        with self._validator.  If the validity of the input changes, the
        ``validity_changed`` signal is emitted with the new validity.

        Args:
            validation_warnings (list): A list of string validation warnings
                returned from the validation callable.

        Returns:
            ``None``
        """
        new_validity = not bool(validation_warnings)
        if self.args_key:
            applicable_warnings = [w[1] for w in validation_warnings
                                   if self.args_key in w[0]]
        else:
            applicable_warnings = [w[1] for w in validation_warnings]

        LOGGER.info('Cleaning up validation for %s.  Warnings: %s.  Valid: %s',
                    self, applicable_warnings, new_validity)
        if applicable_warnings:
            self.valid_button.set_errors(applicable_warnings)
            tooltip_errors = '<br/>'.join(applicable_warnings)
        else:
            self.valid_button.set_errors([])
            tooltip_errors = ''

        for widget in self.widgets[0:2]:  # skip file selection, help buttons
            if widget is not None:
                widget.setToolTip(tooltip_errors)

        current_validity = self._valid
        self._valid = new_validity
        if current_validity != new_validity:
            self.validity_changed.emit(new_validity)

    def valid(self):
        """Check the validity of the input.

        Returns:
            The boolean validity of the input.
        """
        return self._valid

    def clear(self):
        """Reset validity, sufficiency and the valid button state.

        Returns:
            None.
        """
        self._valid = None
        self.sufficient = False
        self.valid_button.clear()

    @QtCore.Slot(int)
    def _hideability_changed(self, show_widgets):
        """Set the hidden state of component widgets.

        This is a private method that actually handles the hiding/showing of
        component widgets.

        This method causes the ``hidden_changed`` signal to be emitted with
        the new hidden state.

        Args:
            show_widgets (bool): Whether the component widgets should
                be shown or hidden.

        Returns:
            ``None``
        """
        for widget in self.widgets[2:]:
            if widget is None:
                continue
            widget.setHidden(not bool(show_widgets))
        self.hidden_changed.emit(bool(show_widgets))

    @QtCore.Slot(int)
    def set_hidden(self, hidden):
        """Set the hidden state of component widgets.

        Args:
            hidden (bool): The new hidden state.  ``False`` indicates that
                component widgets should be visible.  ``True`` indicates that
                component widgets should be hidden.

        Raises:
            ValueError: When the GriddedInput has not been initialized with
                ``hideable=True``.

        Returns:
            ``None``
        """
        if not self.hideable:
            raise ValueError('Input is not hideable.')
        self.label_widget.setChecked(not hidden)

    def hidden(self):
        """Whether the input's component widgets are hidden.

        Returns:
            A boolean.  If the input is not hideable, this will always
            return ``False``.
        """
        if self.hideable:
            return not self.label_widget.isChecked()
        return False


class Text(GriddedInput):
    """A GriddedInput for handling single-line, text-based input."""

    class TextField(QtWidgets.QLineEdit):
        """A custom QLineEdit widget with tweaks for use by Text instances."""

        def __init__(self, starting_value=''):
            """Initialize the TextField instance.

            This textfield may accept ``DragEnterEvent``s and ``DropEvent``s,
            but will only do so if the event has text MIME data.

            Args:
                starting_value='' (string): The starting value of the
                    QLineEdit.

            Returns:
                ``None``
            """
            QtWidgets.QLineEdit.__init__(self, starting_value)
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event):
            """Handle a Qt QDragEnterEvent.

            Overridden from QtWidget.dragEnterEvent.  Will only accept the
            event if the event's mime data has text, but does not have URLs.

            Args:
                event (QDragEnterEvent): The QDragEnterEvent to analyze.

            Returns:
                ``None``
            """
            if event.mimeData().hasText() and not event.mimeData().hasUrls():
                LOGGER.info('Accepting drag enter event for "%s"',
                            event.mimeData().text())
                event.accept()
            else:
                LOGGER.info('Rejecting drag enter event for "%s"',
                            event.mimeData().text())
                event.ignore()

        def dropEvent(self, event):
            """Handle a Qt QDropEvent.

            Reimplemented from QtWidget.dropEvent.  Will always accept the
            event.  Any text in the MIME data of the ``event`` provided will
            be set as the text of this textfield.

            Args:
                event (QDropEvent): The QDropEvent to analyze.

            Returns:
                ``None``
            """
            text = event.mimeData().text()
            LOGGER.info('Accepting and inserting dropped text: "%s"', text)
            event.accept()
            self.setText(text)

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        """Initialize a Text input.

        Args:
            label (string): The string label to use for the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.
            hideable=False (bool): If ``True``, the input will have a
                checkbox that, when triggered, will show/hide the other
                component widgets in this Input.
            validator=None (callable): The validator callable to use for
                validation.  This callable must adhere to the InVEST
                Validation API.

        Returns:
            ``None``
        """
        GriddedInput.__init__(self, label=label, helptext=helptext,
                              interactive=interactive,
                              args_key=args_key, hideable=hideable,
                              validator=validator)
        self.textfield = Text.TextField()
        self.textfield.textChanged.connect(self._text_changed)
        self.widgets[2] = self.textfield

    def _text_changed(self, new_text):
        """A slot to emit the ``value_changed`` signal.

        NOTE: Validation is no longer triggered by this signal handler,
        everything is handled at the model level.

        Args:
            new_text (string): The new string value of the textfield.

        Returns:
            ``None``
        """
        self.dirty = True
        self.value_changed.emit(new_text)

    def value(self):
        """Fetch the value of the textfield.

        Returns:
            The string value of the textfield.
        """
        return self.textfield.text()

    def set_value(self, value):
        """Set the value of the textfield.

        If this Text instance is hideable and ``value`` is not an empty
        string, the input will be shown.  A hideable input shown in this way
        may be hidden again by calling ``self.set_hidden(True)``.  Note that
        the value of the input will be preserved.

        Args:
            value (string, int, or float): The value to use for the new value
                of the textfield.

        Returns:
            ``None``
        """
        try:
            if isinstance(value, (int, float)):
                value = str(value)

            # If it isn't a unicode string, attempt to detect the source
            # encoding.
            if not isinstance(value, str):
                most_likely_encoding = chardet.detect(value)['encoding']
                if most_likely_encoding is None:
                    # When string is empty, assume UTF-8
                    most_likely_encoding = 'UTF-8'

                LOGGER.info('Guessing that string "%s" is encoded as %s ',
                            value, most_likely_encoding)
                encoded_value = value.decode(
                    most_likely_encoding).encode('utf-8')
            else:
                # value is already unicode, should be UTF-8 or ASCII.
                encoded_value = value
        except (UnicodeEncodeError, UnicodeDecodeError):
            # If we can't encode or decode, there's a serious problem.  Log to
            # the console and allow the user to provide their own input
            # directly to the text element.
            LOGGER.exception('Could determine encoding; using value as-is.')
            encoded_value = value

        if len(encoded_value) > 0 and self.hideable:
            self.set_hidden(False)

        self.textfield.setText(encoded_value)

    def clear(self):
        """Reset the input to a 'blank' state.

        Returns:
            None.
        """
        self.textfield.clear()
        GriddedInput.clear(self)


class _Path(Text):
    """Shared code for filepath-based UI inputs."""

    class FileField(QtWidgets.QLineEdit):
        """A class for handling file-related text input and events.

        This file field may accept ``DragEnterEvent``s and ``DropEvent``s,
        but will only do so if the event has exactly 1 URL in its MIME data.
        """

        def __init__(self, starting_value=''):
            """Initialize the FileField instance.

            Args:
                starting_value='' (string): The starting value of the
                    QLineEdit.

            Returns:
                ``None``
            """
            QtWidgets.QLineEdit.__init__(self, starting_value)
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event=None):
            """Handle a Qt QDragEnterEvent.

            Overridden from QtWidget.dragEnterEvent.  Will only accept the
            event if the event's mime data has exactly 1 URL in its MIME data.

            Args:
                event (QDragEnterEvent): The QDragEnterEvent to analyze.

            Returns:
                ``None``
            """
            # If the user tries to drag multiple files into this text field,
            # reject the event!
            if (event.mimeData().hasUrls() and
                    len(event.mimeData().urls()) == 1):
                LOGGER.info('Accepting drag enter event for "%s"',
                            event.mimeData().text())
                event.accept()
            else:
                LOGGER.info('Rejecting drag enter event for "%s"',
                            event.mimeData().text())
                event.ignore()

        def dropEvent(self, event=None):
            """Handle a Qt QDropEvent.

            Reimplemented from QtWidget.dropEvent.  Will always accept the
            event.  Any text in the MIME data of the ``event`` provided will
            be set as the text of this textfield, but the text may be modified
            slightly to correct for filesystem-specific issues in the path
            given.

            Args:
                event (QDropEvent): The QDropEvent to analyze.

            Returns:
                ``None``
            """
            path = event.mimeData().urls()[0].path()
            if platform.system() == 'Windows':
                path = path[1:]  # Remove the '/' ahead of disk letter
            elif platform.system() == 'Darwin':
                # On mac, we need to ask the OS nicely for the fileid.
                # This is only needed on Qt<5.4.1.
                # See bug report at https://bugreports.qt.io/browse/QTBUG-40449
                command = (
                    "osascript -e 'get posix path of my posix file \""
                    "file://{fileid}\" -- kthx. bai'").format(
                        fileid=path)
                process = subprocess.Popen(
                    command, shell=True,
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE)
                path = process.communicate()[0].lstrip().rstrip()

            LOGGER.info('Accepting drop event with path: "%s"', path)
            event.accept()
            if isinstance(path, bytes):
                path = path.decode('utf-8')
            self.setText(path)

        @QtCore.Slot(bool)
        def _emit_textchanged(self, triggered):
            """Slot for re-emitting the textchanged signal with current text.

            Args:
                triggered (bool): Ignored.

            Returns:
                ``None``
            """
            self.textChanged.emit(self.text())

        def contextMenuEvent(self, event):
            """Show a custom context menu for the input.

            This context menu adds a "Refresh" option to the default context
            menu. When clicked, this menu action will cause the
            ``textChanged`` signal to be emitted.

            Args:
                event (QEvent): The context menu event.

            Returns:
                ``None``
            """
            menu = self.createStandardContextMenu()
            refresh_action = QtWidgets.QAction('Refresh', menu)
            refresh_action.setIcon(qtawesome.icon('fa5s.sync'))
            refresh_action.triggered.connect(self._emit_textchanged)
            menu.addAction(refresh_action)

            menu.exec_(event.globalPos())

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        """Initialize the _Path instance.

        Args:
            label (string): The string label to use for the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.
            hideable=False (bool): If ``True``, the input will have a
                checkbox that, when triggered, will show/hide the other
                component widgets in this Input.
            validator=None (callable): The validator callable to use for
                validation.  This callable must adhere to the InVEST
                Validation API.

        Returns:
            ``None``
        """
        Text.__init__(self, label, helptext, interactive, args_key,
                      hideable, validator=validator)
        self.textfield = _Path.FileField()
        self.textfield.textChanged.connect(self._text_changed)

        # None values are filler.  They represent an empty column in this row
        # of inputs in the gridded layout.
        self.widgets = [
            self.valid_button,
            self.label_widget,
            self.textfield,
            None,
            self.help_button,
        ]

    def _handle_file_button_selection(self, value):
        """Handle the case when the user presses 'cancel' in the file dialog.

        Args:
            value (string): The path selected.  This path will be ``''`` if the
                dialog was cancelled.

        Returns:
            ``None``
        """
        if value != '':
            self.textfield.setText(value)


class Folder(_Path):
    """An InVESTModelInput for selecting a folder."""

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        """Initialize the Folder instance.

        Args:
            label (string): The string label to use for the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.
            hideable=False (bool): If ``True``, the input will have a
                checkbox that, when triggered, will show/hide the other
                component widgets in this InVESTModelInput.
            validator=None (callable): The validator callable to use for
                validation.  This callable must adhere to the InVEST
                Validation API.

        Returns:
            ``None``
        """
        _Path.__init__(self, label, helptext, interactive, args_key,
                       hideable, validator=validator)
        self.path_select_button = FolderButton('Select folder')
        self.path_select_button.path_selected.connect(self._handle_file_button_selection)

        # index 3 is the column place right before the help button, after the
        # textfield.
        self.widgets[3] = self.path_select_button

        if self.hideable:
            self._hideability_changed(False)


class File(_Path):
    """An InVESTModelInput for selecting a single file."""

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        """Initialize the File instance.

        Args:
            label (string): The string label to use for the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.
            hideable=False (bool): If ``True``, the input will have a
                checkbox that, when triggered, will show/hide the other
                component widgets in this InVESTModelInput.
            validator=None (callable): The validator callable to use for
                validation.  This callable must adhere to the InVEST
                Validation API.

        Returns:
            ``None``
        """
        _Path.__init__(self, label, helptext, interactive, args_key,
                       hideable, validator=validator)
        self.path_select_button = FileButton('Select file')
        self.path_select_button.path_selected.connect(
            self._handle_file_button_selection)

        # Index 3 is the column to the right of the textfield, to the left of
        # the help button.
        self.widgets[3] = self.path_select_button

        if self.hideable:
            self._hideability_changed(False)


class SaveFile(_Path):
    """An InVESTModelInput for selecting a file to save to."""

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None,
                 default_savefile='new_file.txt'):
        """Initialize the SaveFile instance.

        Args:
            label (string): The string label to use for the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.
            hideable=False (bool): If ``True``, the input will have a
                checkbox that, when triggered, will show/hide the other
                component widgets in this InVESTModelInput.
            validator=None (callable): The validator callable to use for
                validation.  This callable must adhere to the InVEST
                Validation API.

        Returns:
            ``None``
        """
        _Path.__init__(self, label, helptext, interactive, args_key,
                       hideable, validator=validator)
        self.path_select_button = SaveFileButton('Select file',
                                                 default_savefile)
        self.path_select_button.path_selected.connect(
            self._handle_file_button_selection)
        self.widgets[3] = self.path_select_button

        if self.hideable:
            self._hideability_changed(False)


class Checkbox(GriddedInput):
    """An InVESTModelInput for boolean user input."""

    # Re-setting value_changed to adapt to the type requirement.
    value_changed = QtCore.Signal(bool)
    # Re-setting interactivity_changed to avoid a segfault while testing on
    # linux via `python setup.py nosetests`.
    interactivity_changed = QtCore.Signal(bool)

    def __init__(self, label, helptext=None, interactive=True, args_key=None):
        """Initialize the Checkbox instance.

        Args:
            label (string): The string label to use for the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.

        Returns:
            ``None``
        """
        GriddedInput.__init__(self, label=label, helptext=helptext,
                              interactive=interactive, args_key=args_key,
                              hideable=False, validator=None)

        self.checkbox = QtWidgets.QCheckBox(label)
        self.checkbox.stateChanged.connect(self.value_changed.emit)
        self.widgets[0] = None  # No need for a valid button
        self.widgets[1] = self.checkbox  # replace label with checkbox
        self.satisfied = True

    def clear(self):
        """Clear the checkbox's input by setting to unchecked.

        Returns:
            None.
        """
        self.set_value(False)
        GriddedInput.clear(self)

    def value(self):
        """Get the value of the checkbox.

        Returns:
            A boolean, whether the checkbox is checked.
        """
        return self.checkbox.isChecked()

    def valid(self):
        """Check whether the checkbox's input is valid.

        Note:
            Unlike other inputs, the checkbox's input is always valid.

        Returns:
            Always returns ``True``.
        """
        return True

    def set_value(self, value):
        """Set the value of the checkbox.

        Args:
            value (bool): The new check state of the checkbox. If ``True``,
                the checkbox will be checked.
        """
        self.checkbox.setChecked(value)


class Dropdown(GriddedInput):
    """An InVESTModelInput for selecting one out of a set of defined options."""

    def __init__(self, label, helptext=None, interactive=True, args_key=None,
                 hideable=False, options=(), return_value_map=None):
        """Initialize a Dropdown instance.

        Like the Checkbox class, a Dropdown is always valid.

        Args:
            label (string): The string label to use for the input.
            helptext=None (string): The helptext string used to display more
                information about the input.  If ``None``, no extra information
                will be displayed.
            interactive=True (bool): Whether the user can interact with the
                component widgets of this input.
            args_key=None (string):  The args key of this input.  If ``None``,
                the input will not have an args key.
            hideable=False (bool): If ``True``, the input will have a
                checkbox that, when triggered, will show/hide the other
                component widgets in this InVESTModelInput.
            options=() (iterable): An iterable of options for this Dropdown.
                Options will be added in the order they exist in the iterable.
            return_value_map=None (dict or None): If a dict, keys must exactly
                match the values of ``options``.  Values will be returned when
                the user selects the option indicated by the key.  If ``None``,
                the option selected by the user will be returned verbatim.

        Returns:
            ``None``
        """
        GriddedInput.__init__(self, label=label, helptext=helptext,
                              interactive=interactive, args_key=args_key,
                              hideable=hideable, validator=None)
        self.dropdown = QtWidgets.QComboBox()
        self.widgets[2] = self.dropdown
        self.set_options(options, return_value_map)
        self.dropdown.currentIndexChanged.connect(self._index_changed)
        self.satisfied = True
        self._valid = True  # Dropdown is always valid!
        self._return_value_map = return_value_map

        # Init hideability if needed
        if self.hideable:
            self._hideability_changed(False)

    def clear(self):
        """Reset the dropdown to a 'blank' state.

        If the dropdown has options set, the menu will be reset to the item at
        index 0.  If there are no options, validity and sufficiency is reset
        only.

        Returns:
            None.
        """
        try:
            self.set_value(self.options[0])
        except IndexError:
            # When there are no options
            pass
        GriddedInput.clear(self)

    @QtCore.Slot(int)
    def _index_changed(self, newindex):
        """A slot for emitting ``value_changed``.

        ``value_changed`` will be emitted with the text of the new selection.

        Args:
            newindex (int): The index of the new selection.

        Returns:
            ``None``
        """
        # QComboBoxes are 1-indexed.  An index of -1 means there are no options
        # in the dropdown.
        if newindex >= 0:
            value = self.options[newindex]
        else:
            value = 'None'
        self.value_changed.emit(value)

    def set_options(self, options, return_value_map=None):
        """Set the available options for this dropdown.

        Args:
            options (iterable): The new options for the dropdown.
            return_value_map=None (dict or None): If a dict, keys must exactly
                match the values of ``options``.  Values will be returned when
                the user selects the option indicated by the key.  If ``None``,
                the option selected by the user will be returned verbatim.

        Returns:
            ``None``
        """
        if (return_value_map is not None and
                len(set(options) ^ set(return_value_map.keys())) > 0):
            raise ValueError('Options must exactly match keys in '
                             'return_value_map')

        def _cast_value(value):
            if isinstance(value, (int, float)):
                value = str(value)
            # It's already unicode, so can't decode further.
            return value

        # make sure all values in the return value map are text
        if return_value_map is not None:
            return_value_map = dict(
                (_cast_value(key), _cast_value(value)) for (key, value) in
                return_value_map.items())
        self.return_value_map = return_value_map

        self.dropdown.clear()
        cast_options = []
        self.dropdown.blockSignals(True)
        for label in options:
            cast_value = _cast_value(label)
            self.dropdown.addItem(cast_value)
            cast_options.append(cast_value)
        self.dropdown.blockSignals(False)
        self.options = cast_options
        self.user_options = options

    def value(self):
        """Get the text of the currently-selected option.

        Returns:
            A string with the currently selected option.  If options were
            provided that were not strings, the string version of the option
            is returned.
        """
        dropdown_text = self.dropdown.currentText()
        if self.return_value_map is not None:
            return self.return_value_map[dropdown_text]

        return dropdown_text

    def set_value(self, value):
        """Set the current index of the dropdown based on the value.

        Args:
            value: The option to select in the dropdown. This value should
                match either a value in the options iterable set via
                ``Dropdown.set_options`` or the ``options`` parameter to
                ``Dropdown.__init__``, or else must be the string text of the
                option.

        Raises:
            ValueError: When the value provided cannot be found in either the
            user-defined list of options or the list of options that has been
            cast to a string.

        Returns:
            ``None``
        """
        # If we have known mapped values, try to match the value with the
        # return value map we know about.
        inverted_map = None
        if self.return_value_map is not None:
            inverted_map = dict((v, k) for (k, v) in
                                self.return_value_map.items())

        # Handle case where value is of the type provided by the user,
        # and the case where it's been converted to a utf-8 string.
        for options_attr in ('options', 'user_options'):
            try:
                if inverted_map is not None:
                    try:
                        value = inverted_map[value]
                    except KeyError:
                        pass
                index = getattr(self, options_attr).index(value)
                self.dropdown.setCurrentIndex(index)
                return
            except ValueError:
                # ValueError when the value is not in the list
                pass

        raise ValueError(('Value %s not in options %s, user options %s '
                          'or return value map %s') % (
                              value, self.options, self.user_options,
                              self.return_value_map))


class Label(QtWidgets.QLabel):
    """A widget for displaying information in a UI."""

    def __init__(self, text):
        """Initialize the Label.

        Labels may contain links, which will be opened externally if possible.

        Args:
            text (string): The text of the label.

        Returns:
            ``None``
        """
        QtWidgets.QLabel.__init__(self, text)
        self.setWordWrap(True)
        self.setOpenExternalLinks(True)
        self.setTextFormat(QtCore.Qt.RichText)

    def add_to(self, layout):
        """Add this widget to a QGridLayout.

        Args:
            layout (QGridLayout): The layout to which this Label will be
                added.  The Label will span all columns.

        Returns:
            ``None``
        """
        layout.addWidget(self, layout.rowCount(),  # target row
                         0,  # target starting column
                         1,  # row span
                         layout.columnCount())  # span all columns


class Container(QtWidgets.QGroupBox, InVESTModelInput):
    """An InVESTModelInput that contains other inputs within a QGridLayout."""

    # Unlike other subclasses of InVESTModelInput, we need to redefine all of the signals
    # here because we're changing the type of the parameter emitted by
    # value_changed to a bool.
    value_changed = QtCore.Signal(bool)
    interactivity_changed = QtCore.Signal(bool)
    sufficiency_changed = QtCore.Signal(bool)

    def __init__(self, label, interactive=True, expandable=False,
                 expanded=True, args_key=None):
        """Initialize a Container.

        Args:
            label (string): The label of the Container.
            interactive=True (bool): Whether the user can interact with this
                container.
            expandable=False (bool): Whether the Container may be expanded
                and collapsed at will.
            expanded=True (bool): Whether the Container will start out
                expanded or collapsed.  If ``True``, the Container will be
                initialized to be expanded.
            args_key=None (string): The args key for the Container.

        Returns:
            ``None``
        """
        QtWidgets.QGroupBox.__init__(self)
        InVESTModelInput.__init__(self, label=label, interactive=interactive,
                                  args_key=args_key)
        self.widgets = [self]
        self.setCheckable(expandable)
        if expandable:
            self.setChecked(expanded)
        self.setTitle(label)
        self.setLayout(QtWidgets.QGridLayout())
        self.set_interactive(interactive)
        self.toggled.connect(self.value_changed.emit)
        self.toggled.connect(self._hide_widgets)
        self.value_changed.connect(self._check_sufficiency)
        self.interactivity_changed.connect(self._check_sufficiency)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,  # horizontal
            QtWidgets.QSizePolicy.Maximum)  # vertical

    def clear(self):
        """Reset the container to unchecked if it is checkable.

        If the container is not checkable, nothing is done.

        Returns:
            None.
        """
        if self.expandable:
            self.setChecked(False)

    @QtCore.Slot(bool)
    def _hide_widgets(self, check_state):
        """A slot for hiding and showing all widgets in this Container.

        Args:
            check_state (bool): Whether the container's checkbox is checked.
                If the Container has just been collapsed, ``check_state`` will
                be ``False``.

        Returns:
            ``None``.
        """
        for layout_item in (self.layout().itemAtPosition(*coords)
                            for coords in itertools.product(
                                range(1, self.layout().rowCount()),
                                range(1, self.layout().columnCount()))):
            if layout_item and self.isVisible():
                layout_item.widget().setVisible(self.isChecked())

        # Update size based on sizehint now that widgets changed.
        self.setMinimumSize(self.sizeHint())

    def showEvent(self, event):
        """Initialize hidden state of contained widgets when shown.

        Reimplemented from QWidget.showEvent.

        Args:
            event (QEvent): The current QEvent.  Ignored.
        """
        if self.isCheckable():
            self._hide_widgets(self.value())

    @property
    def expanded(self):
        """Whether the Container is expanded.

        Returns:
            A boolean indicating whether the container is expanded.
        """
        if self.expandable:
            return self.isChecked()
        return True

    @expanded.setter
    def expanded(self, value):
        """Set the container's expanded state.

        Args:
            value (bool): The new expanded state.  ``True`` indicates that
                the container will be expanded.

        Raises:
            ValueError: When the container was not initialized to be
            expandable.
        """
        if not self.expandable:
            raise ValueError('Container cannot be expanded when not '
                             'expandable')
        self.setChecked(value)

    @property
    def expandable(self):
        """Whether the container is expandable.

        Returns:
            A boolean indicating whether the container is expandable.
        """
        return self.isCheckable()

    @expandable.setter
    def expandable(self, value):
        """Set whether the container is expandable.

        Returns:
            ``None``
        """
        self.setCheckable(value)

    def add_input(self, input_obj):
        """Add an input to the Container.

        The input must have an ``add_to`` method that handles how to add the
        InVESTModelInput and/or its component widgets to a QGridLayout that is
        owned by the Container.

        Args:
            input_obj (InVESTModelInput): An instance of Input to add to the
                Container's layout.

        Returns:
            ``None``
        """
        input_obj.add_to(layout=self.layout())
        _apply_sizehint(self.layout().parent())

        if self.expandable:
            input_obj.set_visible(self.expanded)
            input_obj.set_interactive(self.expanded)

            if self.isVisible():
                for widget in input_obj.widgets:
                    if not widget:
                        continue
                    widget.setVisible(self.expanded)
        self.sufficiency_changed.connect(input_obj.set_interactive)
        self.sufficiency_changed.connect(input_obj.set_visible)

    def add_to(self, layout):
        """Define how to add this Container to a QGridLayout.

        The container will occupy all columns.

        Args:
            layout (QGridLayout): A QGridLayout to which this Container will
                be added.

        Returns:
            ``None``
        """
        layout.addWidget(self,
                         layout.rowCount(),  # target row
                         0,  # target starting column
                         1,  # row span
                         layout.columnCount())  # span all columns

    def value(self):
        """Fetch the value of this container.

        The value is the same as ``self.expanded``.

        Returns:
            A boolean, whether the Container is expanded.
        """
        return self.expanded

    def set_value(self, value):
        """Set the value of the Container.

        This is the same as setting the value of ``self.expanded``.

        Args:
            value (bool): The new expanded state of the Container.
        """
        self.expanded = value


class FormScrollArea(QtWidgets.QScrollArea):
    """Object to contain scrollarea-related functionality."""

    def __init__(self):
        """Initialize the ScrollArea."""
        QtWidgets.QScrollArea.__init__(self)
        self.setWidgetResizable(True)
        self.verticalScrollBar().rangeChanged.connect(
            self.update_scroll_border)
        self.update_scroll_border(
            self.verticalScrollBar().minimum(),
            self.verticalScrollBar().maximum())

    def update_scroll_border(self, range_min, range_max):
        """Show or hide the border of the scrolling area as needed.

        Args:
            range_min (int): The scroll area's range minimum.
            range_max (int): The scroll area's range maximum.

        Returns:
            ``None``
        """
        if range_min == 0 and range_max == 0:
            self.setStyleSheet("QScrollArea { border: None } ")
        else:
            self.setStyleSheet("")


class Form(QtWidgets.QWidget):
    """A form that contains multiple InVESTModelInputs."""

    submitted = QtCore.Signal()
    run_finished = QtCore.Signal()

    def __init__(self, parent=None):
        """Initialize the Form.

        Returns:
            ``None``
        """
        QtWidgets.QWidget.__init__(self, parent=parent)

        # self._thread is redefined as an Executor when we run the target
        # callable.
        self._thread = None

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.inputs = Container(label='')
        self.inputs.setFlat(True)

        # Have the inputs take up as much space as needed
        self.inputs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum)

        # Make the inputs container scrollable.
        self.scroll_area = FormScrollArea()
        self.layout().addWidget(self.scroll_area)
        self.scroll_area.setWidget(self.inputs)

        # set the sizehint of the inputs again ... needed after setting
        # scroll_area.
        if self.inputs.sizeHint().isValid():
            self.inputs.setMinimumSize(self.inputs.sizeHint())
        self.layout().setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.inputs.layout().setSizeConstraint(
            QtWidgets.QLayout.SetMinimumSize)

        self.buttonbox = QtWidgets.QDialogButtonBox()
        self.run_button = QtWidgets.QPushButton(' Run')
        self.run_button.setIcon(QtGui.QIcon(ICON_ENTER))

        self.buttonbox.addButton(
            self.run_button, QtWidgets.QDialogButtonBox.AcceptRole)
        self.layout().addWidget(self.buttonbox)
        self.run_button.pressed.connect(self._emit_submitted)

        self.run_dialog = FileSystemRunDialog()

    @QtCore.Slot()
    def _emit_submitted(self):
        """Emit the submitted signal."""
        # PyQt4 won't recognize self.submitted.emit as a bound slot, so
        # creating a bound method of Form to handle this.  Useful for MESH
        # demo.
        self.submitted.emit()

    def run(self, target, args=(), kwargs=None, window_title='',
            out_folder='/'):
        """Run a function within the run dialog.

        This method creates and starts a new execution.Executor thread
        instance for the execution of the target.

        Args:
            target (callable): A function to execute.
            args=() (iterable): Positional arguments to pass to the target.
            kwargs=None (dict): Keyword args to pass to the target.
            window_title (string): The title of the run dialog window.
            out_folder='/' (string): The folder on disk that the run dialog's
                "Open Workspace" button should open when pressed.

        Returns:
            ``None``
        """
        if not hasattr(target, '__call__'):
            raise ValueError('Target %s must be callable' % target)

        # Don't need to log an exception or completion in the Executor thread
        # in the case of a model run; those messages are handled by
        # utils.prepare_workspace.
        self._thread = execution.Executor(target,
                                          args,
                                          kwargs,
                                          log_events=False)
        self._thread.finished.connect(self._run_finished)

        self.run_dialog.show()
        self.run_dialog.start(window_title=window_title,
                              out_folder=out_folder)
        self._thread.start()

    @QtCore.Slot()
    def _run_finished(self):
        """A slot that is called when the executor thread finishes.

        Returns:
            ``None``
        """
        self.run_dialog.finish(
            exception=self._thread.exception)
        self.run_finished.emit()

    def add_input(self, input):
        """Add an input to the Form.

        Args:
            input (InVESTModelInput): The Input instance to add to the Form.

        Returns:
            ``None``
        """
        self.inputs.add_input(input)
