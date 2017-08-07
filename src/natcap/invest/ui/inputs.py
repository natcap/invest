# coding=UTF-8
"""Input classes for the InVEST UI, a Qt-based UI abstraction layer."""

from __future__ import absolute_import

import functools
import os
import threading
import logging
import platform
import subprocess
import warnings
import sys
import atexit
import itertools

import sip
sip.setapi('QString', 2)
import qtpy
from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui
import six
import qtawesome

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

LOGGER = logging.getLogger(__name__)
ICON_FOLDER = qtawesome.icon('fa.folder-o')
ICON_FILE = qtawesome.icon('fa.file-o')
ICON_ENTER = qtawesome.icon('fa.arrow-circle-o-right',
                            color='green')
_QLABEL_STYLE_TEMPLATE = ('QLabel {{padding={padding};'
                          'background-color={bg_color};'
                          'border={border};}}')
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
    # Adding this allows tests to run on linux via `python setup.py nosetests`
    # and `python setup.py test` without segfault.
    global QT_APP
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

    Parameters:
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

    Parameters:
        window_ptr (QtWidgets.QWidget): a reference to a Qt window.
    """
    geometry = window_ptr.frameGeometry()
    center = QtWidgets.QDesktopWidget().availableGeometry().center()
    geometry.moveCenter(center)
    window_ptr.move(geometry.topLeft())


class Validator(QtCore.QObject):
    """A class to manage alidating in a separate Qt thread."""

    started = QtCore.Signal()
    finished = QtCore.Signal(list)

    def __init__(self, parent):
        """Initialize the Validator instance.

        Parameters:
            parent (QtWidgets.QWidget): The parent qwidget.  This will be the
                parent of the validation thread.
        """
        QtCore.QObject.__init__(self, parent)
        self._validation_thread = QtCore.QThread(parent=self)
        self._validation_worker = None

    def in_progress(self):
        """Whether the validation thread is running.

        Returns:
            is_running (bool): Whether the validation thread is running.
        """
        return self._validation_thread.isRunning()

    def validate(self, target, args, limit_to=None):
        """Validate the provided args with the provided target.

        Parameters:
            target (callable): The validation callable.  Must adhere to the
                InVEST validation API.
            args (dict): The arguments dictionary to validate.
            limit_to=None (string or None): Optional. If provided, this is the
                validation key that should be validated.  All other keys will
                be excluded.  Part of the InVEST Validation API.

        Returns:
            ``None``
        """
        for i in xrange(10):
            if self._validation_thread.isRunning():
                break
            self._validation_thread.start()
            QtCore.QThread.currentThread().msleep(5)

        self.started.emit()
        self._validation_worker = ValidationWorker(
            target=target,
            args=args,
            limit_to=limit_to)
        self._validation_worker.moveToThread(self._validation_thread)

        def _finished():
            """Slot to process warnings and emit ``self.finished``."""
            LOGGER.info('Finished validation for args_key %s', limit_to)
            warnings_ = [w for w in self._validation_worker.warnings]

            LOGGER.debug(warnings_)
            self.finished.emit(warnings_)

        # Order matters with these callbacks.
        self._validation_worker.finished.connect(self._validation_thread.quit)
        self._validation_worker.finished.connect(_finished)
        self._validation_worker.finished.connect(
            self._validation_worker.deleteLater)
        QtCore.QThread.currentThread().msleep(25)  # avoids segfault
        self._validation_worker.start()


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

        Parameters:
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

        Parameters:
            stream_widget (QtWidgets.QWidget): A QWidget that supports the
                python streams API.
        """
        logging.StreamHandler.__init__(self, stream=stream_widget)
        self._stream = stream_widget
        self.setLevel(logging.NOTSET)  # capture everything

        self.formatter = logging.Formatter(
            fmt='%(asctime)s %(name)-18s %(levelname)-8s %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S ')
        self.setFormatter(self.formatter)


class LogMessagePane(QtWidgets.QPlainTextEdit):
    """A subclass of ``QtWidgets.QPlainTextEdit`` to support write().

    Uses the signals/slots framework to support writing text to the
    QPlainTextEdit from different threads.
    """

    message_received = QtCore.Signal(six.text_type)

    def __init__(self):
        """Initialize the LogMessagePane instance.

        Sets the stylesheet for the QPlainTextEdit, and sets it to read-only.
        """
        QtWidgets.QPlainTextEdit.__init__(self)

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

        Parameters:
            message (string): The message to be written to the message pane.

        Returns:
            ``None``
        """
        self.message_received.emit(message)

    def _write(self, message):
        """Write the message provided to the message pane.

        Calling this method from a thread other than the main thread will
        cause an error from within Qt.

        Parameters:
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
        self.statusAreaLabel = QtWidgets.QLabel('Messages:')
        self.log_messages_pane = LogMessagePane()
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

        self.openWorkspaceCB = QtWidgets.QCheckBox('Open workspace after success')
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
        self.backButton.clicked.connect(self.close_window)

        # add the buttonBox to the window.
        self.layout().addWidget(self.buttonBox)

        # Indicate that this window should be styled like a dialog.
        self.setWindowFlags(QtCore.Qt.Dialog)

    def __del__(self):
        """Delete/deregister required objects."""
        self.logger.removeHandler(self.loghandler)
        self.deleteLater()

    def start(self, window_title, out_folder):
        """Set the state of the dialog to indicate processing has started."""
        if not window_title:
            window_title = "Running ..."
        self.setWindowTitle(window_title)
        self.out_folder = out_folder

        self.is_executing = True
        self.log_messages_pane.clear()
        self.progressBar.setMaximum(0)  # start the progressbar.
        self.backButton.setDisabled(True)

        self.log_messages_pane.write('Initializing...\n')

    # TODO: consolidate exception_found, thread_exception if possible.
    def finish(self, exception_found, thread_exception=None):
        """Notify the user that model processing has finished.

        Parameters:
            exception_found (bool): Whether an exception was found while
                processing.
            thread_exception=None (Exception or None): The exception object
                If the error encountered.  None if no error found.

        Returns:
            ``None``
        """
        self.is_executing = False
        self.progressBar.setMaximum(1)  # stops the progressbar.
        self.backButton.setDisabled(False)

        if exception_found:
            self.messageArea.set_error(True)
            self.messageArea.setText(
                (u'<b>%s</b> encountered: <em>%s</em> <br/>'
                 'See the log for details.') % (
                    thread_exception.__class__.__name__,
                    thread_exception))
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

    def _request_workspace(self, event=None):
        open_workspace(self.out_folder)

    def close_window(self):
        """Close the window and reset the state of the process window.

        Returns:
            ``None``
        """
        self.openWorkspaceCB.setVisible(True)
        self.openWorkspaceButton.setVisible(False)
        self.messageArea.clear()
        self.cancel = False
        self.done(0)

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
            QtWidgets.QDialog.closeEvent(self, event)


class InfoButton(QtWidgets.QPushButton):
    """An informational button that shows helpful text when clicked."""

    def __init__(self, default_message=None):
        """Initialize an instance of InfoButton.

        Parameters:
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

    def _show_popup(self, clicked=None):
        QtWidgets.QWhatsThis.enterWhatsThisMode()
        QtWidgets.QWhatsThis.showText(self.mapToGlobal(self.pos()),
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

    def set_errors(self, errors):
        """Set the error message and style based on the provided errors.

        Parameters:
            errors (list): A list of strings.  If this list is empty, the
                style of the button is set to green, indicating validation
                success.  If this list is not empty, the strings in this list
                will be formatted and set as the error text, and the button
                style will be set to a red.

        Returns:
            ``None``
        """
        if errors:
            self.setIcon(qtawesome.icon('fa.times',
                                        color='red'))
            error_string = '<br/>'.join(errors)
            self.successful = False
        else:
            self.setIcon(qtawesome.icon('fa.check',
                                        color='green'))
            error_string = 'Validation successful'
            self.successful = True

        self.setWhatsThis(error_string)
        self.setToolTip(error_string)


class HelpButton(InfoButton):
    """An InfoButton with an informational help icon."""

    def __init__(self, default_message=None):
        """Initialize the HelpButton.

        Parameters:
            default_message=None (string): The default message of this button.
                See InfoButton.__init__ for more information.

        Returns:
            ``None``
        """
        InfoButton.__init__(self, default_message)
        self.setIcon(qtawesome.icon('fa.info-circle',
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

        Parameters:
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
            finished (bool): Whether validation has finished."""
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
        LOGGER.info(('Starting validation thread with target=%s, args=%s, '
                     'limit_to=%s'), self.target, self.args, self.limit_to)
        try:
            self.warnings = self.target(self.args, limit_to=self.limit_to)
            LOGGER.info('Validation thread returned warnings: %s',
                        self.warnings)
        except Exception as error:
            self.error = str(error)
            LOGGER.exception('Validation: Error when validating %s:',
                             self.target)
        self._finished = True
        self.finished.emit()


class FileDialog(object):
    def __init__(self):
        object.__init__(self)
        self.file_dialog = QtWidgets.QFileDialog()

    def __del__(self):
        self.file_dialog.deleteLater()

    def save_file(self, title, start_dir=None, savefile=None):
        if not start_dir:
            start_dir = os.path.expanduser(
                INVEST_SETTINGS.value('last_dir', DEFAULT_LASTDIR, unicode))

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
                                 os.path.dirname(six.text_type(filename)))
        return filename

    def open_file(self, title, start_dir=None, filters=()):
        if not start_dir:
            start_dir = os.path.expanduser(
                INVEST_SETTINGS.value('last_dir', DEFAULT_LASTDIR, unicode))

        # Allow us to open folders with spaces in them.
        os.path.normpath(start_dir)

        if filters:
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
                                 os.path.dirname(six.text_type(filename)))
        return filename

    def open_folder(self, title, start_dir=None):
        if not start_dir:
            start_dir = os.path.expanduser(
                INVEST_SETTINGS.value('last_dir', DEFAULT_LASTDIR, unicode))
        dialog_title = 'Select folder: ' + title

        dirname = self.file_dialog.getExistingDirectory(self.file_dialog, dialog_title,
                                                        start_dir)
        dirname = six.text_type(dirname)
        INVEST_SETTINGS.setValue('last_dir', dirname)
        return dirname


class _FileSystemButton(QtWidgets.QPushButton):

    path_selected = QtCore.Signal(six.text_type)

    def __init__(self, dialog_title):
        QtWidgets.QPushButton.__init__(self)
        if not hasattr(self, '_icon'):
            self._icon = ICON_FOLDER
        self.setIcon(self._icon)
        self.dialog_title = dialog_title
        self.dialog = FileDialog()
        self.open_method = None  # This should be overridden
        self.clicked.connect(self._get_path)
        self.save_dialog_kwargs = {
            'title': self.dialog_title,
            'start_dir': INVEST_SETTINGS.value('last_dir',
                                               DEFAULT_LASTDIR,
                                               unicode),
        }

    def _get_path(self):
        selected_path = self.open_method(**self.save_dialog_kwargs)
        self.path_selected.emit(selected_path)

    def set_save_dialog_options(self, **kwargs):
        self.save_dialog_kwargs = kwargs


class FileButton(_FileSystemButton):
    def __init__(self, dialog_title):
        self._icon = ICON_FILE
        _FileSystemButton.__init__(self, dialog_title)
        self.open_method = self.dialog.open_file


class SaveFileButton(_FileSystemButton):
    def __init__(self, dialog_title, default_savefile):
        self._icon = ICON_FILE
        _FileSystemButton.__init__(self, dialog_title)
        self.open_method = functools.partial(
            self.dialog.save_file,
            savefile=default_savefile)


class FolderButton(_FileSystemButton):
    def __init__(self, dialog_title):
        _FileSystemButton.__init__(self, dialog_title)
        self.open_method = self.dialog.open_folder


class Input(QtCore.QObject):

    value_changed = QtCore.Signal(six.text_type)
    interactivity_changed = QtCore.Signal(bool)
    sufficiency_changed = QtCore.Signal(bool)

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None):
        QtCore.QObject.__init__(self)
        self.label = label
        self.widgets = []
        self.dirty = False
        self.interactive = interactive
        self.args_key = args_key
        self.helptext = helptext
        self.lock = threading.Lock()
        self.sufficient = False
        self._visible_hint = True

        self.value_changed.connect(self._check_sufficiency)
        self.interactivity_changed.connect(self._check_sufficiency)

    def _check_sufficiency(self, event=None):
        new_sufficiency = bool(self.value()) and self.interactive

        LOGGER.debug('Sufficiency for %s %s --> %s', self,
                     self.sufficient, new_sufficiency)

        if self.sufficient != new_sufficiency:
            self.sufficient = new_sufficiency
            self.sufficiency_changed.emit(new_sufficiency)

    def visible(self):
        return self._visible_hint

    def set_visible(self, visible_hint):
        # Qt visibility is actually controlled by containers and the parent
        # window.
        # We use self._visible_hint to indicate whether the widgets should
        # be considered by natcap.ui as being visible.
        self._visible_hint = visible_hint
        if any(widget.parent().isVisible() for widget in self.widgets
               if widget and widget.parent()):
            for widget in self.widgets:
                if not widget:
                    continue
                widget.setVisible(self._visible_hint)

    def value(self):
        raise NotImplementedError

    def set_value(self):
        raise NotImplementedError

    def set_noninteractive(self, noninteractive):
        self.set_interactive(not noninteractive)

    def set_interactive(self, enabled):
        self.interactive = enabled
        for widget in self.widgets:
            if not widget:  # widgets to be skipped are None
                continue
            widget.setEnabled(enabled)
        self.interactivity_changed.emit(self.interactive)

    def _add_to(self, layout):
        self.setParent(layout.parent().window())  # all widgets belong to Form
        current_row = layout.rowCount()
        for widget_index, widget in enumerate(self.widgets):
            if not widget:
                continue

            # set the default interactivity based on self.interactive
            widget.setEnabled(self.interactive)

            _apply_sizehint(widget)
            layout.addWidget(
                widget,  # widget
                current_row,  # row
                widget_index)  # column


class GriddedInput(Input):

    hidden_changed = QtCore.Signal(bool)
    validity_changed = QtCore.Signal(bool)

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        Input.__init__(self, label=label, helptext=helptext,
                       interactive=interactive, args_key=args_key)

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

        self.lock = threading.Lock()

        # initialize visibility, as we've changed the input's widgets
        self.set_visible(self._visible_hint)

    def _validate(self):
        self.lock.acquire()

        try:
            if self.validator_ref:
                LOGGER.info('Validation: validator taken from self.validator_ref: %s',
                            self.validator_ref)
                validator_ref = self.validator_ref
            else:
                if not self.args_key:
                    LOGGER.info(('Validation: No validator and no args_id defined; '
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
                # self.parent() is only set when the Input is added to a layout.
                args = {self.args_key: self.value()}

            LOGGER.info(
                ('Starting validation thread for %s with target:%s, args:%s, '
                 'limit_to:%s'),
                self, validator_ref, args, self.args_key)

            self._validator.validate(
                target=validator_ref,
                args=args,
                limit_to=self.args_key)
        except Exception:
            LOGGER.exception('Error found when validating %s, releasing lock.',
                             self)
            self.lock.release()
            raise

    def _validation_finished(self, validation_warnings):
        new_validity = not bool(validation_warnings)
        if self.args_key:
            appliccable_warnings = [w[1] for w in validation_warnings
                                    if self.args_key in w[0]]
        else:
            appliccable_warnings = [w[1] for w in validation_warnings]

        LOGGER.info('Cleaning up validation for %s.  Warnings: %s.  Valid: %s',
                    self, appliccable_warnings, new_validity)
        if appliccable_warnings:
            self.valid_button.set_errors(appliccable_warnings)
            tooltip_errors = '<br/>'.join(appliccable_warnings)
        else:
            self.valid_button.set_errors([])
            tooltip_errors = ''

        for widget in self.widgets[0:2]:  # skip file selection, help buttons
            if widget:
                widget.setToolTip(tooltip_errors)

        current_validity = self._valid
        self._valid = new_validity
        self.lock.release()
        if current_validity != new_validity:
            self.validity_changed.emit(new_validity)

    def valid(self):
        # TODO: wait until the lock is released.
        while self._validator.in_progress():
            QtCore.QThread.msleep(50)
        return self._valid

    @QtCore.Slot(int)
    def _hideability_changed(self, show_widgets):
        for widget in self.widgets[2:]:
            if not widget:
                continue
            widget.setHidden(not bool(show_widgets))
        self.hidden_changed.emit(bool(show_widgets))

    @QtCore.Slot(int)
    def set_hidden(self, hidden):
        if not self.hideable:
            raise ValueError('Input is not hideable.')
        self.label_widget.setChecked(not hidden)

    def hidden(self):
        if self.hideable:
            return not self.label_widget.isChecked()
        return False


class Text(GriddedInput):
    class TextField(QtWidgets.QLineEdit):
        def __init__(self, starting_value=''):
            QtWidgets.QLineEdit.__init__(self, starting_value)
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event=None):
            if event.mimeData().hasText() and not event.mimeData().hasUrls():
                LOGGER.info('Accepting drag enter event for "%s"',
                            event.mimeData().text())
                event.accept()
            else:
                LOGGER.info('Rejecting drag enter event for "%s"',
                            event.mimeData().text())
                event.ignore()

        def dropEvent(self, event=None):
            """Overriding the default Qt DropEvent function when a file is
            dragged and dropped onto this qlineedit."""
            text = event.mimeData().text()
            LOGGER.info('Accepting and inserting dropped text: "%s"', text)
            event.accept()
            self.setText(text)

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        GriddedInput.__init__(self, label=label, helptext=helptext,
                              interactive=interactive,
                              args_key=args_key, hideable=hideable,
                              validator=validator)
        self.textfield = Text.TextField()
        self.textfield.textChanged.connect(self._text_changed)
        self.widgets[2] = self.textfield

    @QtCore.Slot(unicode)
    def _text_changed(self, new_text):
        self.dirty = True
        self.value_changed.emit(new_text)
        self._validate()

    def value(self):
        return self.textfield.text()

    def set_value(self, value):
        if value and self.hideable:
            self.set_hidden(False)

        if isinstance(value, int) or isinstance(value, float):
            value = str(value)

        if isinstance(value, str):
            # convert a bytestring into a UTF-8 string.
            value = value.decode('utf-8')
        self.textfield.setText(value)


class _Path(Text):
    class FileField(QtWidgets.QLineEdit):
        def __init__(self, starting_value=''):
            QtWidgets.QLineEdit.__init__(self, starting_value)
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event=None):
            """Overriding the default dragEnterEvent function for when a file is
            dragged and dropped onto this qlineedit.  This reimplementation is
            necessary for the dropEvent function to work on Windows."""
            # If the user tries to drag multiple files into this text field,
            # reject the event!
            if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
                LOGGER.info('Accepting drag enter event for "%s"',
                            event.mimeData().text())
                event.accept()
            else:
                LOGGER.info('Rejecting drag enter event for "%s"',
                            event.mimeData().text())
                event.ignore()

        def dropEvent(self, event=None):
            """Overriding the default Qt DropEvent function when a file is
            dragged and dropped onto this qlineedit."""

            path = event.mimeData().urls()[0].path()
            if platform.system() == 'Windows':
                path = path[1:]  # Remove the '/' ahead of disk letter
            elif platform.system() == 'Darwin':
                # On mac, we need to ask the OS nicely for the fileid.
                # This is only needed on Qt<5.4.1.
                # See bug report at https://bugreports.qt.io/browse/QTBUG-40449
                command = (
                    u"osascript -e 'get posix path of my posix file \""
                    u"file://{fileid}\" -- kthx. bai'").format(
                        fileid=path)
                process = subprocess.Popen(
                    command, shell=True,
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE)
                path = process.communicate()[0].lstrip().rstrip()

            LOGGER.info('Accepting drop event with path: "%s"', path)
            event.accept()
            self.setText(path)

        @QtCore.Slot(bool)
        def _emit_textchanged(self, triggered):
            """Slot for re-emitting the textchanged signal with current text.

            Parameters:
                triggered (bool): Ignored.

            Returns:
                ``None``
            """
            self.textChanged.emit(self.text())

        def contextMenuEvent(self, event):
            """Reimplemented from QtGui.QLineEdit.contextMenuEvent.

            This function allows me to make changes to the context menu when one
            is requested before I show the menu."""
            menu = self.createStandardContextMenu()
            refresh_action = QtWidgets.QAction('Refresh', menu)
            refresh_action.setIcon(qtawesome.icon('fa.refresh'))
            refresh_action.triggered.connect(self._emit_textchanged)
            menu.addAction(refresh_action)

            menu.exec_(event.globalPos())

    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        Text.__init__(self, label, helptext, interactive, args_key,
                      hideable, validator=validator)
        self.textfield = _Path.FileField()
        self.textfield.textChanged.connect(self._text_changed)

        self.widgets = [
            self.valid_button,
            self.label_widget,
            self.textfield,
            None,
            self.help_button,
        ]


class Folder(_Path):
    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        _Path.__init__(self, label, helptext, interactive, args_key,
                       hideable, validator=validator)
        self.path_select_button = FolderButton('Select folder')
        self.path_select_button.path_selected.connect(self.textfield.setText)
        self.widgets[3] = self.path_select_button

        if self.hideable:
            self._hideability_changed(False)


class File(_Path):
    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None):
        _Path.__init__(self, label, helptext, interactive, args_key,
                       hideable, validator=validator)
        self.path_select_button = FileButton('Select file')
        self.path_select_button.path_selected.connect(self.textfield.setText)
        self.widgets[3] = self.path_select_button

        if self.hideable:
            self._hideability_changed(False)


class SaveFile(_Path):
    def __init__(self, label, helptext=None, interactive=True,
                 args_key=None, hideable=False, validator=None,
                 default_savefile='new_file.txt'):
        _Path.__init__(self, label, helptext, interactive, args_key,
                       hideable, validator=validator)
        self.path_select_button = SaveFileButton('Select file',
                                                 default_savefile)
        self.path_select_button.path_selected.connect(self.textfield.setText)
        self.widgets[3] = self.path_select_button

        if self.hideable:
            self._hideability_changed(False)


class Checkbox(GriddedInput):

    # Re-setting value_changed to adapt to the type requirement.
    value_changed = QtCore.Signal(bool)
    # Re-setting interactivity_changed to avoid a segfault while testing on
    # linux via `python setup.py nosetests`.
    interactivity_changed = QtCore.Signal(bool)

    def __init__(self, label, helptext=None, interactive=True, args_key=None):
        GriddedInput.__init__(self, label=label, helptext=helptext,
                              interactive=interactive, args_key=args_key,
                              hideable=False, validator=None, )

        self.checkbox = QtWidgets.QCheckBox(label)
        self.checkbox.stateChanged.connect(self.value_changed.emit)
        self.widgets[0] = None  # No need for a valid button
        self.widgets[1] = self.checkbox  # replace label with checkbox
        self.satisfied = True

    def value(self):
        return self.checkbox.isChecked()

    def valid(self):
        return True

    def set_value(self, value):
        self.checkbox.setChecked(value)


class Dropdown(GriddedInput):
    def __init__(self, label, helptext=None, interactive=True, args_key=None,
                 hideable=False, options=()):
        GriddedInput.__init__(self, label=label, helptext=helptext,
                              interactive=interactive, args_key=args_key,
                              hideable=hideable, validator=None)
        self.dropdown = QtWidgets.QComboBox()
        self.widgets[2] = self.dropdown
        self.set_options(options)
        self.dropdown.currentIndexChanged.connect(self._index_changed)
        self.satisfied = True

        # Init hideability if needed
        if self.hideable:
            self._hideability_changed(False)

    def _index_changed(self, newindex):
        self.value_changed.emit(self.options[newindex])

    def set_options(self, options):
        self.dropdown.clear()
        cast_options = []
        for label in options:
            if type(label) in (int, float):
                label = str(label)
            try:
                cast_value = six.text_type(label, 'utf-8')
            except TypeError:
                # It's already unicode, so can't decode further.
                cast_value = label
            self.dropdown.addItem(cast_value)
            cast_options.append(cast_value)
        self.options = cast_options
        self.user_options = options

    def value(self):
        return self.dropdown.currentText()

    def set_value(self, value):
        # Handle case where value is of the type provided by the user,
        # and the case where it's been converted to a utf-8 string.
        for options_attr in ('options', 'user_options'):
            try:
                index = getattr(self, options_attr).index(value)
                self.dropdown.setCurrentIndex(index)
                return
            except ValueError:
                # ValueError when the value is not in the list
                pass
        raise ValueError('Value %s not in options %s or user options %s' % (
            value, self.options, self.user_options))


class Label(QtWidgets.QLabel):
    def __init__(self, text):
        QtWidgets.QLabel.__init__(self, text)
        self.setWordWrap(True)
        self.setOpenExternalLinks(True)

    def _add_to(self, layout):
        layout.addWidget(self, layout.rowCount(),  # target row
                         0,  # target starting column
                         1,  # row span
                         layout.columnCount())  # span all columns


class Container(QtWidgets.QGroupBox, Input):

    # need to redefine signals here.
    value_changed = QtCore.Signal(bool)
    interactivity_changed = QtCore.Signal(bool)
    sufficiency_changed = QtCore.Signal(bool)

    def __init__(self, label, interactive=True, expandable=False,
                 expanded=True, args_key=None):
        QtWidgets.QGroupBox.__init__(self)
        Input.__init__(self, label=label, interactive=interactive,
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
            QtWidgets.QSizePolicy.MinimumExpanding)  # vertical

    @QtCore.Slot(bool)
    def _hide_widgets(self, check_state):
        for layout_item in (self.layout().itemAtPosition(*coords)
                            for coords in itertools.product(
                                xrange(1, self.layout().rowCount()),
                                xrange(1, self.layout().columnCount()))):
            if layout_item and self.isVisible():
                layout_item.widget().setVisible(self.isChecked())

        # Update size based on sizehint now that widgets changed.
        self.setMinimumSize(self.sizeHint())
        #self.resize(self.sizeHint())

    def showEvent(self, event=None):
        if self.isCheckable():
            self._hide_widgets(self.value())
        self.resize(self.sizeHint())

    @property
    def expanded(self):
        if self.expandable:
            return self.isChecked()
        return True

    @expanded.setter
    def expanded(self, value):
        if not self.expandable:
            raise ValueError('Container cannot be expanded when not '
                             'expandable')
        self.setChecked(value)

    @property
    def expandable(self):
        return self.isCheckable()

    @expandable.setter
    def expandable(self, value):
        return self.setCheckable(value)

    def add_input(self, input):
        input._add_to(layout=self.layout())
        _apply_sizehint(self.layout().parent())

        if self.expandable:
            input.set_visible(self.expanded)
            input.set_interactive(self.expanded)

            if self.isVisible():
                for widget in input.widgets:
                    if not widget:
                        continue
                    widget.setVisible(self.expanded)
        self.sufficiency_changed.connect(input.set_interactive)
        self.sufficiency_changed.connect(input.set_visible)

        if isinstance(input, Multi):
            def _update_sizehints():
                self.setMinimumSize(self.sizeHint())
            input.input_added.connect(_update_sizehints)

    def _add_to(self, layout):
        layout.addWidget(self,
                         layout.rowCount(),  # target row
                         0,  # target starting column
                         1,  # row span
                         layout.columnCount())  # span all columns

    def value(self):
        return self.expanded

    def set_value(self, value):
        self.expanded = value


class Multi(Container):

    value_changed = QtCore.Signal(list)
    input_added = QtCore.Signal()

    class _RemoveButton(QtWidgets.QPushButton):

        remove_requested = QtCore.Signal(int)

        def __init__(self, label, index):
            QtWidgets.QPushButton.__init__(self, label)
            self.index = index
            self.clicked.connect(self._remove)

        def _remove(self, checked=False):
            self.remove_requested.emit(self.index)

    def __init__(self, label, callable_, interactive=True, args_key=None,
                 link_text='Add Another', helptext=None):
        self.items = []
        Container.__init__(self,
                           label=label,
                           interactive=interactive,
                           args_key=args_key,
                           expandable=False,
                           expanded=True)

        if not hasattr(callable_, '__call__'):
            raise ValueError("Callable passed to Multi is not callable.")

        self.callable_ = callable_
        self.add_link = QtWidgets.QLabel('<a href="add_new">%s</a>' % link_text)
        self.add_link.linkActivated.connect(self._add_templated_item)
        self._append_add_link()
        self.remove_buttons = []

    def value(self):
        return [input_.value() for input_ in self.items]

    def set_value(self, values):
        self.clear()
        for input_value in values:
            new_input_instance = self.callable_()
            new_input_instance.set_value(input_value)
            self.add_item(new_input_instance)

        self.value_changed.emit(list(values))

    def _add_templated_item(self, label=None):
        self.add_item()

    def add_item(self, new_input=None):
        if not new_input:
            new_input = self.callable_()

        new_input._add_to(self.layout())
        self.items.append(new_input)

        layout = self.layout()
        rightmost_item = layout.itemAtPosition(
            layout.rowCount()-1, layout.columnCount()-1)
        if not rightmost_item:
            col_index = layout.columnCount()-1
        else:
            col_index = layout.columnCount()

        new_remove_button = Multi._RemoveButton(
            '-R-', index=max(0, len(self.items)-1))
        new_remove_button.remove_requested.connect(self.remove)
        self.remove_buttons.append(new_remove_button)

        layout.addWidget(new_remove_button,
                         layout.rowCount()-1,  # current last row
                         col_index,
                         1,  # span 1 row
                         1)  # span 1 column
        self.setMinimumSize(self.sizeHint())
        self.update()
        self.input_added.emit()

    def _append_add_link(self):
        layout = self.layout()
        layout.addWidget(self.add_link,
                         layout.rowCount(),  # make new last row
                         0,  # target starting column
                         1,  # row span
                         layout.columnCount())  # span all columns

    def clear(self):
        layout = self.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        self._append_add_link()

    def remove(self, index):
        # clear all widgets from the layout.
        self.clear()

        self.items.pop(index)
        self.remove_buttons.pop(index)
        old_items = self.items[:]
        self.items = []
        self.remove_buttons = []
        for item in old_items:
            self.add_item(item)


class Form(QtWidgets.QWidget):

    submitted = QtCore.Signal()
    run_finished = QtCore.Signal()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

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
        self.scroll_area = QtWidgets.QScrollArea()
        self.layout().addWidget(self.scroll_area)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.verticalScrollBar().rangeChanged.connect(
            self.update_scroll_border)
        self.update_scroll_border(
            self.scroll_area.verticalScrollBar().minimum(),
            self.scroll_area.verticalScrollBar().maximum())
        self.scroll_area.setWidget(self.inputs)

        # set the sizehint of the inputs again ... needed after setting
        # scroll_area.
        if self.inputs.sizeHint().isValid():
            self.inputs.setMinimumSize(self.inputs.sizeHint())
        self.layout().setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.inputs.layout().setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)

        self.buttonbox = QtWidgets.QDialogButtonBox()
        self.run_button = QtWidgets.QPushButton(' Run')
        self.run_button.setIcon(QtGui.QIcon(ICON_ENTER))

        self.buttonbox.addButton(
            self.run_button, QtWidgets.QDialogButtonBox.AcceptRole)
        self.layout().addWidget(self.buttonbox)
        self.run_button.pressed.connect(self.emit_submitted)

        self.run_dialog = FileSystemRunDialog()

    def emit_submitted(self):
        # PyQt4 won't recognize self.submitted.emit as a bound slot, so
        # creating a bound method of Form to handle this.  Useful for MESH
        # demo.
        self.submitted.emit()

    def update_scroll_border(self, min, max):
        if min == 0 and max == 0:
            self.scroll_area.setStyleSheet("QScrollArea { border: None } ")
        else:
            self.scroll_area.setStyleSheet("")

    def run(self, target, args=(), kwargs=None, window_title='',
            out_folder='/'):

        if not hasattr(target, '__call__'):
            raise ValueError('Target %s must be callable' % target)

        self._thread = execution.Executor(target,
                                          args,
                                          kwargs)
        self._thread.finished.connect(self._run_finished)

        self.run_dialog.show()
        self.run_dialog.start(window_title=window_title,
                              out_folder=out_folder)
        self._thread.start()

    def _run_finished(self):
        # When the thread finishes.
        self.run_dialog.finish(
            exception_found=(self._thread.exception is not None),
            thread_exception=self._thread.exception)
        self.run_finished.emit()

    def add_input(self, input):
        self.inputs.add_input(input)
