# coding=utf-8
import unittest
import functools
import warnings
import threading
import logging
import contextlib
import sys
import os
import requests
import time
import tempfile
import shutil
import textwrap
import imp
import uuid
import json

if sys.version_info >= (3,):
    # Need to force PySide2 import in python3.  It's the only set of bindings I
    # can seem to get to work here.
    basestring = str
    import unittest.mock as mock
else:
    import mock

try:
    import PyQt4
except ImportError:
    import PySide2

import faulthandler
faulthandler.enable()
import qtpy
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtTest import QTest
import six

LOGGER = logging.getLogger(__name__)

@contextlib.contextmanager
def wait_on_signal(qt_app, signal, timeout=250):
    """Block loop until signal emitted, or timeout (ms) elapses."""
    loop = QtCore.QEventLoop()
    signal.connect(loop.quit)

    try:
        yield
        if qt_app.hasPendingEvents():
            qt_app.processEvents()
    except Exception as error:
        LOGGER.exception('Error encountered while witing for signal %s',
                         signal)
        raise error
    finally:
        if timeout is not None:
            QtCore.QTimer.singleShot(timeout, loop.quit)
        loop.exec_()
    loop = None


class _QtTest(unittest.TestCase):
    def setUp(self):
        """Set up a QAppplication on each test case."""
        try:
            QApplication = QtGui.QApplication
        except AttributeError:
            QApplication = QtWidgets.QApplication

        self.qt_app = QApplication.instance()

        # If we're running PyQt4, we need to instruct Qt to use UTF-8 strings
        # internally.
        if qtpy.API in ('pyqt', 'pyqt4'):
            QtCore.QTextCodec.setCodecForCStrings(
                QtCore.QTextCodec.codecForName('UTF-8'))

        if self.qt_app is None:
            self.qt_app = QApplication(sys.argv)

    def tearDown(self):
        """Clear the QApplication's event queue."""
        # After each test, empty the event queue.
        # This should help to make sure that there aren't any event-based race
        # conditions where a C/C++ object is deleted before a slot is called.
        self.qt_app.sendPostedEvents()

class _SettingsSandbox(_QtTest):
    def setUp(self):
        _QtTest.setUp(self)
        from natcap.invest.ui import inputs

        # back up the QSettings options for the test run so we don't disrupt
        # whatever settings exist on this computer
        self.settings = dict(
            (key, inputs.INVEST_SETTINGS.value(key)) for key in
            inputs.INVEST_SETTINGS.allKeys())
        inputs.INVEST_SETTINGS.clear()

    def tearDown(self):
        _QtTest.tearDown(self)
        from natcap.invest.ui import inputs
        inputs.INVEST_SETTINGS.clear()
        for key, value in self.settings.items():
            inputs.INVEST_SETTINGS.setValue(key, value)


class InVESTModelInputTest(_QtTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import InVESTModelInput
        return InVESTModelInput(*args, **kwargs)

    def test_label(self):
        input_instance = self.__class__.create_input(label='foo')
        self.assertEqual(input_instance.label, 'foo')

    def test_helptext(self):
        input_instance = self.__class__.create_input(label='foo', helptext='bar')
        self.assertEqual(input_instance.helptext, 'bar')

    def test_clear(self):
        input_instance = self.__class__.create_input(label='foo')
        if input_instance.__class__.__name__ == 'InVESTModelInput':
            with self.assertRaises(NotImplementedError):
                input_instance.clear()
        else:
            self.fail('Test class must reimplement this test method.')

    def test_interactive(self):
        input_instance = self.__class__.create_input(label='foo', interactive=True)
        self.assertEqual(input_instance.interactive, True)

    def test_noninteractive(self):
        input_instance = self.__class__.create_input(label='foo', interactive=False)
        # Silence notimplementederror exceptions on input.value in some cases.
        try:
            input_instance.value()
        except NotImplementedError:
            input_instance.value = lambda: 'Value!'
        self.assertEqual(input_instance.interactive, False)

    def test_set_interactive(self):
        input_instance = self.__class__.create_input(label='foo', interactive=False)
        self.assertEqual(input_instance.interactive, False)
        # Silence notimplementederror exceptions on input.value in some cases.
        try:
            input_instance.value()
        except NotImplementedError:
            input_instance.value = lambda: 'Value!'
        input_instance.set_interactive(True)
        self.assertEqual(input_instance.interactive, True)

    def test_interactivity_changed(self):
        input_instance = self.__class__.create_input(label='foo', interactive=False)
        callback = mock.Mock()
        input_instance.interactivity_changed.connect(callback)

        with wait_on_signal(self.qt_app, input_instance.interactivity_changed):
            try:
                input_instance.value()
            except NotImplementedError:
                input_instance.value = lambda: 'Value!'
            input_instance.set_interactive(True)

        callback.assert_called_with(True)

    def test_add_to_layout(self):
        base_widget = QtWidgets.QWidget()
        base_widget.setLayout(QtWidgets.QGridLayout())

        input_instance = self.__class__.create_input(label='foo')
        input_instance.add_to(base_widget.layout())

    def test_value(self):
        input_instance = self.__class__.create_input(label='foo')
        if input_instance.__class__.__name__ in ('InVESTModelInput', 'GriddedInput'):
            with self.assertRaises(NotImplementedError):
                input_instance.value()
        else:
            self.fail('Test class must reimplement this test method')

    def test_set_value(self):
        input_instance = self.__class__.create_input(label='foo')
        if input_instance.__class__.__name__ in ('InVESTModelInput', 'GriddedInput'):
            with self.assertRaises(NotImplementedError):
                input_instance.set_value('foo')
        else:
            self.fail('Test class must reimplement this test method')

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(label='some_label')
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)

        if input_instance.__class__.__name__ in ('InVESTModelInput', 'GriddedInput'):
            try:
                with self.assertRaises(NotImplementedError):
                    self.assertEqual(input_instance.value(), '')
                    with wait_on_signal(self.qt_app, input_instance.value_changed):
                        input_instance.set_value('foo')
                    callback.assert_called_with(u'foo')
            finally:
                input_instance.value_changed.disconnect(callback)
        else:
            self.fail('Test class must reimplement this test method')

    def test_value_changed_signal(self):
        input_instance = self.__class__.create_input(label='foo')
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)

        try:
            with wait_on_signal(self.qt_app, input_instance.value_changed):
                try:
                    input_instance.value()
                except NotImplementedError:
                    input_instance.value = lambda: 'Value!'
                input_instance.value_changed.emit('value')

            callback.assert_called_with('value')
        finally:
            input_instance.value_changed.disconnect(callback)

    def test_interactivity_changed_signal(self):
        input_instance = self.__class__.create_input(label='foo')
        callback = mock.Mock()
        input_instance.interactivity_changed.connect(callback)

        with wait_on_signal(self.qt_app, input_instance.interactivity_changed):
            try:
                input_instance.value()
            except NotImplementedError:
                input_instance.value = lambda: 'Value!'

            input_instance.interactivity_changed.emit(True)

        callback.assert_called_with(True)

    def test_args_key(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     args_key='some_key')
        self.assertEqual(input_instance.args_key, 'some_key')

    def test_no_args_key(self):
        input_instance = self.__class__.create_input(label='foo')
        self.assertEqual(input_instance.args_key, None)

    def test_add_to_container(self):
        from natcap.invest.ui.inputs import Container
        input_instance = self.__class__.create_input(label='foo')
        container = Container(label='Some container')
        container.add_input(input_instance)

    def test_visibility(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     interactive=False)
        self.assertEqual(input_instance.visible(), True)

        input_instance.set_visible(False)
        if len(input_instance.widgets) > 0:  # only works if input has widgets
            self.assertEqual(input_instance.visible(), False)

        input_instance.set_visible(True)
        if len(input_instance.widgets) > 0:  # only works if input has widgets
            self.assertEqual(input_instance.visible(), True)

    def test_visibility_when_shown(self):
        from natcap.invest.ui import inputs
        container = inputs.Container(label='sample container')
        input_instance = self.__class__.create_input(label='foo',
                                                     interactive=False)
        container.add_input(input_instance)
        container.show()

        self.assertEqual(input_instance.visible(), True)

        input_instance.set_visible(False)
        if len(input_instance.widgets) > 0:  # only works if input has widgets
            self.assertEqual(input_instance.visible(), False)

        input_instance.set_visible(True)
        if len(input_instance.widgets) > 0:  # only works if input has widgets
            self.assertEqual(input_instance.visible(), True)


class GriddedInputTest(InVESTModelInputTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import GriddedInput
        return GriddedInput(*args, **kwargs)

    def test_label(self):
        input_instance = self.__class__.create_input(label='foo')
        label_text = input_instance.label
        self.assertEqual(label_text, 'foo')

    def test_validator(self):
        _callback = mock.Mock()
        input_instance = self.__class__.create_input(
            label='foo', validator=_callback)
        self.assertEqual(input_instance.validator_ref, _callback)

    def test_helptext(self):
        from natcap.invest.ui.inputs import HelpButton
        input_instance = self.__class__.create_input(label='foo',
                                                     helptext='bar')
        self.assertTrue(isinstance(input_instance.help_button, HelpButton))

    def test_no_helptext(self):
        input_instance = self.__class__.create_input(label='foo')
        self.assertTrue(isinstance(input_instance.help_button,
                                   QtWidgets.QWidget))

    def test_clear(self):
        input_instance = self.__class__.create_input(label='foo')
        # simulate successful validation completion.
        input_instance._validation_finished([])
        self.assertEqual(input_instance._valid, True)
        input_instance.clear()
        self.assertEqual(input_instance.valid_button.whatsThis(), '')
        self.assertEqual(input_instance.valid_button.toolTip(), '')
        self.assertEqual(input_instance._valid, None)
        self.assertEqual(input_instance.sufficient, False)

    def test_validate_passes(self):
        #"""UI: Validation that passes should affect validity."""
        _validation_func = mock.Mock(return_value=[])
        input_instance = self.__class__.create_input(
            label='some_label', args_key='some_key',
            validator=_validation_func)
        try:
            input_instance.value()
        except NotImplementedError:
            input_instance.value = lambda: 'value!'

        input_instance._validate()

        # Wait for validation to finish.
        self.assertEqual(input_instance.valid(), True)

    def test_validate_missing_args_key(self):
        from natcap.invest.ui import inputs
        input_instance = self.__class__.create_input(
            label='some_label')

        input_instance.value = mock.Mock(
            input_instance, return_value=u'something')

        # Verify we're starting with an unvalidated input
        self.assertEqual(input_instance.valid(), None)
        with warnings.catch_warnings(record=True) as messages:
            input_instance._validate()
            time.sleep(0.25)  # wait for warnings to register
        self.qt_app.processEvents()

        # Validation still passes
        self.assertEqual(input_instance.valid(), True)

    def test_validate_fails(self):
        #"""UI: Validation that fails should affect validity."""
        _validation_func = mock.Mock(
            return_value=[('some_key', 'some warning')])
        input_instance = self.__class__.create_input(
            label='some_label', args_key='some_key',
            validator=_validation_func)
        try:
            input_instance.value()
        except NotImplementedError:
            input_instance.value = lambda: 'value!'

        input_instance._validate()

        # Wait for validation to finish and assert Failure.
        self.assertEqual(input_instance.valid(), False)

    def test_validate_required_validator(self):
        from natcap.invest.ui import inputs
        input_instance = self.__class__.create_input(
            label='some_label', args_key='foo'
        )

        input_instance.value = mock.Mock(
            input_instance, return_value=u'something')

        # Verify we're starting with an unvalidated input
        self.assertEqual(input_instance.valid(), None)
        with warnings.catch_warnings(record=True) as messages:
            input_instance._validate()
            time.sleep(0.25)  # wait for warnings to register
        self.qt_app.processEvents()

        # Validation still passes, but verify warning raised
        self.assertEqual(len(messages), 1)
        self.assertEqual(input_instance.valid(), True)

    def test_validate_error(self):
        input_instance = self.__class__.create_input(
            label='some_label', args_key='foo',
            validator=lambda args, limit_to=None: []
        )

        input_instance.value = mock.Mock(
            input_instance, return_value=u'something')

        input_instance._validator.validate = mock.Mock(
            input_instance._validator.validate, side_effect=ValueError('foo'))

        with self.assertRaises(ValueError):
            input_instance._validate()

    def test_nonhideable_default_state(self):
        sample_widget = QtWidgets.QWidget()
        sample_widget.setLayout(QtWidgets.QGridLayout())
        input_instance = self.__class__.create_input(
            label='some_label', hideable=False)
        input_instance.add_to(sample_widget.layout())
        sample_widget.show()

        self.assertEqual(input_instance.hideable, False)
        self.assertEqual(input_instance.hidden(), False)

        for widget, hidden in zip(input_instance.widgets,
                                  [False, False, False, False, False]):
            if not widget:
                continue
            if not widget.isHidden() == hidden:
                self.fail('Widget %s hidden: %s, expected: %s' % (
                    widget, widget.isHidden(), hidden))

    def test_nonhideable_set_hidden_fails(self):
        input_instance = self.__class__.create_input(
            label='some_label', hideable=False)
        with self.assertRaises(ValueError):
            input_instance.set_hidden(False)

    def test_hideable_set_hidden(self):
        sample_widget = QtWidgets.QWidget()
        sample_widget.setLayout(QtWidgets.QGridLayout())
        input_instance = self.__class__.create_input(
            label='some_label', hideable=True)
        input_instance.add_to(sample_widget.layout())
        sample_widget.show()

        self.assertEqual(input_instance.hidden(), True)  # default is hidden
        input_instance.set_hidden(False)
        self.assertEqual(input_instance.hidden(), False)
        for widget, hidden in zip(input_instance.widgets,
                                  [False, False, False, False, False]):
            if not widget:
                continue
            if not widget.isHidden() == hidden:
                self.fail('Widget %s hidden: %s, expected: %s' % (
                    widget, widget.isHidden(), hidden))

        input_instance.set_hidden(True)
        self.assertEqual(input_instance.hidden(), True)
        for widget, hidden in zip(input_instance.widgets,
                                  [False, False, True, True, True]):
            if not widget:
                continue
            if not widget.isHidden() == hidden:
                self.fail('Widget %s hidden: %s, expected: %s' % (
                    widget, widget.isHidden(), hidden))

    def test_hidden_change_signal(self):
        input_instance = self.__class__.create_input(
            label='some_label', hideable=True)
        callback = mock.Mock()
        input_instance.hidden_changed.connect(callback)
        self.assertEqual(input_instance.hidden(), True)

        with wait_on_signal(self.qt_app, input_instance.hidden_changed):
            input_instance.set_hidden(False)

        callback.assert_called_with(True)

    def test_hidden_when_not_hideable(self):
        """UI: Verify non-hideable Text input has expected behavior."""
        input_instance = self.__class__.create_input(
            label='Some label', hideable=False)

        self.assertEqual(input_instance.hideable, False)
        self.assertEqual(input_instance.hidden(), False)

        with self.assertRaises(ValueError):
            input_instance.set_hidden(True)


class TextTest(GriddedInputTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import Text
        return Text(*args, **kwargs)

    def test_value(self):
        input_instance = self.__class__.create_input(label='text')
        self.assertEqual(input_instance.value(), '')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_set_value(self):
        input_instance = self.__class__.create_input(label='text')
        self.assertEqual(input_instance.value(), '')
        input_instance.set_value('foo')
        self.assertEqual(input_instance.value(), u'foo')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_set_value_latin1(self):
        input_instance = self.__class__.create_input(label='text')
        self.assertEqual(input_instance.value(), '')
        input_instance.set_value('gr\xe9gory')  # Latin-1 encoded string
        self.assertEqual(input_instance.value().encode('utf-8'),
                         b'gr\xc3\xa9gory')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_set_value_int(self):
        input_instance = self.__class__.create_input(label='text')
        input_instance.set_value(1)
        self.assertEqual(input_instance.value(), u'1')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_set_value_float(self):
        input_instance = self.__class__.create_input(label='text')
        input_instance.set_value(3.14159)
        self.assertEqual(input_instance.value(), u'3.14159')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_set_value_when_hideable(self):
        input_instance = self.__class__.create_input(label='text',
                                                     hideable=True)
        self.assertEqual(input_instance.value(), '')
        self.assertEqual(input_instance.hideable, True)
        self.assertEqual(input_instance.hidden(), True)
        input_instance.set_value('foo')
        self.assertEqual(input_instance.value(), u'foo')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))
        self.assertFalse(input_instance.hidden())

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(label='text')
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)

        self.assertEqual(input_instance.value(), '')

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.set_value('foo')

        callback.assert_called_with(u'foo')

    def test_textfield_settext(self):
        input_instance = self.__class__.create_input(label='text')

        input_instance.textfield.setText('foo')
        self.assertEqual(input_instance.value(), u'foo')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_textfield_settext_signal(self):
        input_instance = self.__class__.create_input(label='text')
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.textfield.setText('foo')

        callback.assert_called_with(u'foo')

    def test_textfield_drag_n_drop(self):
        input_instance = self.__class__.create_input(label='text')

        mime_data = QtCore.QMimeData()
        mime_data.setText('Hello world!')

        event = QtGui.QDragEnterEvent(
            input_instance.textfield.pos(),
            QtCore.Qt.CopyAction,
            mime_data,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)

        input_instance.textfield.dragEnterEvent(event)
        self.assertEqual(event.isAccepted(), True)

    def test_textfield_drag_n_drop_urls(self):
        input_instance = self.__class__.create_input(label='text')

        mime_data = QtCore.QMimeData()
        mime_data.setText('Hello world!')
        mime_data.setUrls([QtCore.QUrl('/foo/bar')])

        event = QtGui.QDragEnterEvent(
            input_instance.textfield.pos(),
            QtCore.Qt.CopyAction,
            mime_data,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)

        input_instance.textfield.dragEnterEvent(event)
        self.assertEqual(event.isAccepted(), False)

    def test_textfield_drop(self):
        input_instance = self.__class__.create_input(label='text')

        mime_data = QtCore.QMimeData()
        mime_data.setText('Hello world!')
        mime_data.setUrls([QtCore.QUrl('/foo/bar')])

        event = QtGui.QDropEvent(
            input_instance.textfield.pos(),
            QtCore.Qt.CopyAction,
            mime_data,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)

        input_instance.textfield.dropEvent(event)
        self.assertEqual(event.isAccepted(), True)
        self.assertEqual(input_instance.value(), 'Hello world!')

    def test_clear(self):
        input_instance = self.__class__.create_input(label='text')
        input_instance.set_value('foo')
        input_instance.clear()
        self.assertEqual(input_instance.value(), '')
        self.assertEqual(input_instance.valid(), None)


class PathTest(TextTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import _Path
        return _Path(*args, **kwargs)

    def test_path_context_menu_coverage(self):
        input_instance = self.__class__.create_input(label='foo')

        _callback = mock.Mock()
        input_instance.textfield.textChanged.connect(_callback)

        event = QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Mouse,
            input_instance.textfield.mapToGlobal(
                input_instance.textfield.pos()))

        def _click_out_of_contextmenu():
            # In case the popup isn't shown until after the callback is called,
            # we should be sure to wait for when it is shown.
            popup = None
            n_tries = 0
            while True:
                if n_tries > 5:
                    raise RuntimeError(
                        'Something happened where we could not get the '
                        'context menu')
                popup = self.qt_app.activePopupWidget()
                try:
                    popup.close()
                    break
                except AttributeError:
                    # When popup is None
                    n_tries += 1
                    self.qt_app.processEvents()
                    time.sleep(0.25)


        QtCore.QTimer.singleShot(25, _click_out_of_contextmenu)
        input_instance.textfield.contextMenuEvent(event)

        # simulate textchanged signal (expects a bool)
        input_instance.textfield._emit_textchanged(True)
        self.qt_app.processEvents()
        _callback.assert_called_once()

    def test_path_selected(self):
        input_instance = self.__class__.create_input(label='foo')
        # Only run this test on subclasses of path
        if input_instance.__class__.__name__ != '_Path':
            input_instance.path_select_button.path_selected.emit(u'/tmp/foo')
            self.assertTrue(input_instance.value(), '/tmp/foo')

    def test_path_dialog_cancelled(self):
        input_instance = self.__class__.create_input(label='foo')
        # Only run this test on subclasses of path
        if input_instance.__class__.__name__ != '_Path':
            # initial value should not be overwritten by the new value.
            input_instance.set_value('some_value')

            # path is blank when the dialog was cancelled.
            input_instance.path_select_button.path_selected.emit(u'')
            self.assertTrue(input_instance.value(), 'some_value')

    def test_path_selected_cyrillic(self):
        input_instance = self.__class__.create_input(label='foo')
        # Only run this test on subclasses of path
        if input_instance.__class__.__name__ != '_Path':
            input_instance.set_value(u'/tmp/fooДЖЩя')
            input_instance.path_select_button.path_selected.emit(
                u'/tmp/fooДЖЩя')
            self.assertEquals(input_instance.value(), u'/tmp/fooДЖЩя')

    def test_textfield_drag_n_drop(self):
        input_instance = self.__class__.create_input(label='text')

        mime_data = QtCore.QMimeData()
        mime_data.setText('Hello world!ДЖЩя')

        event = QtGui.QDragEnterEvent(
            input_instance.textfield.pos(),
            QtCore.Qt.CopyAction,
            mime_data,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)

        input_instance.textfield.dragEnterEvent(event)
        self.assertEqual(event.isAccepted(), False)

    def test_textfield_drag_n_drop_urls(self):
        input_instance = self.__class__.create_input(label='text')

        mime_data = QtCore.QMimeData()
        mime_data.setText(u'Hello world!ДЖЩя')
        mime_data.setUrls([QtCore.QUrl('/foo/bar/ДЖЩя')])

        event = QtGui.QDragEnterEvent(
            input_instance.textfield.pos(),
            QtCore.Qt.CopyAction,
            mime_data,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)

        input_instance.textfield.dragEnterEvent(event)
        self.assertEqual(event.isAccepted(), True)

    def test_textfield_drop(self):
        pass

    def test_textfield_drop_windows(self):
        input_instance = self.__class__.create_input(label='text')

        mime_data = QtCore.QMimeData()
        mime_data.setText(u'Hello world!ДЖЩя')
        # this is what paths look like when Qt receives them.
        mime_data.setUrls([QtCore.QUrl(u'/C:/foo/bar/ДЖЩя')])

        event = QtGui.QDropEvent(
            input_instance.textfield.pos(),
            QtCore.Qt.CopyAction,
            mime_data,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)

        with mock.patch('platform.system', return_value='Windows'):
            input_instance.textfield.dropEvent(event)

        self.assertEqual(event.isAccepted(), True)
        self.assertEqual(input_instance.value(), u'C:/foo/bar/ДЖЩя')

    def test_textfield_drop_mac(self):
        # NOTE: Mac OS's filesystem is UTF-8.
        input_instance = self.__class__.create_input(label='text')

        text_path = u'/foo/bar/ДЖЩя'
        mime_data = QtCore.QMimeData()
        mime_data.setText(u'Hello world!ДЖЩя')
        mime_data.setUrls([QtCore.QUrl(text_path)])

        event = QtGui.QDropEvent(
            input_instance.textfield.pos(),
            QtCore.Qt.CopyAction,
            mime_data,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)

        with mock.patch('platform.system', return_value='Darwin'):
            with mock.patch('subprocess.Popen') as mock_popen:
                mock_process = mock.Mock()
                mock_process.configure_mock(
                    **{'communicate.return_value': [text_path]})
                mock_popen.return_value = mock_process

                input_instance.textfield.dropEvent(event)

        self.assertTrue(mock_popen.called)
        self.assertTrue(mock_popen.call_args[0][0].startswith('osascript'))
        self.assertEqual(event.isAccepted(), True)
        self.assertEqual(input_instance.value(), u'/foo/bar/ДЖЩя')


class FolderTest(PathTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import Folder
        return Folder(*args, **kwargs)


class FileTest(PathTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import File
        return File(*args, **kwargs)


class SaveFileTest(PathTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import SaveFile
        return SaveFile(*args, **kwargs)


class CheckboxTest(GriddedInputTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import Checkbox
        return Checkbox(*args, **kwargs)

    def test_value(self):
        input_instance = self.__class__.create_input(label='new_label')
        self.assertEqual(input_instance.value(), False)  # default value

        # set the value using the qt method
        input_instance.checkbox.setChecked(True)
        self.assertEqual(input_instance.value(), True)

    def test_set_value(self):
        input_instance = self.__class__.create_input(label='new_label')
        self.assertEqual(input_instance.value(), False)
        input_instance.set_value(True)
        self.assertEqual(input_instance.value(), True)

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(label='new_label')
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), False)

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.set_value(True)

        callback.assert_called_with(True)

    def test_value_changed_signal(self):
        input_instance = self.__class__.create_input(label='new_label')
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.value_changed.emit(True)

        callback.assert_called_with(True)

    def test_clear(self):
        input_instance = self.__class__.create_input(label='new_label')
        input_instance.set_value(True)
        input_instance.clear()
        self.assertEqual(input_instance.value(), False)

    def test_valid(self):
        input_instance = self.__class__.create_input(label='new_label')
        self.assertEqual(input_instance.value(), False)
        self.assertEqual(input_instance.valid(), True)
        input_instance.set_value(True)
        self.assertEqual(input_instance.valid(), True)

    def test_validate_required_validator(self):
        # Override from GriddedInputTest, as checkbox is always valid.
        from natcap.invest.ui import inputs
        input_instance = self.__class__.create_input(
            label='some_label', args_key='foo'
        )

        input_instance.value = mock.Mock(
            input_instance, return_value=u'something')

        # Verify we're starting with an unvalidated input
        self.assertEqual(input_instance.valid(), True)
        with warnings.catch_warnings(record=True) as messages:
            input_instance._validate()
            time.sleep(0.25)  # wait for warnings to register
        self.qt_app.processEvents()

        # Validation still passes, but verify warning raised
        self.assertEqual(len(messages), 1)
        self.assertEqual(input_instance.valid(), True)

    def test_validate_missing_args_key(self):
        # Override from GriddedInputTest, as checkbox is always valid.
        from natcap.invest.ui import inputs
        input_instance = self.__class__.create_input(
            label='some_label')

        input_instance.value = mock.Mock(
            input_instance, return_value=u'something')

        # Verify we're starting with an unvalidated input
        self.assertEqual(input_instance.valid(), True)
        with warnings.catch_warnings(record=True) as messages:
            input_instance._validate()
            time.sleep(0.25)  # wait for warnings to register
        self.qt_app.processEvents()

        # Validation still passes
        self.assertEqual(input_instance.valid(), True)


    def test_label(self):
        # Override, sinve 'Optional' is irrelevant for Checkbox.
        pass

    def test_validator(self):
        pass

    def test_validate_required(self):
        pass

    def test_validate_passes(self):
        pass

    def test_validate_fails(self):
        pass

    def test_required(self):
        pass

    def test_set_required(self):
        pass

    def test_nonrequired(self):
        pass

    def test_nonhideable_set_hidden_fails(self):
        pass

    def test_nonhideable_default_state(self):
        pass

    def test_label_required(self):
        pass

    def test_hideable_set_hidden(self):
        pass

    def test_hidden_when_not_hideable(self):
        pass

    def test_hidden_change_signal(self):
        pass

    def test_validate_required_args_key(self):
        pass

    def test_validate_error(self):
        pass


class DropdownTest(GriddedInputTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import Dropdown
        return Dropdown(*args, **kwargs)

    def test_options(self):
        input_instance = self.__class__.create_input(
            label='label', options=('foo', 'bar', 'baz'))
        self.assertEqual(input_instance.options, [u'foo', u'bar', u'baz'])

    def test_options_with_return_value_map(self):
        return_value_map = {'foo': 1, 'bar': 2, 'baz': 3}
        input_instance = self.__class__.create_input(
            label='label', options=('foo', 'bar', 'baz'),
            return_value_map=return_value_map)
        self.assertEqual(input_instance.options, [u'foo', u'bar', u'baz'])
        self.assertEqual(input_instance.return_value_map,
                         {'foo': '1', 'bar': '2', 'baz': '3'})

    def test_options_return_value_mismatch(self):
        with self.assertRaises(ValueError):
            self.__class__.create_input(
                label='label', options=(1, 2, 3),
                return_value_map={'foo': 4, 1: 'bar'})

    def test_options_typecast(self):
        input_instance = self.__class__.create_input(
            label='label', options=(1, 2, 3))
        self.assertEqual(input_instance.options, [u'1', u'2', u'3'])

    def test_set_options_unicode(self):
        input_instance = self.__class__.create_input(
            label='label', options=(u'Þingvellir',))
        self.assertEqual(input_instance.options, [u'Þingvellir'])

    def test_clear(self):
        input_instance = self.__class__.create_input(
            label='label', options=('foo', 'bar', 'baz'))
        input_instance.set_value('bar')
        self.assertEqual(input_instance.value(), 'bar')
        input_instance.clear()
        self.assertEqual(input_instance.value(), 'foo')

    def test_clear_no_options(self):
        input_instance = self.__class__.create_input(
            label='label', options=())
        try:
            input_instance.clear()
        except Exception as e:
            self.fail("Unexpected exception: %s" % repr(e))

    def test_set_value(self):
        input_instance = self.__class__.create_input(
            label='label', options=('foo', 'bar', 'baz'))
        input_instance.set_value('foo')
        self.assertEqual(input_instance.value(), u'foo')

    def test_set_value_noncast(self):
        input_instance = self.__class__.create_input(
            label='label', options=(1, 2, 3))
        input_instance.set_value(1)
        self.assertEqual(input_instance.value(), u'1')

    def test_set_value_not_in_options(self):
        input_instance = self.__class__.create_input(
            label='label', options=(1, 2, 3))
        with self.assertRaises(ValueError):
            input_instance.set_value('foo')

    def test_set_value_from_return_map(self):
        input_instance = self.__class__.create_input(
            label='label', options=(1, 2, 3),
            return_value_map={1: 'a', 2: 'b', 3: 'c'}
        )
        input_instance.set_value('a')
        self.assertEqual(input_instance.value(), 'a')
        self.assertEqual(input_instance.dropdown.currentIndex(), 0)

        input_instance.set_value(3)
        self.assertEqual(input_instance.value(), 'c')
        self.assertEqual(input_instance.dropdown.currentIndex(), 2)

    def test_value(self):
        input_instance = self.__class__.create_input(
            label='label', options=('foo', 'bar', 'baz'))
        self.assertEqual(input_instance.value(), u'foo')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(
            label='label', options=('foo', 'bar', 'baz'))
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), u'foo')

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.set_value('bar')

        callback.assert_called_with('bar')

    def test_label(self):
        # Override, since 'Optional' is irrelevant for Dropdown.
        pass

    def test_validator(self):
        pass

    def test_validate_required(self):
        pass

    def test_validate_passes(self):
        pass

    def test_validate_fails(self):
        pass

    def test_required(self):
        pass

    def test_set_required(self):
        pass

    def test_nonrequired(self):
        pass

    def test_label_required(self):
        pass

    def test_validate_required_args_key(self):
        pass

    def test_validate_error(self):
        pass

    def test_validate_missing_args_key(self):
        pass

    def test_validate_required_validator(self):
        pass


class LabelTest(_QtTest):
    def test_add_to_layout(self):
        from natcap.invest.ui.inputs import Label

        super_widget = QtWidgets.QWidget()
        super_widget.setLayout(QtWidgets.QGridLayout())
        label = Label('Hello, World!')
        label.add_to(super_widget.layout())


class ContainerTest(InVESTModelInputTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import Container
        return Container(*args, **kwargs)

    def test_expandable(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=False)

        self.assertEqual(input_instance.expandable, False)
        self.assertEqual(input_instance.expanded, True)

        input_instance.expandable = True
        self.assertEqual(input_instance.expandable, True)

    def test_expanded(self):
        from natcap.invest.ui import inputs
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True,
                                                     expanded=True)
        input_instance.show()

        # Add an input so we can text that the input becomes visible.
        contained_input = inputs.Text(label='some text!')
        input_instance.add_input(contained_input)

        self.assertEqual(input_instance.expandable, True)
        self.assertEqual(input_instance.expanded, True)

        input_instance.expanded = False
        self.assertEqual(input_instance.expanded, False)

    def test_clear(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True)
        input_instance.expanded = True
        input_instance.clear()
        self.assertEqual(input_instance.expanded, False)

    def test_value_changed_signal(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True)
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.value_changed.emit(True)

        callback.assert_called_with(True)

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True,
                                                     expanded=False)
        callback = mock.Mock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), False)

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.set_value(True)

        callback.assert_called_with(True)

    def test_value(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True)

        input_instance.setChecked(False)
        self.assertEqual(input_instance.value(), False)
        input_instance.setChecked(True)
        self.assertEqual(input_instance.value(), True)

    def test_set_value(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True,
                                                     expanded=False)

        self.assertEqual(input_instance.value(), False)
        input_instance.set_value(True)
        self.assertEqual(input_instance.value(), True)

    def test_set_value_nonexpandable(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=False)
        with self.assertRaises(ValueError):
            input_instance.set_value(False)

    def test_add_input_multi_coverage(self):
        # Multis need a special case because the whole container needs to be
        # resized via a callback when a new input is added to the multi.
        from natcap.invest.ui import inputs
        input_instance = self.__class__.create_input(label='foo')
        multi = inputs.Multi(label='Some multi element',
                             callable_=functools.partial(inputs.Text,
                                                         label='text input'))
        input_instance.add_input(multi)
        multi.add_item()

    def test_helptext(self):
        pass

    def test_nonrequired(self):
        pass

    def test_required(self):
        pass

    def test_set_required(self):
        pass


@unittest.skip('skipping MultiTest class. See issue #3936')
class MultiTest(ContainerTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import Multi

        if 'callable_' not in kwargs:
            kwargs['callable_'] = MultiTest.create_sample_callable(
                label='some text')
        return Multi(*args, **kwargs)

    @staticmethod
    def create_sample_callable(*args, **kwargs):
        from natcap.invest.ui.inputs import Text
        return functools.partial(Text, *args, **kwargs)

    def test_setup_callable_not_callable(self):
        with self.assertRaises(ValueError):
            self.__class__.create_input(
                label='foo',
                callable_=None)

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(
            label='foo',
            callable_=self.__class__.create_sample_callable(label='foo'))

        callback = mock.Mock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), [])

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.set_value(('aaa', 'bbb'))

        callback.assert_called_with(['aaa', 'bbb'])

    def test_value_changed_signal(self):
        input_instance = self.__class__.create_input(
            label='foo',
            callable_=self.__class__.create_sample_callable(label='foo'))

        callback = mock.Mock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), [])

        with wait_on_signal(self.qt_app, input_instance.value_changed):
            input_instance.value_changed.emit(['aaa', 'bbb'])

        callback.assert_called_with(['aaa', 'bbb'])

    def test_value(self):
        input_instance = self.__class__.create_input(
            label='foo',
            callable_=self.__class__.create_sample_callable(label='foo'))

        self.assertEqual(input_instance.value(), [])  # default value

    def test_set_value(self):
        input_instance = self.__class__.create_input(
            label='foo',
            callable_=self.__class__.create_sample_callable(label='foo'))

        self.assertEqual(input_instance.value(), [])  # default value
        input_instance.set_value(('aaa', 'bbb'))
        self.assertEqual(input_instance.value(), ['aaa', 'bbb'])

    def test_remove_item_by_button(self):
        input_instance = self.__class__.create_input(
            label='foo',
            callable_=self.__class__.create_sample_callable(label='foo'))

        self.assertEqual(input_instance.value(), [])  # default value
        input_instance.set_value(('aaa', 'bbb', 'ccc'))
        self.assertEqual(input_instance.value(), ['aaa', 'bbb', 'ccc'])

        # reach into the Multi and press the 'bbb' remove button
        QTest.mouseClick(input_instance.remove_buttons[1],
                         QtCore.Qt.LeftButton)

        self.assertEqual(input_instance.value(), ['aaa', 'ccc'])

    def test_add_item_by_link(self):
        input_instance = self.__class__.create_input(
            label='foo',
            callable_=self.__class__.create_sample_callable(label='foo'))
        input_instance.add_link.linkActivated.emit('add_new')

        self.assertEqual(input_instance.value(), [''])

    def test_clear(self):
        input_instance = self.__class__.create_input(
            label='foo',
            callable_=self.__class__.create_sample_callable(label='foo'))

        # Add a few text inputs to this multi
        for _ in range(3):
            input_instance.add_item()

        input_instance.clear()
        self.assertEqual(input_instance.value(), [])

    def test_set_value_nonexpandable(self):
        pass

    def test_expanded(self):
        pass

    def test_expandable(self):
        pass

    def test_set_required(self):
        pass


class ValidationWorkerTest(_QtTest):
    def test_run(self):
        from natcap.invest.ui.inputs import ValidationWorker
        _callable = mock.Mock(return_value=[])
        worker = ValidationWorker(
            target=_callable,
            args={'foo': 'bar'},
            limit_to='foo')
        worker.start()
        while not worker.isFinished():
            QTest.qWait(50)
        self.assertEqual(worker.warnings, [])
        self.assertEqual(worker.error, None)

    def test_error(self):
        from natcap.invest.ui.inputs import ValidationWorker
        _callable = mock.Mock(side_effect=KeyError('missing'))
        worker = ValidationWorker(
            target=_callable,
            args={'foo': 'bar'},
            limit_to='foo')
        worker.start()
        while not worker.isFinished():
            QTest.qWait(50)
        self.assertEqual(worker.warnings, [])
        self.assertEqual(worker.error, "'missing'")


class FileButtonTest(_QtTest):
    def test_button_clicked(self):
        from natcap.invest.ui.inputs import FileButton
        button = FileButton('Some title')

        # Patch up the open_method to return a known path.
        # Would block on user input otherwise.
        button.open_method = mock.Mock(return_value='/some/path')
        _callback = mock.Mock()
        button.path_selected.connect(_callback)

        QTest.mouseClick(button, QtCore.Qt.LeftButton)
        self.qt_app.processEvents()

        _callback.assert_called_with('/some/path')
        self.qt_app.processEvents()

    def test_button_title(self):
        from natcap.invest.ui.inputs import FileButton
        button = FileButton('Some title')
        self.assertEqual(button.dialog_title, 'Some title')


class FolderButtonTest(_QtTest):
    def test_button_clicked(self):
        from natcap.invest.ui.inputs import FolderButton
        button = FolderButton('Some title')

        # Patch up the open_method to return a known path.
        # Would block on user input otherwise.
        button.open_method = mock.Mock(return_value='/some/path')
        _callback = mock.Mock()
        button.path_selected.connect(_callback)

        QTest.mouseClick(button, QtCore.Qt.LeftButton)
        self.qt_app.processEvents()

        _callback.assert_called_with('/some/path')

    def test_button_title(self):
        from natcap.invest.ui.inputs import FolderButton
        button = FolderButton('Some title')
        self.assertEqual(button.dialog_title, 'Some title')


class FileDialogTest(_SettingsSandbox):
    def test_save_file_title_and_last_selection(self):
        from natcap.invest.ui.inputs import FileDialog, INVEST_SETTINGS
        dialog = FileDialog()
        dialog.file_dialog.getSaveFileName = mock.Mock(
            spec=dialog.file_dialog.getSaveFileName,
            return_value='/new/file')

        INVEST_SETTINGS.setValue('last_dir', '/tmp/foo/bar')

        out_file = dialog.save_file(title='foo', start_dir=None)
        self.assertEqual(
            dialog.file_dialog.getSaveFileName.call_args[0],  # pos. args
            (dialog.file_dialog, 'foo', '/tmp/foo/bar'))
        self.assertEqual(out_file, '/new/file')
        self.assertEqual(INVEST_SETTINGS.value('last_dir', ''),
                         u'/new')

    def test_save_file_defined_savefile(self):
        from natcap.invest.ui.inputs import FileDialog
        dialog = FileDialog()
        dialog.file_dialog.getSaveFileName = mock.Mock(
            spec=dialog.file_dialog.getSaveFileName,
            return_value=os.path.join('/new','file'))

        out_file = dialog.save_file(title='foo', start_dir='/tmp',
                                    savefile='file.txt')
        self.assertEqual(
            dialog.file_dialog.getSaveFileName.call_args[0],  # pos. args
            (dialog.file_dialog, 'foo', os.path.join('/tmp', 'file.txt')))

        self.assertEqual(out_file, os.path.join('/new', 'file'))

    def test_open_file_qt5(self):
        from natcap.invest.ui.inputs import FileDialog, INVEST_SETTINGS
        dialog = FileDialog()

        # patch up the Qt method to get the path to the file to open
        # Qt4 and Qt5 have different return values.  Mock up accordingly.
        # Simulate Qt5 return value.
        try:
            _old_qtpy_version = qtpy.QT_VERSION
            qtpy.QT_VERSION = ('5', '5', '5')
            dialog.file_dialog.getOpenFileName = mock.Mock(
                spec=dialog.file_dialog.getOpenFileName,
                return_value=('/new/file', 'filter'))

            INVEST_SETTINGS.setValue('last_dir', '/tmp/foo/bar')
            out_file = dialog.open_file(title='foo')
        finally:
            qtpy.QT_VERSION = _old_qtpy_version

        self.assertEqual(
            dialog.file_dialog.getOpenFileName.call_args[0],  # pos. args
            (dialog.file_dialog, 'foo', '/tmp/foo/bar', ''))
        self.assertEqual(out_file, '/new/file')
        self.assertEqual(INVEST_SETTINGS.value('last_dir', ''), '/new')

    def test_open_file_qt4(self):
        from natcap.invest.ui.inputs import FileDialog, INVEST_SETTINGS
        dialog = FileDialog()

        # patch up the Qt method to get the path to the file to open
        # Qt4 and Qt5 have different return values.  Mock up accordingly.
        # Simulate Qt4 return value.
        try:
            _old_qtpy_version = qtpy.QT_VERSION
            qtpy.QT_VERSION = ('4', '5', '6')
            dialog.file_dialog.getOpenFileName = mock.Mock(
                spec=dialog.file_dialog.getOpenFileName,
                return_value='/new/file')

            INVEST_SETTINGS.setValue('last_dir', '/tmp/foo/bar')
            out_file = dialog.open_file(title='foo')
        finally:
            qtpy.QT_VERSION = _old_qtpy_version

        self.assertEqual(
            dialog.file_dialog.getOpenFileName.call_args[0],  # pos. args
            (dialog.file_dialog, 'foo', '/tmp/foo/bar', ''))
        self.assertEqual(out_file, '/new/file')
        self.assertEqual(INVEST_SETTINGS.value('last_dir', ''), '/new')

    def test_open_folder(self):
        from natcap.invest.ui.inputs import FileDialog, INVEST_SETTINGS
        dialog = FileDialog()

        # patch up the Qt method to get the path to the file to open
        dialog.file_dialog.getExistingDirectory = mock.Mock(
            spec=dialog.file_dialog.getExistingDirectory,
            return_value='/existing/folder')

        INVEST_SETTINGS.setValue('last_dir', '/tmp/foo/bar')
        new_folder = dialog.open_folder('foo', start_dir=None)

        self.assertEqual(dialog.file_dialog.getExistingDirectory.call_args[0],
                         (dialog.file_dialog, 'Select folder: foo', '/tmp/foo/bar'))
        self.assertEqual(new_folder, '/existing/folder')
        self.assertEqual(INVEST_SETTINGS.value('last_dir', ''),
                         '/existing/folder')


class InfoButtonTest(_QtTest):
    def test_buttonpress(self):
        from natcap.invest.ui.inputs import InfoButton
        button = InfoButton('some text')
        self.assertEqual(button.whatsThis(), 'some text')

        # Necessary to mock up the QWhatsThis module because it always
        # segfaults in a test if I don't.  Haven't yet been able to figure out
        # why or how to work around, and this allows me to have the test
        # coverage.
        with mock.patch('qtpy.QtWidgets.QWhatsThis'):
            button.show()
            QTest.mouseClick(button, QtCore.Qt.LeftButton)


class FormTest(_QtTest):
    @staticmethod
    def validate(args, limit_to=None):
        return []

    @staticmethod
    def execute(args, limit_to=None):
        pass

    @staticmethod
    def make_ui():
        from natcap.invest.ui.inputs import Form

        return Form()

    def test_run_button_pressed(self):
        form = FormTest.make_ui()
        mock_object = mock.Mock()
        form.submitted.connect(mock_object)

        QTest.mouseClick(form.run_button,
                         QtCore.Qt.LeftButton)

        self.qt_app.processEvents()
        mock_object.assert_called_once()

    def test_run_noerror(self):
        form = FormTest.make_ui()
        def _target():
            return
        with wait_on_signal(self.qt_app, form.run_finished, timeout=250):
            form.run(target=_target)

        self.qt_app.processEvents()
        # At the end of the run, the button should be visible.
        self.assertTrue(form.run_dialog.openWorkspaceButton.isVisible())

        # close the window by pressing the back button.
        QTest.mouseClick(form.run_dialog.backButton,
                         QtCore.Qt.LeftButton)

    def test_run_target_error(self):
        form = FormTest.make_ui()
        with self.assertRaises(ValueError):
            form.run(target='str does not have a __call__()')

    def test_open_workspace_on_success(self):
        class _SampleTarget(object):
            @staticmethod
            def validate(args, limit_to=None):
                return []

            @staticmethod
            def execute(args):
                pass

        form = FormTest.make_ui()
        target = _SampleTarget().execute

        # patch open_workspace to avoid lots of open file dialogs.
        with mock.patch('natcap.invest.ui.inputs.open_workspace',
                        mock.Mock(return_value=None)) as open_workspace:
            with wait_on_signal(self.qt_app, form.run_finished):
                form.run(target=target)

                self.assertTrue(form.run_dialog.openWorkspaceCB.isVisible())
                self.assertFalse(
                    form.run_dialog.openWorkspaceButton.isVisible())

                form.run_dialog.openWorkspaceCB.setChecked(True)
                self.assertTrue(form.run_dialog.openWorkspaceCB.isChecked())

        def _close_modal_dialog():
            # close the window by pressing the back button.
            QTest.mouseClick(form.run_dialog.backButton,
                             QtCore.Qt.LeftButton)

        QtCore.QTimer.singleShot(0, _close_modal_dialog)
        self.qt_app.processEvents()
        open_workspace.assert_called()

    def test_run_prevent_dialog_close_esc(self):
        thread_event = threading.Event()

        class _SampleTarget(object):
            @staticmethod
            def validate(args, limit_to=None):
                return []

            @staticmethod
            def execute(args):
                thread_event.wait()

        target_mod = _SampleTarget().execute
        form = FormTest.make_ui()
        form.run(target=target_mod)
        QTest.keyPress(form.run_dialog, QtCore.Qt.Key_Escape)
        self.assertTrue(form.run_dialog.isVisible())

        # when the execute function finishes, pressing escape should
        # close the window.
        thread_event.set()
        QTest.keyPress(form.run_dialog, QtCore.Qt.Key_Escape)
        self.assertEqual(form.run_dialog.result(), QtWidgets.QDialog.Rejected)
        self.assertEqual(form.run_dialog.result(), QtWidgets.QDialog.Rejected)

    def test_run_prevent_dialog_close_event(self):
        thread_event = threading.Event()

        class _SampleTarget(object):
            @staticmethod
            def validate(args, limit_to=None):
                return []

            @staticmethod
            def execute(args):
                thread_event.wait()

        form = FormTest.make_ui()
        target_mod = _SampleTarget().execute
        try:
            form.run(target=target_mod, kwargs={'args': {'a': 1}})
            if self.qt_app.hasPendingEvents():
                self.qt_app.processEvents()
            self.assertTrue(form.run_dialog.isVisible())
            form.run_dialog.close()
            if self.qt_app.hasPendingEvents():
                self.qt_app.processEvents()
            self.assertTrue(form.run_dialog.isVisible())

            # when the execute function finishes, pressing escape should
            # close the window.
            thread_event.set()
            form._thread.join()
            if self.qt_app.hasPendingEvents():
                self.qt_app.processEvents()
            form.run_dialog.close()
            if self.qt_app.hasPendingEvents():
                self.qt_app.processEvents()
            self.assertFalse(form.run_dialog.isVisible())
        except Exception as error:
            LOGGER.exception('Something failed')
            # If something happens while executing, be sure the thread executes
            # cleanly.
            thread_event.set()
            form._thread.join()
            self.fail(error)

    def test_run_error(self):
        class _SampleTarget(object):
            @staticmethod
            def validate(args, limit_to=None):
                return []

            @staticmethod
            def execute(args):
                raise RuntimeError('Something broke!')

        target_mod = _SampleTarget().execute
        form = FormTest.make_ui()
        form.run(target=target_mod, kwargs={'args': {}})
        form._thread.join()
        if self.qt_app.hasPendingEvents():
            self.qt_app.processEvents()

        self.assertTrue('encountered' in form.run_dialog.messageArea.text())

    def test_show(self):
        form = FormTest.make_ui()
        form.show()

    def test_resize_scrollbar(self):
        form = FormTest.make_ui()
        form.show()
        self.assertTrue('border: None' in form.scroll_area.styleSheet())
        form.scroll_area.update_scroll_border(50, 50)  # simulate form resize
        self.assertTrue(len(form.scroll_area.styleSheet()) == 0)

    def test_add_input(self):
        from natcap.invest.ui import inputs
        form = FormTest.make_ui()
        text_input = inputs.Text('hello there')
        form.add_input(text_input)


class OpenWorkspaceTest(_QtTest):
    def test_windows(self):
        from natcap.invest.ui.inputs import open_workspace
        with mock.patch('natcap.invest.ui.inputs.subprocess.Popen') as method:
            with mock.patch('platform.system', return_value='Windows'):
                with mock.patch('os.path.normpath', return_value='/foo\\bar'):
                    open_workspace(os.path.join('/foo', 'bar'))
                    method.assert_called_with('explorer "/foo\\bar"')

    def test_mac(self):
        from natcap.invest.ui.inputs import open_workspace
        with mock.patch('natcap.invest.ui.inputs.subprocess.Popen') as method:
            with mock.patch('platform.system', return_value='Darwin'):
                with mock.patch('os.path.normpath', return_value='/foo/bar'):
                    open_workspace('/foo/bar')
                    method.assert_called_with('open /foo/bar', shell=True)

    def test_linux(self):
        from natcap.invest.ui.inputs import open_workspace
        with mock.patch('natcap.invest.ui.inputs.subprocess.Popen') as method:
            with mock.patch('platform.system', return_value='Linux'):
                with mock.patch('os.path.normpath', return_value='/foo/bar'):
                    open_workspace('/foo/bar')
                    method.assert_called_with(['xdg-open', '/foo/bar'])

    def test_error_in_subprocess(self):
        from natcap.invest.ui.inputs import open_workspace
        popen_import_path = 'natcap.invest.ui.inputs.subprocess.Popen'
        normpath_import_path = 'natcap.invest.ui.inputs.os.path.normpath'
        with mock.patch(popen_import_path,
                        side_effect=OSError('error message')) as patch:
            with mock.patch(normpath_import_path, return_value='/foo/bar'):
                open_workspace('/foo/bar')
                patch.assert_called()


class ExecutionTest(_QtTest):
    def test_executor_run(self):
        from natcap.invest.ui.execution import Executor

        thread_event = threading.Event()

        def _waiting_func(*args, **kwargs):
            thread_event.wait()

        target = mock.Mock(wraps=_waiting_func)
        callback = mock.Mock()
        args = ('a', 'b', 'c')
        kwargs = {'d': 1, 'e': 2, 'f': 3}

        executor = Executor(
            target=target,
            args=args,
            kwargs=kwargs)

        self.assertEqual(executor.target, target)
        self.assertEqual(executor.args, args)
        self.assertEqual(executor.kwargs, kwargs)

        # register the callback with the finished signal.
        executor.finished.connect(callback)

        executor.start()
        thread_event.set()
        executor.join()
        if self.qt_app.hasPendingEvents():
            self.qt_app.processEvents()
        callback.assert_called_once()
        target.assert_called_once()
        target.assert_called_with(*args, **kwargs)

    def test_executor_exception(self):
        from natcap.invest.ui.execution import Executor

        thread_event = threading.Event()

        def _waiting_func(*args, **kwargs):
            thread_event.wait()
            raise ValueError('Some demo exception')

        target = mock.Mock(wraps=_waiting_func)
        callback = mock.Mock()
        args = ('a', 'b', 'c')
        kwargs = {'d': 1, 'e': 2, 'f': 3}

        executor = Executor(
            target=target,
            args=args,
            kwargs=kwargs)

        self.assertEqual(executor.target, target)
        self.assertEqual(executor.args, args)
        self.assertEqual(executor.kwargs, kwargs)

        # register the callback with the finished signal.
        executor.finished.connect(callback)

        executor.start()
        thread_event.set()
        executor.join()
        if self.qt_app.hasPendingEvents():
            self.qt_app.processEvents()
        callback.assert_called_once()
        target.assert_called_once()
        target.assert_called_with(*args, **kwargs)

        self.assertTrue(executor.failed)
        self.assertEqual(str(executor.exception),
                         'Some demo exception')
        self.assertTrue(isinstance(executor.traceback, basestring))

    def test_default_args(self):
        from natcap.invest.ui.execution import Executor

        executor = Executor(target=mock.Mock())

        # We didn't define args or kwargs (which default to None), so verify
        # that the parameters are set correctly.
        self.assertEqual(executor.args, ())
        self.assertEqual(executor.kwargs, {})


class IntegrationTests(_QtTest):
    def test_checkbox_enables_collapsible_container(self):
        from natcap.invest.ui import inputs
        checkbox = inputs.Checkbox(label='Trigger')
        container = inputs.Container(label='Container',
                                     expandable=True,
                                     expanded=False,
                                     interactive=False)
        # Interactivity of the contained file is dependent upon the
        # collapsed state of the container.
        contained_file = inputs.File(label='some file input')
        container.add_input(contained_file)
        checkbox.value_changed.connect(container.set_interactive)

        # Assert everything starts out fine.
        self.assertTrue(checkbox.interactive)
        self.assertFalse(container.interactive)
        self.assertFalse(container.expanded)
        self.assertFalse(contained_file.interactive)
        self.assertFalse(contained_file.visible())

        # When the checkbox is enabled, the container should become enabled,
        # but the container's contained widgets should still be noninteractive
        checkbox.set_value(True)
        if self.qt_app.hasPendingEvents():
            self.qt_app.processEvents()

        self.assertTrue(container.interactive)
        self.assertFalse(container.expanded)
        self.assertFalse(contained_file.interactive)
        self.assertFalse(contained_file.visible())

        # When the container is expanded, the contained input should become
        # interactive and visible
        container.set_value(True)
        if self.qt_app.hasPendingEvents():
            self.qt_app.processEvents()

        self.assertTrue(container.interactive)
        self.assertTrue(container.expanded)
        self.assertTrue(contained_file.interactive)
        self.assertTrue(contained_file.visible())


class OptionsDialogTest(_QtTest):
    def test_postprocess_not_implemented_coverage(self):
        """UI OptionsDialog: Coverage for postprocess method."""
        from natcap.invest.ui import model
        from natcap.invest.ui import inputs

        options_dialog = model.OptionsDialog()
        options_dialog.open()
        options_dialog.accept()
        self.qt_app.processEvents()

    def test_postprocess_not_implemented(self):
        """UI OptionsDialog: postprocess() raises NotImplementedError."""
        from natcap.invest.ui import model

        options_dialog = model.OptionsDialog()
        with self.assertRaises(NotImplementedError):
            options_dialog.postprocess(0)


class SettingsDialogTest(_SettingsSandbox):
    def test_cache_dir_initialized_correctly(self):
        """UI SettingsDialog: check initialization of inputs."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()
        try:
            # Qt4
            cache_dir = QtGui.QDesktopServices.storageLocation(
                QtGui.QDesktopServices.CacheLocation)
        except AttributeError:
            # Package location changed in Qt5
            cache_dir = QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.CacheLocation)

        self.assertEqual(settings_dialog.cache_directory.value(),
                         cache_dir)

    def test_cache_dir_setting_set_correctly(self):
        """UI SettingsDialog: check settings values when input changed."""
        from natcap.invest.ui import model
        from natcap.invest.ui import inputs

        settings_dialog = model.SettingsDialog()
        settings_dialog.show()
        settings_dialog.cache_directory.set_value('new_dir')
        QTest.mouseClick(settings_dialog.ok_button,
                         QtCore.Qt.LeftButton)
        self.qt_app.processEvents()
        try:
            self.assertEqual(settings_dialog.cache_directory.value(),
                             'new_dir')
        finally:
            settings_dialog.close()

    def test_dialog_log_level_initialized(self):
        """UI SettingsDialog: check initialization of dialog log level."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()
        self.assertEqual(settings_dialog.dialog_logging_level.value(),
                         'INFO')


    def test_dialog_log_level_set_correctly(self):
        """UI SettingsDialog: check settings value for dialog threshold."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()

        settings_dialog.show()
        settings_dialog.dialog_logging_level.set_value('CRITICAL')
        QTest.mouseClick(settings_dialog.ok_button,
                         QtCore.Qt.LeftButton)
        self.qt_app.processEvents()
        try:
            self.assertEqual(settings_dialog.dialog_logging_level.value(),
                             'CRITICAL')
        finally:
            settings_dialog.close()

    def test_logfile_log_level_initialized(self):
        """UI SettingsDialog: check initialization of logfile log level."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()
        self.assertEqual(settings_dialog.logfile_logging_level.value(),
                         'NOTSET')

    def test_logfile_log_level_set_correctly(self):
        """UI SettingsDialog: check settings value for logfile threshold."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()

        settings_dialog.show()
        settings_dialog.logfile_logging_level.set_value('CRITICAL')
        QTest.mouseClick(settings_dialog.ok_button,
                         QtCore.Qt.LeftButton)
        self.qt_app.processEvents()
        try:
            self.assertEqual(settings_dialog.logfile_logging_level.value(),
                             'CRITICAL')
        finally:
            settings_dialog.close()

    def test_taskgraph_log_level_initialized(self):
        """UI SettingsDialog: check initialization of taskgraph log level."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()
        self.assertEqual(settings_dialog.taskgraph_logging_level.value(),
                         'ERROR')

    def test_taskgraph_log_level_set_correctly(self):
        """UI SettingsDialog: check settings value for taskgraph threshold."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()

        settings_dialog.show()
        settings_dialog.taskgraph_logging_level.set_value('CRITICAL')
        QTest.mouseClick(settings_dialog.ok_button,
                         QtCore.Qt.LeftButton)
        self.qt_app.processEvents()
        try:
            self.assertEqual(settings_dialog.taskgraph_logging_level.value(),
                             'CRITICAL')
        finally:
            settings_dialog.close()

    def test_n_workers_initialized(self):
        """UI SettingsDialog: check initialization of n_workers."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()
        self.assertEqual(settings_dialog.taskgraph_n_workers.value(),
                         '-1')

    def test_n_workers_set_correctly(self):
        """UI SettingsDialog: check settings value for n_workers."""
        from natcap.invest.ui import model

        settings_dialog = model.SettingsDialog()

        settings_dialog.show()
        settings_dialog.taskgraph_n_workers.set_value('3')
        QTest.mouseClick(settings_dialog.ok_button,
                         QtCore.Qt.LeftButton)
        self.qt_app.processEvents()
        try:
            self.assertEqual(settings_dialog.taskgraph_n_workers.value(),
                             '3')
        finally:
            settings_dialog.close()


class DatastackOptionsDialogTests(_QtTest):
    def setUp(self):
        _QtTest.setUp(self)
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        _QtTest.tearDown(self)
        shutil.rmtree(self.workspace)

    def test_dialog_save_button_enable(self):
        """UI Datastack Options: verify save button enable/disable."""
        from natcap.invest.ui import model

        options_dialog = model.DatastackOptionsDialog(
            paramset_basename='test_model')
        new_paramset_path = os.path.join(
            self.workspace, 'testdir1', 'test.invs.json')

        options_dialog.save_parameters.set_value(new_paramset_path)
        self.assertTrue(options_dialog.ok_button.isEnabled())

        # whitespace and empty string are the only values that disable
        invalids = ['  ', '']
        for val in invalids:
            options_dialog.save_parameters.set_value(val)
            self.assertFalse(options_dialog.ok_button.isEnabled())

    def test_dialog_return_value(self):
        """UI Datastack Options: Verify return value of dialog."""
        from natcap.invest.ui import model
        from natcap.invest.ui import inputs

        options_dialog = model.DatastackOptionsDialog(
            paramset_basename='test_model')

        # set this option to ensure coverage of the slot
        options_dialog.datastack_type.set_value(model._DATASTACK_DATA_ARCHIVE)
        options_dialog.datastack_type.set_value(model._DATASTACK_PARAMETER_SET)
        self.qt_app.processEvents()

        new_paramset_path = os.path.join(self.workspace, 'test.invs.json')
        options_dialog.save_parameters.set_value(new_paramset_path)

        def _press_accept():
            options_dialog.accept()

        QtCore.QTimer.singleShot(25, _press_accept)
        return_options = options_dialog.exec_()

        self.assertEqual(
            model.DatastackSaveOpts(
                model._DATASTACK_PARAMETER_SET,  # datastack type
                False,  # use relative paths
                False,  # include workpace
                new_paramset_path),  # datastack path
            return_options)

        # verify that the options have been cleared.
        for input_obj, expected_value in (
                (options_dialog.datastack_type,
                 options_dialog.datastack_type.options[0]),
                (options_dialog.use_relative_paths, False),
                (options_dialog.include_workspace, False),
                (options_dialog.save_parameters, '')):
            self.assertEqual(input_obj.value(), expected_value)


    def test_dialog_cancelled(self):
        """UI Datastack Options: Verify return value when dialog cancelled."""
        from natcap.invest.ui import model
        from natcap.invest.ui import inputs

        options_dialog = model.DatastackOptionsDialog(
            paramset_basename='test_model')

        # set this option to ensure coverage of the slot
        options_dialog.datastack_type.set_value(model._DATASTACK_DATA_ARCHIVE)
        options_dialog.datastack_type.set_value(model._DATASTACK_PARAMETER_SET)
        self.qt_app.processEvents()

        def _press_accept():
            options_dialog.reject()

        QtCore.QTimer.singleShot(25, _press_accept)
        return_options = options_dialog.exec_()

        self.assertEqual(return_options, None)

@unittest.skip('These tests are segfaulting. '
               'See github.com/natcap/invest/issues/72')
class ModelTests(_QtTest):
    def setUp(self):
        _QtTest.setUp(self)
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        _QtTest.tearDown(self)
        shutil.rmtree(self.workspace)

    @staticmethod
    def build_model(validate_func=None, target_func=None):
        from natcap.invest.ui import model
        from natcap.invest import validation

        if target_func is None:
            def _target(args):
                pass
            target_func = _target

        if validate_func is None:
            @validation.invest_validator
            def _validate(args, limit_to=None):
                return []
            validate_func = _validate

        # Fetch settings and clear them before the UI constructs.
        # This avoids a scenario where invalid settings have been
        # constructed and not destroyed properly before the test model object
        # could complete its setup, causing lots of test failures.
        label = 'Test model'
        settings = model.SETTINGS_TEMPLATE(label)
        settings.clear()

        class _TestInVESTModel(model.InVESTModel):
            def __init__(self):
                model.InVESTModel.__init__(
                    self,
                    label=label,
                    target=target_func,
                    validator=validate_func,
                    localdoc='testmodel.html')

            # Default model class already has workspace and suffix input.
            def assemble_args(self):
                return {
                    self.workspace.args_key: self.workspace.value(),
                    self.suffix.args_key: self.suffix.value(),
                }

            def __del__(self):
                # clear the settings for future runs.
                try:
                    self.settings.clear()
                except RuntimeError:
                    pass

        return _TestInVESTModel()

    def test_model_defaults(self):
        """UI Model: Check that workspace and suffix are added."""
        from natcap.invest.ui import inputs

        model_ui = ModelTests.build_model()
        try:
            workspace_input = getattr(model_ui, 'workspace')
            self.assertTrue(isinstance(workspace_input, inputs.Folder))

            suffix_input = getattr(model_ui, 'suffix')
            self.assertTrue(isinstance(suffix_input, inputs.Text))
        except AttributeError as missing_input:
            self.fail(str(missing_input))
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_lastrun(self):
        """UI Model: Check that lastrun saving/loading works."""
        model_ui = ModelTests.build_model()
        try:
            # Set input values and save the lastrun.
            model_ui.workspace.set_value('foo')
            model_ui.suffix.set_value('bar')
            model_ui.save_lastrun()

            # change the input values
            model_ui.workspace.set_value('new workspace')
            model_ui.suffix.set_value('new suffix')
            self.assertEqual(model_ui.workspace.value(), 'new workspace')
            self.assertEqual(model_ui.suffix.value(), 'new suffix')

            # load the values from lastrun and assert that the values are correct.
            model_ui.load_lastrun()
            self.assertEqual(model_ui.workspace.value(), 'foo')
            self.assertEqual(model_ui.suffix.value(), 'bar')
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_close_window_confirm(self):
        """UI Model: Close confirmation dialog 'remember lastrun' checkbox."""
        model_ui = ModelTests.build_model()
        try:
            model_ui.show()
            try:
                QTest.qWaitForWindowShown(model_ui)
            except AttributeError:
                # pyqt5 has different wait methods.
                QTest.qWaitForWindowExposed(model_ui)

            threading_event = threading.Event()

            def _tests():
                # verify 'remember inputs' is checked by default.
                self.assertTrue(model_ui.quit_confirm_dialog.checkbox.isChecked())

                # click yes.
                QTest.mouseClick(
                    model_ui.quit_confirm_dialog.button(QtWidgets.QMessageBox.Yes),
                    QtCore.Qt.LeftButton)

                threading_event.set()

            QtCore.QTimer.singleShot(25, _tests)
            model_ui.close()
            self.qt_app.processEvents()

            threading_event.wait(0.5)
            self.qt_app.processEvents()

            # verify the 'remember inputs' state
            self.assertEqual(model_ui.settings.value('remember_lastrun'),
                             True)
            self.assertFalse(model_ui.quit_confirm_dialog.isVisible())
            self.assertFalse(model_ui.isVisible())
        finally:
            model_ui.destroy()

    def test_close_window_cancel(self):
        """UI Model: Close confirmation dialog cancel"""
        model_ui = ModelTests.build_model()
        model_ui.show()

        threading_event = threading.Event()

        def _tests():
            # click cancel.
            button = QtWidgets.QMessageBox.Cancel
            QTest.mouseClick(
                model_ui.quit_confirm_dialog.button(button),
                QtCore.Qt.LeftButton)
            threading_event.set()

        QtCore.QTimer.singleShot(25, _tests)
        model_ui.close()
        threading_event.wait()

        self.assertFalse(model_ui.quit_confirm_dialog.isVisible())
        self.assertTrue(model_ui.isVisible())
        # then close it for real so it doesn't hang around
        model_ui.close(prompt=False)
        model_ui.destroy()

    def test_validation_passes(self):
        """UI Model: Check what happens when validation passes."""
        from natcap.invest import validation
        from natcap.invest.ui import inputs

        @validation.invest_validator
        def _sample_validate(args, limit_to=None):
            # no validation errors!
            return []

        model_ui = ModelTests.build_model(_sample_validate)
        try:
            model_ui.show()

            model_ui.validate(block=True)
            self.qt_app.processEvents()
            self.assertEqual(len(model_ui.validation_report_dialog.warnings), 0)
            self.assertTrue(model_ui.is_valid())
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_validate_blocking(self):
        """UI Model: Validate that the blocking validation call works."""
        from natcap.invest import validation
        from natcap.invest.ui import inputs

        @validation.invest_validator
        def _sample_validate(args, limit_to=None):
            return [(('workspace_dir',), 'some error')]

        model_ui = ModelTests.build_model(_sample_validate)
        try:
            model_ui.show()

            model_ui.validate(block=True)
            self.qt_app.processEvents()
            self.assertEqual(len(model_ui.validation_report_dialog.warnings), 1)
            self.assertFalse(model_ui.is_valid())
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_validate_nonblocking(self):
        """UI Model: Validate that the nonblocking validation call works."""
        from natcap.invest import validation
        from natcap.invest.ui import inputs

        @validation.invest_validator
        def _sample_validate(args, limit_to=None):
            return [(('workspace_dir',), 'some error')]

        model_ui = ModelTests.build_model(_sample_validate)
        try:
            model_ui.show()

            model_ui.validate(block=False)
            self.qt_app.processEvents()
            self.assertEqual(len(model_ui.validation_report_dialog.warnings), 1)
            self.assertFalse(model_ui.is_valid())
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_assemble_args_not_implemented(self):
        """UI Model: Validate exception when assemble_args not implemented."""
        from natcap.invest.ui import model

        with self.assertRaises(NotImplementedError):
            try:
                model_ui = model.InVESTModel(
                    label='foo',
                    target=lambda args: None,
                    validator=lambda args, limit_to=None: [],
                    localdoc='sometextfile.html'
                )
                model_ui.assemble_args()
            except Exception:
                LOGGER.exception('Could not create the model UI')
                raise
            finally:
                model_ui.close(prompt=False)
                model_ui.destroy()

    def test_load_args(self):
        """UI Model: Check that we can load args as expected."""
        model_ui = ModelTests.build_model()
        try:
            args = {
                'workspace_dir': 'new workspace!',
                'results_suffix': 'a',
            }
            model_ui.load_args(args)
            self.assertEqual(model_ui.workspace.value(), args['workspace_dir'])
            self.assertEqual(model_ui.suffix.value(), args['results_suffix'])
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_args_bad_key(self):
        """UI Model: Check that we can handle loading of bad keys."""
        model_ui = ModelTests.build_model()
        try:
            model_ui.workspace.set_value('')
            args = {
                'bad_key': 'something unexpected!',
                'results_suffix': 'a',
            }
            model_ui.load_args(args)
            self.assertEqual(model_ui.workspace.value(), '')  # was never changed
            self.assertEqual(model_ui.suffix.value(), args['results_suffix'])
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_args_error(self):
        """UI Model: Check that we can handle errors when loading args."""
        model_ui = ModelTests.build_model()
        try:
            model_ui.workspace.set_value('')
            args = {
                'workspace_dir': 'workspace',
                'results_suffix': 'a',
            }

            def _raise_valueerror(new_value):
                raise ValueError('foo!')

            model_ui.workspace.set_value = _raise_valueerror
            model_ui.load_args(args)

            self.assertEqual(model_ui.workspace.value(), '')  # was never changed
            self.assertEqual(model_ui.suffix.value(), args['results_suffix'])
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_run(self):
        """UI Model: Check that we can run the model."""
        from natcap.invest.ui import inputs
        model_ui = ModelTests.build_model()
        try:
            model_ui.test_container = inputs.Container('test')
            model_ui.add_input(model_ui.test_container)

            def _close_window():
                # trigger whole-model validation for coverage of callback.
                model_ui.workspace.set_value('foo')

                self.qt_app.processEvents()

                model_ui.close(prompt=False)

            model_ui.run()
            self.assertTrue(model_ui.isVisible())
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_execute_with_n_workers(self):
        """UI Model: Check that model runs with n_workers parameter."""

        from natcap.invest.ui import inputs

        n_workers_setting = inputs.INVEST_SETTINGS.value(
            'taskgraph/n_workers', '-1')

        def target_func(args):
            """n_workers is required in args."""
            if 'n_workers' not in args:
                raise ValueError('n_workers should be in args but is not.')

            if args['n_workers'] != n_workers_setting:
                raise ValueError('n_workers value is not set correctly')

        model_ui = ModelTests.build_model(target_func=target_func)
        model_ui.workspace.set_value(os.path.join(self.workspace, 'new_dir'))

        try:
            # Show the window
            model_ui.run()
            self.assertTrue(model_ui.isVisible())

            # This should execute without exception.
            model_ui.execute_model()

            # I don't particularly like spinning here, but it should be
            # reliable.
            while model_ui.form.run_dialog.is_executing:
                self.qt_app.processEvents()
                time.sleep(0.1)
            self.assertFalse(model_ui.form._thread.failed)
        finally:
            model_ui.form.run_dialog.close()
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_execute_error_with_n_workers(self):
        """UI Model: Check that model fails with n_workers parameter."""
        from natcap.invest.ui import inputs, model
        from natcap.invest import validation

        n_workers_setting = inputs.INVEST_SETTINGS.value(
            'taskgraph/n_workers', 1)

        def target_func(args):
            raise AssertionError(
                'Target should not have been called due to n_workers being '
                'in args.')

        @validation.invest_validator
        def _validate(args, limit_to=None):
            return []

        class _TestInVESTModel(model.InVESTModel):
            def __init__(self):
                model.InVESTModel.__init__(
                    self,
                    label='Test model',
                    target=target_func,
                    validator=_validate,
                    localdoc='testmodel.html')

            # Default model class already has workspace and suffix input.
            def assemble_args(self):
                return {
                    self.workspace.args_key: self.workspace.value(),
                    self.suffix.args_key: self.suffix.value(),
                    'n_workers': n_workers_setting  # this should cause an error.
                }

            def __del__(self):
                # clear the settings for future runs.
                try:
                    self.settings.clear()
                except RuntimeError:
                    pass

        model_ui = _TestInVESTModel()
        model_ui.workspace.set_value(os.path.join(self.workspace, 'new_dir'))

        try:
            # Show the window
            model_ui.run()
            self.assertTrue(model_ui.isVisible())

            # This should execute without exception.
            model_ui.execute_model()

            # I don't particularly like spinning here, but it should be
            # reliable.
            while model_ui.form.run_dialog.is_executing:
                self.qt_app.processEvents()
                time.sleep(0.1)
            self.assertTrue(model_ui.form._thread.failed)

            # We expect RuntimeError to be raised by the UI in the function
            # that wraps the target, not by the target itself.
            self.assertTrue(isinstance(model_ui.form._thread.exception,
                                       RuntimeError))
            self.assertTrue('n_workers defined in args.' in
                            repr(model_ui.form._thread.exception))
        finally:
            model_ui.form.run_dialog.close()
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_local_docs_from_hyperlink(self):
        """UI Model: Check that we can open the local docs missing dialog."""
        model_ui = ModelTests.build_model()
        try:
            model_ui.run()

            def _check_dialog_and_close():
                self.assertTrue(model_ui.local_docs_missing_dialog.isVisible())
                model_ui.local_docs_missing_dialog.accept()

            QtCore.QTimer.singleShot(25, _check_dialog_and_close)

            if hasattr(QtCore, 'QDesktopServices'):
                patch_object = mock.patch('natcap.invest.ui.inputs'
                                          '.QtCore.QDesktopServices.openUrl')
            else:
                # PyQt5 changed the location of this.
                patch_object = mock.patch('natcap.invest.ui.inputs'
                                          '.QtGui.QDesktopServices.openUrl')

            # simulate a mouse click on the localdocs hyperlink.
            with patch_object as patched_openurl:
                # simulate a mouse click on the localdocs hyperlink.
                model_ui._check_local_docs('localdocs')
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_local_docs_launch(self):
        """UI Model: Check that we can launch local documentation."""
        model_ui = ModelTests.build_model()
        try:
            model_ui.run()

            if hasattr(QtCore, 'QDesktopServices'):
                patch_object = mock.patch('natcap.invest.ui.inputs'
                                          '.QtCore.QDesktopServices.openUrl')
            else:
                # PyQt5 changed the location of this.
                patch_object = mock.patch('natcap.invest.ui.inputs'
                                          '.QtGui.QDesktopServices.openUrl')

            # simulate a mouse click on the localdocs hyperlink.
            with patch_object as patched_openurl:
                with mock.patch('os.path.exists', return_value=True):
                    # simulate about --> view documentation menu.
                    model_ui._check_local_docs('http://some_file_that_exists')

            patched_openurl.assert_called_once_with(
                QtCore.QUrl('http://some_file_that_exists'))
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_forums_link_launch(self):
        """UI Model: Check that we can link to the forums."""
        model_ui = ModelTests.build_model()
        try:
            model_ui.run()

            if hasattr(QtCore, 'QDesktopServices'):
                patch_object = mock.patch('natcap.invest.ui.inputs'
                                          '.QtCore.QDesktopServices.openUrl')
            else:
                # PyQt5 changed the location of this.
                patch_object = mock.patch('natcap.invest.ui.inputs'
                                          '.QtGui.QDesktopServices.openUrl')

            # simulate a mouse click on the localdocs hyperlink.
            with patch_object as patched_openurl:
                # simulate forums link clicked
                model_ui._check_local_docs(
                    'https://forums.naturalcapitalproject.org')

            patched_openurl.assert_called_once_with(
                QtCore.QUrl('https://forums.naturalcapitalproject.org'))
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_version_update(self):
        """UI Model: Check update button & dialog exist with a new version."""
        import natcap.invest
        current_version = natcap.invest.__version__

        try:
            # Assert that update button and dialog do not exist if the current
            # version is already the latest
            with mock.patch('natcap.invest.ui.model.'
                            'InVESTModel._get_latest_version',
                            return_value=current_version):
                model_ui = ModelTests.build_model()
                assert not all([hasattr(model_ui, 'update_button'),
                                hasattr(model_ui, 'version_update_dialog')])

            # Assert that button and dialog exists if there is a new version,
            # and the page can be linked successfully
            with mock.patch('natcap.invest.ui.model.'
                            'InVESTModel._get_latest_version',
                            return_value='1' + current_version):
                model_ui = ModelTests.build_model()
                assert all([hasattr(model_ui, 'update_button'),
                            hasattr(model_ui, 'version_update_dialog')])

                if hasattr(QtCore, 'QDesktopServices'):
                    patch_object = mock.patch('natcap.invest.ui.inputs.QtCore.'
                                              'QDesktopServices.openUrl')
                else:
                    # PyQt5 changed the location of this.
                    patch_object = mock.patch('natcap.invest.ui.inputs.QtGui.'
                                              'QDesktopServices.openUrl')

                # Simulate InVEST download link clicked
                with patch_object as patched_openurl:
                    model_ui._activate_link(
                        model_ui.version_update_dialog.download_qurl)

                patched_openurl.assert_called_once_with(QtCore.QUrl(
                    'https://naturalcapitalproject.stanford.edu/invest/'))
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_version_update_slow_connection(self):
        """UI Model: Check no/slow Internet connection won't interrupt UI."""
        import natcap.invest
        try:
            # Assert that update button and dialog don't exist if connection
            # is slow
            with mock.patch('requests.get', side_effect=requests.Timeout):
                model_ui = ModelTests.build_model()
                assert not all([hasattr(model_ui, 'update_button'),
                                hasattr(model_ui, 'version_update_dialog')])

            # Assert that update button and dialog don't exist if no connection
            with mock.patch('requests.get',
                            side_effect=requests.ConnectionError):
                model_ui = ModelTests.build_model()
                assert not all([hasattr(model_ui, 'update_button'),
                                hasattr(model_ui, 'version_update_dialog')])

        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_datastack_paramset(self):
        """UI Model: Check that we can load a parameter set datastack."""
        from natcap.invest import datastack
        model_ui = ModelTests.build_model()
        try:
            args = {
                'workspace_dir': 'foodir',
                'results_suffix': 'suffix',
            }
            datastack_filepath = os.path.join(self.workspace, 'paramset.json')
            datastack.build_parameter_set(
                args=args,
                model_name=model_ui.target.__module__,
                paramset_path=datastack_filepath,
                relative=False)

            model_ui.load_datastack(datastack_filepath)

            self.assertEqual(model_ui.workspace.value(), args['workspace_dir'])
            self.assertEqual(model_ui.suffix.value(), args['results_suffix'])
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_datastack_archive(self):
        """UI Model: Check that we can load a parameter archive."""
        from natcap.invest import datastack
        model_ui = ModelTests.build_model()
        try:
            args = {
                'workspace_dir': 'foodir',
                'results_suffix': 'suffix',
            }
            datastack_filepath = os.path.join(self.workspace, 'archive.tar.gz')
            datastack.build_datastack_archive(args, model_ui.target.__module__,
                                              datastack_filepath)

            extracted_archive = os.path.join(self.workspace, 'archive_dir')
            def _set_extraction_dir():
                model_ui.datastack_archive_extract_dialog.extraction_point.set_value(
                    extracted_archive)
                model_ui.datastack_archive_extract_dialog.accept()

            QtCore.QTimer.singleShot(25, _set_extraction_dir)

            # close the archive progress dialog automatically when extraction
            # finishes.
            model_ui.datastack_progress_dialog.checkbox.setChecked(True)

            model_ui.load_datastack(datastack_filepath)

            # Workspace isn't saved in a parameter archive, so just test suffix
            self.assertEqual(model_ui.suffix.value(), args['results_suffix'])
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_datastack_from_logfile(self):
        """UI Model: Check that we can load parameters from a logfile."""
        import natcap.invest
        model_ui = ModelTests.build_model()
        try:
            # write a sample logfile
            logfile_path = os.path.join(self.workspace, 'logfile')
            with open(logfile_path, 'w') as logfile:
                logfile.write(textwrap.dedent("""
                    07/20/2017 16:37:48  natcap.invest.ui.model INFO
                    Arguments for InVEST %s %s:
                    results_suffix                   foo
                    workspace_dir                    some_workspace_dir

                """ % (model_ui.target.__module__, natcap.invest.__version__)))

            model_ui.load_datastack(logfile_path)

            self.assertEqual(model_ui.workspace.value(), 'some_workspace_dir')
            self.assertEqual(model_ui.suffix.value(), 'foo')
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_datastack_extraction_dialog_cancelled(self):
        """UI Model: coverage when user clicks cancel in datastack dialog."""
        from natcap.invest import datastack
        model_ui = ModelTests.build_model()
        try:
            args = {
                'workspace_dir': 'foodir',
                'results_suffix': 'suffix',
            }
            datastack_filepath = os.path.join(self.workspace, 'archive.tar.gz')
            datastack.build_datastack_archive(args, model_ui.target.__module__,
                                              datastack_filepath)

            def _cancel_dialog():
                model_ui.datastack_archive_extract_dialog.reject()

            QtCore.QTimer.singleShot(25, _cancel_dialog)
            model_ui.load_datastack(datastack_filepath)
            self.assertFalse(model_ui.isVisible())
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_datastack_file_dialog_cancelled(self):
        """UI Model: coverage for when the file select dialog is cancelled."""
        # I'm mocking up the file dialog because I can't figure out how to
        # programmatically press the cancel button in a way that works on both
        # mac and linux.
        with mock.patch('qtpy.QtWidgets.QFileDialog.getOpenFileName',
                        return_value=(None, None)):
            try:
                model_ui = ModelTests.build_model()
                model_ui.load_datastack()
            finally:
                model_ui.close(prompt=False)
                model_ui.destroy()

    def test_model_quickrun(self):
        """UI Model: Test the quickrun path through model.run()."""
        model_ui = ModelTests.build_model()
        try:
            def _confirm_workspace_overwrite():
                # Just using dialog.accept() didn't work here, and I can't seem to
                # figure out why.
                QTest.mouseClick(
                    model_ui.workspace_overwrite_confirm_dialog.button(
                        QtWidgets.QMessageBox.Yes),
                    QtCore.Qt.LeftButton)

            # this line used to be in a singleshot, but i don't see why we
            # can't set it directly, works when i do it.
            model_ui.workspace.set_value(self.workspace)
            # Need to wait a little longer on this one to compensate for other
            # singleshot timers in model.run().
            QtCore.QTimer.singleShot(1000, _confirm_workspace_overwrite)
            model_ui.run(quickrun=True)
            try:
                QTest.qWaitForWindowShown(model_ui)
            except AttributeError:
                # pyqt5 has different wait methods.
                QTest.qWaitForWindowExposed(model_ui)
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_save_datastack_cancel_coverage(self):
        """UI Model: Test coverage for cancelling save datastack dialog."""
        model_ui = ModelTests.build_model()
        try:
            def _cancel_datastack_dialog():
                model_ui.datastack_options_dialog.reject()

            QtCore.QTimer.singleShot(25, _cancel_datastack_dialog)
            model_ui._save_datastack_as()
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_save_datastack_as_archive(self):
        """UI Model: Test coverage for saving parameter archives."""
        from natcap.invest.ui import model
        model_ui = ModelTests.build_model()
        try:

            starting_window_title = model_ui.windowTitle()

            archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')

            def _set_archive_options():
                model_ui.datastack_options_dialog.datastack_type.set_value(
                    model._DATASTACK_DATA_ARCHIVE)
                model_ui.datastack_options_dialog.save_parameters.set_value(
                    archive_path)
                self.qt_app.processEvents()
                model_ui.datastack_options_dialog.accept()

            # close the archive progress dialog automatically when extraction
            # finishes.
            model_ui.datastack_progress_dialog.checkbox.setChecked(True)

            QtCore.QTimer.singleShot(25, _set_archive_options)
            model_ui._save_datastack_as()
            self.assertNotEqual(starting_window_title, model_ui.windowTitle())
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_save_datastack_as_parameter_set(self):
        """UI Model: Test coverage for saving parameter set."""
        from natcap.invest.ui import model
        model_ui = ModelTests.build_model()
        try:
            starting_window_title = model_ui.windowTitle()

            archive_path = os.path.join(self.workspace, 'parameters.invs.json')

            def _set_archive_options():
                model_ui.datastack_options_dialog.datastack_type.set_value(
                    model._DATASTACK_PARAMETER_SET)
                model_ui.datastack_options_dialog.use_relative_paths.set_value(
                    True)
                model_ui.datastack_options_dialog.include_workspace.set_value(
                    True)
                model_ui.datastack_options_dialog.save_parameters.set_value(
                    archive_path)
                self.qt_app.processEvents()
                model_ui.datastack_options_dialog.accept()

            QtCore.QTimer.singleShot(25, _set_archive_options)
            model_ui._save_datastack_as()
            self.assertNotEqual(starting_window_title, model_ui.windowTitle())
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_settings_saved_message(self):
        """UI Model: Verify that saving settings posts status to statusbar."""
        model_ui = ModelTests.build_model()
        try:
            # this used to have a singleeshot to get the dialog to exec_, but
            # open seems to work just fine
            model_ui.settings_dialog.open()
            model_ui.settings_dialog.accept()
            self.assertEqual(
                model_ui.statusBar().currentMessage(), 'Settings saved')
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_run_with_validation_errors(self):
        """UI Model: Verify coverage when validation errors before a run."""
        from natcap.invest import validation

        @validation.invest_validator
        def _validate(args, limit_to=None):
            return [(['workspace_dir'], 'Some error message')]

        model_ui = ModelTests.build_model(_validate)
        try:
            model_ui.workspace.set_value('')
            model_ui.validate(block=True)
            self.assertEqual(len(model_ui.validation_report_dialog.warnings), 1)

            def _close_validation_report():
                model_ui.validation_report_dialog.accept()

            QtCore.QTimer.singleShot(25, _close_validation_report)
            model_ui.execute_model()
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_exception_raised_in_target(self):
        """UI Model: Verify coverage when exception raised in target."""
        def _target(args):
            raise Exception('foo!')

        model_ui = ModelTests.build_model(target_func=_target)
        try:
            model_ui.workspace.set_value(os.path.join(self.workspace,
                                                      'dir_not_there'))
            self.qt_app.processEvents()

            # Wait until the InVESTModel object is about to be deleted.
            with wait_on_signal(self.qt_app, model_ui.destroyed):
                model_ui.execute_model()

            self.assertEqual(str(model_ui.form._thread.exception), 'foo!')
            model_ui.form.run_dialog.close()
            model_ui.form.run_dialog.destroy()
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_workspace_overwrite_reject(self):
        """UI Model: Verify coverage when overwrite dialog is rejected."""

        model_ui = ModelTests.build_model()
        try:
            def _cancel_workspace_overwrite():
                model_ui.workspace_overwrite_confirm_dialog.reject()

            QtCore.QTimer.singleShot(50, _cancel_workspace_overwrite)
            model_ui.execute_model()
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_save_to_python(self):
        """UI Model: Verify that we can make a python script from params."""

        model_ui = ModelTests.build_model()
        try:
            python_file = os.path.join(self.workspace, 'python_script.py')

            model_ui.save_to_python(python_file)

            self.assertTrue(os.path.exists(python_file))

            module_name = str(uuid.uuid4()) + 'testscript'
            try:
                module = imp.load_source(module_name, python_file)
                self.assertEqual(module.args, model_ui.assemble_args())
            finally:
                del sys.modules[module_name]
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_drag_n_drop_datastack(self):
        """UI Model: Verify that we can drag-n-drop a valid datastack."""
        model = ModelTests.build_model()
        try:
            # Write a sample parameter set file to drop
            datastack_filepath = os.path.join(self.workspace, 'datastack.invest.json')
            with open(datastack_filepath, 'w') as sample_datastack:
                sample_datastack.write(json.dumps(
                    {'args': {'workspace_dir': '/foo/bar',
                              'results_suffix': 'baz'},
                     'model_name': model.target.__module__,
                     'invest_version': 'testing'}))

            mime_data = QtCore.QMimeData()
            mime_data.setText('Some datastack')
            mime_data.setUrls([QtCore.QUrl.fromLocalFile(datastack_filepath)])

            drag_event = QtGui.QDragEnterEvent(
                model.pos(),
                QtCore.Qt.CopyAction,
                mime_data,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier)
            model.dragEnterEvent(drag_event)
            self.assertTrue(drag_event.isAccepted())

            event = QtGui.QDropEvent(
                model.pos(),
                QtCore.Qt.CopyAction,
                mime_data,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier)

            # When the datastack is dropped, the datastack is loaded.
            model.dropEvent(event)
            self.assertEqual(model.workspace.value(),
                             os.path.normpath('/foo/bar'))
            self.assertEqual(model.suffix.value(), 'baz')
        finally:
            model.close(prompt=False)
            model.destroy()

    def test_drag_n_drop_datastack_no_mime_text(self):
        """UI Model: Verify drag-n-drop a datastack without MIME text."""
        model = ModelTests.build_model()
        try:
            # Write a sample parameter set file to drop
            datastack_filepath = os.path.join(self.workspace, 'datastack.invest.json')
            with open(datastack_filepath, 'w') as sample_datastack:
                sample_datastack.write(json.dumps(
                    {'args': {'workspace_dir': '/foo/bar',
                              'results_suffix': 'baz'},
                     'model_name': model.target.__module__,
                     'invest_version': 'testing'}))

            # Deliberately not setting any text in the mimedata.  We should
            # still be able to load the datastack.
            mime_data = QtCore.QMimeData()
            mime_data.setUrls([QtCore.QUrl.fromLocalFile(datastack_filepath)])

            drag_event = QtGui.QDragEnterEvent(
                model.pos(),
                QtCore.Qt.CopyAction,
                mime_data,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier)
            model.dragEnterEvent(drag_event)
            self.assertTrue(drag_event.isAccepted())

            event = QtGui.QDropEvent(
                model.pos(),
                QtCore.Qt.CopyAction,
                mime_data,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier)

            # When the datastack is dropped, the datastack is loaded.
            model.dropEvent(event)
            self.assertEqual(model.workspace.value(),
                             os.path.normpath('/foo/bar'))
            self.assertEqual(model.suffix.value(), 'baz')
        finally:
            model.close(prompt=False)
            model.destroy()

    def test_drag_n_drop_rejected_multifile(self):
        """UI Model: Drag-n-drop fails when dragging several files."""
        model = ModelTests.build_model()
        try:
            mime_data = QtCore.QMimeData()
            mime_data.setText('Some datastack')
            mime_data.setUrls([QtCore.QUrl('/path/1'),
                               QtCore.QUrl('/path/2')])

            drag_event = QtGui.QDragEnterEvent(
                model.pos(),
                QtCore.Qt.CopyAction,
                mime_data,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier)
            model.dragEnterEvent(drag_event)
            self.assertFalse(drag_event.isAccepted())
        finally:
            model.close(prompt=False)
            model.destroy()

    def test_open_recent_menu(self):
        """UI Model: Check for correct behavior of the open-recent menu."""
        from natcap.invest import datastack

        model_ui = ModelTests.build_model()
        try:

            datastack_created = []
            for datastack_index in range(11):
                datastack_path = os.path.join(self.workspace,
                                              'datastack_%s.invest.json' %
                                              datastack_index)
                args = {
                    'workspace_dir': 'workspace_%s' % datastack_index,
                }

                datastack.build_parameter_set(args,
                                              model_ui.target.__module__,
                                              datastack_path)
                datastack_created.append(datastack_path)
                model_ui.load_datastack(datastack_path)

            previous_datastack_actions = []
            for action in model_ui.open_menu.actions():
                if action.isSeparator() or action is model_ui.open_file_action:
                    continue
                previous_datastack_actions.append(action.data())

            # We should only have the 10 most recent datastack
            self.assertEqual(len(previous_datastack_actions), 10)
            self.assertEqual(max(previous_datastack_actions),
                             max(datastack_created))

            # The earliest datastack should have been booted off the list since
            # we're only keeping the 10 most recent.
            self.assertEqual(min(previous_datastack_actions),
                             sorted(datastack_created)[1])
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_datastack_display_short_filepaths(self):
        """UI Model: Check handling of long filepaths in open-recent menu."""
        model_ui = ModelTests.build_model()
        try:
            # synthesize a recent datastack path by adding it to the right setting.
            model_ui._add_to_open_menu('/foo.invest.json')

            last_run_datastack_actions = []
            for action in model_ui.open_menu.actions():
                if action.isSeparator() or action is model_ui.open_file_action:
                    continue
                last_run_datastack_actions.append((action.text(), action.data()))

            self.assertEqual(len(last_run_datastack_actions), 1)
            self.assertTrue('/foo.invest.json' in last_run_datastack_actions[0][0])
            self.assertEqual(last_run_datastack_actions[0][1], '/foo.invest.json')
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_datastack_display_long_filepaths(self):
        """UI Model: Check handling of long filepaths in open-recent menu."""
        model_ui = ModelTests.build_model()
        try:
            # synthesize a recent datastack path by adding it to the right setting.
            deep_directory = os.path.join(*[str(uuid.uuid4()) for x in range(10)])
            filepath = os.path.join(deep_directory, 'something.invest.json')
            model_ui._add_to_open_menu(filepath)

            last_run_datastack_actions = []
            for action in model_ui.open_menu.actions():
                if action.isSeparator() or action is model_ui.open_file_action:
                    continue
                last_run_datastack_actions.append((action.text(), action.data()))

            self.assertEqual(len(last_run_datastack_actions), 1)
            self.assertTrue('something.invest.json' in last_run_datastack_actions[0][0])
            self.assertEqual(last_run_datastack_actions[0][1], filepath)
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_load_datastack_from_open_recent(self):
        """UI Model: Check loading of datastack via open-recent menu."""
        from natcap.invest import datastack
        model_ui = ModelTests.build_model()
        try:

            datastack_filepath = os.path.join(self.workspace,
                                              'datastack.invest.json')
            args = {
                'workspace_dir': 'workspace_foo',
            }
            datastack.build_parameter_set(args,
                                          model_ui.target.__module__,
                                          datastack_filepath)

            model_ui.load_datastack(datastack_filepath)
            self.assertEqual(model_ui.workspace.value(), 'workspace_foo')
            model_ui.workspace.set_value('some_other_workspace')
            self.assertEqual(model_ui.workspace.value(), 'some_other_workspace')

            # Check to make sure that the loaded datastack was added to the
            # last-run menu.
            last_run_datastack_actions = []
            for action in model_ui.open_menu.actions():
                if action.isSeparator() or action is model_ui.open_file_action:
                    continue
                last_run_datastack_actions.append(action)

            # There should be exactly one recently-loaded datastack
            self.assertEqual(len(last_run_datastack_actions), 1)
            self.assertEqual(last_run_datastack_actions[0].data(), datastack_filepath)

            # When we trigger the action and process events, the datastack should
            # be loaded into the UI.
            action = last_run_datastack_actions[0]
            def _accept_parameter_overwrite():
                QTest.mouseClick(
                    model_ui.input_overwrite_confirm_dialog.button(
                        QtWidgets.QMessageBox.Yes),
                    QtCore.Qt.LeftButton)

            QtCore.QTimer.singleShot(25, _accept_parameter_overwrite)
            action.activate(QtWidgets.QAction.Trigger)
            self.qt_app.processEvents()
            time.sleep(0.25)
            self.qt_app.processEvents()

            self.assertEqual(model_ui.workspace.value(), args['workspace_dir'])
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_clear_local_settings(self):
        """UI Model: Check that we can clear local settings."""
        model_ui = ModelTests.build_model()
        try:
            # write something to settings and check it's been saved
            model_ui.save_lastrun()
            self.assertEqual(model_ui.settings.allKeys(), ['lastrun'])

            # clear settings and verify it's been cleared.
            model_ui.clear_local_settings()
            self.assertEqual(model_ui.settings.allKeys(), [])
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()

    def test_reject_on_modelname_mismatch(self):
        """UI Model: confirm when datastack modelname != model modelname."""
        from natcap.invest.ui import model
        from natcap.invest import datastack

        filepath = os.path.join(self.workspace, 'paramset.json')
        args = {'foo': 'foo', 'bar': 'bar'}
        datastack.build_parameter_set(args, 'test_model', filepath)

        model_ui = ModelTests.build_model()
        try:

            def _confirm_datastack_load():
                self.assertTrue(
                    model_ui.model_mismatch_confirm_dialog.isVisible())

                # Both the parameter's model name ('test_model') and the target
                # model's module name ('test_ui_inputs') should be in the
                # informative text.
                info_text = model_ui.model_mismatch_confirm_dialog.informativeText()
                self.assertTrue(model_ui.target.__module__ in info_text)
                self.assertTrue('test_model' in info_text)

                # Reject the dialog.
                QTest.mouseClick(
                    model_ui.model_mismatch_confirm_dialog.button(
                        QtWidgets.QMessageBox.Cancel),
                    QtCore.Qt.LeftButton)

            QtCore.QTimer.singleShot(500, _confirm_datastack_load)

            # Verify that we haven't changed anything when the dialog is cancelled.
            previous_args = model_ui.assemble_args()
            model_ui.load_datastack(filepath)
            self.assertEqual(previous_args, model_ui.assemble_args())
        finally:
            model_ui.close(prompt=False)
            model_ui.destroy()


class ValidatorTest(_QtTest):
    def test_in_progress(self):
        from natcap.invest.ui import inputs

        parent_widget = QtWidgets.QWidget()
        validator = inputs.Validator(parent_widget)

        self.assertFalse(validator.in_progress())

        # validator should be in progress while _validate() is executing.
        def _validate(args, limit_to=None):
            self.assertTrue(validator.in_progress())
            return []

        validator.validate(target=_validate, args={})


class IsProbablyDatastackTests(unittest.TestCase):
    """Tests for our quick check for whether a file is a datastack."""

    def setUp(self):
        """Create a new workspace for each test."""
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the workspace created for each test."""
        shutil.rmtree(self.workspace)

    def test_directory(self):
        """Model UI datastack: directory is not a datastack."""
        from natcap.invest.ui import model

        dirpath = os.path.join(self.workspace, 'foo_dir')
        os.makedirs(dirpath)
        self.assertFalse(model.is_probably_datastack(dirpath))

    def test_json_extension(self):
        """Model UI datastack: invest.json extension is a datastack"""
        from natcap.invest.ui import model

        filepath = 'some_model.invest.json'
        self.assertTrue(model.is_probably_datastack(filepath))

    def test_tar_gz_extension(self):
        """Model UI datastack: invest.tar.gz extension is a datastack"""
        from natcap.invest.ui import model

        filepath = 'some_model.invest.tar.gz'
        self.assertTrue(model.is_probably_datastack(filepath))

    def test_parameter_set(self):
        """Model UI datastack: a parameter set should be a datastack."""
        from natcap.invest.ui import model
        from natcap.invest import datastack

        filepath = os.path.join(self.workspace, 'paramset.json')
        args = {'foo': 'foo', 'bar': 'bar'}
        datastack.build_parameter_set(args, 'test_model', filepath)

        self.assertTrue(model.is_probably_datastack(filepath))

    def test_parameter_archive(self):
        """Model UI datastack: a parameter archive should be a datastack."""
        from natcap.invest.ui import model
        from natcap.invest import datastack

        filepath = os.path.join(self.workspace, 'paramset.tar.gz')
        args = {'foo': 'foo', 'bar': 'bar'}
        datastack.build_datastack_archive(args, 'test_model', filepath)

        self.assertTrue(model.is_probably_datastack(filepath))

    def test_parameter_logfile(self):
        """Model UI datastack: a logfile should be a datastack."""
        from natcap.invest.ui import model

        filepath = os.path.join(self.workspace, 'logfile.txt')
        with open(filepath, 'w') as logfile:
            logfile.write(textwrap.dedent("""
                07/20/2017 16:37:48  natcap.invest.ui.model INFO
                Arguments:
                suffix                           foo
                some_int                         1
                some_float                       2.33
                workspace_dir                    some_workspace_dir

                07/20/2017 16:37:48  natcap.invest.ui.model INFO post args.
            """))

        self.assertTrue(model.is_probably_datastack(filepath))

    def test_parameter_logfile_startswith_args(self):
        """Model UI datastack: logfile starting with arguments is datastack."""
        from natcap.invest.ui import model

        filepath = os.path.join(self.workspace, 'logfile.txt')
        with open(filepath, 'w') as logfile:
            logfile.write(textwrap.dedent(
                """Arguments:
                carbon_pools_path file_a.csv
                lulc_cur_path     file_b.tif
                workspace_dir     new_workspace_dir"""))
        self.assertTrue(model.is_probably_datastack(filepath))


    def test_csv_not_a_parameter(self):
        """Model UI datastack: a CSV is probably not a parameter set."""
        from natcap.invest.ui import model

        filepath = os.path.join(self.workspace, 'sample.csv')
        with open(filepath, 'w') as sample_csv:
            sample_csv.write('A,B,C\n')
            sample_csv.write('"aaa","bbb","ccc"\n')

        self.assertFalse(model.is_probably_datastack(filepath))


class FileSystemRunDialogTests(_QtTest):
    def setUp(self):
        _QtTest.setUp(self)
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        _QtTest.tearDown(self)
        shutil.rmtree(self.workspace)

    def test_window_close(self):
        from natcap.invest.ui import inputs

        dialog = inputs.FileSystemRunDialog()
        dialog.show()
        dialog.start('Window title', self.workspace)
        self.qt_app.processEvents()

        # Try to close the dialog, verify that it did not close.
        dialog.close()  # This shouldn't do anything.
        self.qt_app.processEvents()
        self.assertTrue(dialog.isVisible())

        # Finish the dialog, verify it is no longer visible.
        dialog.finish(None)  # None = no exception encountered.
        dialog.close()  # this should work now.
        self.qt_app.processEvents()
        self.assertFalse(dialog.isVisible())

        # launch the window again and verify that the things that should have
        # been cleared have been cleared.
        dialog.show()
        self.assertTrue(dialog.openWorkspaceCB.isVisible())
        self.assertFalse(dialog.openWorkspaceButton.isVisible())
        self.assertEqual(dialog.messageArea.text(), '')
        self.assertEqual(dialog.cancel, False)
