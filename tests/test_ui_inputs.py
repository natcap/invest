
import unittest
import functools
import warnings
import threading
import logging
import tempfile
import shutil
import os
import contextlib
import sys

import sip
sip.setapi('QString', 2)  # qtpy assumes api version 2
import qtpy
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtTest import QTest
import six

if sys.version_info >= (3,):
    import unittest.mock as mock
else:
    import mock

try:
    QApplication = QtGui.QApplication
except AttributeError:
    QApplication = QtWidgets.QApplication

QT_APP = QApplication.instance()
if QT_APP is None:
    QT_APP = QApplication(sys.argv)

LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def wait_on_signal(signal, timeout=250):
    """Block loop until signal emitted, or timeout (ms) elapses."""
    loop = QtCore.QEventLoop()

    wait_on_signal.signal_called = False

    def _set_signal_called(*args, **kwargs):
        wait_on_signal.signal_called = True

    signal.connect(_set_signal_called)
    signal.connect(loop.quit)

    try:
        yield
    finally:
        if timeout is not None:
            QtCore.QTimer.singleShot(timeout, loop.quit)

        if not wait_on_signal.signal_called:
            loop.exec_()
        signal.disconnect(loop.quit)


class InputTest(unittest.TestCase):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import Input
        return Input(*args, **kwargs)

    def test_label(self):
        input_instance = self.__class__.create_input(label='foo')
        self.assertEqual(input_instance.label, 'foo')

    def test_helptext(self):
        input_instance = self.__class__.create_input(label='foo', helptext='bar')
        self.assertEqual(input_instance.helptext, 'bar')

    def test_required(self):
        input_instance = self.__class__.create_input(label='foo', required=True)
        self.assertEqual(input_instance.required, True)

    def test_set_required(self):
        input_instance = self.__class__.create_input(label='foo', required=True)
        input_instance.set_required(False)
        self.assertEqual(input_instance.required, False)

    def test_nonrequired(self):
        input_instance = self.__class__.create_input(label='foo', required=False)
        self.assertEqual(input_instance.required, False)

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

    def test_set_noninteractive(self):
        input_instance = self.__class__.create_input(label='foo', interactive=False)
        self.assertEqual(input_instance.interactive, False)
        # Silence notimplementederror exceptions on input.value in some cases.
        try:
            input_instance.value()
        except NotImplementedError:
            input_instance.value = lambda: 'Value!'
        input_instance.set_noninteractive(False)
        self.assertEqual(input_instance.interactive, True)

    def test_interactivity_changed(self):
        input_instance = self.__class__.create_input(label='foo', interactive=False)
        callback = mock.MagicMock()
        input_instance.interactivity_changed.connect(callback)

        with wait_on_signal(input_instance.interactivity_changed):
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
        input_instance._add_to(base_widget.layout())

    def test_value(self):
        input_instance = self.__class__.create_input(label='foo')
        if input_instance.__class__.__name__ in ('Input', 'GriddedInput'):
            with self.assertRaises(NotImplementedError):
                input_instance.value()
        else:
            self.fail('Test class must reimplement this test method')

    def test_set_value(self):
        input_instance = self.__class__.create_input(label='foo')
        if input_instance.__class__.__name__ in ('Input', 'GriddedInput'):
            with self.assertRaises(NotImplementedError):
                input_instance.set_value()
        else:
            self.fail('Test class must reimplement this test method')

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(label='some_label')
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)

        if input_instance.__class__.__name__ in ('Input', 'GriddedInput'):
            try:
                with self.assertRaises(NotImplementedError):
                    self.assertEqual(input_instance.value(), '')
                    with wait_on_signal(input_instance.value_changed):
                        input_instance.set_value('foo')
                    callback.assert_called_with(u'foo')
            finally:
                input_instance.value_changed.disconnect(callback)
        else:
            self.fail('Test class must reimplement this test method')

    def test_value_changed_signal(self):
        input_instance = self.__class__.create_input(label='foo')
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)

        try:
            with wait_on_signal(input_instance.value_changed):
                try:
                    input_instance.value()
                except NotImplementedError:
                    input_instance.value = lambda: 'Value!'
                input_instance.value_changed.emit(six.text_type('value', 'utf-8'))

            callback.assert_called_with(six.text_type('value', 'utf-8'))
        finally:
            input_instance.value_changed.disconnect(callback)

    def test_interactivity_changed_signal(self):
        input_instance = self.__class__.create_input(label='foo')
        callback = mock.MagicMock()
        input_instance.interactivity_changed.connect(callback)

        with wait_on_signal(input_instance.interactivity_changed):
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

    def test_visiblity_when_shown(self):
        from natcap.invest.ui import inputs
        container = inputs.Container(label='sample container')
        input_instance = self.__class__.create_input(label='foo',
                                                     interactive=False)
        container.add_input(input_instance)
        container.show()
        inputs.QT_APP.processEvents()

        self.assertEqual(input_instance.visible(), True)

        input_instance.set_visible(False)
        inputs.QT_APP.processEvents()
        if len(input_instance.widgets) > 0:  # only works if input has widgets
            self.assertEqual(input_instance.visible(), False)

        input_instance.set_visible(True)
        inputs.QT_APP.processEvents()
        if len(input_instance.widgets) > 0:  # only works if input has widgets
            self.assertEqual(input_instance.visible(), True)


class GriddedInputTest(InputTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import GriddedInput
        return GriddedInput(*args, **kwargs)

    def test_label(self):
        input_instance = self.__class__.create_input(label='foo')
        label_text = input_instance.label
        self.assertEqual(label_text,  u'foo (Optional)')

    def test_label_required(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     required=True)
        label_text = input_instance.label
        self.assertEqual(label_text, 'foo')

    def test_validator(self):
        _callback = mock.MagicMock()
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

    def test_validate_passes(self):
        #"""UI: Validation that passes should affect validity."""
        _validation_func = mock.MagicMock(return_value=[])
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

    def test_validate_fails(self):
        #"""UI: Validation that fails should affect validity."""
        _validation_func = mock.MagicMock(
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

    def test_validate_required(self):
        """UI: Requirement with no input should affect validity."""
        input_instance = self.__class__.create_input(
            label='some_label', required=True)
        try:
            input_instance.value()
        except NotImplementedError:
            input_instance.value = lambda: ''

        self.assertEqual(input_instance.valid(), True)
        input_instance._validate()
        # Wait for validation to finish and assert Failure.
        self.assertEqual(input_instance.valid(), False)

    def test_validate_required_args_key(self):
        """UI: Requirement with no input should raise a warning."""
        input_instance = self.__class__.create_input(
            label='some_label', required=True,
            validator=lambda args, limit_to=None: []
        )

        input_instance.value = mock.MagicMock(
            input_instance, return_value=u'something')

        self.assertEqual(input_instance.valid(), True)
        with warnings.catch_warnings(record=True) as messages:
            input_instance._validate()

        # Validation still passes, but verify warning raised
        self.assertEqual(len(messages), 1)
        self.assertEqual(input_instance.valid(), True)

    def test_validate_required_validator(self):
        input_instance = self.__class__.create_input(
            label='some_label', args_key='foo'
        )

        input_instance.value = mock.MagicMock(
            input_instance, return_value=u'something')

        self.assertEqual(input_instance.valid(), True)
        with warnings.catch_warnings(record=True) as messages:
            input_instance._validate()

        # Validation still passes, but verify warning raised
        self.assertEqual(len(messages), 1)
        self.assertEqual(input_instance.valid(), True)

    def test_validate_error(self):
        input_instance = self.__class__.create_input(
            label='some_label', args_key='foo',
            validator=lambda args, limit_to=None: []
        )

        input_instance.value = mock.MagicMock(
            input_instance, return_value=u'something')

        input_instance._validator.validate = mock.MagicMock(
            input_instance._validator.validate, side_effect=ValueError('foo'))

        with self.assertRaises(ValueError):
            input_instance._validate()

    def test_nonhideable_default_state(self):
        sample_widget = QtWidgets.QWidget()
        sample_widget.setLayout(QtWidgets.QGridLayout())
        input_instance = self.__class__.create_input(
            label='some_label', hideable=False)
        input_instance._add_to(sample_widget.layout())
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
        input_instance._add_to(sample_widget.layout())
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
        callback = mock.MagicMock()
        input_instance.hidden_changed.connect(callback)
        self.assertEqual(input_instance.hidden(), True)

        with wait_on_signal(input_instance.hidden_changed):
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

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(label='text')
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)

        self.assertEqual(input_instance.value(), '')

        with wait_on_signal(input_instance.value_changed):
            input_instance.set_value('foo')

        callback.assert_called_with(u'foo')

    def test_textfield_settext(self):
        input_instance = self.__class__.create_input(label='text')

        input_instance.textfield.setText('foo')
        self.assertEqual(input_instance.value(), u'foo')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_textfield_settext_signal(self):
        input_instance = self.__class__.create_input(label='text')
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)

        with wait_on_signal(input_instance.value_changed):
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


class PathTest(TextTest):
    @staticmethod
    def create_input(*args, **kwargs):
        from natcap.invest.ui.inputs import _Path
        return _Path(*args, **kwargs)

    def test_path_selected(self):
        input_instance = self.__class__.create_input(label='foo')
        # Only run this test on subclasses of path
        if input_instance.__class__.__name__ != '_Path':
            input_instance.path_select_button.path_selected.emit(u'/tmp/foo')
            self.assertTrue(input_instance.value(), '/tmp/foo')

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
        self.assertEqual(event.isAccepted(), False)

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
        self.assertEqual(event.isAccepted(), True)

    def test_textfield_drop(self):
        pass

    def test_textfield_drop_windows(self):
        input_instance = self.__class__.create_input(label='text')

        mime_data = QtCore.QMimeData()
        mime_data.setText('Hello world!')
        # this is what paths look like when Qt receives them.
        mime_data.setUrls([QtCore.QUrl('/C:/foo/bar')])

        event = QtGui.QDropEvent(
            input_instance.textfield.pos(),
            QtCore.Qt.CopyAction,
            mime_data,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)

        with mock.patch('platform.system', return_value='Windows'):
            input_instance.textfield.dropEvent(event)

        self.assertEqual(event.isAccepted(), True)
        self.assertEqual(input_instance.value(), 'C:/foo/bar')

    def test_textfield_drop_mac(self):
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

        with mock.patch('platform.system', return_value='Darwin'):
            input_instance.textfield.dropEvent(event)

        self.assertEqual(event.isAccepted(), True)
        self.assertEqual(input_instance.value(), '/foo/bar')


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
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), False)

        with wait_on_signal(input_instance.value_changed):
            input_instance.set_value(True)

        callback.assert_called_with(True)

    def test_value_changed_signal(self):
        input_instance = self.__class__.create_input(label='new_label')
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)

        with wait_on_signal(input_instance.value_changed):
            input_instance.value_changed.emit(True)

        callback.assert_called_with(True)

    def test_valid(self):
        input_instance = self.__class__.create_input(label='new_label')
        self.assertEqual(input_instance.value(), False)
        self.assertEqual(input_instance.valid(), True)
        input_instance.set_value(True)
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

    def test_options_typecast(self):
        input_instance = self.__class__.create_input(
            label='label', options=(1, 2, 3))
        self.assertEqual(input_instance.options, [u'1', u'2', u'3'])

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

    def test_value(self):
        input_instance = self.__class__.create_input(
            label='label', options=('foo', 'bar', 'baz'))
        self.assertEqual(input_instance.value(), u'foo')
        self.assertTrue(isinstance(input_instance.value(), six.text_type))

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(
            label='label', options=('foo', 'bar', 'baz'))
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), u'foo')

        with wait_on_signal(input_instance.value_changed):
            input_instance.set_value('bar')

        callback.assert_called_with('bar')

    def test_label(self):
        # Override, sinve 'Optional' is irrelevant for Dropdown.
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


class LabelTest(unittest.TestCase):
    def test_add_to_layout(self):
        from natcap.invest.ui.inputs import Label

        super_widget = QtWidgets.QWidget()
        super_widget.setLayout(QtWidgets.QGridLayout())
        label = Label('Hello, World!')
        label._add_to(super_widget.layout())


class ContainerTest(InputTest):
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
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True,
                                                     expanded=True)

        self.assertEqual(input_instance.expandable, True)
        self.assertEqual(input_instance.expanded, True)

        input_instance.expanded = False
        self.assertEqual(input_instance.expanded, False)

    def test_value_changed_signal(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True)
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)

        with wait_on_signal(input_instance.value_changed):
            input_instance.value_changed.emit(True)

        callback.assert_called_with(True)

    def test_value_changed_signal_emitted(self):
        input_instance = self.__class__.create_input(label='foo',
                                                     expandable=True,
                                                     expanded=False)
        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), False)

        with wait_on_signal(input_instance.value_changed):
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

    def test_helptext(self):
        pass

    def test_nonrequired(self):
        pass

    def test_required(self):
        pass

    def test_set_required(self):
        pass


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

        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), [])

        with wait_on_signal(input_instance.value_changed):
            input_instance.set_value(('aaa', 'bbb'))

        callback.assert_called_with(['aaa', 'bbb'])

    def test_value_changed_signal(self):
        input_instance = self.__class__.create_input(
            label='foo',
            callable_=self.__class__.create_sample_callable(label='foo'))

        callback = mock.MagicMock()
        input_instance.value_changed.connect(callback)
        self.assertEqual(input_instance.value(), [])

        with wait_on_signal(input_instance.value_changed):
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

    def test_set_value_nonexpandable(self):
        pass

    def test_expanded(self):
        pass

    def test_expandable(self):
        pass

    def test_set_required(self):
        pass


class ValidationWorkerTest(unittest.TestCase):
    def test_run(self):
        from natcap.invest.ui.inputs import ValidationWorker
        _callable = mock.MagicMock(return_value=[])
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
        _callable = mock.MagicMock(side_effect=KeyError('missing'))
        worker = ValidationWorker(
            target=_callable,
            args={'foo': 'bar'},
            limit_to='foo')
        worker.start()
        while not worker.isFinished():
            QTest.qWait(50)
        self.assertEqual(worker.warnings, [])
        self.assertEqual(worker.error, "'missing'")


class FileButtonTest(unittest.TestCase):
    def test_button_clicked(self):
        from natcap.invest.ui.inputs import FileButton
        button = FileButton('Some title')

        # Patch up the open_method to return a known path.
        # Would block on user input otherwise.
        button.open_method = mock.MagicMock(return_value='/some/path')
        _callback = mock.MagicMock()
        button.path_selected.connect(_callback)

        with wait_on_signal(button.path_selected):
            QTest.mouseClick(button, QtCore.Qt.LeftButton)

        _callback.assert_called_with('/some/path')

    def test_button_title(self):
        from natcap.invest.ui.inputs import FileButton
        button = FileButton('Some title')
        self.assertEqual(button.dialog_title, 'Some title')


class FolderButtonTest(unittest.TestCase):
    def test_button_clicked(self):
        from natcap.invest.ui.inputs import FolderButton
        button = FolderButton('Some title')

        # Patch up the open_method to return a known path.
        # Would block on user input otherwise.
        button.open_method = mock.MagicMock(return_value='/some/path')
        _callback = mock.MagicMock()
        button.path_selected.connect(_callback)

        with wait_on_signal(button.path_selected):
            QTest.mouseClick(button, QtCore.Qt.LeftButton)

        _callback.assert_called_with('/some/path')

    def test_button_title(self):
        from natcap.invest.ui.inputs import FolderButton
        button = FolderButton('Some title')
        self.assertEqual(button.dialog_title, 'Some title')


class FileDialogTest(unittest.TestCase):
    def test_save_file_title_and_last_selection(self):
        from natcap.invest.ui.inputs import FileDialog, DATA
        dialog = FileDialog()
        dialog.file_dialog.getSaveFileName = mock.MagicMock(
            spec=dialog.file_dialog.getSaveFileName,
            return_value='/new/file')

        DATA['last_dir'] = '/tmp/foo/bar'

        out_file = dialog.save_file(title='foo', start_dir=None)
        self.assertEqual(
            dialog.file_dialog.getSaveFileName.call_args[0],  # pos. args
            (dialog.file_dialog, 'foo', '/tmp/foo/bar'))
        self.assertEqual(out_file, '/new/file')
        self.assertEqual(DATA['last_dir'], u'/new')

    def test_save_file_defined_savefile(self):
        from natcap.invest.ui.inputs import FileDialog
        dialog = FileDialog()
        dialog.file_dialog.getSaveFileName = mock.MagicMock(
            spec=dialog.file_dialog.getSaveFileName,
            return_value='/new/file')

        out_file = dialog.save_file(title='foo', start_dir='/tmp',
                                    savefile='file.txt')
        self.assertEqual(
            dialog.file_dialog.getSaveFileName.call_args[0],  # pos. args
            (dialog.file_dialog, 'foo', '/tmp/file.txt'))

    def test_open_file(self):
        from natcap.invest.ui.inputs import FileDialog, DATA
        dialog = FileDialog()

        # patch up the Qt method to get the path to the file to open
        # Qt4 and Qt4 have different return values.  Mock up accordingly.
        if int(qtpy.QT_VERSION[0]) == 5:
            return_value = ('/new/file', 'filter')
        else:
            return_value = '/new/file'

        dialog.file_dialog.getOpenFileName = mock.MagicMock(
            spec=dialog.file_dialog.getOpenFileName,
            return_value=return_value)

        DATA['last_dir'] = '/tmp/foo/bar'

        out_file = dialog.open_file(title='foo')
        self.assertEqual(
            dialog.file_dialog.getOpenFileName.call_args[0],  # pos. args
            (dialog.file_dialog, 'foo', '/tmp/foo/bar'))
        self.assertEqual(out_file, '/new/file')
        self.assertEqual(DATA['last_dir'], '/new')

    def test_open_folder(self):
        from natcap.invest.ui.inputs import FileDialog, DATA
        dialog = FileDialog()

        # patch up the Qt method to get the path to the file to open
        dialog.file_dialog.getExistingDirectory = mock.MagicMock(
            spec=dialog.file_dialog.getExistingDirectory,
            return_value='/existing/folder')

        DATA['last_dir'] = '/tmp/foo/bar'
        new_folder = dialog.open_folder('foo', start_dir=None)

        self.assertEqual(dialog.file_dialog.getExistingDirectory.call_args[0],
                         (dialog.file_dialog, 'Select folder: foo', '/tmp/foo/bar'))
        self.assertEqual(new_folder, '/existing/folder')
        self.assertEqual(DATA['last_dir'], '/existing/folder')


class InfoButtonTest(unittest.TestCase):
    @unittest.skip("'Always segfaults, don't know why")
    def test_buttonpress(self):
        from natcap.invest.ui.inputs import InfoButton
        button = InfoButton('some text')
        self.assertEqual(button.whatsThis(), 'some text')

        # Execute this, for coverage.
        button.show()
        QTest.mouseClick(button, QtCore.Qt.LeftButton)
        #self.assertTrue(QtGui.QWhatsThis.inWhatsThisMode())

class FormTest(unittest.TestCase):
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

    def test_run_noerror(self):
        form = FormTest.make_ui()
        form.run(target=lambda: None)
        form._thread.join()
        QT_APP.processEvents()

        # At the end of the run, the button should be visible.
        self.assertTrue(form.run_dialog.openWorkspaceButton.isVisible())

        # close the window by pressing the back button.
        QTest.mouseClick(form.run_dialog.backButton,
                         QtCore.Qt.LeftButton)

    def test_open_workspace_on_success(self):
        thread_event = threading.Event()

        class _SampleTarget(object):
            @staticmethod
            def validate(args, limit_to=None):
                return []

            @staticmethod
            def execute(args):
                thread_event.wait()

        form = FormTest.make_ui()
        target = _SampleTarget().execute
        form.run(target=target)

        self.assertTrue(form.run_dialog.openWorkspaceCB.isVisible())
        self.assertFalse(form.run_dialog.openWorkspaceButton.isVisible())

        form.run_dialog.openWorkspaceCB.setChecked(True)
        self.assertTrue(form.run_dialog.openWorkspaceCB.isChecked())

        thread_event.set()

        # close the window by pressing the back button.
        QTest.mouseClick(form.run_dialog.backButton,
                         QtCore.Qt.LeftButton)

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
        QT_APP.processEvents()
        target_mod = _SampleTarget().execute
        try:
            form.run(target=target_mod, kwargs={'args': {'a': 1}})
            self.assertTrue(form.run_dialog.isVisible())
            QT_APP.processEvents()
            form.run_dialog.close()
            QT_APP.processEvents()
            self.assertTrue(form.run_dialog.isVisible())

            # when the execute function finishes, pressing escape should
            # close the window.
            thread_event.set()
            form._thread.join()
            QT_APP.processEvents()
            form.run_dialog.close()
            QT_APP.processEvents()
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
        QT_APP.processEvents()

        self.assertTrue('encountered' in form.run_dialog.messageArea.text())

    def test_show(self):
        form = FormTest.make_ui()
        form.show()


class OpenWorkspaceTest(unittest.TestCase):
    def test_windows(self):
        from natcap.invest.ui.inputs import open_workspace
        with mock.patch('subprocess.Popen') as method:
            with mock.patch('platform.system', return_value='Windows'):
                open_workspace('/foo/bar')
                method.assert_called_with('explorer "/foo/bar"')

    def test_mac(self):
        from natcap.invest.ui.inputs import open_workspace
        with mock.patch('subprocess.Popen') as method:
            with mock.patch('platform.system', return_value='Darwin'):
                open_workspace('/foo/bar')
                method.assert_called_with('open /foo/bar', shell=True)

    def test_linux(self):
        from natcap.invest.ui.inputs import open_workspace
        with mock.patch('subprocess.Popen') as method:
            with mock.patch('platform.system', return_value='Linux'):
                open_workspace('/foo/bar')
                method.assert_called_with(['xdg-open', '/foo/bar'])

    def test_error_in_subprocess(self):
        from natcap.invest.ui.inputs import open_workspace
        with mock.patch('subprocess.Popen',
                        side_effect=OSError('error message')) as patch:
            open_workspace('/foo/bar')
            patch.assert_called_once()


class ExecutionTest(unittest.TestCase):
    def test_executor_run(self):
        from natcap.invest.ui.execution import Executor

        thread_event = threading.Event()

        def _waiting_func(*args, **kwargs):
            thread_event.wait()

        target = mock.MagicMock(wraps=_waiting_func)
        callback = mock.MagicMock()
        args = ('a', 'b', 'c')
        kwargs = {'d': 1, 'e': 2, 'f': 3}
        tempdir = tempfile.mkdtemp()
        log_file = os.path.join(tempdir, 'logfile.txt')

        try:
            executor = Executor(
                target=target,
                args=args,
                kwargs=kwargs,
                logfile=log_file,
                tempdir=tempdir)

            self.assertEqual(executor.target, target)
            self.assertEqual(executor.args, args)
            self.assertEqual(executor.kwargs, kwargs)
            self.assertEqual(executor.logfile, log_file)
            self.assertEqual(executor.tempdir, tempdir)

            # register the callback with the finished signal.
            executor.finished.connect(callback)

            executor.start()
            thread_event.set()
            executor.join()
            QtWidgets.QApplication.instance().processEvents()
            callback.assert_called_once()
            target.assert_called_once()
            target.assert_called_with(*args, **kwargs)

        finally:
            shutil.rmtree(tempdir)

    def test_executor_exception(self):
        from natcap.invest.ui.execution import Executor

        thread_event = threading.Event()

        def _waiting_func(*args, **kwargs):
            thread_event.wait()
            raise ValueError('Some demo exception')

        target = mock.MagicMock(wraps=_waiting_func)
        callback = mock.MagicMock()
        args = ('a', 'b', 'c')
        kwargs = {'d': 1, 'e': 2, 'f': 3}
        tempdir = tempfile.mkdtemp()
        log_file = os.path.join(tempdir, 'logfile.txt')

        try:
            executor = Executor(
                target=target,
                args=args,
                kwargs=kwargs,
                logfile=log_file,
                tempdir=tempdir)

            self.assertEqual(executor.target, target)
            self.assertEqual(executor.args, args)
            self.assertEqual(executor.kwargs, kwargs)
            self.assertEqual(executor.logfile, log_file)
            self.assertEqual(executor.tempdir, tempdir)

            # register the callback with the finished signal.
            executor.finished.connect(callback)

            executor.start()
            thread_event.set()
            executor.join()
            QtWidgets.QApplication.instance().processEvents()
            callback.assert_called_once()
            target.assert_called_once()
            target.assert_called_with(*args, **kwargs)

            self.assertTrue(executor.failed)
            self.assertEqual(str(executor.exception),
                             'Some demo exception')
            self.assertTrue(isinstance(executor.traceback, basestring))
        finally:
            shutil.rmtree(tempdir)


class IntegrationTests(unittest.TestCase):
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
        inputs.QT_APP.processEvents()

        self.assertTrue(container.interactive)
        self.assertFalse(container.expanded)
        self.assertFalse(contained_file.interactive)
        self.assertFalse(contained_file.visible())

        # When the container is expanded, the contained input should become
        # interactive and visible
        container.set_value(True)
        inputs.QT_APP.processEvents()

        self.assertTrue(container.interactive)
        self.assertTrue(container.expanded)
        self.assertTrue(contained_file.interactive)
        self.assertTrue(contained_file.visible())

