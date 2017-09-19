"""A class for a Qt-enabled python Thread."""
import threading
import logging
import traceback

from qtpy import QtCore


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class Executor(QtCore.QObject, threading.Thread):
    """A thread of control that will emit a Qt signal when finished."""

    finished = QtCore.Signal()

    def __init__(self, target, args=None, kwargs=None):
        """Initialize the Executor object.

        Parameters:
            target (callable): A function or unbound method that should be
                called within the separate thread of control.
            args (iterable): An iterable of positional arguments that will
                be passed to ``target``.  If ``None``, positional
                arguments will be ignored.
            kwargs (dict): A dict mapping string parameter names to parameter
                values that should be passed to ``target``.  If ``None``,
                keyword arguments will be ignored.

        Returns:
            ``None``.
        """
        QtCore.QObject.__init__(self)
        threading.Thread.__init__(self)
        self.target = target

        if args is None:
            args = ()
        self.args = args

        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

        self.failed = False
        self.exception = None
        self.traceback = None

    def run(self):
        """Run the target callable in a separate thread of control.

        The callable will be run with whatever ``args`` and ``kwargs`` are
        provided to the thread's ``__init__`` method.

        If an exception is encountered while executing the target, several
        things happen:

            * The exception is logged.
            * ``self.failed`` is set to ``True``.
            * ``self.exception`` refers to the exception object that was raised.
            * ``self.traceback`` refers to the formatted traceback.

        Finally, the signal ``self.finished`` is emitted, regardless of whether
        an exception was raised.
        """
        try:
            self.target(*self.args, **self.kwargs)
        except Exception as error:
            # We deliberately want to catch all possible exceptions.
            LOGGER.exception('Target %s failed with exception', self.target)
            self.failed = True
            self.exception = error
            self.traceback = traceback.format_exc()
        finally:
            LOGGER.info('Execution finished')
            self.finished.emit()
