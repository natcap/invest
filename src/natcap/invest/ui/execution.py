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

    def __init__(self, target, args=None, kwargs=None, log_events=True):
        """Initialize the Executor object.

        Args:
            target (callable): A function or unbound method that should be
                called within the separate thread of control.
            args (iterable): An iterable of positional arguments that will
                be passed to ``target``.  If ``None``, positional
                arguments will be ignored.
            kwargs (dict): A dict mapping string parameter names to parameter
                values that should be passed to ``target``.  If ``None``,
                keyword arguments will be ignored.
            log_events=True (bool): If ``True``, exceptions raised when calling
                ``target`` as well as completion of ``target`` will
                be logged.

        Returns:
            ``None``.
        """
        QtCore.QObject.__init__(self)
        threading.Thread.__init__(self)
        self.target = target
        self.log_events = log_events

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
        except BaseException as error:
            # We deliberately want to catch all possible exceptions, so
            # BaseException is the way to go.  This is in part because we have
            # a flaky test that failed where self.exception wasn't set and we
            # don't know if that's because this exception handler wasn't being
            # called (we were only capturing Exception at the time), or if
            # there was something else very strange going on.
            # Failed build:
            # http://builds.naturalcapitalproject.org/job/test-natcap.invest.ui/100/

            # When we're running a model, the exception is logged via
            # utils.prepare_workspace.  But this thread is also used in the Qt
            # interface by validation and the datastack archive creation
            # function, and we for sure want to log exceptions in that case.
            if self.log_events:
                LOGGER.exception('Target %s failed with exception', self.target)
            self.failed = True
            self.exception = error
            self.traceback = traceback.format_exc()
        finally:
            if self.log_events:
                LOGGER.info('Execution finished')
            self.finished.emit()
