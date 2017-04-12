import threading
import os
import logging
import pprint
import traceback
import tempfile

from qtpy import QtCore


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class Executor(QtCore.QObject, threading.Thread):
    """Executor represents a thread of control that runs a python function with
    a single input.  Once created with the proper inputs, threading.Thread has
    the following attributes:
        self.module - the loaded module object provided to __init__()
        self.args   - the argument to the target function.  Usually a dict.
        self.func_name - the function name that will be called.
        self.log_manager - the LogManager instance managing logs for this script
        self.failed - defaults to False.  Indicates whether the thread raised an
            exception while running.
        self.execption - defaults to None.  If not None, points to the exception
            raised while running the thread.
    The Executor.run() function is an overridden function from threading.Thread
    and is started in the same manner by calling Executor.start().  The run()
    function is extremely simple by design: Print the arguments to the logfile
    and run the specified function.  If an execption is raised, it is printed
    and saved locally for retrieval later on.
    In keeping with convention, a single Executor thread instance is only
    designed to be run once.  To run the same function again, it is best to
    create a new Executor instance and run that."""

    finished = QtCore.Signal()

    def __init__(self, target, args, kwargs, logfile, tempdir=None):
        QtCore.QObject.__init__(self)
        threading.Thread.__init__(self)
        self.target = target
        self.tempdir = tempdir

        if not args:
            args = ()
        self.args = args

        if not kwargs:
            kwargs = {}
        self.kwargs = kwargs

        if logfile is None:
            logfile = os.path.join(tempfile.mkdtemp(), 'logfile.txt')
        self.logfile = logfile

        self.failed = False
        self.exception = None
        self.traceback = None

    def run(self):
        """Run the python script provided by the user with the arguments
        specified.  This function also prints the arguments to the logfile
        handler.  If an exception is raised in either the loading or execution
        of the module or function, a traceback is printed and the exception is
        saved."""
        try:
            self.target(*self.args, **self.kwargs)
        except Exception as error:
            # We deliberately want to catch all possible exceptions.
            LOGGER.exception(error)
            self.failed = True
            self.exception = error
            self.traceback = traceback.format_exc()
        finally:
            LOGGER.info('Execution finished')

        self.finished.emit()
