"""Task graph framework."""
import datetime
import hashlib
import json
import pickle
import os
import logging
import multiprocessing
import threading
import errno
import Queue
import inspect

logging.basicConfig(
    format='%(asctime)s %(name)-10s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
LOGGER = logging.getLogger('ipbes-cv')


def _worker(work_queue):
    """Thread worker.  `work_queue` has func/args tuple or 'STOP'."""
    for func, args in iter(work_queue.get, 'STOP'):
        func(*args)

class TaskGraph(object):
    """Encapsulates the worker and tasks states for parallel processing."""

    def __init__(self, token_storage_path, n_workers):
        """Create a task graph.

        Creates an object for building task graphs, executing them,
        parallelizing independent work notes, and avoiding repeated calls.

        Parameters:
            token_storage_path (string): path to a directory where work tokens
                (files) can be stored.  Task graph checks this directory to
                see if a task has already been completed.
            n_workers (int): number of parallel workers to allow during
                task graph execution.
        """
        # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        try:
            os.makedirs(token_storage_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.token_storage_path = token_storage_path
        self.worker_pool = multiprocessing.Pool(n_workers)

        self.global_thread_semaphore = threading.Semaphore(n_workers * 2)

        # used to lock global resources
        self.global_lock = threading.Lock()

        # if a Task pair is in here, it's been previously created
        self.global_working_task_dict = {}

    def add_task(
            self, target=None, args=None, kwargs=None,
            dependent_task_list=None):
        """Add a task to the task graph.

        Parameters:
            target (function): target function
            args (list): argument list for `target`
            kwargs (dict): keyword arguments for `target`
            dependent_task_list (list): list of `Task`s that this task is
                dependent on.

        Returns:
            Task which was just added to the graph.
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if dependent_task_list is None:
            dependent_task_list = []
        if target is None:
            target = lambda: None
        task = Task(
            target, args, kwargs, dependent_task_list,
            self.token_storage_path)

        self.global_thread_semaphore.acquire()
        threading.Thread(
            target=task, args=(
                self.global_lock,
                self.global_working_task_dict,
                self.worker_pool,
                self.global_thread_semaphore)).start()
        with self.global_lock:
            self.global_working_task_dict[task.task_id] = task
        return task

    def join(self):
        """Join all threads in the graph."""
        for task in self.global_working_task_dict.itervalues():
            task.join()


class Task(object):
    """Encapsulates work/task state for multiprocessing."""

    def __init__(
            self, target, args, kwargs, dependent_task_list,
            token_storage_path):
        """Make a task.

        Parameters:
            target (function): a function that takes the argument list
                `args`
            args (tuple): a list of arguments to pass to `target`.  Can be
                None.
            kwargs (dict): keyword arguments to pass to `target`.  Can be
                None.
            dependent_task_list (list of Task): a list of other
                `Task`s that are to be invoked before `target(args)` is
                invoked.
            token_storage_path (string): path to a directory that exists
                where task can store a file to indicate completion of task.
        """
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.dependent_task_list = dependent_task_list

        # Used to ensure only one attempt at executing and also a mechanism
        # to see when Task is complete
        self.lock = threading.Lock()

        # Make a unique hash of the input parameters of the function call
        # TODO: consider file date/time, dependent tasks,
        # TODO: new implementations (code versions)')
        task_string = '%s:%s:%s:%s' % (
            target.__name__, pickle.dumps(args),
            json.dumps(kwargs, sort_keys=True),
            inspect.getsource(target))
        self.task_id = '%s_%s' % (
            target.__name__, hashlib.sha1(task_string).hexdigest())

        # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        try:
            os.makedirs(token_storage_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        # The following file will be written when work is complete
        self.token_path = os.path.join(token_storage_path, self.task_id)

    def __call__(
            self, global_lock, global_working_task_dict,
            global_worker_pool, global_thread_semaphore):
        """Invoke this method when ready to execute task.

        Parameters:
            global_lock (threading.Lock): use this to lock global
                the global resources to the task graph.
            global_working_task_dict (dict): contains a dictionary of task_ids
                to Tasks that are currently executing.  Global resource and
                should acquire lock before modifying it.
            global_worker_pool (multiprocessing.Pool): a process pool used to
                execute subprocesses.
            global_thread_semaphore (threading.Semaphore): a semaphore to
                throttle the total number of global threads that are
                spun up.  This task releases a semaphore when complete.

        Returns:
            None
        """
        try:
            with self.lock:
                LOGGER.debug("Starting task %s", self.task_id)
                if self.is_complete():
                    LOGGER.info(
                        "Completion token exists for %s so not executing",
                        self.task_id)
                    return

                # if this Task is currently running somewhere, wait for it.
                if len(self.dependent_task_list) > 0:
                    LOGGER.debug("joining dependent threads %s", self.task_id)
                    for task in self.dependent_task_list:
                        task.join()
                        if not task.is_complete():
                            raise RuntimeError(
                                "Task %s didn't complete, discontinuing "
                                "execution of %s" % (task.task_id, self.task_id))

                # Do this task's work
                LOGGER.debug("Starting process for %s", self.task_id)
                result = global_worker_pool.apply_async(
                    func=self.target, args=self.args, kwds=self.kwargs)
                result.get()
                with open(self.token_path, 'w') as token_file:
                    token_file.write(str(datetime.datetime.now()))
        finally:
            global_thread_semaphore.release()

    def is_complete(self):
        """Return true if complete token exists."""
        return os.path.exists(self.token_path)

    def join(self):
        """Block until task is complete."""
        with self.lock:
            pass
