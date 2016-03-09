"""executor module for natcap.invest.iui"""

import uuid
import urllib
import locale
import platform
import hashlib
import json
import Pyro4
import threading
import os
import sys
import imp
from collections import deque
import traceback
import logging
import time
import subprocess
import datetime
import shutil
import codecs
import errno
import tempfile
from types import StringType
import importlib

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pygeoprocessing
import natcap.invest
import natcap.invest.iui
from natcap.invest import fileio as fileio
import pygeoprocessing.geoprocessing

LOGGER = natcap.invest.iui.get_ui_logger(None)
ENCODING = sys.getfilesystemencoding()
INVEST_USAGE_LOGGER_URL = ('http://data.naturalcapitalproject.org/'
                           'server_registry/invest_usage_logger/')

# If we're not on windows, python doesn't know what a
# WindowsError is.  Using shutil to import an empty WindowsError
CURRENT_OS = platform.system()
if CURRENT_OS != 'Windows':
    from shutil import WindowsError


class InsufficientDiskSpace(Exception):
    """This class is to be used if certain WindowsErrors or IOErrors are
    encountered."""
    pass


def locate_module(module_list, path=None):
    """Search for and return an executable module object as long as the target
        module is within the pythonpath.  This method recursively uses the
        find_module and load_module functions of the python imp module to
        locate the target module by its heirarchical module name.

        module_list - a python list of strings, where each element is the name
            of a contained module.  For example, os.path would be represented
            here as ['os', 'path'].
        path=None - the base path to search.  If None, the pythonpath will be
            used.

        returns an executeable python module object if it can be found.
        Returns None if not."""

    current_name = module_list[0]
    module_info = imp.find_module(current_name, path)
    imported_module = imp.load_module(current_name, *module_info)

    if len(module_list) > 1:
        return locate_module(module_list[1:], imported_module.__path__)
    else:
        return imported_module


class Controller(object):
    """The Controller class manages two Thread objects: Executor and
        PrintQueueChecker.  Executor runs models and queues up print statements
        in a local printqueue list.  Printqueue checks on Executor's printqueue
        and fetches the next message at a specified interval.

        The printqueuechecker exists to offload the work of list-related
        operations from the main thread, which leaves the main thread free to
        perform UI-related tasks."""

    def __init__(self):
        object.__init__(self)
        self.executor = Executor()
        self.msg_checker = PrintQueueChecker(self.executor)
        self.thread_finished = False
        self.thread_failed = False
        self.thread_exception = None

    def get_message(self):
        """Check to see if the message checker thread is alive and returns the
            current message if so.  If the message checker thread is not alive,
            None is returned and self.finished() is called."""

        if self.msg_checker.is_alive():
            return self.msg_checker.get_message()
        else:
            self.finished()

    def start_executor(self):
        """Starts the executor and message checker threads.  Returns nothing.
        """

        if not self.executor:
            self.executor = Executor()
            self.msg_checker = PrintQueueChecker(self.executor)

        self.thread_finished = False
        self.executor.start()
        self.msg_checker.start()

    def cancel_executor(self):
        """Trigger the executor's cancel event. Returns nothing."""

        self.executor.cancel()

    def finished(self):
        """Set the executor and message checker thread objects to none and set
            the thread_finished variable to True.

            Returns nothing."""

        self.thread_failed = self.executor.isThreadFailed()
        self.thread_exception = self.executor.failure_exception
        del self.executor
        self.executor = None
        del self.msg_checker
        self.executor = None
        self.thread_finished = True

    def is_finished(self):
        """Returns True if the threads are finished.  False if not."""

        return self.thread_finished

    def add_operation(self, op, args=None, uri=None, index=None):
        """Wrapper method for Executor.addOperation.  Creates new executor
            and message checker thread instances if necessary.

            Returns nothing."""

        if not self.executor:
            self.executor = Executor()
            self.msg_checker = PrintQueueChecker(self.executor)

        self.executor.addOperation(op, args, uri, index)


class PrintQueueChecker(threading.Thread):
    """PrintQueueChecker is a thread class that checks on a specified executor
        thread object.  By placing the responsibility of this operation in a
        separate thread, we allow the main thread to attend to more pressing UI
        related tasks."""

    def __init__(self, executor_object):
        threading.Thread.__init__(self)
        self.executor = executor_object
        self.message = None

    def get_message(self):
        """Check to see if there is a new message available.

        Returns the string message, if one is available.  None if not."""

        message = self.message
        self.message = None  # indicates the current message has been retrieved
        return message

    def run(self):
        """Fetch messages as long as the executor is alive or has messages.

            This method is reimplemented from threading.Thread and is started
            by calling self.start().

            This function calls the executor object function getMessage(),
            which uses the collections.deque queue object to manage the
            printqueue.

            The new message is only fetched from the executor if the main
            thread has fetched the current message from this
            PrintQueueChecker instance.

            returns nothing."""
        while self.executor.is_alive() or self.executor.hasMessages():
            #new_message = self.executor.getMessage()
            if self.message == None:
                #self.message = new_message
                self.message = self.executor.getMessage()
            #else:
            #    self.message += new_message
            time.sleep(0.025)


class Executor(threading.Thread):
    def __init__(self):
        format_string = '%(asctime)s %(name)-18s %(levelname)-8s %(message)s'
        date_format = '%m/%d/%Y %H:%M:%S '
        formatter = logging.Formatter(format_string, date_format)
        handler = logging.StreamHandler(self)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)

        threading.Thread.__init__(self)

        self.log_file = None
        self.printQueue = deque([])
        self.printQueueLock = threading.Lock()
        self.threadFailed = False
        self.failure_exception = None
        self.cancelFlag = threading.Event()
        self.operations = []
        self.funcMap = {'validator': self.runValidator,
                        'model': self.runModel,
                        'saveParams': self.saveParamsToDisk}

    def flush(self):
        # This function is required, since I point sys.stdout to self.  Flush()
        # is called by some parts of the multiprocessing module.
        pass

    def write(self, string):
        self.printQueueLock.acquire()
        self.printQueue.append(string)
        self.printQueueLock.release()

        # Convert unix line endings to windows line endings if we are writing to
        # a log file on windows.
        if platform.system() == 'Windows':
            string = string.replace('\n', '\r\n')

        encoding = sys.getfilesystemencoding()
        if self.log_file != None:
            if not self.log_file.closed:
                if type(string) is StringType:
                    string = string.decode('utf-8')
                else:
                    # assume UnicodeType.  If it's a UnicodeType string, then
                    # we should be able to re-encode it to UTF-8 from whatever
                    # the encoding might happen to be.
                    string = string.encode('utf-8')
                self.log_file.write(string)

    def hasMessages(self):
        self.printQueueLock.acquire()
        has_messages = len(self.printQueue) > 0
        self.printQueueLock.release()
        return has_messages

    def getMessage(self):
        self.printQueueLock.acquire()
        msg = None
        try:
            msg = self.printQueue.popleft()
            #For some reason this thing barfs out if the printQueue only has
            #1 element on it.  I have *no* idea what is going on here. Can't
            #debug it either.  If you change this to > 0 nothing will print out.
            while len(self.printQueue) > 1:
                msg += self.printQueue.popleft()
            self.printQueueLock.release()
            return msg
        except IndexError:
            self.printQueueLock.release()
            return msg

    def cancel(self):
        LOGGER.debug('Cancellation request received; finishing current ' +
                     'operation')
        self.cancelFlag.set()

    def isCancelled(self):
        return self.cancelFlag.isSet()

    def setThreadFailed(self, state, exception=None):
        """Set the flag of whether the thread has failed.  exception should be a
        pointer to a python Exception or a boolean."""
        self.threadFailed = state
        self.failure_exception = exception

    def isThreadFailed(self):
        return self.threadFailed

    def printTraceback(self):
        LOGGER.debug('Printing traceback')
        print(str(traceback.format_exc()) + '\n')

    def addOperation(self, op, args=None, uri=None, index=None):
        #op is a string index to self.funcmap
        opDict = {'type': op,
                  'args': args,
                  'uri': uri}

        if index == None:
            self.operations.append(opDict)
        else:
            self.operations.insert(index, opDict)

    def print_args(self, args_dict):
        """Write args_dict to a formatted string to the self.write() function.
            args_dict - a dictionary.

        returns noting"""

        self.write("Arguments:\n")
        sorted_args = sorted(args_dict.iteritems(), key=lambda x: x[0])
        max_key_width = max(map(lambda x:len(x[0]), sorted_args))
        format_str = "%-" + str(max_key_width) + "s %s\n"
        for name, value in sorted_args:
            if name == '_iui_meta':
                continue  # skip over the meta tag if provided.

            self.write(format_str % (name, value))
        self.write("\n\n")

    def print_system_info(self, function=None):
        if function == None:
            function = lambda x: self.write(x + '\n')

        run_details = [
            ('Current temp dir', tempfile.gettempdir()),
            ('tempfile.tempdir', tempfile.tempdir),
        ]

        system_details = [
            ('OS', platform.platform()),
            ('Processor architecture', platform.machine()),
            ('FS encoding', sys.getfilesystemencoding()),
            ('Preferred encoding', locale.getpreferredencoding()),
        ]

        python_details = [
            ('Version', platform.python_version()),
            ('Build', platform.python_build()),
            ('Compiler', platform.python_compiler()),
            ('Implementation', platform.python_implementation()),
            ('Architecture', platform.architecture()[0]),
            ('Linkage format', platform.architecture()[1]),
        ]

        def pkg_ver(name):
            try:
                return __import__(name).__version__
            except (ImportError, AttributeError):
                return '?'
        packages = [
            ('Cython', pkg_ver('Cython')),
            ('Numpy', pkg_ver('numpy')),
            ('Scipy', pkg_ver('scipy')),
            ('OSGEO', pkg_ver('osgeo')),
            ('Shapely', pkg_ver('shapely')),
            ('InVEST', natcap.invest.__version__),
        ]

        function('Build details')
        detail_lists = [
            ('Interpreter', run_details),
            ('System', system_details),
            ('Python', python_details),
            ('Packages',packages),
        ]
        for list_name, detail_list in detail_lists:
            function(list_name)
            fmt_string = '%-16s: %s'
            for detail_name, detail in detail_list:
                function(fmt_string % (detail_name, detail))
            function('')
        function('')

    def format_time(self, seconds):
        """Render the integer number of seconds in a string.  Returns a string.
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        hours = int(hours)
        minutes = int(minutes)

        if hours > 0:
            return "%sh %sm %ss" % (hours, minutes, seconds)
        else:
            if minutes > 0:
                return "%sm %ss" % (minutes, seconds)
            else:
                return "%ss" % seconds

    def run(self):
        sys.stdout = self
        sys.stderr = self

        # Get the current time so that we can time the overall operations
        overall_start = time.time()

        #determine what type of function needs to be run
        for operation in self.operations:
            self.setThreadFailed(False)

            if self.isCancelled():
                LOGGER.debug('Remaining operations cancelled')
                break
            else:
                try:
                    self.funcMap[operation['type']](operation['uri'],
                                                    operation['args'])
                except Exception as e:
                    self.setThreadFailed(True)
                    self.printTraceback()
                    LOGGER.error('Exception %s, %s found', e.__class__, e)

            if self.isThreadFailed():
                LOGGER.error('Exiting due to failures')
                break

        # Log the elapsed time.
        elapsed_time = round(time.time() - overall_start, 2)
        LOGGER.info('Elapsed time: %s', self.format_time(elapsed_time))

        if not self.isThreadFailed():
            LOGGER.info('Operations completed successfully')

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def runValidator(self, uri, args):
        LOGGER.info('Starting validator')
        validator = imp.load_source('validator', uri)
        outputList = []

        try:
            validator.execute(args, outputList)

            if len(outputList) > 0:
                print('ERRORS:\n')
                for error in outputList:
                    self.write(error + '\n')
                self.setThreadFailed(True)
            else:
                print('Validation complete.')
        except Exception:
            print('\nProblem occurred while running validation.')
            self.printTraceback()
            self.setThreadFailed(True)

    def saveParamsToDisk(self, data=None):
        LOGGER.info('Saving parameters to disk')
        self.outputObj.saveLastRun()
        LOGGER.info('Parameters saved to disk')

    def runModel(self, module, args):
        session_id = str(uuid.uuid4())
        args = args.copy()
        try:
            workspace = args['workspace_dir']
        except KeyError:
            # KeyError thrown when the key 'workspace_dir' is not used in the
            # args dictionary, print an inconsequential error.
            LOGGER.error('Cannot find args id \'workspace_dir\'.')
            workspace = None

        try:
            workspace = workspace.encode(ENCODING)
        except (UnicodeDecodeError, UnicodeEncodeError):
            # UnicodeDecodeError is thrown if the workspace path cannot be
            # correctly decoded according to the filesystem's encoding.
            LOGGER.warn('Cannot decode workspace with the fs encoding. '
                        'workspace=%s, Encoding=%s', workspace, ENCODING)

        try:
            os.makedirs(workspace)
        except OSError as exception:
            #It's okay if the directory already exists, if it fails for
            #some other reason, raise that exception
            if exception.errno != errno.EEXIST:
                raise

        # the module name needs to be extracted differently if it's a python
        # module or if it's a file on disk.  While we're at it, we can also
        # locate the model to be loaded.
        try:
            if os.path.isfile(module):
                model = importlib.import_module(
                    imp.load_source('model', module))
            else:
                model = importlib.import_module(module)
                LOGGER.debug('Loading %s in frozen environment', model)
        except ImportError as exception:
            LOGGER.error('ImportError found when locating %s', module)
            self.printTraceback()
            self.setThreadFailed(True, exception)
            return

        # Create the log filename from the current time and save that in the
        # root of the user's workspace.  The file is actually written to
        # whenever self.write() is called.
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
        filename = '%s-log-%s.txt' % (module, timestamp)
        log_file_uri = os.path.abspath(
            os.path.join(workspace, filename))
        self.log_file = codecs.open(
            log_file_uri, 'w', encoding='utf-8', buffering=0)

        # Now that the log file is open, write the arguments to it.
        self.print_args(args)
        LOGGER.info('Logging will be saved to %s', filename)

        LOGGER.debug('Loaded the model from %s', module)
        LOGGER.info('Executing the loaded model')

        try:
            LOGGER.info('Running InVEST version "%s"', natcap.invest.__version__)
            LOGGER.info('Python architecture: %s', platform.architecture())
            LOGGER.info('Disk space remaining for workspace: %s',
                        fileio.get_free_space(workspace))
            log_thread = threading.Thread(
                target=_log_model, args=(module, args, session_id))
            log_thread.start()

            LOGGER.info(
                'Pointing temporary directory at the workspace at %s',
                args['workspace_dir'])
            temporary_path = os.path.join(args['workspace_dir'], 'tmp')
            try:
                os.makedirs(temporary_path)
            except OSError as exception:
                #It's okay if the directory already exists, if it fails for
                #some other reason, raise that exception
                if exception.errno != errno.EEXIST:
                    raise
            for tmp_variable in ['TMP', 'TEMP', 'TMPDIR']:
                if tmp_variable in os.environ:
                    LOGGER.info(
                        'Updating os.environ["%s"]=%s to %s', tmp_variable,
                        os.environ[tmp_variable], args['workspace_dir'])
                else:
                    LOGGER.info(
                        'Setting os.environ["%s"]=%s', tmp_variable,
                        args['workspace_dir'])

                os.environ[tmp_variable] = temporary_path

            LOGGER.info('Setting tempfile.tempdir to %s', temporary_path)
            tempfile.tempdir = temporary_path

            model_start_time = time.time()
            LOGGER.info('Starting %s', module)

            # If we're including metadata, add per-run metadata here.
            if '_iui_meta' in args:
                logfile_data = {
                    'uri': log_file_uri,
                    'timestamp': timestamp,
                }
                args['_iui_meta']['logfile'] = logfile_data

            model.execute(args)

            log_exit_thread = threading.Thread(
                target=_log_exit_status, args=(session_id, ':)'))
            log_exit_thread.start()
        except Exception as exception:
            # We are explicitly handling all exceptions and below we have a
            # special case for out of disk space
            LOGGER.error('---------------------------------------------------')
            LOGGER.error('---------------------- ERROR ----------------------')
            LOGGER.error('---------------------------------------------------')
            LOGGER.error('Error: exception found while running %s', module)
            LOGGER.debug('')
            self.print_system_info(LOGGER.debug)

            # If the exception indicates that we ran out of disk space, convert
            # e to a more informative exception.
            # EnvironmentError exception contains the errno and __class__
            # attributes
            if issubclass(exception.__class__, EnvironmentError):
                LOGGER.debug(
                    'error %s number %s', exception.__class__, exception.errno)
                LOGGER.debug('Error message: %s', str(exception))

                # Actually check the error code for the exception and use a new
                # custom exception with more information.
                # The exception.__class__  is necessary as the second argument.
                # If not included, will throw an exception, which will cause
                # the thread to silently crash.
                if ((isinstance(exception, WindowsError.__class__) and
                     (exception.errno in [8, 28])) or
                        (isinstance(exception, IOError.__class__) and
                         (exception.errno == 28))):
                    LOGGER.debug('Insufficient disk space detected')
                    exception = InsufficientDiskSpace(
                        'You do not have sufficient disk  space available for '
                        ' this model to finish running.')
                else:
                    LOGGER.debug('Exception not disk-space related')
            else:
                LOGGER.debug('Exception not environment-related')

            #This intentionally handles all exceptions
            self.printTraceback()
            self.setThreadFailed(True, exception)
            log_exit_thread = threading.Thread(
                target=_log_exit_status, args=(
                    session_id, traceback.format_exc()))
            log_exit_thread.start()
            return

        # clean up the temporary folder, but only if we've completed the model
        # successfully.
        try:
            shutil.rmtree(temporary_path)
        except Exception as exception:
            LOGGER.warn(exception)

        if workspace is not None:
            #Try opening up a file explorer to see the results.
            try:
                LOGGER.info('Opening file explorer to workspace directory')
                if platform.system() == 'Windows':
                    # Try to launch a windows file explorer to visit the
                    # workspace directory now that the operation has finished
                    # executing.
                    LOGGER.info('Using windows explorer to view files')
                    subprocess.Popen(
                        'explorer "%s"' % os.path.normpath(workspace))
                else:
                    # Assume we're on linux.  No biggie, just use xdg-open to
                    # use the default file opening scheme.
                    LOGGER.info('Not on windows, using default file browser')
                    subprocess.Popen(['xdg-open', workspace])
            except OSError:
                # OSError is thrown if the given file browser program (whether
                # explorer or xdg-open) cannot be found.  No biggie, just pass.
                LOGGER.error(
                    'Cannot find default file browser. Platform: %s |'
                    ' folder: %s', platform.system(), workspace)

        LOGGER.info('Disk space free: %s', fileio.get_free_space(workspace))
        elapsed_time = round(time.time() - model_start_time, 2)
        # wait 5 seconds just in case it took that long to resolve the remote
        # logging server for whatever reason
        log_thread.join(5.0)
        log_exit_thread.join(5.0)
        LOGGER.info('Elapsed time: %s', self.format_time(elapsed_time))
        LOGGER.info('Finished.')


def _calculate_args_bounding_box(args_dict):
    """Parse through an args dict and calculate the bounding boxes of any GIS
    types found there.

    Args:
        args_dict (dict): a string key and any value pair dictionary.

    Returns:
        bb_intersection, bb_union tuple that's either the lat/lng bounding
            intersection and union bounding boxes of the gis types referred to
            in args_dict.  If no GIS types are present, this is a (None, None)
            tuple."""

    def _merge_bounding_boxes(bb1, bb2, mode):
        """Helper function to merge two bounding boxes through union or
            intersection

            Parameters:
                bb1 (list of float): bounding box of the form
                    [minx, maxy, maxx, miny] or None
                bb2 (list of float): bounding box of the form
                    [minx, maxy, maxx, miny] or None
                mode (string): either "union" or "intersection" indicating the
                    how to combine the two bounding boxes.

            Returns:
                either the intersection or union of bb1 and bb2 depending
                on mode.  If either bb1 or bb2 is None, the other is returned.
                If both are None, None is returned.
            """
        if bb1 is None:
            return bb2
        if bb2 is None:
            return bb1

        if mode == "union":
            comparison_ops = [min, max, max, min]
        if mode == "intersection":
            comparison_ops = [max, min, min, max]

        bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
        return bb_out

    def _merge_local_bounding_boxes(arg, bb_intersection=None, bb_union=None):
        """Allows us to recursively walk a potentially nested dictionary
        and merge the bounding boxes that might be found in the GIS
        types

        Args:
            arg (dict): contains string keys and pairs that might be files to
                gis types.  They can be any other type, including dictionaries.
            bb_intersection (list or None): if list, has the form
                [xmin, ymin, xmax, ymax], where coordinates are in lng, lat
            bb_union (list or None): if list, has the form
                [xmin, ymin, xmax, ymax], where coordinates are in lng, lat

        Returns:
            (intersection, union) bounding box tuples of all filepaths to GIS
            data types found in the dictionary and bb_intersection and bb_union
            inputs.  None, None if no arguments were GIS data types and input
            bounding boxes are None."""

        def _is_gdal(arg):
            """tests if input argument is a path to a gdal raster"""
            if (isinstance(arg, str) or
                    isinstance(arg, unicode)) and os.path.exists(arg):
                raster = gdal.Open(arg)
                if raster is not None:
                    return True
            return False

        def _is_ogr(arg):
            """tests if input argument is a path to an ogr vector"""
            if (isinstance(arg, str) or
                    isinstance(arg, unicode)) and os.path.exists(arg):
                vector = ogr.Open(arg)
                if vector is not None:
                    return True
            return False

        if isinstance(arg, dict):
            # if dict, grab the bb's for all the members in it
            for value in arg.itervalues():
                bb_intersection, bb_union = _merge_local_bounding_boxes(
                    value, bb_intersection, bb_union)
        elif isinstance(arg, list):
            # if list, grab the bb's for all the members in it
            for value in arg:
                bb_intersection, bb_union = _merge_local_bounding_boxes(
                    value, bb_intersection, bb_union)
        else:
            # singular value, test if GIS type, if not, don't update bb's
            # this is an undefined bounding box that gets returned when ogr
            # opens a table only
            local_bb = [0., 0., 0., 0.]
            if _is_gdal(arg):
                local_bb = pygeoprocessing.get_bounding_box(arg)
                projection_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(
                    arg)
                spatial_ref = osr.SpatialReference()
                spatial_ref.ImportFromWkt(projection_wkt)
            elif _is_ogr(arg):
                local_bb = pygeoprocessing.get_datasource_bounding_box(arg)
                spatial_ref = pygeoprocessing.get_spatial_ref_uri(arg)

            try:
                # means there's a GIS type with a well defined bounding box
                # create transform, and reproject local bounding box to lat/lng
                lat_lng_ref = osr.SpatialReference()
                lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng
                to_lat_trans = osr.CoordinateTransformation(
                    spatial_ref, lat_lng_ref)
                for point_index in [0, 2]:
                    local_bb[point_index], local_bb[point_index + 1], _ = (
                        to_lat_trans.TransformPoint(
                            local_bb[point_index], local_bb[point_index + 1]))

                bb_intersection = _merge_bounding_boxes(
                    local_bb, bb_intersection, 'intersection')
                bb_union = _merge_bounding_boxes(
                    local_bb, bb_union, 'union')
            except Exception:
                # All kinds of exceptions from bad transforms or CSV files
                # or dbf files could get us to this point, just don't bother
                # with the local_bb at all
                pass

        return bb_intersection, bb_union

    return _merge_local_bounding_boxes(args_dict)


def _log_exit_status(session_id, status):
    """Log the completion of a model with the given status.

    Parameters:
        session_id (string): a unique string that can be used to identify
            the current session between the model initial start and exit.
        status (string): a string describing the exit status of the model,
            'success' would indicate the successful completion while an
            exception string could indicate a failure.

    Returns:
        None
    """
    logger = logging.getLogger('natcap.invest.iui.executor._log_exit_status')

    try:
        payload = {
            'session_id': session_id,
            'status': status,
        }

        logging_server = _get_logging_server()
        logging_server.log_invest_run(payload, 'exit')
    except Exception as exception:
        # An exception was thrown, we don't care.
        logger.warn(
            'an exception encountered when _log_exit_status %s',
            str(exception))


def _log_model(model_name, model_args, session_id=None):
    """Log information about a model run to a remote server.

    Parameters:
        model_name (string): a python string of the package version.
        model_args (dict): the traditional InVEST argument dictionary.

    Returns:
        None
    """

    logger = logging.getLogger('natcap.invest.iui._log_model')

    def _node_hash():
        """Returns a hash for the current computational node."""
        data = {
            'os': platform.platform(),
            'hostname': platform.node(),
            'userdir': os.path.expanduser('~')
        }
        md5 = hashlib.md5()
        # a json dump will handle non-ascii encodings
        md5.update(json.dumps(data))
        return md5.hexdigest()

    try:
        bounding_box_intersection, bounding_box_union = (
            _calculate_args_bounding_box(model_args))

        payload = {
            'model_name': model_name,
            'invest_release': natcap.invest.__version__,
            'node_hash': _node_hash(),
            'system_full_platform_string': platform.platform(),
            'system_preferred_encoding': locale.getdefaultlocale()[1],
            'system_default_language': locale.getdefaultlocale()[0],
            'bounding_box_intersection': str(bounding_box_intersection),
            'bounding_box_union': str(bounding_box_union),
            'session_id': session_id,
        }

        logging_server = _get_logging_server()
        logging_server.log_invest_run(payload, 'log')
    except Exception as exception:
        # An exception was thrown, we don't care.
        logger.warn(
            'an exception encountered when logging %s', str(exception))


def _get_logging_server(path=None):
    """Returns a remote procedure call logging server from the
    https://bitbucket.org/natcap/natcap_model_logger project.

    Parameters:
        path (string): A Pyro4 compatible url for getting a Proxy object.

    """

    if path is None:
        path = urllib.urlopen(INVEST_USAGE_LOGGER_URL).read().rstrip()
    logging_server = Pyro4.Proxy(path)
    return logging_server
