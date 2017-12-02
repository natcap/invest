"""InVEST specific code utils."""
import math
import os
import contextlib
import logging
import threading
import tempfile
import shutil
from datetime import datetime
import time
import csv

import numpy
from osgeo import gdal
from osgeo import osr
import pygeoprocessing

LOGGER = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s %(name)-18s %(levelname)-8s %(message)s"
DATE_FMT = "%m/%d/%Y %H:%M:%S "

# GDAL has 5 error levels, python's logging has 6.  We skip logging.INFO.
# A dict clarifies the mapping between levels.
GDAL_ERROR_LEVELS = {
    gdal.CE_None: logging.NOTSET,
    gdal.CE_Debug: logging.DEBUG,
    gdal.CE_Warning: logging.WARNING,
    gdal.CE_Failure: logging.ERROR,
    gdal.CE_Fatal: logging.CRITICAL,
}


@contextlib.contextmanager
def capture_gdal_logging():
    """Context manager for logging GDAL errors with python logging.

    GDAL error messages are logged via python's logging system, at a severity
    that corresponds to a log level in ``logging``.  Error messages are logged
    with the ``osgeo.gdal`` logger.

    Parameters:
        ``None``

    Returns:
        ``None``"""
    osgeo_logger = logging.getLogger('osgeo')

    def _log_gdal_errors(err_level, err_no, err_msg):
        """Log error messages to osgeo.

        All error messages are logged with reasonable ``logging`` levels based
        on the GDAL error level.

        Parameters:
            err_level (int): The GDAL error level (e.g. ``gdal.CE_Failure``)
            err_no (int): The GDAL error number.  For a full listing of error
                codes, see: http://www.gdal.org/cpl__error_8h.html
            err_msg (string): The error string.

        Returns:
            ``None``"""
        osgeo_logger.log(
            level=GDAL_ERROR_LEVELS[err_level],
            msg='[errno {err}] {msg}'.format(
                err=err_no, msg=err_msg.replace('\n', ' ')))

    gdal.PushErrorHandler(_log_gdal_errors)
    try:
        yield
    finally:
        gdal.PopErrorHandler()


def _format_time(seconds):
    """Render the integer number of seconds as a string.  Returns a string.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    hours = int(hours)
    minutes = int(minutes)

    if hours > 0:
        return "%sh %sm %ss" % (hours, minutes, seconds)

    if minutes > 0:
        return "%sm %ss" % (minutes, seconds)
    return "%ss" % seconds


@contextlib.contextmanager
def prepare_workspace(workspace, name, logging_level=logging.NOTSET):
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    logfile = os.path.join(
        workspace,
        'InVEST-{modelname}-log-{timestamp}.txt'.format(
            modelname='-'.join(name.replace(':', '').split(' ')),
            timestamp=datetime.now().strftime("%Y-%m-%d--%H_%M_%S")))

    with capture_gdal_logging(), log_to_file(logfile,
                                             logging_level=logging_level):
        with sandbox_tempdir(dir=workspace):
            logging.captureWarnings(True)
            LOGGER.info('Writing log messages to %s', logfile)
            start_time = time.time()
            try:
                yield
            finally:
                LOGGER.info('Elapsed time: %s',
                            _format_time(round(time.time() - start_time, 2)))
                logging.captureWarnings(False)


class ThreadFilter(logging.Filter):
    """When used, this filters out log messages that were recorded from other
    threads.  This is especially useful if we have logging coming from several
    concurrent threads.
    Arguments passed to the constructor:
        thread_name - the name of the thread to identify.  If the record was
            reported from this thread name, it will be passed on.
    """
    def __init__(self, thread_name):
        logging.Filter.__init__(self)
        self.thread_name = thread_name

    def filter(self, record):
        if record.threadName == self.thread_name:
            return True
        return False


@contextlib.contextmanager
def log_to_file(logfile, threadname=None, logging_level=logging.NOTSET,
                log_fmt=LOG_FMT, date_fmt=DATE_FMT):
    """Log all messages within this context to a file.

    Parameters:
        logfile (string): The path to where the logfile will be written.
            If there is already a file at this location, it will be
            overwritten.
        threadname=None (string): If None, logging from all threads will be
            included in the log.  If a string, only logging from the thread
            with the same name will be included in the logfile.
        logging_level=logging.NOTSET (int): The logging threshold.  Log
            messages with a level less than this will be automatically
            excluded from the logfile.  The default value (``logging.NOTSET``)
            will cause all logging to be captured.
        log_fmt=LOG_FMT (string): The logging format string to use.  If not
            provided, ``utils.LOG_FMT`` will be used.
        date_fmt=DATE_FMT (string): The logging date format string to use.
            If not provided, ``utils.DATE_FMT`` will be used.


    Yields:
        ``handler``: An instance of ``logging.FileHandler`` that
            represents the file that is being written to.

    Returns:
        ``None``"""
    if os.path.exists(logfile):
        LOGGER.warn('Logfile %s exists and will be overwritten', logfile)

    if not threadname:
        threadname = threading.current_thread().name

    handler = logging.FileHandler(logfile, 'w', encoding='UTF-8')
    formatter = logging.Formatter(log_fmt, date_fmt)
    thread_filter = ThreadFilter(threadname)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)
    root_logger.addHandler(handler)
    handler.addFilter(thread_filter)
    handler.setFormatter(formatter)
    handler.setLevel(logging_level)
    try:
        yield handler
    finally:
        handler.close()
        root_logger.removeHandler(handler)


@contextlib.contextmanager
def sandbox_tempdir(suffix='', prefix='tmp', dir=None):
    """Create a temporary directory for this context and clean it up on exit.

    Parameters are identical to those for :py:func:`tempfile.mkdtemp`.

    When the context manager exits, the created temporary directory is
    recursively removed.

    Parameters:
        suffix='' (string): a suffix for the name of the directory.
        prefix='tmp' (string): the prefix to use for the directory name.
        dir=None (string or None): If a string, a directory that should be
            the parent directory of the new temporary directory.  If None,
            tempfile will determine the appropriate tempdir to use as the
            parent folder.

    Yields:
        ``sandbox`` (string): The path to the new folder on disk.

    Returns:
        ``None``"""
    sandbox = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)

    try:
        yield sandbox
    finally:
        try:
            shutil.rmtree(sandbox)
        except OSError:
            LOGGER.exception('Could not remove sandbox %s', sandbox)


def make_suffix_string(args, suffix_key):
    """Make an InVEST appropriate suffix string.

    Creates an InVEST appropriate suffix string  given the args dictionary and
    suffix key.  In general, prepends an '_' when necessary and generates an
    empty string when necessary.

    Parameters:
        args (dict): the classic InVEST model parameter dictionary that is
            passed to `execute`.
        suffix_key (string): the key used to index the base suffix.

    Returns:
        If `suffix_key` is not in `args`, or `args['suffix_key']` is ""
            return "",
        If `args['suffix_key']` starts with '_' return `args['suffix_key']`
            else return '_'+`args['suffix_key']`
    """
    try:
        file_suffix = args[suffix_key]
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    return file_suffix


def exponential_decay_kernel_raster(expected_distance, kernel_filepath):
    """Create a raster-based exponential decay kernel.

    The raster created will be a tiled GeoTiff, with 256x256 memory blocks.

    Parameters:
        expected_distance (int or float): The distance (in pixels) of the
            kernel's radius, the distance at which the value of the decay
            function is equal to `1/e`.
        kernel_filepath (string): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.

    Returns:
        None
    """
    max_distance = expected_distance * 5
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_filepath.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    srs = osr.SpatialReference()
    srs.SetUTM(11, 1)
    srs.SetWellKnownGeogCS('NAD27')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    cols_per_block, rows_per_block = kernel_band.GetBlockSize()

    n_cols = kernel_dataset.RasterXSize
    n_rows = kernel_dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    integration = 0.0
    for row_block_index in xrange(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in xrange(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # Numpy creates index rasters as ints by default, which sometimes
            # creates problems on 32-bit builds when we try to add Int32
            # matrices to float64 matrices.
            row_indices, col_indices = numpy.indices((row_block_width,
                                                      col_block_width),
                                                     dtype=numpy.float)

            row_indices += numpy.float(row_offset - max_distance)
            col_indices += numpy.float(col_offset - max_distance)

            kernel_index_distances = numpy.hypot(
                row_indices, col_indices)
            kernel = numpy.where(
                kernel_index_distances > max_distance, 0.0,
                numpy.exp(-kernel_index_distances / expected_distance))
            integration += numpy.sum(kernel)

            kernel_band.WriteArray(kernel, xoff=col_offset,
                                   yoff=row_offset)

    # Need to flush the kernel's cache to disk before opening up a new Dataset
    # object in interblocks()
    kernel_dataset.FlushCache()

    for block_data, kernel_block in pygeoprocessing.iterblocks(
            kernel_filepath):
        kernel_block /= integration
        kernel_band.WriteArray(kernel_block, xoff=block_data['xoff'],
                               yoff=block_data['yoff'])


def build_file_registry(base_file_path_list, file_suffix):
    """Combine file suffixes with key names, base filenames, and directories.

    Parameters:
        base_file_tuple_list (list): a list of (dict, path) tuples where
            the dictionaries have a 'file_key': 'basefilename' pair, or
            'file_key': list of 'basefilename's.  'path'
            indicates the file directory path to prepend to the basefile name.
        file_suffix (string): a string to append to every filename, can be
            empty string

    Returns:
        dictionary of 'file_keys' from the dictionaries in
        `base_file_tuple_list` mapping to full file paths with suffixes or
        lists of file paths with suffixes depending on the original type of
        the 'basefilename' pair.

    Raises:
        ValueError if there are duplicate file keys or duplicate file paths.
        ValueError if a path is not a string or a list of strings.
    """
    all_paths = set()
    duplicate_keys = set()
    duplicate_paths = set()
    f_reg = {}

    def _build_path(base_filename, path):
        """Internal helper to avoid code duplication."""
        pre, post = os.path.splitext(base_filename)
        full_path = os.path.join(path, pre+file_suffix+post)

        # Check for duplicate keys or paths
        if full_path in all_paths:
            duplicate_paths.add(full_path)
        else:
            all_paths.add(full_path)
        return full_path

    for base_file_dict, path in base_file_path_list:
        for file_key, file_payload in base_file_dict.iteritems():
            # check for duplicate keys
            if file_key in f_reg:
                duplicate_keys.add(file_key)
            else:
                # handle the case whether it's a filename or a list of strings
                if isinstance(file_payload, basestring):
                    full_path = _build_path(file_payload, path)
                    f_reg[file_key] = full_path
                elif isinstance(file_payload, list):
                    f_reg[file_key] = []
                    for filename in file_payload:
                        full_path = _build_path(filename, path)
                        f_reg[file_key].append(full_path)
                else:
                    raise ValueError(
                        "Unknown type in base_file_dict[%s]=%s" % (
                            file_key, path))

    if len(duplicate_paths) > 0 or len(duplicate_keys):
        raise ValueError(
            "Cannot consolidate because of duplicate paths or keys: "
            "duplicate_keys: %s duplicate_paths: %s" % (
                duplicate_keys, duplicate_paths))

    return f_reg


def _attempt_float(value):
    """Attempt to cast `value` to a float.  If fail, return original value."""
    try:
        return float(value)
    except ValueError:
        return value


def build_lookup_from_csv(
        table_path, key_field, to_lower=True, numerical_cast=True,
        warn_if_missing=True):
    """Read a CSV table into a dictionary indexed by `key_field`.

    Creates a dictionary from a CSV whose keys are unique entries in the CSV
    table under the column named by `key_field` and values are dictionaries
    indexed by the other columns in `table_path` including `key_field` whose
    values are the values on that row of the CSV table.

    Parameters:
        table_path (string): path to a CSV file containing at
            least the header key_field
        key_field: (string): a column in the CSV file at `table_path` that
            can uniquely identify each row in the table.  If `numerical_cast`
            is true the values will be cast to floats/ints/unicode if
            possible.
        to_lower (string): if True, converts all unicode in the CSV,
            including headers and values to lowercase, otherwise uses raw
            string values.
        numerical_cast (bool): If true, all values in the CSV table will
            attempt to be cast to a floating point type; if it fails will be
            left as unicode.  If false, all values will be considered raw
            unicode.
        warn_if_missing (bool): If True, warnings are logged if there are
            empty headers or value rows.

    Returns:
        lookup_dict (dict): a dictionary of the form {
                key_field_0: {csv_header_0: value0, csv_header_1: value1...},
                key_field_1: {csv_header_0: valuea, csv_header_1: valueb...}
            }

        if `to_lower` all strings including key_fields and values are
        converted to lowercase unicde.  if `numerical_cast` all values
        that can be represented as floats are, otherwise unicode.
    """
    with open(table_path, 'rbU') as table_file:
        reader = csv.reader(table_file)
        header_row = reader.next()
        header_row = [unicode(x) for x in header_row]
        key_field = unicode(key_field)
        if to_lower:
            key_field = key_field.lower()
            header_row = [x.lower() for x in header_row]
        if key_field not in header_row:
            raise ValueError(
                '%s expected in %s for the CSV file at %s' % (
                    key_field, header_row, table_path))
        if warn_if_missing and '' in header_row:
            LOGGER.warn(
                "There are empty strings in the header row at %s", table_path)
        key_index = header_row.index(key_field)
        lookup_dict = {}
        for row in reader:
            if to_lower:
                row = [x.lower() for x in row]
            if numerical_cast:
                row = [_attempt_float(x) for x in row]
            if warn_if_missing and '' in row:
                LOGGER.warn(
                    "There are empty strings in row %s in %s", row,
                    table_path)
            lookup_dict[row[key_index]] = dict(zip(header_row, row))
        return lookup_dict


def make_directories(directory_list):
    """Create directories in `directory_list` if they do not already exist."""
    if not isinstance(directory_list, list):
        raise ValueError(
            "Expected `directory_list` to be an instance of `list` instead "
            "got type %s instead", type(directory_list))

    for path in directory_list:
        # From http://stackoverflow.com/a/14364249/42897
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
