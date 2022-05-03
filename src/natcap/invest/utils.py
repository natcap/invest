"""InVEST specific code utils."""
import codecs
import math
import os
import contextlib
import logging
import re
import tempfile
import shutil
from datetime import datetime
import time

import pandas
import numpy
from shapely.wkt import loads
from osgeo import gdal
from osgeo import osr
import pygeoprocessing


LOGGER = logging.getLogger(__name__)
_OSGEO_LOGGER = logging.getLogger('osgeo')
LOG_FMT = (
    "%(asctime)s "
    "(%(name)s) "
    "%(module)s.%(funcName)s(%(lineno)d) "
    "%(levelname)s %(message)s")

# GDAL has 5 error levels, python's logging has 6.  We skip logging.INFO.
# A dict clarifies the mapping between levels.
GDAL_ERROR_LEVELS = {
    gdal.CE_None: logging.NOTSET,
    gdal.CE_Debug: logging.DEBUG,
    gdal.CE_Warning: logging.WARNING,
    gdal.CE_Failure: logging.ERROR,
    gdal.CE_Fatal: logging.CRITICAL,
}

# In GDAL 3.0 spatial references no longer ignore Geographic CRS Axis Order
# and conform to Lat first, Lon Second. Transforms expect (lat, lon) order
# as opposed to the GIS friendly (lon, lat). See
# https://trac.osgeo.org/gdal/wiki/rfc73_proj6_wkt2_srsbarn Axis order
# issues. SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) swaps the
# axis order, which will use Lon,Lat order for Geographic CRS, but otherwise
# leaves Projected CRS alone
DEFAULT_OSR_AXIS_MAPPING_STRATEGY = osr.OAMS_TRADITIONAL_GIS_ORDER


def _log_gdal_errors(*args, **kwargs):
    """Log error messages to osgeo.

    All error messages are logged with reasonable ``logging`` levels based
    on the GDAL error level.

    Note:
        This function is designed to accept any number of positional and
        keyword arguments because of some odd forums questions where this
        function was being called with an unexpected number of arguments.
        With this catch-all function signature, we can at least guarantee
        that in the off chance this function is called with the wrong
        parameters, we can at least log what happened.  See the issue at
        https://github.com/natcap/invest/issues/630 for details.

    Args:
        err_level (int): The GDAL error level (e.g. ``gdal.CE_Failure``)
        err_no (int): The GDAL error number.  For a full listing of error
            codes, see: http://www.gdal.org/cpl__error_8h.html
        err_msg (string): The error string.

    Returns:
        ``None``
    """
    if len(args) + len(kwargs) != 3:
        LOGGER.error(
            '_log_gdal_errors was called with an incorrect number of '
            f'arguments.  args: {args}, kwargs: {kwargs}')

    try:
        gdal_args = {}
        for index, key in enumerate(('err_level', 'err_no', 'err_msg')):
            try:
                parameter = args[index]
            except IndexError:
                parameter = kwargs[key]
            gdal_args[key] = parameter
    except KeyError as missing_key:
        LOGGER.exception(
            f'_log_gdal_errors called without the argument {missing_key}. '
            f'Called with args: {args}, kwargs: {kwargs}')

        # Returning from the function because we don't have enough
        # information to call the ``osgeo_logger`` in the way we intended.
        return

    err_level = gdal_args['err_level']
    err_no = gdal_args['err_no']
    err_msg = gdal_args['err_msg'].replace('\n', '')
    _OSGEO_LOGGER.log(
        level=GDAL_ERROR_LEVELS[err_level],
        msg=f'[errno {err_no}] {err_msg}')


@contextlib.contextmanager
def capture_gdal_logging():
    """Context manager for logging GDAL errors with python logging.

    GDAL error messages are logged via python's logging system, at a severity
    that corresponds to a log level in ``logging``.  Error messages are logged
    with the ``osgeo.gdal`` logger.

    Args:
        ``None``

    Returns:
        ``None``
    """
    gdal.PushErrorHandler(_log_gdal_errors)
    try:
        yield
    finally:
        gdal.PopErrorHandler()


def _format_time(seconds):
    """Render the integer number of seconds as a string. Returns a string."""
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
def prepare_workspace(
        workspace, name, logging_level=logging.NOTSET, exclude_threads=None):
    """Prepare the workspace."""
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    modelname = '-'.join(name.replace(':', '').split(' '))
    logfile = os.path.join(
        workspace,
        'InVEST-{modelname}-log-{timestamp}.txt'.format(
            modelname=modelname,
            timestamp=datetime.now().strftime("%Y-%m-%d--%H_%M_%S")))

    with capture_gdal_logging(), log_to_file(logfile,
                                             exclude_threads=exclude_threads,
                                             logging_level=logging_level):
        with sandbox_tempdir(dir=workspace):
            logging.captureWarnings(True)
            # If invest is launched as a subprocess (e.g. the Workbench)
            # the parent process can rely on this announcement to know the
            # logfile path (within []), and to know the invest process started.
            LOGGER.log(100, 'Writing log messages to [%s]', logfile)
            start_time = time.time()
            try:
                yield
            except Exception:
                LOGGER.exception(f'Exception while executing {modelname}')
                raise
            finally:
                LOGGER.info('Elapsed time: %s',
                            _format_time(round(time.time() - start_time, 2)))
                logging.captureWarnings(False)
                LOGGER.info('Execution finished')


class ThreadFilter(logging.Filter):
    """Filters out log messages issued by the given thread.

    Any log messages generated by a thread with the name matching the
    threadname provided to the constructor will be excluded.
    """

    def __init__(self, thread_name):
        """Construct a ThreadFilter.

        Args:
            thread_name (string): The thread name to filter on.

        """
        logging.Filter.__init__(self)
        self.thread_name = thread_name

    def filter(self, record):
        """Filter the given log record.

        Args:
            record (log record): The log record to filter.

        Returns:
            True if the record should be included, false if not.
        """
        if record.threadName == self.thread_name:
            return False
        return True


@contextlib.contextmanager
def log_to_file(logfile, exclude_threads=None, logging_level=logging.NOTSET,
                log_fmt=LOG_FMT, date_fmt=None):
    """Log all messages within this context to a file.

    Args:
        logfile (string): The path to where the logfile will be written.
            If there is already a file at this location, it will be
            overwritten.
        exclude_threads=None (list): If None, logging from all threads will be
            included in the log. If a list, it must be a list of string thread
            names that should be excluded from logging in this file.
        logging_level=logging.NOTSET (int): The logging threshold.  Log
            messages with a level less than this will be automatically
            excluded from the logfile.  The default value (``logging.NOTSET``)
            will cause all logging to be captured.
        log_fmt=LOG_FMT (string): The logging format string to use.  If not
            provided, ``utils.LOG_FMT`` will be used.
        date_fmt (string): The logging date format string to use.
            If not provided, ISO8601 format will be used.


    Yields:
        ``handler``: An instance of ``logging.FileHandler`` that
            represents the file that is being written to.

    Returns:
        ``None``
    """
    try:
        if os.path.exists(logfile):
            LOGGER.warning('Logfile %s exists and will be overwritten',
                           logfile)
    except SystemError:
        # This started happening in Windows tests:
        #  SystemError: <built-in function stat> returned NULL without
        #  setting an error
        # Looking at https://bugs.python.org/issue28040#msg276223, this might
        # be a low-level python error.
        pass

    handler = logging.FileHandler(logfile, 'w', encoding='UTF-8')
    formatter = logging.Formatter(log_fmt, date_fmt)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)
    root_logger.addHandler(handler)
    handler.setFormatter(formatter)
    handler.setLevel(logging_level)

    if exclude_threads is not None:
        for threadname in exclude_threads:
            thread_filter = ThreadFilter(threadname)
            handler.addFilter(thread_filter)

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

    Args:
        suffix='' (string): a suffix for the name of the directory.
        prefix='tmp' (string): the prefix to use for the directory name.
        dir=None (string or None): If a string, a directory that should be
            the parent directory of the new temporary directory.  If None,
            tempfile will determine the appropriate tempdir to use as the
            parent folder.

    Yields:
        ``sandbox`` (string): The path to the new folder on disk.

    Returns:
        ``None``
    """
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

    Args:
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

    Args:
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
    kernel_dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    cols_per_block, rows_per_block = kernel_band.GetBlockSize()

    n_cols = kernel_dataset.RasterXSize
    n_rows = kernel_dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    integration = 0.0
    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # Numpy creates index rasters as ints by default, which sometimes
            # creates problems on 32-bit builds when we try to add Int32
            # matrices to float64 matrices.
            row_indices, col_indices = numpy.indices((row_block_width,
                                                      col_block_width),
                                                     dtype=float)

            row_indices += float(row_offset - max_distance)
            col_indices += float(col_offset - max_distance)

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
    kernel_band.FlushCache()
    kernel_dataset.FlushCache()

    for block_data in pygeoprocessing.iterblocks(
            (kernel_filepath, 1), offset_only=True):
        kernel_block = kernel_band.ReadAsArray(**block_data)
        kernel_block /= integration
        kernel_band.WriteArray(kernel_block, xoff=block_data['xoff'],
                               yoff=block_data['yoff'])

    kernel_band.FlushCache()
    kernel_dataset.FlushCache()
    kernel_band = None
    kernel_dataset = None


def build_file_registry(base_file_path_list, file_suffix):
    """Combine file suffixes with key names, base filenames, and directories.

    Args:
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
        for file_key, file_payload in base_file_dict.items():
            # check for duplicate keys
            if file_key in f_reg:
                duplicate_keys.add(file_key)
            else:
                # handle the case whether it's a filename or a list of strings
                if isinstance(file_payload, str):
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


def build_lookup_from_csv(
        table_path, key_field, column_list=None, to_lower=True):
    """Read a CSV table into a dictionary indexed by ``key_field``.

    Creates a dictionary from a CSV whose keys are unique entries in the CSV
    table under the column named by ``key_field`` and values are dictionaries
    indexed by the other columns in ``table_path`` including ``key_field``
    whose values are the values on that row of the CSV table.

    If an entire row is NA/NaN (including ``key_field``) then it is dropped
    from the table and a warning is given of the dropped rows.

    Args:
        table_path (string): path to a CSV file containing at
            least the header key_field
        key_field: (string): a column in the CSV file at `table_path` that
            can uniquely identify each row in the table and sets the row index.
        column_list (list): a list of column names to subset from the CSV
            file, default=None
        to_lower (bool): if True, converts all unicode in the CSV,
            including headers and values to lowercase, otherwise uses raw
            string values. default=True.

    Returns:
        lookup_dict (dict): a dictionary of the form
        {key_field_0: {csv_header_0: value0, csv_header_1: value1...},
        key_field_1: {csv_header_0: valuea, csv_header_1: valueb...}}

        if ``to_lower`` all strings including key_fields and values are
        converted to lowercase unicode.

    Raise:
        ValueError
            If ValueError occurs during conversion to dictionary.
        KeyError
            If ``key_field`` is not present during ``set_index`` call.
    """
    # Reassign to avoid mutation
    col_list = column_list
    # if a list of columns are provided to use and return, make sure
    # 'key_field' is one of them.
    if col_list and key_field not in col_list:
        col_list.append(key_field)

    table = read_csv_to_dataframe(
        table_path, to_lower=to_lower, sep=None, index_col=False,
        engine='python')

    # if 'to_lower`, case handling is done before trying to access the data.
    # the columns are stripped of leading/trailing whitespace in
    # ``read_csv_to_dataframe``, and also lowercased if ``to_lower`` so we only
    # need to convert the rest of the table.
    if to_lower:
        key_field = key_field.lower()
        # lowercase column names
        if col_list:
            col_list = [col.lower() for col in col_list]
        # lowercase values
        table = table.applymap(
            lambda x: x.lower() if isinstance(x, str) else x)

    # Set 'key_field' as the index of the dataframe
    try:
        table.set_index(key_field, drop=False, inplace=True)
    except KeyError:
        # If 'key_field' is not a column then KeyError is raised for using
        # it as the index column
        LOGGER.error(f"'key_field' : '{key_field}' could not be found as a"
                     f" column in the table. Table path: {table_path}.")
        raise

    # Subset dataframe by columns if desired
    if col_list:
        table = table.loc[:, col_list]

    # look for NaN values and warn if any are found.
    table_na = table.isna()
    if table_na.values.any():
        LOGGER.warning(
            f"Empty or NaN values were found in the table: {table_path}.")
    # look to see if an entire row is NA values
    table_na_rows = table_na.all(axis=1)
    na_rows = table_na_rows.index[table_na_rows].tolist()
    # if a completely empty row, drop it
    if na_rows:
        LOGGER.warning(
            "Encountered an entirely blank row on line(s)"
            f" {[x+2 for x in na_rows]}. Dropping rows from table.")
        table.dropna(how="all", inplace=True)
    # fill the rest of empty or NaN values with empty string
    table.fillna(value="", inplace=True)
    try:
        lookup_dict = table.to_dict(orient='index')
    except ValueError:
        # If 'key_field' is not unique then a value error is raised.
        LOGGER.error(f"The 'key_field' : '{key_field}' column values are not"
                     f" unique: {table.index.tolist()}")
        raise

    return lookup_dict


def read_csv_to_dataframe(
        path, to_lower=False, sep=None, encoding=None, engine='python',
        **kwargs):
    """Return a dataframe representation of the CSV.

    Wrapper around ``pandas.read_csv`` that standardizes the column names by
    stripping leading/trailing whitespace and optionally making all lowercase.
    This helps avoid common errors caused by user-supplied CSV files with
    column names that don't exactly match the specification. Strips
    leading/trailing whitespace from data entries as well.

    Args:
        path (string): path to a CSV file
        to_lower (bool): if True, convert all column names to lowercase
        sep: separator to pass to pandas.read_csv. Defaults to None, which
            lets the Python engine infer the separator (if engine='python').
        encoding (string): name of encoding codec to pass to `pandas.read_csv`.
            Defaults to None. Setting engine='python' when encoding=None allows
            a lot of non-UTF8 encodings to be read without raising an error.
            Any special characters in other encodings may get replaced with the
            replacement character.
            If encoding=None, and the file begins with a BOM, the encoding gets
            set to 'utf-8-sig'; otherwise the BOM causes an error.
        engine (string): kwarg for pandas.read_csv: 'c', 'python', or None.
            Defaults to 'python' (see note about encoding).
        **kwargs: any kwargs that are valid for ``pandas.read_csv``

    Returns:
        pandas.DataFrame with the contents of the given CSV

    """
    # Check if the file encoding is UTF-8 BOM first
    # allow encoding kwarg to override this if it's provided
    if not encoding and has_utf8_bom(path):
        encoding = 'utf-8-sig'
    try:
        dataframe = pandas.read_csv(path, engine=engine, encoding=encoding,
                                    sep=sep, **kwargs)
    except UnicodeDecodeError as error:
        LOGGER.error(
            f'{path} must be encoded as utf-8 or ASCII')
        raise error

    # this won't work on integer types, which happens if you set header=None
    # however, there's little reason to use this function if there's no header
    dataframe.columns = dataframe.columns.str.strip()
    if to_lower:
        dataframe.columns = dataframe.columns.str.lower()

    # Remove values with leading ('^ +') and trailing (' +$') whitespace.
    # Regular expressions using 'replace' only substitute on strings.
    dataframe = dataframe.replace(r"^ +| +$", r"", regex=True)

    return dataframe


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


def mean_pixel_size_and_area(pixel_size_tuple):
    """Convert to mean and raise Exception if they are not close.

    Parameter:
        pixel_size_tuple (tuple): a 2 tuple indicating the x/y size of a
            pixel.

    Returns:
        tuple of (mean absolute average of pixel_size, area of pixel size)

    Raises:
        ValueError if the dimensions of pixel_size_tuple are not almost
            square.

    """
    x_size, y_size = abs(pixel_size_tuple[0]), abs(pixel_size_tuple[1])
    if not numpy.isclose(x_size, y_size):
        raise ValueError(
            "pixel size is not square. dimensions: %s" % repr(
                pixel_size_tuple))

    return (x_size, x_size*y_size)


def create_coordinate_transformer(
        base_ref, target_ref,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Create a spatial reference coordinate transformation function.

    Args:
        base_ref (osr spatial reference): A defined spatial reference to
            transform FROM
        target_ref (osr spatial reference): A defined spatial reference
            to transform TO
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``utils.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter should
            not be changed unless you know what you are doing.

    Returns:
        An OSR Coordinate Transformation object

    """
    # Make a copy of the base and target spatial references to avoid side
    # effects from mutation of setting the axis mapping strategy
    base_ref_wkt = base_ref.ExportToWkt()
    target_ref_wkt = target_ref.ExportToWkt()

    base_ref_copy = osr.SpatialReference()
    target_ref_copy = osr.SpatialReference()

    base_ref_copy.ImportFromWkt(base_ref_wkt)
    target_ref_copy.ImportFromWkt(target_ref_wkt)

    base_ref_copy.SetAxisMappingStrategy(osr_axis_mapping_strategy)
    target_ref_copy.SetAxisMappingStrategy(osr_axis_mapping_strategy)

    transformer = osr.CreateCoordinateTransformation(
        base_ref_copy, target_ref_copy)
    return transformer


def _assert_vectors_equal(
        expected_vector_path, actual_vector_path, field_value_atol=1e-3):
    """Assert two vectors are equal.

    Assert spatial reference, feature count, geometries, field names, and
    values are equal with no respect to order of field names or geometries.

    Note: this assertion may fail incorrectly. The comparison algorithm
    allows for float imprecision if geometry coords are in the same order. It
    also allows coords to be in a different order if the numbers match exactly.
    If the coords are in a different order and have float imprecision, the
    assertion will fail.

    Args:
        actual_vector_path (string): path on disk to a gdal Vector dataset.
        expected_vector_path (string): path on disk to a gdal Vector dataset
            to use as the ground truth.
        field_value_atol (float): the absolute tolerance for comparing field
            attribute values, default=1e-3.

    Returns:
        None on success

    Raise:
        AssertionError
           If vector projections, feature counts, field names, or geometries
           do not match.
    """
    try:
        # Open vectors
        actual_vector = gdal.OpenEx(actual_vector_path, gdal.OF_VECTOR)
        actual_layer = actual_vector.GetLayer()
        expected_vector = gdal.OpenEx(expected_vector_path, gdal.OF_VECTOR)
        expected_layer = expected_vector.GetLayer()

        # Check projections
        expected_projection = expected_layer.GetSpatialRef()
        expected_projection_wkt = expected_projection.ExportToWkt()
        actual_projection = actual_layer.GetSpatialRef()
        actual_projection_wkt = actual_projection.ExportToWkt()
        if expected_projection_wkt != actual_projection_wkt:
            raise AssertionError(
                "Vector projections are not the same. \n"
                f"Expected projection wkt: {expected_projection_wkt}. \n"
                f"Actual projection wkt: {actual_projection_wkt}. ")

        # Check feature count
        actual_feat_count = actual_layer.GetFeatureCount()
        expected_feat_count = expected_layer.GetFeatureCount()
        if expected_feat_count != actual_feat_count:
            raise AssertionError(
                "Vector feature counts are not the same. \n"
                f"Expected feature count: {expected_feat_count}. \n"
                f"Actual feature count: {actual_feat_count}. ")

        # Check field names
        expected_field_names = [field.name for field in expected_layer.schema]
        actual_field_names = [field.name for field in actual_layer.schema]
        if sorted(expected_field_names) != sorted(actual_field_names):
            raise AssertionError(
                "Vector field names are not the same. \n"
                f"Expected field names: {sorted(expected_field_names)}. \n"
                f"Actual field names: {sorted(actual_field_names)}. ")

        # Check field values and geometries
        for expected_feature in expected_layer:
            fid = expected_feature.GetFID()
            expected_values = [
                expected_feature.GetField(field)
                for field in expected_field_names]

            actual_feature = actual_layer.GetFeature(fid)
            actual_values = [
                actual_feature.GetField(field)
                for field in expected_field_names]

            for av, ev in zip(actual_values, expected_values):
                if av is not None:
                    # Number comparison
                    if isinstance(av, int) or isinstance(av, float):
                        if not numpy.allclose(numpy.array([av]),
                                              numpy.array([ev]),
                                              atol=field_value_atol):
                            raise AssertionError(
                                "Vector field values are not equal: \n"
                                f"Expected value: {ev}. \n"
                                f"Actual value: {av}. ")
                    # String and other comparison
                    else:
                        if av != ev:
                            raise AssertionError(
                                "Vector field values are not equal. \n"
                                f"Expected value : {ev}. \n"
                                f"Actual value : {av}. ")
                else:
                    if ev is not None:
                        raise AssertionError(
                            "Vector field values are not equal: \n"
                            f"Expected value: {ev}. \n"
                            f"Actual value: {av}. ")

            expected_geom = expected_feature.GetGeometryRef()
            expected_geom_wkt = expected_geom.ExportToWkt()
            actual_geom = actual_feature.GetGeometryRef()
            actual_geom_wkt = actual_geom.ExportToWkt()
            expected_geom_shapely = loads(expected_geom_wkt)
            actual_geom_shapely = loads(actual_geom_wkt)
            # confusingly named, `equals` compares numbers exactly but
            # allows coords to be in a different order.
            # `equals_exact` compares numbers with a tolerance but
            # coords must be in the same order.
            geoms_equal = (
                expected_geom_shapely.equals(actual_geom_shapely) or
                expected_geom_shapely.equals_exact(actual_geom_shapely, 1e-6))
            if not geoms_equal:
                raise AssertionError(
                    "Vector geometry assertion fail. \n"
                    f"Expected geometry: {expected_geom_wkt}. \n"
                    f"Actual geometry: {actual_geom_wkt}. ")

            expected_feature = None
            actual_feature = None
    finally:
        actual_layer = None
        actual_vector = None
        expected_layer = None
        expected_vector = None

    return None


def has_utf8_bom(textfile_path):
    """Determine if the text file has a UTF-8 byte-order marker.

    Args:
        textfile_path (str): The path to a file on disk.

    Returns:
        A bool indicating whether the textfile has a BOM.  If ``True``, a BOM
        is present.

    """
    with open(textfile_path, 'rb') as file_obj:
        first_line = file_obj.readline()
        return first_line.startswith(codecs.BOM_UTF8)


def reclassify_raster(
        raster_path_band, value_map, target_raster_path, target_datatype,
        target_nodata, error_details):
    """A wrapper function for calling ``pygeoprocessing.reclassify_raster``.

    This wrapper function is helpful when added as a ``TaskGraph.task`` so
    a better error message can be provided to the users if a
    ``pygeoprocessing.ReclassificationMissingValuesError`` is raised.

    Args:
        raster_path_band (tuple): a tuple including file path to a raster
            and the band index to operate over. ex: (path, band_index)
        value_map (dictionary): a dictionary of values of
            {source_value: dest_value, ...} where source_value's type is the
            same as the values in ``base_raster_path`` at band ``band_index``.
            Must contain at least one value.
        target_raster_path (string): target raster output path; overwritten if
            it exists
        target_datatype (gdal type): the numerical type for the target raster
        target_nodata (numerical type): the nodata value for the target raster
            Must be the same type as target_datatype
        error_details (dict): a dictionary with key value pairs that provide
            more context for a raised
            ``pygeoprocessing.ReclassificationMissingValuesError``.
            keys must be {'raster_name', 'column_name', 'table_name'}. Values
            each key represent:

                'raster_name' - string for the raster name being reclassified
                'column_name' - name of the table column that ``value_map``
                dictionary keys came from.
                'table_name' - table name that ``value_map`` came from.

    Returns:
        None

    Raises:
        ValueError if ``values_required`` is ``True`` and a pixel value from
        ``raster_path_band`` is not a key in ``value_map``.
    """
    # Error early if 'error_details' keys are invalid
    raster_name = error_details['raster_name']
    column_name = error_details['column_name']
    table_name = error_details['table_name']

    try:
        pygeoprocessing.reclassify_raster(
            raster_path_band, value_map, target_raster_path, target_datatype,
            target_nodata, values_required=True)
    except pygeoprocessing.ReclassificationMissingValuesError as err:
        error_message = (
            f"Values in the {raster_name} raster were found that are not"
            f" represented under the '{column_name}' column of the"
            f" {table_name} table. The missing values found in the"
            f" {raster_name} raster but not the table are:"
            f" {err.missing_values}.")
        raise ValueError(error_message)


def array_equals_nodata(array, nodata):
    """Check for the presence of ``nodata`` values in ``array``.

    The comparison supports ``numpy.nan`` nodata values.

    Args:
        array (numpy array): the array to mask for nodata values.
        nodata (number): the nodata value to check for. Supports ``numpy.nan``.

    Returns:
        A boolean numpy array with values of 1 where ``array`` is equal to
        ``nodata`` and 0 otherwise.
    """
    # comparing an integer array against numpy.nan works correctly and is
    # faster than using numpy.isclose().
    if numpy.issubdtype(array.dtype, numpy.integer):
        return array == nodata
    return numpy.isclose(array, nodata, equal_nan=True)


def matches_format_string(test_string, format_string):
    """Assert that a given string matches a given format string.

    This means that the given test string could be derived from the given
    format string by replacing replacement fields with any text. For example,
    the string 'Value "foo" is invalid.' matches the format string
    'Value "{value}" is invalid.'

    Args:
        test_string (str): string to test.
        format_string (str): format string, which may contain curly-brace
            delimited replacement fields

    Returns:
        True if test_string matches format_string, False if not.
    """
    # replace all curly-braced substrings of the format string with '.*'
    # to make a regular expression
    pattern = re.sub(r'\{.*\}', '.*', format_string)
    # check if the given string matches the format string pattern
    if re.fullmatch(pattern, test_string):
        return True
    return False
