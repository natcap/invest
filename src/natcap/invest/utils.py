"""InVEST specific code utils."""
import ast
import codecs
import contextlib
import functools
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from urllib.parse import urlparse
from datetime import datetime

import natcap.invest
import numpy
import pandas
import pygeoprocessing
from osgeo import gdal
from osgeo import osr
from shapely.wkt import loads


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
    on the GDAL error level. While we are now using ``gdal.UseExceptions()``,
    we still need this to handle GDAL logging that does not get raised as
    an exception.

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

    While we are now using ``gdal.UseExceptions()``, we still need this to
    handle GDAL logging that does not get raised as an exception.
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
        workspace, model_id, logging_level=logging.NOTSET, exclude_threads=None):
    """Prepare the workspace."""
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    logfile = os.path.join(
        workspace,
        f'InVEST-{model_id}-log-{datetime.now().strftime("%Y-%m-%d--%H_%M_%S")}.txt')

    with capture_gdal_logging(), log_to_file(logfile,
                                             exclude_threads=exclude_threads,
                                             logging_level=logging_level):
        logging.captureWarnings(True)
        # If invest is launched as a subprocess (e.g. the Workbench)
        # the parent process can rely on this announcement to know the
        # logfile path (within []), and to know the invest process started.
        LOGGER.log(100, f'Writing log messages to [{logfile}]', )
        start_time = time.time()
        try:
            yield
        except Exception:
            LOGGER.exception(f'Exception while executing {model_id}')
            raise
        finally:
            LOGGER.info(
                f'Elapsed time: {_format_time(round(time.time() - start_time, 2))}')
            logging.captureWarnings(False)
            LOGGER.info(f'Execution finished; version: {natcap.invest.__version__}')


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
            LOGGER.warning(
                f'Logfile {logfile} exists and will be overwritten')
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

    if exclude_threads:
        for threadname in exclude_threads:
            thread_filter = ThreadFilter(threadname)
            handler.addFilter(thread_filter)

    try:
        yield handler
    finally:
        handler.close()
        root_logger.removeHandler(handler)


def expand_path(path, base_path):
    """Check if a path is relative, and if so, expand it using the base path.

    Args:
        path (string): path to check and expand if necessary
        base_path (string): path to expand the first path relative to

    Returns:
        path as an absolute path
    """
    if not path:
        return None
    if platform.system() in {'Darwin', 'Linux'} and '\\' in path:
        path = path.replace('\\', '/')
    if os.path.isabs(path):
        return os.path.abspath(path)  # normalize path separators
    return os.path.abspath(os.path.join(os.path.dirname(base_path), path))


def read_csv_to_dataframe(path, **kwargs):
    """Return a dataframe representation of the CSV.

    Wrapper around ``pandas.read_csv`` that performs some common data cleaning.
    Column names are lowercased and whitespace is stripped off. Empty rows and
    columns are dropped. Sets custom defaults for some kwargs passed to
    ``pandas.read_csv``, which you can override with kwargs:

    - sep=None: lets the Python engine infer the separator
    - engine='python': The 'python' engine supports the sep=None option.
    - encoding='utf-8-sig': 'utf-8-sig' handles UTF-8 with or without BOM.
    - index_col=False: force pandas not to index by any column, useful in
        case of trailing separators

    Args:
        path (str): path to a CSV file
        **kwargs: additional kwargs will be passed to ``pandas.read_csv``

    Returns:
        pandas.DataFrame with the contents of the given CSV
    """
    try:
        df = pandas.read_csv(
            path,
            **{
                'index_col': False,
                'sep': None,
                'engine': 'python',
                'encoding': 'utf-8-sig',
                **kwargs
            })
    except UnicodeDecodeError as error:
        raise ValueError(
            f'The file {path} must be encoded as UTF-8 or ASCII')

    # drop columns whose header is NA
    df = df[[col for col in df.columns if not pandas.isna(col)]]

    # strip whitespace from column names and convert to lowercase
    # this won't work on integer types, which happens if you set header=None
    # however, there's little reason to use this function if there's no header
    df.columns = df.columns.astype(str).str.strip().str.lower()

    return df


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
        ValueError:
            - if ``values_required`` is ``True`` and a pixel value from
            ``raster_path_band`` is not a key in ``value_map``.
        TypeError:
            - if there is a ``None`` or ``NA`` key in ``value_map``.
    """
    # Error early if 'error_details' keys are invalid
    raster_name = error_details['raster_name']
    column_name = error_details['column_name']
    table_name = error_details['table_name']

    # check keys in value map to ensure none are NA or None
    if any((key is pandas.NA or key is None)
           for key in value_map):
        error_message = (f"Missing or NA value in '{column_name}' column"
                         f" in {table_name} table.")
        raise TypeError(error_message)

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


def copy_spatial_files(spatial_filepath, target_dir):
    """Copy spatial files to a new directory.

    Args:
        spatial_filepath (str): The filepath to a GDAL-supported file.
        target_dir (str): The directory where all component files of
            ``spatial_filepath`` should be copied.  If this directory does not
            exist, it will be created.

    Returns:
        filepath (str): The path to a representative file copied into the
        ``target_dir``.  If possible, this will match the basename of
        ``spatial_filepath``, so if someone provides an ESRI Shapefile called
        ``my_vector.shp``, the return value will be ``os.path.join(target_dir,
        my_vector.shp)``.
    """
    LOGGER.info(f'Copying {spatial_filepath} --> {target_dir}')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    source_basename = os.path.basename(spatial_filepath)
    return_filepath = None

    # For ArcGIS Binary/Grid format the filepath will likely be a directory.
    # Even though we don't strictly need to, open the file to make
    # sure it is a gdal dataset before copying an entire directory tree.
    spatial_file = gdal.OpenEx(spatial_filepath)
    if os.path.isdir(spatial_filepath):
        target_filepath = os.path.join(target_dir, source_basename)
        shutil.copytree(spatial_filepath, target_filepath)
        spatial_file = None
        return target_filepath
    for member_file in spatial_file.GetFileList():
        # If the file is an ArcGIS Binary/Grid format but the given filepath
        # was _not_ the parent directory -- it can be an '.adf' file --
        # just skip the parent directory.
        if os.path.isdir(member_file):
            continue
        target_basename = os.path.basename(member_file)
        target_filepath = os.path.join(target_dir, target_basename)
        if source_basename == target_basename:
            return_filepath = target_filepath
        shutil.copyfile(member_file, target_filepath)
    spatial_file = None

    # I can't conceive of a case where the basename of the source file does not
    # match any of the member file basenames, but just in case there's a
    # weird GDAL driver that does this, it seems reasonable to fall back to
    # whichever of the member files was most recent.
    if not return_filepath:
        return_filepath = target_filepath

    return return_filepath


class _GDALPath:
    """Result of parsing a dataset URI/Path

    This class is largely copied from rasterio._path.

    In this context, "scheme" refers to the first part of a URL in the format
    scheme://netloc/path;parameters?query#fragment

    Attributes
    ----------
    path : str
        Parsed path. Includes the hostname and query string in the case
        of a URI.
    archive : str
        Parsed archive path.
    scheme : str
        URI scheme such as "https" or "zip+s3".
    """
    
    def __init__(self, path, archive, scheme):
        self.path = path
        self.archive = archive
        self.scheme = scheme

        # Supported URI schemes and their mapping to GDAL's VSI suffix.
        self.schemes = {
            'ftp': 'curl',
            'gzip': 'gzip',
            'http': 'curl',
            'https': 'curl',
            's3': 's3',
            'tar': 'tar',
            'zip': 'zip',
            'file': 'file',
            'oss': 'oss',
            'gs': 'gs',
            'az': 'az',
        }
        self.curlschemes = {k for k, v in self.schemes.items() if v == "curl"}

        self.remoteschemes = {
            k for k, v in self.schemes.items() if v in (
                "curl",
                "s3",
                "oss",
                "gs",
                "az",
            )
        }

    @classmethod
    def from_uri(cls, uri):
        """Instantiate a _GDALPath for a given URI.

        Args:
            uri: URI string to parse

        Returns:
            _GDALPath object
        """
        parts = urlparse(uri)
        if sys.platform == "win32" and re.match(r"^[a-zA-Z]\:", uri):
            parsed_path = uri
            parsed_netloc = None
            scheme = None
        else:
            parsed_path = parts.path
            parsed_netloc = parts.netloc
            scheme = parts.scheme or None

        path = parsed_path

        if parts.query:
            path += "?" + parts.query

        if scheme and scheme.startswith(("gzip", "tar", "zip")):
            path_parts = path.split('!')
            path = path_parts.pop() if path_parts else None
            archive = path_parts.pop() if path_parts else None
        else:
            archive = None

        if scheme and parsed_netloc:
            if archive:
                archive = parsed_netloc + archive
            else:
                path = parsed_netloc + path

        return _GDALPath(path, archive, scheme)

    def to_normalized_path(self):
        """Return path represented as a string normalized to a gdal-ready scheme.

        This means that URI schemes are converted to gdal /vsi prefixes,
        for example "zip+https" -> "/vsizip/vsicurl/".
        """
        if not self.scheme:
            return self.path

        if self.scheme.split('+')[-1] in self.curlschemes:
            suffix = f'{self.scheme.split("+")[-1]}://'
        else:
            suffix = ''

        vsi_prefix = '/'.join(
            f'vsi{self.schemes[p]}' for p in self.scheme.split('+') if p != 'file'
        )

        if vsi_prefix:
            if self.archive:
                result = f'/{vsi_prefix}/{suffix}{self.archive}/{self.path.lstrip("/")}'
            else:
                result = f'/{vsi_prefix}/{suffix}{self.path}'
        else:
            result = self.path
        return result

    @property
    def is_remote(self):
        """Test if the path is a remote, network URI"""
        return bool(self.scheme) and self.scheme.split("+")[-1] in self.remoteschemes

    @property
    def is_local(self):
        """Test if the path is a local URI"""
        return not self.is_remote


def evaluate_expression(expression, variable_map):
    """Evaluate a python expression.

    The expression must be able to be evaluated as a python expression.

    Args:
        expression (string): A string expression that returns a value.
        variable_map (dict): A dict mapping string variable names to their
            python object values.  This is the variable map that will be used
            when evaluating the expression.

    Returns:
        Whatever value is returned from evaluating ``expression`` with the
        variables stored in ``variable_map``.

    """
    # __builtins__ can be either a dict or a module.  We need its contents as a
    # dict in order to use ``eval``.
    if not isinstance(__builtins__, dict):
        builtins = __builtins__.__dict__
    else:
        builtins = __builtins__
    builtin_symbols = set(builtins.keys())

    active_symbols = set()
    for tree_node in ast.walk(ast.parse(expression)):
        if isinstance(tree_node, ast.Name):
            active_symbols.add(tree_node.id)

    # This should allow any builtin functions, exceptions, etc. to be handled
    # correctly within an expression.
    missing_symbols = (active_symbols -
                       set(variable_map.keys()).union(builtin_symbols))
    if missing_symbols:
        raise AssertionError(
            'Identifiers expected in the expression "%s" are missing: %s' % (
                expression, ', '.join(missing_symbols)))

    # The usual warnings should go with this call to eval:
    # Don't run untrusted code!!!
    return eval(expression, builtins, variable_map)


def format_args_dict(args_dict, model_id):
    """Nicely format an arguments dictionary for writing to a stream.

    If printed to a console, the returned string will be aligned in two columns
    representing each key and value in the arg dict.  Keys are in ascending,
    sorted order.  Both columns are left-aligned.

    Args:
        args_dict (dict): The args dictionary to format.
        model_id (string): The model ID (e.g. carbon)

    Returns:
        A formatted, unicode string.
    """
    sorted_args = sorted(args_dict.items(), key=lambda x: x[0])

    max_key_width = 0
    if len(sorted_args) > 0:
        max_key_width = max(len(x[0]) for x in sorted_args)

    format_str = f"%-{max_key_width}s %s"

    args_string = '\n'.join([format_str % (arg) for arg in sorted_args])
    args_string = (
        f"Arguments for InVEST {model_id} {natcap.invest.__version__}:"
        f"\n{args_string}\n")
    return args_string
