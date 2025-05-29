import dataclasses
import importlib
import json
import logging
import os
import pprint
import queue
import re
import threading
import types
import typing
import warnings

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import geometamaker
import natcap.invest
import pandas
import pint
import pygeoprocessing
from pydantic import ValidationError

from natcap.invest import utils
from natcap.invest.validation import get_message, _evaluate_expression
from . import gettext
from .unit_registry import u


LOGGER = logging.getLogger(__name__)

# accessing a file could take a long time if it's in a file streaming service
# to prevent the UI from hanging due to slow validation,
# set a timeout for these functions.
def timeout(func, timeout=5):
    """Stop a function after a given amount of time.

    Args:
        func (function): function to apply the timeout to
        args: arguments to pass to the function
        timeout (number): how many seconds to allow the function to run.
            Defaults to 5.

    Returns:
        A string warning message if the thread completed in time and returned
        warnings, ``None`` otherwise.

    Raises:
        ``RuntimeWarning`` if the thread does not complete in time.
    """
    # use a queue to share the return value from the file checking thread
    # the target function puts the return value from `func` into shared memory
    message_queue = queue.Queue()

    def wrapper(*args, **kwargs):
        def put_fn():
            message_queue.put(func(*args, **kwargs))
        thread = threading.Thread(target=put_fn)
        LOGGER.debug(f'Starting file checking thread with timeout={timeout}')
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            # first arg to `check_csv`, `check_raster`, `check_vector` is the path
            warnings.warn(
                f'Validation of file {args[0]} timed out. If this file '
                'is stored in a file streaming service, it may be taking a long '
                'time to download. Try storing it locally instead.')

        else:
            LOGGER.debug('File checking thread completed.')
            # get any warning messages returned from the thread
            a = message_queue.get()
            return a

    return wrapper


def check_headers(expected_headers, actual_headers, header_type='header'):
    """Validate that expected headers are in a list of actual headers.

    - Each expected header should be found exactly once.
    - Actual headers may contain extra headers that are not expected.
    - Headers are converted to lowercase before matching.

    Args:
        expected_headers (list[str]): A list of headers that are expected to
            exist in `actual_headers`.
        actual_headers (list[str]): A list of actual headers to validate
            against `expected_headers`.
        header_type (str): A string to use in the error message to refer to the
            header (typically one of 'column', 'row', 'field')

    Returns:
        None, if validation passes; or a string describing the problem, if a
        validation rule is broken.
    """
    actual_headers = [header.lower()
                      for header in actual_headers]  # case insensitive
    for expected in expected_headers:
        count = actual_headers.count(expected)
        if count == 0:
            return get_message('MATCHED_NO_HEADERS').format(
                header=header_type,
                header_name=expected)
        elif count > 1:
            return get_message('DUPLICATE_HEADER').format(
                header=header_type,
                header_name=expected,
                number=count)


def _check_projection(srs, projected, projection_units):
    """Validate a GDAL projection.

    Args:
        srs (osr.SpatialReference): A GDAL Spatial Reference object
            representing the spatial reference of a GDAL dataset.
        projected (bool): Whether the spatial reference must be projected in
            linear units.
        projection_units (pint.Unit): The projection's required linear units.

    Returns:
        A string error message if an error was found. ``None`` otherwise.

    """
    empty_srs = osr.SpatialReference()
    if srs is None or srs.IsSame(empty_srs):
        return get_message('INVALID_PROJECTION')

    if projected:
        if not srs.IsProjected():
            return get_message('NOT_PROJECTED')

    if projection_units:
        # pint uses underscores in multi-word units e.g. 'survey_foot'
        # it is case-sensitive
        layer_units_name = srs.GetLinearUnitsName().lower().replace(' ', '_')
        try:
            # this will parse common synonyms: m, meter, meters, metre, metres
            layer_units = u.Unit(layer_units_name)
            # Compare pint Unit objects
            if projection_units != layer_units:
                return get_message('WRONG_PROJECTION_UNIT').format(
                    unit_a=projection_units, unit_b=layer_units_name)
        except pint.errors.UndefinedUnitError:
            return get_message('WRONG_PROJECTION_UNIT').format(
                unit_a=projection_units, unit_b=layer_units_name)


class IterableWithDotAccess():
    """Iterable that supports dot notation access by id attribute."""

    def __init__(self, *args):
        self.args = args
        self.inputs_dict = {i.id: i for i in args}
        self.iter_index = 0

    def __iter__(self):
        return iter(self.args)

    def get(self, key):
        """Get an item by its id.

        Args:
            key: the item id

        Returns:
            the corresponding item
        """
        return self.inputs_dict[key]

    def to_json(self):
        """Return a JSON serializable representation of self.

        Returns:
            dict mapping item IDs to items
        """
        return self.inputs_dict


@dataclasses.dataclass
class Input:
    """A data input, or parameter, of an invest model.

    This represents an abstract input or parameter, which is rendered as an
    input field in the InVEST workbench. This does not store the value of the
    parameter for a specific run of the model.
    """
    id: str = ''
    """Input identifier that should be unique within a model"""

    name: str = ''
    """The user-facing name of the input. The workbench UI displays this
    property as a label for each input. The name should be as short as
    possible. Any extra description should go in ``about``. The name should
    be all lower-case, except for things that are always capitalized (acronyms,
    proper names).

    Good examples: ``precipitation``, ``Kc factor``, ``valuation table``

    Bad examples: ``PRECIPITATION``, ``kc_factor``, ``table of valuation parameters``
    """

    about: str = ''
    """User-facing description of the input"""

    required: typing.Union[bool, str] = True
    """Whether the input is required to be provided. Defaults to True. Set to
    False if the input is always optional. If the input is conditionally
    required depending on the state of other inputs, provide a string
    expression that evaluates to a boolean to describe this condition."""

    allowed: typing.Union[bool, str] = True
    """Defaults to True. If the input is not allowed to be provided under a
    certain condition (such as when running the model in a mode where the
    input is not used), provide a string expression that evaluates to a
    boolean to describe this condition."""

    hidden: bool = False
    """Whether to hide the input from the model input form in the workbench.
    Use this if the value should not be configurable from the input form, such
    as if it's pulled in from another source. Defaults to False."""


@dataclasses.dataclass
class Output:
    """A data output, or result, of an invest model.

    This represents an abstract output which is produced as a result of running
    an invest model. This does not store the value of the output for a specific
    run of the model.
    """
    id: str = ''
    """Output identifier that should be unique within a model"""

    about: str = ''
    """User-facing description of the output"""

    created_if: typing.Union[bool, str] = True
    """Defaults to True. If the input is only created under a certain condition
    (such as when running the model in a specific mode), provide a string
    expression that evaluates to a boolean to describe this condition."""


@dataclasses.dataclass
class FileInput(Input):
    """A generic file input, or parameter, of an invest model.

    This represents a not-otherwise-specified file input type. Use this only if
    a more specific type, such as `CSVInput` or `VectorInput`, does not apply.
    """
    permissions: str = 'r'
    """A string that includes the lowercase characters ``r``, ``w`` and/or
    ``x``, indicating read, write, and execute permissions (respectively)
    required for this file."""

    type: typing.ClassVar[str] = 'file'

    @timeout
    def validate(self, filepath: str):
        """Validate a file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        if not os.path.exists(filepath):
            return get_message('FILE_NOT_FOUND')

        for letter, mode, descriptor in (
                ('r', os.R_OK, 'read'),
                ('w', os.W_OK, 'write'),
                ('x', os.X_OK, 'execute')):
            if letter in self.permissions and not os.access(filepath, mode):
                return get_message('NEED_PERMISSION_FILE').format(permission=descriptor)

    @staticmethod
    def format_column(col: pandas.Series, base_path: str) -> pandas.Series:
        """Format a column of a pandas dataframe that contains FileInput values.

        File path values are cast to `pandas.StringDtype`. Relative paths are
        expanded to absolute paths relative to `base_path`. NA values remain NA.

        Args:
            col: Column of a pandas dataframe to format
            base_path: Base path of the source CSV. Relative file path values
                will be expanded relative to this base path.

        Returns:
            Transformed dataframe column
        """
        return col.apply(
            lambda p: p if pandas.isna(p) else utils.expand_path(str(p).strip(), base_path)
        ).astype(pandas.StringDtype())


@dataclasses.dataclass
class RasterBand():
    """A single-band raster input, or parameter, of an invest model.

    This represents a raster file input (all GDAL-supported raster file types
    are allowed), where only the first band is needed.
    """
    band_id: typing.Union[int, str] = 1
    """band index used to access the raster band"""

    data_type: typing.Type = float
    """float or int"""

    units: typing.Union[pint.Unit, None] = None
    """units of measurement of the raster band values"""


@dataclasses.dataclass
class RasterInput(FileInput):
    """A raster input, or parameter, of an invest model.

    This represents a raster file input (all GDAL-supported raster file types
    are allowed), which may have multiple bands.
    """
    bands: typing.Iterable[RasterBand] = dataclasses.field(default_factory=list)
    """An iterable of `RasterBand` representing the bands expected to be in
    the raster."""

    projected: typing.Union[bool, None] = None
    """Defaults to None, indicating a projected (as opposed to geographic)
    coordinate system is not required. Set to True if a projected coordinate
    system is required."""

    projection_units: typing.Union[pint.Unit, None] = None
    """Defaults to None. If `projected` is `True`, and a specific unit of
    projection (such as meters) is required, indicate it here."""

    type: typing.ClassVar[str] = 'raster'

    @timeout
    def validate(self, filepath: str):
        """Validate a raster file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        file_warning = super().validate(filepath)
        if file_warning:
            return file_warning

        try:
            gdal_dataset = gdal.OpenEx(filepath, gdal.OF_RASTER)
        except RuntimeError:
            return get_message('NOT_GDAL_RASTER')

        # Check that an overview .ovr file wasn't opened.
        if os.path.splitext(filepath)[1] == '.ovr':
            return get_message('OVR_FILE')

        srs = gdal_dataset.GetSpatialRef()
        projection_warning = _check_projection(srs, self.projected, self.projection_units)
        if projection_warning:
            return projection_warning


@dataclasses.dataclass
class SingleBandRasterInput(FileInput):
    """A single-band raster input, or parameter, of an invest model.

    This represents a raster file input (all GDAL-supported raster file types
    are allowed), where only the first band is needed. While the same thing can
    be achieved using a `RasterInput`, this class exists to simplify access to
    the band properties when there is only one band.
    """
    data_type: typing.Type = float
    """float or int"""

    units: typing.Union[pint.Unit, None] = None
    """units of measurement of the raster values"""

    projected: typing.Union[bool, None] = None
    """Defaults to None, indicating a projected (as opposed to geographic)
    coordinate system is not required. Set to True if a projected coordinate
    system is required."""

    projection_units: typing.Union[pint.Unit, None] = None
    """Defaults to None. If `projected` is `True`, and a specific unit of
    projection (such as meters) is required, indicate it here."""

    type: typing.ClassVar[str] = 'raster'

    @timeout
    def validate(self, filepath: str):
        """Validate a raster file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        file_warning = super().validate(filepath)
        if file_warning:
            return file_warning

        try:
            gdal_dataset = gdal.OpenEx(filepath, gdal.OF_RASTER)
        except RuntimeError:
            return get_message('NOT_GDAL_RASTER')

        # Check that an overview .ovr file wasn't opened.
        if os.path.splitext(filepath)[1] == '.ovr':
            return get_message('OVR_FILE')

        srs = gdal_dataset.GetSpatialRef()
        projection_warning = _check_projection(srs, self.projected, self.projection_units)
        if projection_warning:
            return projection_warning


@dataclasses.dataclass
class VectorInput(FileInput):
    """A vector input, or parameter, of an invest model.

    This represents a vector file input (all GDAL-supported vector file types
    are allowed). It is assumed that only the first layer is used.
    """
    geometry_types: set = dataclasses.field(default_factory=dict)
    """A set of geometry type(s) that are allowed for this vector"""

    fields: typing.Union[typing.Iterable[Input], None] = None
    """An iterable of `Input`s representing the fields that this vector is
    expected to have. The `key` of each input must match the corresponding
    field name."""

    projected: typing.Union[bool, None] = None
    """Defaults to None, indicating a projected (as opposed to geographic)
    coordinate system is not required. Set to True if a projected coordinate
    system is required."""

    projection_units: typing.Union[pint.Unit, None] = None
    """Defaults to None. If `projected` is `True`, and a specific unit of
    projection (such as meters) is required, indicate it here."""

    type: typing.ClassVar[str] = 'vector'

    def __post_init__(self):
        if self.fields:
            self.fields = IterableWithDotAccess(*self.fields)

    @timeout
    def validate(self, filepath: str):
        """Validate a vector file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        file_warning = super().validate(filepath)
        if file_warning:
            return file_warning

        try:
            gdal_dataset = gdal.OpenEx(filepath, gdal.OF_VECTOR)
        except RuntimeError:
            return get_message('NOT_GDAL_VECTOR')

        geom_map = {
            'POINT': [ogr.wkbPoint, ogr.wkbPointM, ogr.wkbPointZM,
                      ogr.wkbPoint25D],
            'LINESTRING': [ogr.wkbLineString, ogr.wkbLineStringM,
                           ogr.wkbLineStringZM, ogr.wkbLineString25D],
            'POLYGON': [ogr.wkbPolygon, ogr.wkbPolygonM,
                        ogr.wkbPolygonZM, ogr.wkbPolygon25D],
            'MULTIPOINT': [ogr.wkbMultiPoint, ogr.wkbMultiPointM,
                           ogr.wkbMultiPointZM, ogr.wkbMultiPoint25D],
            'MULTILINESTRING': [ogr.wkbMultiLineString, ogr.wkbMultiLineStringM,
                                ogr.wkbMultiLineStringZM,
                                ogr.wkbMultiLineString25D],
            'MULTIPOLYGON': [ogr.wkbMultiPolygon, ogr.wkbMultiPolygonM,
                             ogr.wkbMultiPolygonZM, ogr.wkbMultiPolygon25D]
        }

        allowed_geom_types = []
        for geom in self.geometry_types:
            allowed_geom_types += geom_map[geom]

        # NOTE: this only checks the layer geometry type, not the types of the
        # actual geometries (layer.GetGeometryTypes()). This is probably equivalent
        # in most cases, and it's more efficient than checking every geometry, but
        # we might need to change this in the future if it becomes a problem.
        # Currently not supporting ogr.wkbUnknown which allows mixed types.
        layer = gdal_dataset.GetLayer()
        if layer.GetGeomType() not in allowed_geom_types:
            return get_message('WRONG_GEOM_TYPE').format(allowed=self.geometry_types)

        if self.fields:
            field_patterns = []
            for spec in self.fields:
                # brackets are a special character for our args spec syntax
                # they surround the part of the key that's user-defined
                # user-defined rows/columns/fields are not validated here, so skip
                if spec.required is True and '[' not in spec.id:
                    field_patterns.append(spec.id)

            fieldnames = [defn.GetName() for defn in layer.schema]
            required_field_warning = check_headers(
                field_patterns, fieldnames, 'field')
            if required_field_warning:
                return required_field_warning

        srs = layer.GetSpatialRef()
        projection_warning = _check_projection(srs, self.projected, self.projection_units)
        return projection_warning


@dataclasses.dataclass
class RasterOrVectorInput(FileInput):
    """An invest model input that can be either a single-band raster or a vector."""

    data_type: typing.Type = float
    """Data type for the raster values (float or int)"""

    units: typing.Union[pint.Unit, None] = None
    """Units of measurement of the raster values"""

    geometry_types: set = dataclasses.field(default_factory=dict)
    """A set of geometry type(s) that are allowed for this vector"""

    fields: typing.Union[typing.Iterable[Input], None] = None
    """An iterable of `Input`s representing the fields that this vector is
    expected to have. The `key` of each input must match the corresponding
    field name."""

    projected: typing.Union[bool, None] = None
    """Defaults to None, indicating a projected (as opposed to geographic)
    coordinate system is not required. Set to True if a projected coordinate
    system is required."""

    projection_units: typing.Union[pint.Unit, None] = None
    """Defaults to None. If `projected` is `True`, and a specific unit of
    projection (such as meters) is required, indicate it here."""

    type: typing.ClassVar[str] = 'raster_or_vector'

    def __post_init__(self):
        self.single_band_raster_input = SingleBandRasterInput(
            data_type=self.data_type,
            units=self.units,
            projected=self.projected,
            projection_units=self.projection_units)
        self.vector_input = VectorInput(
            geometry_types=self.geometry_types,
            fields=self.fields,
            projected=self.projected,
            projection_units=self.projection_units)

    @timeout
    def validate(self, filepath: str):
        """Validate a raster or vector file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        try:
            gis_type = pygeoprocessing.get_gis_type(filepath)
        except ValueError as err:
            return str(err)
        if gis_type == pygeoprocessing.RASTER_TYPE:
            return self.single_band_raster_input.validate(filepath)
        else:
            return self.vector_input.validate(filepath)


@dataclasses.dataclass
class CSVInput(FileInput):
    """A CSV table input, or parameter, of an invest model.

    For CSVs with a simple layout, `columns` or `rows` (but not both) may be
    specified. For more complex table structures that cannot be described by
    `columns` or `rows`, you may omit both attributes. Note that more complex
    table structures are often more difficult to use; consider dividing them
    into multiple, simpler tabular inputs.
    """
    columns: typing.Union[typing.Iterable[Input], None] = None
    """An iterable of `Input`s representing the columns that this CSV is
    expected to have. The `key` of each input must match the corresponding
    column header."""

    rows: typing.Union[typing.Iterable[Input], None] = None
    """An iterable of `Input`s representing the rows that this CSV is
    expected to have. The `key` of each input must match the corresponding
    row header."""

    index_col: typing.Union[str, None] = None
    """The header name of the column to use as the index. When processing a
    CSV file to a dataframe, the dataframe index will be set to this column."""

    type: typing.ClassVar[str] = 'csv'

    def __post_init__(self):
        if self.rows:
            self.rows = IterableWithDotAccess(*self.rows)
        if self.columns:
            self.columns = IterableWithDotAccess(*self.columns)

    @timeout
    def validate(self, filepath: str):
        """Validate a CSV file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        file_warning = super().validate(filepath)
        if file_warning:
            return file_warning
        if self.columns or self.rows:
            try:
                self.get_validated_dataframe(filepath)
            except Exception as e:
                return str(e)

    def get_validated_dataframe(self, csv_path: str, read_csv_kwargs={}):
        """Read a CSV into a dataframe that is guaranteed to match the spec.

        This is only supported when `columns` or `rows` is provided. Each
        column will be read in to a dataframe and values will be pre-processed
        according to that column input type. Column/row headers are matched
        case-insensitively. Values are cast to the appropriate
        type and relative paths are expanded.

        Args:
            csv_path: Path to the CSV to process
            read_csv_kwargs: Additional kwargs to pass to `pandas.read_csv`

        Returns:
            pandas dataframe

        Raises:
            ValueError if the CSV cannot be parsed to fulfill the requirements
            for this input - if a required column or row is missing, or if the
            values in a column cannot be interpreted as the expected type.
        """
        if not (self.columns or self.rows):
            raise ValueError('One of columns or rows must be provided')

        # build up a list of regex patterns to match columns against columns from
        # the table that match a pattern in this list (after stripping whitespace
        # and lowercasing) will be included in the dataframe
        axis = 'column' if self.columns else 'row'

        if self.rows:
            read_csv_kwargs = read_csv_kwargs.copy()
            read_csv_kwargs['header'] = None

        df = utils.read_csv_to_dataframe(csv_path, **read_csv_kwargs)

        if self.rows:
            # swap rows and column
            df = df.set_index(df.columns[0]).rename_axis(
                None, axis=0).T.reset_index(drop=True)

        columns = self.columns if self.columns else self.rows

        patterns = []
        for column in columns:
            column = column.id.lower()
            match = re.match(r'(.*)\[(.+)\](.*)', column)
            if match:
                # for column name patterns, convert it to a regex pattern
                groups = match.groups()
                patterns.append(f'{groups[0]}(.+){groups[2]}')
            else:
                # for regular column names, use the exact name as the pattern
                patterns.append(column.replace('(', r'\(').replace(')', r'\)'))

        # select only the columns that match a pattern
        df = df[[col for col in df.columns if any(
            re.fullmatch(pattern, col) for pattern in patterns)]]

        # drop any empty rows
        df = df.dropna(how="all").reset_index(drop=True)

        available_cols = set(df.columns)

        for col_spec, pattern in zip(columns, patterns):
            matching_cols = [c for c in available_cols if re.fullmatch(pattern, c)]
            if col_spec.required is True and '[' not in col_spec.id and not matching_cols:
                raise ValueError(get_message('MATCHED_NO_HEADERS').format(
                    header=axis,
                    header_name=col_spec.id))
            available_cols -= set(matching_cols)
            for col in matching_cols:
                try:
                    df[col] = col_spec.format_column(df[col], csv_path)
                except Exception as err:
                    raise ValueError(
                        f'Value(s) in the "{col}" column could not be interpreted '
                        f'as {type(col_spec).__name__}s. Original error: {err}')

                if (isinstance(col_spec, SingleBandRasterInput) or
                    isinstance(col_spec, VectorInput) or
                    isinstance(col_spec, RasterOrVectorInput)):
                    # recursively validate the files within the column
                    def check_value(value):
                        if pandas.isna(value):
                            return
                        err_msg = col_spec.validate(value)
                        if err_msg:
                            raise ValueError(
                                f'Error in {axis} "{col}", value "{value}": {err_msg}')
                    df[col].apply(check_value)

        if any(df.columns.duplicated()):
            duplicated_columns = df.columns[df.columns.duplicated]
            return get_message('DUPLICATE_HEADER').format(
                header=header_type,
                header_name=expected,
                number=count)

        # set the index column, if specified
        if self.index_col is not None:
            index_col = self.index_col.lower()
            try:
                df = df.set_index(index_col, verify_integrity=True)
            except KeyError:
                # If 'index_col' is not a column then KeyError is raised for using
                # it as the index column
                LOGGER.error(f"The column '{index_col}' could not be found "
                             f"in the table {csv_path}")
                raise

        return df


@dataclasses.dataclass
class DirectoryInput(Input):
    """A directory input, or parameter, of an invest model.

    Use this type when you need to specify a group of many file-based inputs,
    or an unknown number of file-based inputs, by grouping them together in a
    directory. This may also be used to describe an empty directory where model
    outputs will be written to.
    """
    contents: typing.Union[typing.Iterable[Input], None] = None
    """An iterable of `Input`s representing the contents of this directory. The
    `key` of each input must be the file name or pattern."""

    permissions: str = ''
    """A string that includes the lowercase characters ``r``, ``w`` and/or ``x``,
    indicating read, write, and execute permissions (respectively) required for
    this directory."""

    must_exist: bool = True
    """Defaults to True, indicating the directory must already exist before
    running the model. Set to False if the directory will be created."""

    type: typing.ClassVar[str] = 'directory'

    def __post_init__(self):
        if self.contents:
            self.contents = IterableWithDotAccess(*self.contents)

    @timeout
    def validate(self, dirpath: str):
        """Validate a directory path against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        if self.must_exist:
            if not os.path.exists(dirpath):
                return get_message('DIR_NOT_FOUND')

        if os.path.exists(dirpath):
            if not os.path.isdir(dirpath):
                return get_message('NOT_A_DIR')
        else:
            # find the parent directory that does exist and check permissions
            child = dirpath
            parent = os.path.normcase(os.path.abspath(dirpath))
            while child:
                # iterate child because if this gets back to the root dir,
                # child becomes an empty string and parent remains root string.
                parent, child = os.path.split(parent)
                if os.path.exists(parent):
                    dirpath = parent
                    break

        MESSAGE_KEY = 'NEED_PERMISSION_DIRECTORY'

        if 'r' in self.permissions:
            try:
                os.scandir(dirpath).close()
            except OSError:
                return get_message(MESSAGE_KEY).format(permission='read')

        # Check for x access before checking for w,
        # since w operations to a dir are dependent on x access
        if 'x' in self.permissions:
            try:
                cwd = os.getcwd()
                os.chdir(dirpath)
            except OSError:
                return get_message(MESSAGE_KEY).format(permission='execute')
            finally:
                os.chdir(cwd)

        if 'w' in self.permissions:
            try:
                temp_path = os.path.join(dirpath, 'temp__workspace_validation.txt')
                with open(temp_path, 'w') as temp:
                    temp.close()
                    os.remove(temp_path)
            except OSError:
                return get_message(MESSAGE_KEY).format(permission='write')


@dataclasses.dataclass
class NumberInput(Input):
    """A floating-point number input, or parameter, of an invest model.

    Use a more specific type (such as `IntegerInput`, `RatioInput`, or
    `PercentInput`) where applicable.
    """
    units: typing.Union[pint.Unit, None] = None
    """The units of measurement for this numeric value"""

    expression: typing.Union[str, None] = None
    """A string expression that can be evaluated to a boolean indicating whether
    the value meets a required condition. The expression must contain the string
    ``value``, which will represent the user-provided value (after it has been
    cast to a float). Example: ``"(value >= 0) & (value <= 1)"``."""

    type: typing.ClassVar[str] = 'number'

    def validate(self, value):
        """Validate a numeric value against the requirements for this input.

        Args:
            value: The value to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        try:
            float(value)
        except (TypeError, ValueError):
            return get_message('NOT_A_NUMBER').format(value=value)

        if self.expression:
            # Check to make sure that 'value' is in the expression.
            if 'value' not in self.expression:
                raise AssertionError(
                    'The variable name value is not found in the '
                    f'expression: {self.expression}')

            # Expression is assumed to return a boolean, something like
            # "value > 0" or "(value >= 0) & (value < 1)".  An exception will
            # be raised if asteval can't evaluate the expression.
            result = _evaluate_expression(self.expression, {'value': float(value)})
            if not result:  # A python bool object is returned.
                return get_message('INVALID_VALUE').format(condition=self.expression)

    @staticmethod
    def format_column(col, *args):
        """Format a column of a pandas dataframe that contains NumberInput values.

        Values are cast to float.

        Args:
            col: Column of a pandas dataframe to format

        Returns:
            Transformed dataframe column
        """
        return col.astype(float)


@dataclasses.dataclass
class IntegerInput(Input):
    """An integer input, or parameter, of an invest model."""
    type: typing.ClassVar[str] = 'integer'

    def validate(self, value):
        """Validate a value against the requirements for this input.

        Args:
            value: The value to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        try:
            # must first cast to float, to handle both string and float inputs
            as_float = float(value)
            if not as_float.is_integer():
                return get_message('NOT_AN_INTEGER').format(value=value)
        except (TypeError, ValueError):
            return get_message('NOT_A_NUMBER').format(value=value)

    @staticmethod
    def format_column(col, *args):
        """Format a column of a pandas dataframe that contains IntegerInput values.

        Values are cast to `pandas.Int64Dtype`.

        Args:
            col: Column of a pandas dataframe to format

        Returns:
            Transformed dataframe column
        """
        return col.astype(pandas.Int64Dtype())


@dataclasses.dataclass
class RatioInput(NumberInput):
    """A ratio input, or parameter, of an invest model.

    A ratio is a proportion expressed as a value from 0 to 1 (in contrast to a
    percent, which ranges from 0 to 100). Values are restricted to the
    range [0, 1].
    """
    type: typing.ClassVar[str] = 'ratio'

    def validate(self, value):
        """Validate a value against the requirements for this input.

        Args:
            value: The value to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        message = super().validate(value)
        if message:
            return message
        as_float = float(value)
        if as_float < 0 or as_float > 1:
            return get_message('NOT_WITHIN_RANGE').format(
                value=as_float,
                range='[0, 1]')


@dataclasses.dataclass
class PercentInput(NumberInput):
    """A percent input, or parameter, of an invest model.

    A percent is a proportion expressed as a value from 0 to 100 (in contrast to
    a ratio, which ranges from 0 to 1). Values are restricted to the range [0, 100].
    """
    type: typing.ClassVar[str] = 'percent'

    def validate(self, value):
        """Validate a value against the requirements for this input.

        Args:
            value: The value to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        message = super().validate(value)
        if message:
            return message
        as_float = float(value)
        if as_float < 0 or as_float > 100:
            return get_message('NOT_WITHIN_RANGE').format(
                value=as_float,
                range='[0, 100]')


@dataclasses.dataclass
class BooleanInput(Input):
    """A boolean input, or parameter, of an invest model."""
    type: typing.ClassVar[str] = 'boolean'

    def validate(self, value):
        """Validate a value against the requirements for this input.

        Args:
            value: The value to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        if not isinstance(value, bool):
            return get_message('NOT_BOOLEAN').format(value=value)

    @staticmethod
    def format_column(col, *args):
        """Format a column of a pandas dataframe that contains BooleanInput values.

        Values are cast to boolean.

        Args:
            col: Column of a pandas dataframe to format

        Returns:
            Transformed dataframe column
        """
        return col.astype('boolean')


@dataclasses.dataclass
class StringInput(Input):
    """A string input, or parameter, of an invest model.

    This represents a textual input. Do not use this to represent numeric or
    file-based inputs which can be better represented by another type.
    """
    regexp: typing.Union[str, None] = None
    """An optional regex pattern which the text value must match"""

    type: typing.ClassVar[str] = 'string'

    def validate(self, value):
        """Validate a value against the requirements for this input.

        Args:
            value: The value to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        if self.regexp:
            matches = re.fullmatch(self.regexp, str(value))
            if not matches:
                return get_message('REGEXP_MISMATCH').format(regexp=self.regexp)

    @staticmethod
    def format_column(col, *args):
        """Format a column of a pandas dataframe that contains StringInput values.

        Values are cast to `pandas.StringDtype`, lowercased, and leading and
        trailing whitespace is stripped. NA values remain NA.

        Args:
            col: Column of a pandas dataframe to format

        Returns:
            Transformed dataframe column
        """
        return col.apply(
            lambda s: s if pandas.isna(s) else str(s).strip().lower()
        ).astype(pandas.StringDtype())


@dataclasses.dataclass
class OptionStringInput(Input):
    """A string input, or parameter, which is limited to a set of options.

    This corresponds to a dropdown menu in the workbench, where the user
    is limited to a set of pre-defined options.
    """
    options: typing.Union[list, None] = None
    """A list of the values that this input may take. Use this if the set of
    options is predetermined."""

    dropdown_function: typing.Union[typing.Callable, None] = None
    """A function that returns a list of the values that this input may take.
    Use this if the set of options must be dynamically generated."""

    type: typing.ClassVar[str] = 'option_string'

    def validate(self, value):
        """Validate a value against the requirements for this input.

        Args:
            value: The value to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        # if options is empty, that means it's dynamically populated
        # so validation should be left to the model's validate function.
        if self.options and str(value) not in self.options:
            return get_message('INVALID_OPTION').format(option_list=sorted(self.options))

    @staticmethod
    def format_column(col, *args):
        """Format a pandas dataframe column that contains OptionStringInput values.

        Values are cast to `pandas.StringDtype`, lowercased, and leading and
        trailing whitespace is stripped. NA values remain NA.

        Args:
            col: Column of a pandas dataframe to format

        Returns:
            Transformed dataframe column
        """
        return col.apply(
            lambda s: s if pandas.isna(s) else str(s).strip().lower()
        ).astype(pandas.StringDtype())


@dataclasses.dataclass
class SingleBandRasterOutput(Output):
    """A single-band raster output, or result, of an invest model.

    This represents a raster file output (all GDAL-supported raster file types
    are allowed), where only the first band is used.
    """
    data_type: typing.Type = float
    """float or int"""

    units: typing.Union[pint.Unit, None] = None
    """units of measurement of the raster values"""


@dataclasses.dataclass
class RasterOutput(Output):
    """A raster output, or result, of an invest model.

    This represents a raster file output (all GDAL-supported raster file types
    are allowed), which may have multiple bands.
    """
    bands: typing.Iterable[RasterBand] = dataclasses.field(default_factory=list)
    """An iterable of `RasterBand` representing the bands expected to be in
    the raster."""


@dataclasses.dataclass
class VectorOutput(Output):
    """A vector output, or result, of an invest model.

    This represents a vector file output (all GDAL-supported vector file types
    are allowed). It is assumed that only the first layer is used.
    """
    geometry_types: set = dataclasses.field(default_factory=set)
    """A set of geometry type(s) that are produced in this vector"""

    fields: typing.Union[typing.Iterable[Output], None] = None
    """An iterable of `Output`s representing the fields created in this vector.
    The `key` of each input must match the corresponding field name."""


@dataclasses.dataclass
class CSVOutput(Output):
    """A CSV table output, or result, of an invest model.

    For CSVs with a simple layout, `columns` or `rows` (but not both) may be
    specified. For more complex table structures that cannot be described by
    `columns` or `rows`, you may omit both attributes. Note that more complex
    table structures are often more difficult to use; consider dividing them
    into multiple, simpler tabular outputs.
    """
    columns: typing.Union[typing.Iterable[Output], None] = None
    """An iterable of `Output`s representing the table's columns. The `key` of
    each input must match the corresponding column header."""

    rows: typing.Union[typing.Iterable[Output], None] = None
    """An iterable of `Output`s representing the table's rows. The `key` of
    each input must match the corresponding row header."""

    index_col: typing.Union[str, None] = None
    """The header name of the column that is the index of the table."""


@dataclasses.dataclass
class DirectoryOutput(Output):
    """A directory output, or result, of an invest model.

    Use this type when you need to specify a group of many file-based outputs,
    or an unknown number of file-based outputs, by grouping them together in a
    directory.
    """
    contents: typing.Union[typing.Iterable[Output], None] = None
    """An iterable of `Output`s representing the contents of this directory.
    The `key` of each output must be the file name or pattern."""


@dataclasses.dataclass
class FileOutput(Output):
    """A generic file output, or result, of an invest model.

    This represents a not-otherwise-specified file output type. Use this only if
    a more specific type, such as `CSVOutput` or `VectorOutput`, does not apply.
    """
    pass


@dataclasses.dataclass
class NumberOutput(Output):
    """A floating-point number output, or result, of an invest model.

    Use a more specific type (such as `IntegerOutput`, `RatioOutput`, or
    `PercentOutput`) where applicable.
    """
    units: typing.Union[pint.Unit, None] = None
    """The units of measurement for this numeric value"""


@dataclasses.dataclass
class IntegerOutput(Output):
    """An integer output, or result, of an invest model."""
    pass


@dataclasses.dataclass
class RatioOutput(Output):
    """A ratio output, or result, of an invest model.

    A ratio is a proportion expressed as a value from 0 to 1 (in contrast to a
    percent, which ranges from 0 to 100).
    """
    pass


@dataclasses.dataclass
class PercentOutput(Output):
    """A percent output, or result, of an invest model.

    A percent is a proportion expressed as a value from 0 to 100 (in contrast to
    a ratio, which ranges from 0 to 1).
    """
    pass


@dataclasses.dataclass
class StringOutput(Output):
    """A string output, or result, of an invest model.

    This represents a textual output. Do not use this to represent numeric or
    file-based inputs which can be better represented by another type.
    """
    pass


@dataclasses.dataclass
class OptionStringOutput(Output):
    """A string output, or result, which is limited to a set of options."""

    options: typing.Union[list, None] = None
    """A list of the values that this input may take"""


@dataclasses.dataclass
class ModelSpec:
    """Specification of an invest model describing metadata, inputs, and outputs."""

    model_id: str
    """The unique identifier for the plugin model, used internally by invest.
    This identifier should be concise, meaningful, and unique - a good choice
    is often a short version of the ``model_title``, or the name of the github
    repo. Using snake-case is recommended for consistancy. Including the word
    "model" is redundant and not recommended.

    Good example: ``'carbon_storage'``
    Bad examples: ``'Carbon storage'``, ``carbon_storage_model``
    """

    model_title: str
    """The user-facing title for the plugin. This is displayed in the workbench.
    Title-case is recommended. Including the word "model" is redundant and not
    recommended.

    Good example: ``'Carbon Storage'``
    Bad examples: ``'Carbon storage'``, ``The Carbon Storage Model``, ``carbon_storage``
    """

    userguide: str
    """Optional. For core invest models, this is the name of the models'
    documentation file in the core invest user guide. For plugins, this field
    is currently unused. It may be set to a URL pointing to the plugin
    documentation. A future invest release will display it as a link.

    Example: ``"https://github.com/natcap/invest-demo-plugin/blob/main/README.md"``
    """

    input_field_order: list[list[str]]
    """A list that specifies the order and grouping of model inputs.
    Inputs will be displayed in the input form from top to bottom in the order
    listed here. Sub-lists represent groups of inputs that will be visually
    separated by a horizontal line. This improves UX by breaking up long lists
    and visually grouping related inputs. If you do not wish to use groups,
    all inputs may go in the same sub-list. It is a convention to begin with a
    group of ``workspace_dir`` and ``results_suffix``. Each item in the
    sub-lists must match the key of an ``Input`` in ``inputs``. The key of each
    ``Input`` must be included exactly once, unless it is hidden.

    Example: ``[['workspace_dir', 'results_suffix'], ['foo'], ['bar', baz']]``
    """

    inputs: typing.Iterable[Input]
    """An iterable of the data inputs, or parameters, to the model."""

    outputs: typing.Iterable[Output]
    """An iterable of the data outputs, or results, of the model."""

    validate_spatial_overlap: bool = True
    """If True, validation will check that the bounding boxes of all
    top-level spatial inputs overlap (after reprojecting all to the same
    coordinate reference system)."""

    different_projections_ok: bool = True
    """Whether spatial inputs are allowed to have different projections. If
    False, validation will check that all top-level spatial inputs have the
    same projection. This is only considered if ``validate_spatial_overlap``
    is ``True``."""

    aliases: set = dataclasses.field(default_factory=set)
    """Optional. A set of alternative names by which the model can be called
    from the invest command line interface, in addition to the ``model_id``."""

    def __post_init__(self):
        self.inputs_dict = {_input.id: _input for _input in self.inputs}
        self.outputs_dict = {_output.id: _output for _output in self.outputs}

    def get_input(self, key):
        """Get an Input of this model by its key."""
        return self.inputs_dict[key]

    def to_json(self):
        """Serialize an MODEL_SPEC dict to a JSON string.

        Args:
            spec (dict): An invest model's MODEL_SPEC.

        Raises:
            TypeError if any object type within the spec is not handled by
            json.dumps or by the fallback serializer.

        Returns:
            JSON String
        """

        def fallback_serializer(obj):
            """Serialize objects that are otherwise not JSON serializeable."""
            if isinstance(obj, pint.Unit):
                return format_unit(obj)
            # Sets are present in 'geometry_types' attributes of some args
            # We don't need to worry about deserializing back to a set/array
            # so casting to string is okay.
            elif isinstance(obj, set):
                return str(obj)
            elif isinstance(obj, types.FunctionType):
                return str(obj)
            elif dataclasses.is_dataclass(obj):
                as_dict = dataclasses.asdict(obj)
                if hasattr(obj, 'type'):
                    as_dict['type'] = obj.type
                return as_dict
            elif isinstance(obj, IterableWithDotAccess):
                return obj.to_json()
            elif obj is int:
                return 'integer'
            elif obj is float:
                return 'number'
            raise TypeError(f'fallback serializer is missing for {type(obj)}')

        spec_dict = self.__dict__.copy()
        # rename 'inputs' to 'args' to stay consistent with the old api
        spec_dict.pop('inputs')
        spec_dict.pop('inputs_dict')
        spec_dict.pop('outputs_dict')
        spec_dict['args'] = self.inputs_dict
        spec_dict['outputs'] = self.outputs_dict
        return json.dumps(spec_dict, default=fallback_serializer, ensure_ascii=False)


def build_model_spec(model_spec):
    """Convert an old-style MODEL_SPEC dictionary to the new class-based style."""
    inputs = [
        build_input_spec(argkey, argspec)
        for argkey, argspec in model_spec['args'].items()]
    outputs = [
        build_output_spec(argkey, argspec) for argkey, argspec in model_spec['outputs'].items()]
    different_projections_ok = False

    spatial_keys = set()
    for i in inputs:
        if i.type in['raster', 'vector']:
            spatial_keys.add(i.id)

    # validate_spatial_overlap is True if all top-level spatial inputs should overlap,
    # or a list of keys, if only a subset of the inputs must overlap
    validate_spatial_overlap = True
    if 'args_with_spatial_overlap' in model_spec:
        different_projections_ok = model_spec['args_with_spatial_overlap'].get('different_projections_ok', False)
        if set(spatial_keys) != set(model_spec['args_with_spatial_overlap']['spatial_keys']):
            validate_spatial_overlap = model_spec['args_with_spatial_overlap']['spatial_keys']

    return ModelSpec(
        model_id=model_spec['model_id'],
        model_title=model_spec['model_title'],
        userguide=model_spec['userguide'],
        aliases=model_spec['aliases'],
        inputs=inputs,
        outputs=outputs,
        input_field_order=model_spec['ui_spec']['order'],
        validate_spatial_overlap=validate_spatial_overlap,
        different_projections_ok=different_projections_ok)


def build_input_spec(argkey, arg):
    """Convert an old-style input spec dictionary to the new class-based style."""
    base_attrs = {
        'id': argkey,
        'name': arg.get('name', None),
        'about': arg.get('about', None),
        'required': arg.get('required', True),
        'allowed': arg.get('allowed', True),
        'hidden': arg.get('hidden', False)
    }

    t = arg['type']

    if t == 'option_string':
        return OptionStringInput(
            **base_attrs,
            options=arg['options'],
            dropdown_function=arg.get('dropdown_function', None))

    elif t == 'freestyle_string':
        return StringInput(
            **base_attrs,
            regexp=arg.get('regexp', None))

    elif t == 'number':
        return NumberInput(
            **base_attrs,
            units=arg['units'],
            expression=arg.get('expression', None))

    elif t == 'integer':
        return IntegerInput(**base_attrs)

    elif t == 'ratio':
        return RatioInput(**base_attrs)

    elif t == 'percent':
        return PercentInput(**base_attrs)

    elif t == 'boolean':
        return BooleanInput(**base_attrs)

    elif t == 'raster':
        return SingleBandRasterInput(
            **base_attrs,
            data_type=int if arg['bands'][1]['type'] == 'integer' else float,
            units=arg['bands'][1].get('units', None),
            projected=arg.get('projected', None),
            projection_units=arg.get('projection_units', None))

    elif t == 'vector':
        return VectorInput(
            **base_attrs,
            geometry_types=arg['geometries'],
            fields=[build_input_spec(key, field_spec)
                    for key, field_spec in arg['fields'].items()],
            projected=arg.get('projected', None),
            projection_units=arg.get('projection_units', None))

    elif t == 'csv':
        columns = None
        rows = None
        if 'columns' in arg:
            columns = [build_input_spec(col_name, col_spec)
                for col_name, col_spec in arg['columns'].items()]
        elif 'rows' in arg:
            rows = [build_input_spec(row_name, row_spec)
                    for row_name, row_spec in arg['rows'].items()]

        return CSVInput(
            **base_attrs,
            columns=columns,
            rows=rows,
            index_col=arg.get('index_col', None))

    elif t == 'directory':
        return DirectoryInput(
            contents=[
                build_input_spec(k, v) for k, v in arg['contents'].items()],
            permissions=arg.get('permissions', 'rx'),
            must_exist=arg.get('must_exist', None),
            **base_attrs)

    elif t == 'file':
        return FileInput(**base_attrs)

    elif t == {'raster', 'vector'}:
        return RasterOrVectorInput(
            **base_attrs,
            geometry_types=arg['geometries'],
            fields=[build_input_spec(key, field_spec)
                    for key, field_spec in arg['fields'].items()],
            data_type=int if arg['bands'][1]['type'] == 'integer' else float,
            units=arg['bands'][1].get('units', None),
            projected=arg.get('projected', None),
            projection_units=arg.get('projection_units', None))

    else:
        raise ValueError


def build_output_spec(key, spec):
    """Convert an old-style output spec dictionary to the new class-based style."""
    base_attrs = {
        'id': key,
        'about': spec.get('about', None),
        'created_if': spec.get('created_if', None)
    }

    if 'type' in spec:
        t = spec['type']
    else:
        file_extension = key.split('.')[-1]
        if file_extension == 'tif':
            t = 'raster'
        elif file_extension in {'shp', 'gpkg', 'geojson'}:
            t = 'vector'
        elif file_extension == 'csv':
            t = 'csv'
        elif file_extension in {'json', 'txt', 'pickle', 'db', 'zip',
                                'dat', 'idx', 'html'}:
            t = 'file'
        else:
            raise Warning(
                f'output {key} has no recognized file extension and '
                'no "type" property')

    if t == 'number':
        return NumberOutput(
            **base_attrs,
            units=spec['units'])

    elif t == 'integer':
        return IntegerOutput(**base_attrs)

    elif t == 'ratio':
        return RatioOutput(**base_attrs)

    elif t == 'percent':
        return PercentOutput(**base_attrs)

    elif t == 'raster':
        return SingleBandRasterOutput(
            **base_attrs,
            data_type=int if spec['bands'][1]['type'] == 'integer' else float,
            units=spec['bands'][1].get('units', None))

    elif t == 'vector':
        return VectorOutput(
            **base_attrs,
            geometry_types=spec['geometries'],
            fields=[build_output_spec(key, field_spec)
                    for key, field_spec in spec['fields'].items()])

    elif t == 'csv':
        return CSVOutput(
            **base_attrs,
            columns=[
                build_output_spec(key, col_spec) for key, col_spec in spec['columns'].items()],
            index_col=spec.get('index_col', None))

    elif t == 'directory':
        return DirectoryOutput(
            contents=[
                build_output_spec(k, v) for k, v in spec['contents'].items()],
            **base_attrs)

    elif t == 'freestyle_string':
        return StringOutput(**base_attrs)

    elif t == 'option_string':
        return OptionStringOutput(options=spec['options'])

    elif t == 'file':
        return FileOutput(**base_attrs)

    else:
        raise ValueError()


# Specs for common arg types ##################################################
WORKSPACE = {
    "name": gettext("workspace"),
    "about": gettext(
        "The folder where all the model's output files will be written. If "
        "this folder does not exist, it will be created. If data already "
        "exists in the folder, it will be overwritten."),
    "type": "directory",
    "contents": {},
    "must_exist": False,
    "permissions": "rwx",
}

SUFFIX = {
    "name": gettext("file suffix"),
    "about": gettext(
        "Suffix that will be appended to all output file names. Useful to "
        "differentiate between model runs."),
    "type": "freestyle_string",
    "required": False,
    "regexp": "[a-zA-Z0-9_-]*"
}

N_WORKERS = {
    "name": gettext("taskgraph n_workers parameter"),
    "about": gettext(
        "The n_workers parameter to provide to taskgraph. "
        "-1 will cause all jobs to run synchronously. "
        "0 will run all jobs in the same process, but scheduling will take "
        "place asynchronously. Any other positive integer will cause that "
        "many processes to be spawned to execute tasks."),
    "type": "number",
    "units": u.none,
    "required": False,
    "expression": "value >= -1",
    "hidden": True
}

METER_RASTER = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.meter
        }
    }
}
AOI = {
    "type": "vector",
    "fields": {},
    "geometries": {"POLYGON", "MULTIPOLYGON"},
    "name": gettext("area of interest"),
    "about": gettext(
        "A map of areas over which to aggregate and "
        "summarize the final results."),
}
LULC = {
    "type": "raster",
    "bands": {1: {"type": "integer"}},
    "about": gettext(
        "Map of land use/land cover codes. Each land use/land cover type "
        "must be assigned a unique integer code."),
    "name": gettext("land use/land cover")
}
DEM = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.meter
        }
    },
    "about": gettext("Map of elevation above sea level."),
    "name": gettext("digital elevation model")
}
PRECIP = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.millimeter/u.year
        }
    },
    "about": gettext("Map of average annual precipitation."),
    "name": gettext("precipitation")
}
ET0 = {
    "name": gettext("reference evapotranspiration"),
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.millimeter
        }
    },
    "about": gettext("Map of reference evapotranspiration values.")
}
SOIL_GROUP = {
    "type": "raster",
    "bands": {1: {"type": "integer"}},
    "about": gettext(
        "Map of soil hydrologic groups. Pixels may have values 1, 2, 3, or 4, "
        "corresponding to soil hydrologic groups A, B, C, or D, respectively."),
    "name": gettext("soil hydrologic group")
}
THRESHOLD_FLOW_ACCUMULATION = {
    "expression": "value >= 0",
    "type": "number",
    "units": u.pixel,
    "about": gettext(
        "The number of upslope pixels that must flow into a pixel "
        "before it is classified as a stream."),
    "name": gettext("threshold flow accumulation")
}
LULC_TABLE_COLUMN = {
    "type": "integer",
    "about": gettext(
        "LULC codes from the LULC raster. Each code must be a unique "
        "integer.")
}

# Specs for common outputs ####################################################
TASKGRAPH_DIR = {
    "type": "directory",
    "about": (
        "Cache that stores data between model runs. This directory contains no "
        "human-readable data and you may ignore it."),
    "contents": {
        "taskgraph.db": {}
    }
}
FILLED_DEM = {
    "about": gettext("Map of elevation after any pits are filled"),
    "bands": {1: {
        "type": "number",
        "units": u.meter
    }}
}
FLOW_ACCUMULATION = {
    "about": gettext("Map of flow accumulation"),
    "bands": {1: {
        "type": "number",
        "units": u.none
    }}
}
FLOW_DIRECTION = {
    "about": gettext(
        "MFD flow direction. Note: the pixel values should not "
        "be interpreted directly. Each 32-bit number consists "
        "of 8 4-bit numbers. Each 4-bit number represents the "
        "proportion of flow into one of the eight neighboring "
        "pixels."),
    "bands": {1: {"type": "integer"}}
}
FLOW_DIRECTION_D8 = {
    "about": gettext(
        "D8 flow direction."),
    "bands": {1: {"type": "integer"}}
}
SLOPE = {
    "about": gettext(
        "Percent slope, calculated from the pit-filled "
        "DEM. 100 is equivalent to a 45 degree slope."),
    "bands": {1: {"type": "percent"}}
}
STREAM = {
    "about": "Stream network, created using flow direction and flow accumulation derived from the DEM and Threshold Flow Accumulation. Values of 1 represent streams, values of 0 are non-stream pixels.",
    "bands": {1: {"type": "integer"}}
}

FLOW_DIR_ALGORITHM = {
    "flow_dir_algorithm": {
        "type": "option_string",
        "options": {
            "D8": {
                "display_name": gettext("D8"),
                "description": "D8 flow direction"
            },
            "MFD": {
                "display_name": gettext("MFD"),
                "description": "Multiple flow direction"
            }
        },
        "about": gettext("Flow direction algorithm to use."),
        "name": gettext("flow direction algorithm")
    }
}

# geometry types ##############################################################
# the full list of ogr geometry types is in an enum in
# https://github.com/OSGeo/gdal/blob/master/gdal/ogr/ogr_core.h

POINT = {'POINT'}
LINESTRING = {'LINESTRING'}
POLYGON = {'POLYGON'}
MULTIPOINT = {'MULTIPOINT'}
MULTILINESTRING = {'MULTILINESTRING'}
MULTIPOLYGON = {'MULTIPOLYGON'}

LINES = LINESTRING | MULTILINESTRING
POLYGONS = POLYGON | MULTIPOLYGON
POINTS = POINT | MULTIPOINT
ALL_GEOMS = LINES | POLYGONS | POINTS


def format_unit(unit):
    """Represent a pint Unit as user-friendly unicode text.

    This attempts to follow the style guidelines from the NIST
    Guide to the SI (https://www.nist.gov/pml/special-publication-811):
    - Use standard symbols rather than spelling out
    - Use '/' to represent division
    - Use the center dot ' · ' to represent multiplication
    - Combine denominators into one, surrounded by parentheses

    Args:
        unit (pint.Unit): the unit to format

    Raises:
        TypeError if unit is not an instance of pint.Unit.

    Returns:
        String describing the unit.
    """
    if unit is None:
        return ''

    if not isinstance(unit, pint.Unit):
        raise TypeError(
            f'{unit} is of type {type(unit)}. '
            f'It should be an instance of pint.Unit')

    # Optionally use a pre-set format for a particular unit
    custom_formats = {
        u.pixel: gettext('number of pixels'),
        u.year_AD: '',  # don't need to mention units for a year input
        u.other: '',    # for inputs that can have any or multiple units
        # For soil erodibility (t*h*ha/(ha*MJ*mm)), by convention the ha's
        # are left on top and bottom and don't cancel out
        # pint always cancels units where it can, so add them back in here
        # this isn't a perfect solution
        # see https://github.com/hgrecco/pint/issues/1364
        u.t * u.hr / (u.MJ * u.mm): 't · h · ha / (ha · MJ · mm)',
        u.none: gettext('unitless')
    }
    if unit in custom_formats:
        return custom_formats[unit]

    # look up the abbreviated symbol for each unit
    # `formatter` expects an iterable of (unit, exponent) pairs, which lives in
    # the pint.Unit's `_units` attribute.
    unit_items = [(u.get_symbol(key), val) for key, val in unit._units.items()]
    formatted_unit = pint.formatting.formatter(
        unit_items,
        as_ratio=True,
        single_denominator=True,
        product_fmt=" · ",
        division_fmt='/',
        power_fmt="{}{}",
        parentheses_fmt="({})",
        exp_call=pint.formatting._pretty_fmt_exponent)

    if 'currency' in formatted_unit:
        formatted_unit = formatted_unit.replace('currency', gettext('currency units'))
    return formatted_unit


# accepted geometry_types for a vector will be displayed in this order
GEOMETRY_ORDER = [
    'POINT',
    'MULTIPOINT',
    'LINESTRING',
    'MULTILINESTRING',
    'POLYGON',
    'MULTIPOLYGON']

INPUT_TYPES_HTML_FILE = 'input_types.html'


def format_required_string(required):
    """Represent an arg's required status as a user-friendly string.

    Args:
        required (bool | str | None): required property of an arg. May be
            `True`, `False`, `None`, or a conditional string.

    Returns:
        string
    """
    if required is None or required is True:
        return gettext('required')
    elif required is False:
        return gettext('optional')
    else:
        # assume that the about text will describe the conditional
        return gettext('conditionally required')


def format_geometry_types_string(geometry_types):
    """Represent a set of allowed vector geometry types as user-friendly text.

    Args:
        geometry_types (set(str)): set of geometry names

    Returns:
        string
    """
    # sort the geometry types so they always display in a consistent order
    sorted_geoms = sorted(
        geometry_types,
        key=lambda g: GEOMETRY_ORDER.index(g))
    return '/'.join(gettext(geom).lower() for geom in sorted_geoms)


def format_permissions_string(permissions):
    """Represent a rwx-style permissions string as user-friendly text.

    Args:
        permissions (str): rwx-style permissions string

    Returns:
        string
    """
    permissions_strings = []
    if 'r' in permissions:
        permissions_strings.append(gettext('read'))
    if 'w' in permissions:
        permissions_strings.append(gettext('write'))
    if 'x' in permissions:
        permissions_strings.append(gettext('execute'))
    return ', '.join(permissions_strings)


def format_options_string_from_dict(options):
    """Represent a dictionary of option: description pairs as a bulleted list.

    Args:
        options (dict): the dictionary of options to document, where keys are
            options and values are dictionaries describing the options.
            They may have either or both 'display_name' and 'description' keys,
            for example:
            {'option1': {'display_name': 'Option 1', 'description': 'the first option'}}

    Returns:
        list of RST-formatted strings, where each is a line in a bullet list
    """
    lines = []
    for key, info in options.items():
        display_name = info['display_name'] if 'display_name' in info else key
        if 'description' in info:
            lines.append(f'- {display_name}: {info["description"]}')
        else:
            lines.append(f'- {display_name}')
    # sort the options alphabetically
    # casefold() is a more aggressive version of lower() that may work better
    # for some languages to remove all case distinctions
    return sorted(lines, key=lambda line: line.casefold())


def format_options_string_from_list(options):
    """Represent options as a comma-separated list.

    Args:
        options (list[str]): the set of options to document

    Returns:
        string of comma-separated options
    """
    return ', '.join(options)


def capitalize(title):
    """Capitalize a string into title case.

    Args:
        title (str): string to capitalize

    Returns:
        capitalized string (each word capitalized except linking words)
    """

    def capitalize_word(word):
        """Capitalize a word, if appropriate."""
        if word in {'of', 'the'}:
            return word
        else:
            return word[0].upper() + word[1:]

    title = ' '.join([capitalize_word(word) for word in title.split(' ')])
    title = '/'.join([capitalize_word(word) for word in title.split('/')])
    return title


def format_type_string(arg_type):
    """Represent an arg type as a user-friendly string.

    Args:
        arg_type (str|set(str)): the type to format. May be a single type or a
            set of types.

    Returns:
        formatted string that links to a description of the input type(s)
    """
    # some types need a more user-friendly name
    # all types are listed here so that they can be marked up for translation
    type_names = {
        BooleanInput: gettext('true/false'),
        CSVInput: gettext('CSV'),
        DirectoryInput: gettext('directory'),
        FileInput: gettext('file'),
        StringInput: gettext('text'),
        IntegerInput: gettext('integer'),
        NumberInput: gettext('number'),
        OptionStringInput: gettext('option'),
        PercentInput: gettext('percent'),
        SingleBandRasterInput: gettext('raster'),
        RatioInput: gettext('ratio'),
        VectorInput: gettext('vector'),
        RasterOrVectorInput: gettext('raster or vector')
    }
    type_sections = {  # names of section headers to link to in the RST
        BooleanInput: 'truefalse',
        CSVInput: 'csv',
        DirectoryInput: 'directory',
        FileInput: 'file',
        StringInput: 'text',
        IntegerInput: 'integer',
        NumberInput: 'number',
        OptionStringInput: 'option',
        PercentInput: 'percent',
        SingleBandRasterInput: 'raster',
        RatioInput: 'ratio',
        VectorInput: 'vector',
        RasterOrVectorInput: 'raster'
    }
    if arg_type is RasterOrVectorInput:
        return (
            f'`{type_names[SingleBandRasterInput]} <{INPUT_TYPES_HTML_FILE}#{type_sections[SingleBandRasterInput]}>`__ or '
            f'`{type_names[VectorInput]} <{INPUT_TYPES_HTML_FILE}#{type_sections[VectorInput]}>`__')
    return f'`{type_names[arg_type]} <{INPUT_TYPES_HTML_FILE}#{type_sections[arg_type]}>`__'


def describe_arg_from_spec(name, spec):
    """Generate RST documentation for an arg, given an arg spec.

    This is used for documenting:
        - a single top-level arg
        - a row or column in a CSV
        - a field in a vector
        - an item in a directory

    Args:
        name (str): Name to give the section. For top-level args this is
            arg['name']. For nested args it's typically their key in the
            dictionary one level up.
        spec (dict): A arg spec dictionary that conforms to the InVEST args
            spec specification. It must at least have the key `'type'`, and
            whatever other keys are expected for that type.
    Returns:
        list of strings, where each string is a line of RST-formatted text.
        The first line has the arg name, type, required state, description,
        and units if applicable. Depending on the type, there may be additional
        lines that are indented, that describe details of the arg such as
        vector fields and geometry types, option_string options, etc.
    """
    type_string = format_type_string(type(spec))
    in_parentheses = [type_string]

    # For numbers and rasters that have units, display the units
    units = spec.units if hasattr(spec, 'units') else None
    if units:
        units_string = format_unit(units)
        if units_string:
            # pybabel can't find the message if it's in the f-string
            translated_units = gettext("units")
            in_parentheses.append(f'{translated_units}: **{units_string}**')

    if type(spec) is VectorInput:
        in_parentheses.append(format_geometry_types_string(spec.geometry_types))

    # Represent the required state as a string, defaulting to required
    # It doesn't make sense to include this for boolean checkboxes
    if type(spec) is not BooleanInput:
        required_string = format_required_string(spec.required)
        in_parentheses.append(f'*{required_string}*')

    # Nested args may not have an about section
    if spec.about:
        sanitized_about_string = spec.about.replace("_", "\\_")
        about_string = f': {sanitized_about_string}'
    else:
        about_string = ''

    first_line = f"**{name}** ({', '.join(in_parentheses)}){about_string}"

    # Add details for the types that have them
    indented_block = []
    if type(spec) is OptionStringInput:
        # may be either a dict or set. if it's empty, the options are
        # dynamically generated. don't try to document them.
        if spec.options:
            if isinstance(spec.options, dict):
                indented_block.append(gettext('Options:'))
                indented_block += format_options_string_from_dict(spec.options)
            else:
                formatted_options = format_options_string_from_list(spec.options)
                indented_block.append(gettext('Options:') + f' {formatted_options}')

    elif type(spec) is CSVInput:
        if not spec.columns and not spec.rows:
            first_line += gettext(
                ' Please see the sample data table for details on the format.')

    # prepend the indent to each line in the indented block
    return [first_line] + ['\t' + line for line in indented_block]


def describe_arg_from_name(module_name, *arg_keys):
    """Generate RST documentation for an arg, given its model and name.

    Args:
        module_name (str): invest model module containing the arg.
        *arg_keys: one or more strings that are nested arg keys.

    Returns:
        String describing the arg in RST format. Contains an anchor named
        <arg_keys[0]>-<arg_keys[1]>...-<arg_keys[n]>
        where underscores in arg keys are replaced with hyphens.
    """
    # import the specified module (that should have an MODEL_SPEC attribute)
    module = importlib.import_module(module_name)
    # start with the spec for all args
    # narrow down to the nested spec indicated by the sequence of arg keys
    spec = module.MODEL_SPEC.get_input(arg_keys[0])
    for i, key in enumerate(arg_keys[1:]):
        # convert raster band numbers to ints
        if arg_keys[i - 1] == 'bands':
            key = int(key)
        if key in {'bands', 'fields', 'contents', 'columns', 'rows'}:
            spec = getattr(spec, key)
        else:
            try:
                spec = spec.get(key)
            except KeyError:
                keys_so_far = '.'.join(arg_keys[:i + 1])
                raise ValueError(
                    f"Could not find the key '{keys_so_far}' in the "
                    f"{module_name} model's MODEL_SPEC")

    # format spec into an RST formatted description string
    if spec.name:
        arg_name = capitalize(spec.name)
    else:
        arg_name = arg_keys[-1]

    # anchor names cannot contain underscores. sphinx will replace them
    # automatically, but lets explicitly replace them here
    anchor_name = '-'.join(arg_keys).replace('_', '-')
    rst_description = '\n\n'.join(describe_arg_from_spec(arg_name, spec))
    return f'.. _{anchor_name}:\n\n{rst_description}'


def write_metadata_file(datasource_path, spec, keywords_list,
                        lineage_statement='', out_workspace=None):
    """Write a metadata sidecar file for an invest dataset.

    Create metadata for invest model inputs or outputs, taking care to
    preserve existing human-modified attributes.

    Note: We do not want to overwrite any existing metadata so if there is
    invalid metadata for the datasource (i.e., doesn't pass geometamaker
    validation in ``describe``), this function will NOT create new metadata.

    Args:
        datasource_path (str) - filepath to the data to describe
        spec (dict) - the invest specification for ``datasource_path``
        keywords_list (list) - sequence of strings
        lineage_statement (str, optional) - string to describe origin of
            the dataset
        out_workspace (str, optional) - where to write metadata if different
            from data location
    Returns:
        None

    """
    def _get_key(key, resource):
        """Map name of actual key in yml from model_spec key name."""
        names = {field.name.lower(): field.name
                 for field in resource.data_model.fields}
        return names[key]

    try:
        resource = geometamaker.describe(datasource_path)
    except ValidationError:
        LOGGER.debug(
            f"Skipping metadata creation for {datasource_path}, as invalid "
            "metadata exists.")
        return None
    # Don't want function to fail bc can't create metadata due to invalid filetype
    except ValueError as e:
        LOGGER.debug(f"Skipping metadata creation for {datasource_path}: {e}")
        return None
    resource.set_lineage(lineage_statement)
    # a pre-existing metadata doc could have keywords
    words = resource.get_keywords()
    resource.set_keywords(set(words + keywords_list))

    if spec.about:
        resource.set_description(spec.about)
    attr_specs = None
    if hasattr(spec, 'columns') and spec.columns:
        attr_specs = spec.columns
    if hasattr(spec, 'fields') and spec.fields:
        attr_specs = spec.fields
    if attr_specs:
        for nested_spec in attr_specs:
            try:
                # field names in attr_spec are always lowercase, but the
                # actual fieldname in the data could be any case because
                # invest does not require case-sensitive fieldnames
                yaml_key = _get_key(nested_spec.id, resource)
                # Field description only gets set if its empty, i.e. ''
                if len(resource.get_field_description(yaml_key)
                       .description.strip()) < 1:
                    about = nested_spec.about
                    resource.set_field_description(yaml_key, description=about)
                # units only get set if empty
                if len(resource.get_field_description(yaml_key)
                       .units.strip()) < 1:
                    units = format_unit(nested_spec.units) if hasattr(nested_spec, 'units') else ''
                    resource.set_field_description(yaml_key, units=units)
            except KeyError as error:
                # fields that are in the spec but missing
                # from model results because they are conditional.
                LOGGER.debug(error)
    if isinstance(spec, SingleBandRasterInput) or isinstance(spec, SingleBandRasterOutput):
        if len(resource.get_band_description(1).units) < 1:
            units = format_unit(spec.units)
            resource.set_band_description(1, units=units)

    resource.write(workspace=out_workspace)


def generate_metadata_for_outputs(model_module, args_dict):
    """Create metadata for all items in an invest model output workspace.

    Args:
        model_module (object) - the natcap.invest module containing
            the MODEL_SPEC attribute
        args_dict (dict) - the arguments dictionary passed to the
            model's ``execute`` function.

    Returns:
        None

    """
    file_suffix = utils.make_suffix_string(args_dict, 'results_suffix')
    formatted_args = pprint.pformat(args_dict)
    lineage_statement = (
        f'Created by {model_module.__name__}.execute(\n{formatted_args})\n'
        f'Version {natcap.invest.__version__}')
    keywords = [model_module.MODEL_SPEC.model_id, 'InVEST']

    def _walk_spec(output_spec, workspace):
        for spec_data in output_spec:
            if type(spec_data) is DirectoryOutput:
                if 'taskgraph.db' in [s.id for s in spec_data.contents]:
                    continue
                _walk_spec(
                    spec_data.contents,
                    os.path.join(workspace, spec_data.id))
            else:
                pre, post = os.path.splitext(spec_data.id)
                full_path = os.path.join(workspace, f'{pre}{file_suffix}{post}')
                if os.path.exists(full_path):
                    try:
                        write_metadata_file(
                            full_path, spec_data, keywords, lineage_statement)
                    except ValueError as error:
                        # Some unsupported file formats, e.g. html
                        LOGGER.debug(error)

    _walk_spec(model_module.MODEL_SPEC.outputs, args_dict['workspace_dir'])
