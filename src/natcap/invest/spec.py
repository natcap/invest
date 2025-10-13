import contextlib
import functools
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
from urllib.parse import urlparse

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import geometamaker
import natcap.invest
import pandas
import pint
import pygeoprocessing
from pygeoprocessing.geoprocessing_core import GDALUseExceptions
from pydantic import AfterValidator, BaseModel, ConfigDict, \
    field_validator, model_validator, ValidationError
import taskgraph

from natcap.invest.file_registry import FileRegistry
from natcap.invest import utils
from natcap.invest.validation import get_message
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
    with GDALUseExceptions():
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


def validate_permissions_string(permissions):
    """
    Validate an rwx-style permissions string.

    Args:
        permissions (str): a string to validate as permissions

    Returns:
        None

    Raises:
        AssertionError if `permissions` isn't a string, if it's
        an empty string, if it has any letters besides 'r', 'w', 'x',
        or if it has any of those letters more than once
    """
    valid_letters = {'r', 'w', 'x'}
    used_letters = set()
    for letter in permissions:
        if letter not in valid_letters:
            raise ValueError('permissions contains a letter other than r,w,x')
        if letter in used_letters:
            raise ValueError('permissions contains a duplicate letter')
        used_letters.add(letter)
    return permissions


class Input(BaseModel):
    """A data input, or parameter, of an invest model.

    This represents an abstract input or parameter, which is rendered as an
    input field in the InVEST workbench. This does not store the value of the
    parameter for a specific run of the model.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Allow fields to have arbitrary types (that don't inherit from BaseModel).
    Needed for pint.Unit."""

    id: str
    """Input identifier that should be unique within a model"""

    name: typing.Union[str, None] = None
    """The user-facing name of the input. The workbench UI displays this
    property as a label for each input. The name should be as short as
    possible. Any extra description should go in ``about``. The name should
    be all lower-case, except for things that are always capitalized (acronyms,
    proper names).

    Good examples: ``precipitation``, ``Kc factor``, ``valuation table``

    Bad examples: ``PRECIPITATION``, ``kc_factor``, ``table of valuation parameters``
    """

    about: typing.Union[str, None] = None
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

    def format_required_string(self) -> str:
        """Represent this input's required status as a user-friendly string."""
        if self.required is True:
            return gettext('required')
        elif self.required is False:
            return gettext('optional')
        else:
            # assume that the about text will describe the conditional
            return gettext('conditionally required')


    def capitalize_name(self) -> str:
        """Capitalize a self.name into title case.

        Returns:
            capitalized string (each word capitalized except linking words)
        """

        def capitalize_word(word):
            """Capitalize a word, if appropriate."""
            if word in {'of', 'the'}:
                return word
            else:
                return word[0].upper() + word[1:]

        title = ' '.join([capitalize_word(word) for word in self.name.split(' ')])
        title = '/'.join([capitalize_word(word) for word in title.split('/')])
        return title

    def preprocess(self, value):
        """Base preprocessing function.

        Override this when specific preprocessing is needed.

        Args:
            value (object): value to preprocess

        Returns:
            unchanged value (object)
        """
        return value


class Output(BaseModel):
    """A data output, or result, of an invest model.

    This represents an abstract output which is produced as a result of running
    an invest model. This does not store the value of the output for a specific
    run of the model.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Allow fields to have arbitrary types (that don't inherit from BaseModel).
    Needed for pint.Unit."""

    id: str
    """Output identifier that should be unique within a model"""

    about: typing.Union[str, None] = None
    """User-facing description of the output"""

    created_if: typing.Union[bool, str] = True
    """Defaults to True. If the input is only created under a certain condition
    (such as when running the model in a specific mode), provide a string
    expression that evaluates to a boolean to describe this condition."""


class FileInput(Input):
    """A generic file input, or parameter, of an invest model.

    This represents a not-otherwise-specified file input type. Use this only if
    a more specific type, such as `CSVInput` or `VectorInput`, does not apply.
    """
    permissions: typing.Annotated[str, AfterValidator(
        validate_permissions_string)] = 'r'
    """A string that includes the lowercase characters ``r``, ``w`` and/or
    ``x``, indicating read, write, and execute permissions (respectively)
    required for this file."""

    type: typing.ClassVar[str] = 'file'

    display_name: typing.ClassVar[str] = gettext('file')

    rst_section: typing.ClassVar[str] = 'file'

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
        def format_path(p):
            if pandas.isna(p):
                return p
            p = str(p).strip()
            # don't expand remote paths
            if utils._GDALPath.from_uri(p).is_local:
                if not utils._GDALPath.from_uri(base_path).is_local:
                    raise ValueError('Remote CSVs cannot reference local file paths')
                return utils.expand_path(p, base_path)
            return p

        return col.apply(format_path).astype(pandas.StringDtype())


class SpatialFileInput(FileInput):
    """Base class for raster and vector spatial inputs."""

    projected: typing.Union[bool, None] = None
    """Defaults to None, indicating a projected (as opposed to geographic)
    coordinate system is not required. Set to True if a projected coordinate
    system is required."""

    projection_units: typing.Union[pint.Unit, None] = None
    """Defaults to None. If `projected` is `True`, and a specific unit of
    projection (such as meters) is required, indicate it here."""

    @model_validator(mode='after')
    def check_projected_projection_units(self):
        if self.projection_units and not self.projected:
            raise ValueError(
                'Cannot specify projection_units when projected is None')
        return self

    def preprocess(self, value):
        """Normalize a path to a GDAL-compatible local or remote path.

        Args:
            value (string): path to normalize

        Returns:
            normalized path string or None
        """
        if value:
            return utils._GDALPath.from_uri(value).to_normalized_path()
        return None  # if None or empty string, return None


class RasterBand(BaseModel):
    """A single-band raster input, or parameter, of an invest model.

    This represents a raster file input (all GDAL-supported raster file types
    are allowed), where only the first band is needed.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Allow fields to have arbitrary types (that don't inherit from BaseModel).
    Needed for pint.Unit."""

    band_id: typing.Union[int, str] = 1
    """band index used to access the raster band"""

    data_type: typing.Type = float
    """float or int"""

    units: typing.Union[pint.Unit, None]
    """units of measurement of the raster band values"""


class RasterInput(SpatialFileInput):
    """A raster input, or parameter, of an invest model.

    This represents a raster file input (all GDAL-supported raster file types
    are allowed), which may have multiple bands.
    """
    bands: list[RasterBand]
    """An iterable of `RasterBand` representing the bands expected to be in
    the raster."""

    type: typing.ClassVar[str] = 'raster'

    display_name: typing.ClassVar[str] = gettext('raster')

    rst_section: typing.ClassVar[str] = 'raster'

    @timeout
    def validate(self, filepath: str):
        """Validate a raster file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        with GDALUseExceptions():
            gdal_path = utils._GDALPath.from_uri(filepath)
            if gdal_path.is_local:
                file_warning = super().validate(filepath)
                if file_warning:
                    return file_warning

            try:
                gdal_dataset = gdal.OpenEx(
                    gdal_path.to_normalized_path(), gdal.OF_RASTER)
            except RuntimeError:
                return get_message('NOT_GDAL_RASTER')

            # Check that an overview .ovr file wasn't opened.
            if os.path.splitext(filepath)[1] == '.ovr':
                return get_message('OVR_FILE')

            srs = gdal_dataset.GetSpatialRef()
            projection_warning = _check_projection(
                srs, self.projected, self.projection_units)
            if projection_warning:
                return projection_warning


class SingleBandRasterInput(SpatialFileInput):
    """A single-band raster input, or parameter, of an invest model.

    This represents a raster file input (all GDAL-supported raster file types
    are allowed), where only the first band is needed. While the same thing can
    be achieved using a `RasterInput`, this class exists to simplify access to
    the band properties when there is only one band.
    """
    data_type: typing.Type = float
    """float or int"""

    units: typing.Union[pint.Unit, None]
    """units of measurement of the raster values"""

    type: typing.ClassVar[str] = 'raster'

    display_name: typing.ClassVar[str] = gettext('raster')

    rst_section: typing.ClassVar[str] = 'raster'

    @timeout
    def validate(self, filepath: str):
        """Validate a raster file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        with GDALUseExceptions():
            gdal_path = utils._GDALPath.from_uri(filepath)
            if gdal_path.is_local:
                file_warning = super().validate(filepath)
                if file_warning:
                    return file_warning

            try:
                gdal_dataset = gdal.OpenEx(
                    gdal_path.to_normalized_path(), gdal.OF_RASTER)
            except RuntimeError:
                return get_message('NOT_GDAL_RASTER')

            # Check that an overview .ovr file wasn't opened.
            if os.path.splitext(filepath)[1] == '.ovr':
                return get_message('OVR_FILE')

            srs = gdal_dataset.GetSpatialRef()
            projection_warning = _check_projection(
                srs, self.projected, self.projection_units)
            if projection_warning:
                return projection_warning


class VectorInput(SpatialFileInput):
    """A vector input, or parameter, of an invest model.

    This represents a vector file input (all GDAL-supported vector file types
    are allowed). It is assumed that only the first layer is used.
    """
    geometry_types: set
    """A set of geometry type(s) that are allowed for this vector"""

    fields: list[Input]
    """An iterable of `Input`s representing the fields that this vector is
    expected to have. The `key` of each input must match the corresponding
    field name."""

    type: typing.ClassVar[str] = 'vector'

    display_name: typing.ClassVar[str] = gettext('vector')

    rst_section: typing.ClassVar[str] = 'vector'

    _fields_dict: dict[str, Input] = {}

    @model_validator(mode='after')
    def check_field_types(self):
        for field in (self.fields or []):
            if type(field) not in {IntegerInput, NumberInput, OptionStringInput,
                                   PercentInput, RatioInput, StringInput}:
                raise ValueError(f'Field {field} is not an allowed type')
        return self

    def model_post_init(self, context):
        self._fields_dict = {field.id: field for field in self.fields}

    def get_field(self, key: str) -> Input:
        return self._fields_dict[key]

    @timeout
    def validate(self, filepath: str):
        """Validate a vector file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        with GDALUseExceptions():
            gdal_path = utils._GDALPath.from_uri(filepath)
            if gdal_path.is_local:
                file_warning = super().validate(filepath)
                if file_warning:
                    return file_warning

            try:
                gdal_dataset = gdal.OpenEx(
                    gdal_path.to_normalized_path(), gdal.OF_VECTOR)
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

    def format_geometry_types_rst(self):
        """Represent self.geometry_types in RST text.

        Args:
            geometry_types (set(str)): set of geometry names

        Returns:
            string
        """
        # sort the geometry types so they always display in a consistent order
        sorted_geoms = sorted(
            self.geometry_types,
            key=lambda g: GEOMETRY_ORDER.index(g))
        return '/'.join(gettext(geom).lower() for geom in sorted_geoms)


class RasterOrVectorInput(SpatialFileInput):
    """An invest model input that can be either a single-band raster or a vector."""

    data_type: typing.Type = float
    """Data type for the raster values (float or int)"""

    units: typing.Union[pint.Unit, None]
    """Units of measurement of the raster values"""

    geometry_types: set
    """A set of geometry type(s) that are allowed for this vector"""

    fields: typing.Union[list[Input]]
    """An iterable of `Input`s representing the fields that this vector is
    expected to have. The `key` of each input must match the corresponding
    field name."""

    type: typing.ClassVar[str] = 'raster_or_vector'

    display_name: typing.ClassVar[str] = gettext('raster or vector')

    rst_section: typing.ClassVar[str] = 'raster'

    _single_band_raster_input: SingleBandRasterInput
    _vector_input: VectorInput
    _fields_dict: dict[str, Input] = {}

    def model_post_init(self, context):
        self._single_band_raster_input = SingleBandRasterInput(
            id=self.id,
            data_type=self.data_type,
            units=self.units,
            projected=self.projected,
            projection_units=self.projection_units)
        self._vector_input = VectorInput(
            id=self.id,
            geometry_types=self.geometry_types,
            fields=self.fields,
            projected=self.projected,
            projection_units=self.projection_units)
        self._fields_dict = {field.id: field for field in self.fields}

    def get_field(self, key: str) -> Input:
        return self.fields_dict[key]

    @timeout
    def validate(self, filepath: str):
        """Validate a raster or vector file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        try:
            gis_type = pygeoprocessing.get_gis_type(
                utils._GDALPath.from_uri(filepath).to_normalized_path())
        except ValueError as err:
            return str(err)
        if gis_type == pygeoprocessing.RASTER_TYPE:
            return self._single_band_raster_input.validate(filepath)
        else:
            return self._vector_input.validate(filepath)


class CSVInput(FileInput):
    """A CSV table input, or parameter, of an invest model.

    For CSVs with a simple layout, `columns` or `rows` (but not both) may be
    specified. For more complex table structures that cannot be described by
    `columns` or `rows`, you may omit both attributes. Note that more complex
    table structures are often more difficult to use; consider dividing them
    into multiple, simpler tabular inputs.
    """
    columns: typing.Union[list[Input], None] = None
    """An iterable of `Input`s representing the columns that this CSV is
    expected to have. The `id` of each input must match the corresponding
    column header."""

    rows: typing.Union[list[Input], None] = None
    """An iterable of `Input`s representing the rows that this CSV is
    expected to have. The `id` of each input must match the corresponding
    row header."""

    index_col: typing.Union[str, None] = None
    """The header name of the column to use as the index. When processing a
    CSV file to a dataframe, the dataframe index will be set to this column."""

    type: typing.ClassVar[str] = 'csv'

    display_name: typing.ClassVar[str] = gettext('CSV')

    rst_section: typing.ClassVar[str] = 'csv'

    _columns_dict: dict[str, Input] = {}
    _fields_dict: dict[str, Input] = {}

    @model_validator(mode='after')
    def check_not_both_rows_and_columns(self):
        if self.rows is not None and self.columns is not None:
            raise ValueError('Cannot have both rows and columns')
        return self

    @model_validator(mode='after')
    def check_index_col_in_columns(self):
        if (self.index_col is not None and
                self.index_col not in [s.id for s in self.columns]):
            raise ValueError(f'index_col {self.index_col} not found in columns')
        return self

    @model_validator(mode='after')
    def check_row_and_column_types(self):
        allowed_types = {
            BooleanInput, IntegerInput, NumberInput, OptionStringInput,
            PercentInput, RasterOrVectorInput, RatioInput, FileInput,
            SingleBandRasterInput, StringInput, VectorInput, CSVInput}
        for row in (self.rows or []):
            if type(row) not in allowed_types:
                raise ValueError(f'Row {row} is not an allowed type')
        for col in (self.columns or []):
            if type(col) not in allowed_types:
                raise ValueError(f'Column {col} is not an allowed type')
        return self

    def model_post_init(self, context):
        if self.columns:
            self._columns_dict = {col.id: col for col in self.columns}
        if self.rows:
            self._rows_dict = {row.id: row for row in self.rows}

    def get_column(self, key: str) -> Input:
        return self._columns_dict[key]

    def get_row(self, key: str) -> Input:
        return self._rows_dict[key]

    @timeout
    def validate(self, filepath: str):
        """Validate a CSV file against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        # don't check existence of remote paths
        if utils._GDALPath.from_uri(filepath).is_local:
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

                    def normalize_path(value):
                        if pandas.isna(value):
                            return value
                        return utils._GDALPath.from_uri(value).to_normalized_path()

                    df[col].apply(check_value)
                    df[col] = df[col].apply(normalize_path)

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

    def preprocess(self, value):
        """Preprocess a CSV path.

        Args:
            value (string): path to process

        Returns:
            path string or None
        """
        return value if value else None


class DirectoryInput(Input):
    """A directory input, or parameter, of an invest model.

    Use this type when you need to specify a group of many file-based inputs,
    or an unknown number of file-based inputs, by grouping them together in a
    directory. This may also be used to describe an empty directory where model
    outputs will be written to.
    """
    contents: list[Input]
    """An iterable of `Input`s representing the contents of this directory. The
    `key` of each input must be the file name or pattern."""

    permissions: typing.Annotated[str, AfterValidator(
        validate_permissions_string)] = ''
    """A string that includes the lowercase characters ``r``, ``w`` and/or ``x``,
    indicating read, write, and execute permissions (respectively) required for
    this directory."""

    must_exist: bool = True
    """Defaults to True, indicating the directory must already exist before
    running the model. Set to False if the directory will be created."""

    type: typing.ClassVar[str] = 'directory'

    display_name: typing.ClassVar[str] = gettext('directory')

    rst_section: typing.ClassVar[str] = 'directory'

    _contents_dict: dict[str, Input] = {}

    @model_validator(mode='after')
    def check_contents_types(self):
        allowed_types = {
            CSVInput, DirectoryInput, FileInput, RasterOrVectorInput,
            SingleBandRasterInput, VectorInput}
        for content in (self.contents or []):
            if type(content) not in allowed_types:
                raise ValueError(
                    f'Directory contents {content} is not an allowed type')
        return self

    def model_post_init(self, context):
        self._contents_dict = {x.id: x for x in self.contents}

    def get_contents(self, key: str) -> Input:
        return self._contents_dict[key]

    @timeout
    def validate(self, dirpath: str):
        """Validate a directory path against the requirements for this input.

        Args:
            filepath (string): The filepath to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        if not utils._GDALPath.from_uri(dirpath).is_local:
            return  # Don't check paths and permissions for remote paths

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


class NumberInput(Input):
    """A floating-point number input, or parameter, of an invest model.

    Use a more specific type (such as `IntegerInput`, `RatioInput`, or
    `PercentInput`) where applicable.
    """
    units: typing.Union[pint.Unit, None]
    """The units of measurement for this numeric value"""

    expression: typing.Union[str, None] = None
    """A string expression that can be evaluated to a boolean indicating whether
    the value meets a required condition. The expression must contain the string
    ``value``, which will represent the user-provided value (after it has been
    cast to a float). Example: ``"(value >= 0) & (value <= 1)"``."""

    type: typing.ClassVar[str] = 'number'

    display_name: typing.ClassVar[str] = gettext('number')

    rst_section: typing.ClassVar[str] = 'number'

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
            result = utils.evaluate_expression(self.expression, {'value': float(value)})
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

    def preprocess(self, value):
        """Normalize a value to a float.

        Args:
            value: value to preprocess

        Returns:
            float or None
        """
        return None if value in {None, ''} else float(value)


class IntegerInput(NumberInput):
    """An integer input, or parameter, of an invest model."""
    type: typing.ClassVar[str] = 'integer'

    display_name: typing.ClassVar[str] = gettext('integer')

    rst_section: typing.ClassVar[str] = 'integer'

    units: typing.Union[pint.Unit, None] = None

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

        # must first cast to float, to handle both string and float inputs
        # since we already called super().validate, we know that the value
        # can be cast to float
        if not float(value).is_integer():
            return get_message('NOT_AN_INTEGER').format(value=value)


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

    def preprocess(self, value):
        """Normalize a value to an integer.

        Args:
            value: value to preprocess

        Returns:
            int or None
        """
        # cast to float first to handle strings and floats
        return None if value in {None, ''} else int(float(value))


class NWorkersInput(IntegerInput):

    def preprocess(self, value):
        # unlike other numeric inputs, we allow n_workers to be None or an
        # empty string, and default to single process mode in that case
        if value is None or value == '':
            return -1
        return super().preprocess(value)


class RatioInput(NumberInput):
    """A ratio input, or parameter, of an invest model.

    A ratio is a proportion expressed as a value from 0 to 1 (in contrast to a
    percent, which ranges from 0 to 100). Values are restricted to the
    range [0, 1].
    """
    type: typing.ClassVar[str] = 'ratio'

    display_name: typing.ClassVar[str] = gettext('ratio')

    rst_section: typing.ClassVar[str] = 'ratio'

    units: typing.ClassVar[None] = None

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


class PercentInput(NumberInput):
    """A percent input, or parameter, of an invest model.

    A percent is a proportion expressed as a value from 0% to 100% (in contrast
    to a ratio, which ranges from 0 to 1). By default there is no restriction on
    the range a percent value can take, so values may be less than 0 or greater
    than 100. Use the ``expression`` parameter to enforce a value range.
    """
    type: typing.ClassVar[str] = 'percent'

    display_name: typing.ClassVar[str] = gettext('percent')

    rst_section: typing.ClassVar[str] = 'percent'

    units: typing.ClassVar[None] = None

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


class BooleanInput(Input):
    """A boolean input, or parameter, of an invest model."""
    type: typing.ClassVar[str] = 'boolean'

    display_name: typing.ClassVar[str] = gettext('true/false')

    rst_section: typing.ClassVar[str] = 'truefalse'

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

    def preprocess(self, value):
        """Normalize a value to a boolean.

        Args:
            value: value to preprocess

        Returns:
            bool or None
        """
        return None if value in {None, ''} else bool(value)


class StringInput(Input):
    """A string input, or parameter, of an invest model.

    This represents a textual input. Do not use this to represent numeric or
    file-based inputs which can be better represented by another type.
    """
    regexp: typing.Union[str, None] = None
    """An optional regex pattern which the text value must match"""

    type: typing.ClassVar[str] = 'string'

    display_name: typing.ClassVar[str] = gettext('text')

    rst_section: typing.ClassVar[str] = 'text'

    @field_validator('regexp', mode='after')
    @classmethod
    def check_regexp(cls, regexp: typing.Union[str, None]) -> typing.Union[str, None]:
        if regexp is not None:
            try:
                re.compile(regexp)
            except Exception:
                raise ValueError(f'Failed to compile regexp {regexp}')
        return regexp

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

    def preprocess(self, value):
        """Normalize a value to a string.

        Args:
            value: value to preprocess

        Returns:
            string or None
        """
        return None if value in {None, ''} else str(value)


class ResultsSuffixInput(StringInput):

    def preprocess(self, value):
        value = super().preprocess(value)
        if value is None:
            return ''
        # suffix should always start with an underscore
        if (value and not value.startswith('_')):
            value = '_' + value
        return value


class Option(BaseModel):
    """An option in an OptionStringInput or OptionStringOutput."""

    key: str
    """The unique key that identifies this option. If the OptionStringInput is
    represented by a dropdown menu, and `display_name` is `None`, this key will
    be displayed in the menu. For options in CSV columns etc, this is the value
    that should be entered in the column."""

    display_name: typing.Union[str, None] = None
    """For OptionStringInputs that are represented by a dropdown menu, this
    optional attribute will be displayed in the menu instead of the `key`."""

    about: typing.Union[str, None] = None
    """Optional description of this option. Only needed for keys that are
    not self-explanatory."""


class OptionStringInput(Input):
    """A string input, or parameter, which is limited to a set of options.

    This corresponds to a dropdown menu in the workbench, where the user
    is limited to a set of pre-defined options.
    """
    options: list[Option]
    """A list of the values that this input may take. Use this if the set of
    options is predetermined. If using `dropdown_function` instead, this
    should be an empty list."""

    dropdown_function: typing.Union[typing.Callable, None] = None
    """A function that returns a list of the values that this input may take.
    Use this if the set of options must be dynamically generated."""

    type: typing.ClassVar[str] = 'option_string'

    display_name: typing.ClassVar[str] = gettext('option')

    rst_section: typing.ClassVar[str] = 'option'

    @model_validator(mode='after')
    def check_options(self):
        if self.dropdown_function and self.options:
            raise ValueError(f'Cannot have both dropdown_function and options')
        return self

    def validate(self, value):
        """Validate a value against the requirements for this input.

        Args:
            value: The value to validate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        # if options is empty, that means it's dynamically populated
        # so validation should be left to the model's validate function.

        if self.options:
            option_keys = self.list_options()
            if str(value).lower() not in option_keys:
                return get_message('INVALID_OPTION').format(option_list=option_keys)

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

    def list_options(self):
        """Return a sorted list of the option keys."""
        if self.options:
            return sorted([option.key.lower() for option in self.options])

    def format_rst(self):
        """Represent `self.options` as a RST-formatted bulleted list.

        Args:
            options: list of Options to format

        Returns:
            list of RST-formatted strings, where each is a line in a bullet list
        """
        lines = []
        for option in self.options:
            display_name = option.display_name if option.display_name else option.key
            if option.about:
                lines.append(f'- "**{display_name}**": {option.about}')
            else:
                lines.append(f'- "**{display_name}**"')

        # sort the options alphabetically
        # casefold() is a more aggressive version of lower() that may work better
        # for some languages to remove all case distinctions
        return sorted(lines, key=lambda line: line.casefold())

    def preprocess(self, value):
        """Normalize an option string value to a lower cased string.

        Args:
            value: value to preprocess

        Returns:
            string
        """
        return None if value in {None, ''} else str(value).lower()


class FileOutput(Output):
    """A generic file output, or result, of an invest model.

    This represents a not-otherwise-specified file output type. Use this only if
    a more specific type, such as `CSVOutput` or `VectorOutput`, does not apply.
    """
    path: str
    """Path to the output file within the workspace directory"""


class SingleBandRasterOutput(FileOutput):
    """A single-band raster output, or result, of an invest model.

    This represents a raster file output (all GDAL-supported raster file types
    are allowed), where only the first band is used.
    """
    data_type: typing.Type = float
    """float or int"""

    units: typing.Union[pint.Unit, None] = None
    """units of measurement of the raster values"""


class RasterOutput(FileOutput):
    """A raster output, or result, of an invest model.

    This represents a raster file output (all GDAL-supported raster file types
    are allowed), which may have multiple bands.
    """
    bands: list[RasterBand]
    """An iterable of `RasterBand` representing the bands expected to be in
    the raster."""


class VectorOutput(FileOutput):
    """A vector output, or result, of an invest model.

    This represents a vector file output (all GDAL-supported vector file types
    are allowed). It is assumed that only the first layer is used.
    """
    geometry_types: set = set()
    """A set of geometry type(s) that are produced in this vector"""

    fields: list[Output]
    """An iterable of `Output`s representing the fields created in this vector.
    The `key` of each input must match the corresponding field name."""

    @model_validator(mode='after')
    def check_field_types(self):
        for field in (self.fields or []):
            if type(field) not in {IntegerOutput, NumberOutput, OptionStringOutput,
                                   PercentOutput, RatioOutput, StringOutput}:
                raise ValueError(f'Field {field} is not an allowed type')
        return self


class CSVOutput(FileOutput):
    """A CSV table output, or result, of an invest model.

    For CSVs with a simple layout, `columns` or `rows` (but not both) may be
    specified. For more complex table structures that cannot be described by
    `columns` or `rows`, you may omit both attributes. Note that more complex
    table structures are often more difficult to use; consider dividing them
    into multiple, simpler tabular outputs.
    """
    columns: typing.Union[list[Output], None] = None
    """An iterable of `Output`s representing the table's columns. The `key` of
    each input must match the corresponding column header."""

    rows: typing.Union[list[Output], None] = None
    """An iterable of `Output`s representing the table's rows. The `key` of
    each input must match the corresponding row header."""

    index_col: typing.Union[str, None] = None
    """The header name of the column that is the index of the table."""

    @model_validator(mode='after')
    def validate_index_col_in_columns(self):
        if (self.index_col is not None and
                self.index_col not in [s.id for s in self.columns]):
            raise ValueError(f'index_col {self.index_col} not found in columns')
        return self

    @model_validator(mode='after')
    def check_row_and_column_types(self):
        allowed_types = {
            IntegerOutput, NumberOutput, OptionStringOutput, PercentOutput,
            FileOutput, RatioOutput, SingleBandRasterOutput, StringOutput,
            VectorOutput}
        for row in (self.rows or []):
            if type(row) not in allowed_types:
                raise ValueError(f'Row {row} is not an allowed type')
        for col in (self.columns or []):
            if type(col) not in allowed_types:
                raise ValueError(f'Column {col} is not an allowed type')
        return self


class NumberOutput(Output):
    """A floating-point number output, or result, of an invest model.

    Use a more specific type (such as `IntegerOutput`, `RatioOutput`, or
    `PercentOutput`) where applicable.
    """
    units: typing.Union[pint.Unit, None] = None
    """The units of measurement for this numeric value"""


class IntegerOutput(Output):
    """An integer output, or result, of an invest model."""
    pass


class RatioOutput(Output):
    """A ratio output, or result, of an invest model.

    A ratio is a proportion expressed as a value from 0 to 1 (in contrast to a
    percent, which ranges from 0 to 100).
    """
    pass


class PercentOutput(Output):
    """A percent output, or result, of an invest model.

    A percent is a proportion expressed as a value from 0 to 100 (in contrast to
    a ratio, which ranges from 0 to 1).
    """
    pass


class StringOutput(Output):
    """A string output, or result, of an invest model.

    This represents a textual output. Do not use this to represent numeric or
    file-based inputs which can be better represented by another type.
    """
    pass


class OptionStringOutput(Output):
    """A string output, or result, which is limited to a set of options."""

    options: list[Option]
    """A list of the values that this input may take"""


class ModelSpec(BaseModel):
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

    inputs: list[Input]
    """A list of the data inputs, or parameters, to the model."""

    outputs: list[Output]
    """A list of the data outputs, or results, of the model."""

    validate_spatial_overlap: typing.Union[bool, list[str]] = True
    """If True, validation will check that the bounding boxes of all
    top-level spatial inputs overlap (after reprojecting all to the same
    coordinate reference system)."""

    different_projections_ok: bool = True
    """Whether spatial inputs are allowed to have different projections. If
    False, validation will check that all top-level spatial inputs have the
    same projection. This is only considered if ``validate_spatial_overlap``
    is ``True``."""

    aliases: set = set()
    """Optional. A set of alternative names by which the model can be called
    from the invest command line interface, in addition to the ``model_id``."""

    module_name: str
    """The importable module name of the model e.g. ``natcap.invest.foo``."""

    @model_validator(mode='after')
    def check_inputs_in_field_order(self):
        """Check that all inputs either appear in `input_field_order`,
        or are marked as hidden."""
        found_keys = set()
        for group in self.input_field_order:
            for key in group:
                if key in found_keys:
                    raise ValueError(
                        f'Key {key} appears more than once in input_field_order')
                found_keys.add(key)
        for _input in self.inputs:
            if _input.hidden is True:
                if _input.id in found_keys:
                    raise ValueError(
                        f'Input {_input.id} is hidden but appears in input_field_order')
                found_keys.add(_input.id)
        if found_keys != set([s.id for s in self.inputs]):
            raise ValueError(
                f'Mismatch between keys in inputs and input_field_order')
        return self

    def get_input(self, key: str) -> Input:
        """Get an Input of this model by its key."""
        return {_input.id: _input for _input in self.inputs}[key]

    def get_output(self, key: str) -> Output:
        """Get an Output of this model by its key."""
        return {_output.id: _output for _output in self.outputs}[key]

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
            elif obj is int:
                return 'integer'
            elif obj is float:
                return 'number'
            elif isinstance(obj, BaseModel):
                as_dict = obj.model_dump()
                # type is a ClassVar, so it won't be included in the default dump
                if hasattr(obj, 'type'):
                    as_dict['type'] = obj.type
                return as_dict
            raise TypeError(f'fallback serializer is missing for {type(obj)}')

        spec_dict = self.__dict__.copy()
        # rename 'inputs' to 'args' to stay consistent with the old api
        spec_dict.pop('inputs')
        spec_dict['args'] = {_input.id: _input for _input in self.inputs}
        spec_dict['outputs'] = {_output.id: _output for _output in self.outputs}
        return json.dumps(spec_dict, default=fallback_serializer, ensure_ascii=False)

    def preprocess_inputs(self, input_values):
        """Preprocess a dictionary of input values.

        The resulting dict will contain exactly the input keys in the model spec.
        Inputs which were not provided will have a value of None. Each provided
        input value is passed through the corresponding Input.preprocess method.

        Args:
            input_values (dict): Dict mapping input keys to input values

        Returns:
            dictionary mapping input keys to preprocessed input values
        """
        values = {}
        for _input in self.inputs:
            values[_input.id] = _input.preprocess(
                input_values.get(_input.id, None))
        return values

    def generate_metadata_for_outputs(self, args_dict):
        """Create metadata for all items in an invest model output workspace.

        Args:
            args_dict (dict) - the arguments dictionary passed to the
                model's ``execute`` function.

        Returns:
            None
        """
        from natcap.invest import models
        file_suffix = SUFFIX.preprocess(args_dict.get('results_suffix', None))
        formatted_args = pprint.pformat(args_dict)
        lineage_statement = (
            f'Created by {self.model_id} execute('
            f'\n{formatted_args})\nVersion {natcap.invest.__version__}')
        keywords = [self.model_id, 'InVEST']

        def _walk_spec(output_spec, workspace):
            for spec_data in output_spec:
                if 'taskgraph.db' in spec_data.path:
                    continue
                pre, post = os.path.splitext(spec_data.path)
                full_path = os.path.join(workspace, f'{pre}{file_suffix}{post}')
                if os.path.exists(full_path):
                    try:
                        write_metadata_file(
                            full_path, spec_data, keywords, lineage_statement)
                    except ValueError as error:
                        # Some unsupported file formats, e.g. html
                        LOGGER.debug(error)

        _walk_spec(self.outputs, args_dict['workspace_dir'])

    def create_output_directories(self, args):
        """Create the necessary output directories given a set of args.

        Args:
            args (dict): maps input keys to values

        Returns:
            None
        """
        # evaluate which outputs we expect to be created, given the
        # model spec and provided input values
        outputs_to_be_created = set([
            output.id for output in self.outputs if bool(
                utils.evaluate_expression(
                    expression=f'{output.created_if}',
                    variable_map=args
                )
            ) is True
        ])
        # Identify all output subdirectories needed, based on the output
        # paths, and create them
        for output in self.outputs:
            if output.id in outputs_to_be_created:
                os.makedirs(os.path.join(
                    args['workspace_dir'], os.path.split(output.path)[0]
                ), exist_ok=True)

    def setup(self, args, taskgraph_key='taskgraph_cache'):
        """Perform boilerplate setup needed in an invest execute function.

        Args:
            args (dict): maps input keys to values
            taskgraph_key (str): Input key that identifies the taskgraph cache.
                Defaults to 'taskgraph_cache'.

        Returns:
            Tuple of ``(args, file_registry, graph)`` where ``args`` is the
            result of passing the input args through ``self.preprocess_inputs``,
            ``file_registry`` is a ``FileRegistry``, and ``graph`` is a
            ``TaskGraph``, all instantiated appropriately for the given args
            and model specification.
        """
        args = self.preprocess_inputs(args)
        self.create_output_directories(args)
        file_registry = FileRegistry(
            outputs=self.outputs,
            workspace_dir=args['workspace_dir'],
            file_suffix=args['results_suffix'])
        graph = taskgraph.TaskGraph(file_registry[taskgraph_key],
                                    n_workers=args['n_workers'])
        return args, file_registry, graph

    def execute(self, args, create_logfile=False, log_level=logging.NOTSET,
            generate_metadata=False, save_file_registry=False,
            check_outputs=False):
        """Invest model execute function wrapper.

        Performs additonal work before and after the execute function runs:
            - GDAL exceptions are enabled
            - Optionally,

        Args:
            args (dict): the raw user input args dictionary
            generate_metadata (bool): Defaults to False. If True, use
                geometamaker to create metadata files in the workspace
                after execution completes.
            save_file_registry (bool): Defaults to False. If True, the
                file registry dictionary will be saved to the workspace
                as a JSON file after execution completes.
            create_logfile (bool): Defaults to False. If True, all logging
                from the execute function as well as all other pre- and
                post-processing will be written to a logfile in the workspace.
            log_level (int): The logging threshold for the log file (only applies
                if ``create_logfile`` is true. Log messages with a level less
                than this will be excluded from the logfile. The default value
                (``logging.NOTSET``) will cause all logging to be captured.
            check_outputs (bool): Defaults to False. If True, will check that
                the expected outputs and no others were created based on the
                given args and the ``created_if`` attribute of each output. An
                error will be raised if a discrepancy is found.

        Returns:
            file registry dictionary

        Raises:
            RuntimeError if ``check_outputs`` is ``True`` and a discrepancy is
            detected between actual and expected outputs
        """
        if create_logfile:
            cm = utils.prepare_workspace(args['workspace_dir'],
                                         model_id=self.model_id,
                                         logging_level=log_level)
        else: # null context manager, has no effect
            cm = contextlib.nullcontext()

        with GDALUseExceptions(), cm:

            LOGGER.log(
                100,  # define high log level so it should always show in logs
                'Starting model with parameters: \n' +
                utils.format_args_dict(args, self.model_id))

            model_module = importlib.import_module(self.module_name)
            registry = model_module.execute(args)

            preprocessed_args = self.preprocess_inputs(args)

            if check_outputs:
                # evaluate which outputs we expect to be created, given the
                # model spec and provided input values
                outputs_to_be_created = set([
                    output.id for output in self.outputs if bool(
                        utils.evaluate_expression(
                            expression=f'{output.created_if}',
                            variable_map=preprocessed_args
                        )
                    ) is True
                ])
                if outputs_to_be_created != set(registry.keys()):
                    raise RuntimeError(
                        'The set of outputs created differs from what was expected.\n'
                        f'Missing outputs: {outputs_to_be_created - set(registry.keys())}\n'
                        f'Extra outputs: {set(registry.keys()) - outputs_to_be_created}')

            # optionally create metadata files for the results
            if generate_metadata:
                LOGGER.info('Generating metadata for results')
                try:
                    # If there's an exception from creating metadata
                    # I don't think we want to indicate a model failure
                    self.generate_metadata_for_outputs(preprocessed_args)
                except Exception as exc:
                    LOGGER.warning(
                        'Something went wrong while generating metadata', exc_info=exc)

            # optionally write the file registry dict to a JSON file in the workspace
            if save_file_registry:
                file_registry_path = os.path.join(
                    preprocessed_args['workspace_dir'],
                    f'file_registry{preprocessed_args["results_suffix"]}.json')
                with open(file_registry_path, 'w') as json_file:
                    json.dump(registry, json_file, indent=4)

            return registry



# Specs for common arg types ##################################################
WORKSPACE = DirectoryInput(
    id="workspace_dir",
    name="workspace",
    about=(
        "The folder where all the model's output files will be written."
        " If this folder does not exist, it will be created. If data"
        " already exists in the folder, it will be overwritten."
    ),
    contents=[],
    permissions="rwx",
    must_exist=False,
)
SUFFIX = ResultsSuffixInput(
    id="results_suffix",
    name=gettext("file suffix"),
    about=gettext(
        "Suffix that will be appended to all output file names. Useful to"
        " differentiate between model runs."
    ),
    required=False,
    regexp="[a-zA-Z0-9_-]*"
)
N_WORKERS = NWorkersInput(
    id="n_workers",
    name=gettext("taskgraph n_workers parameter"),
    about=gettext(
        "The n_workers parameter to provide to taskgraph. -1 will cause all jobs"
        " to run synchronously. 0 will run all jobs in the same process, but"
        " scheduling will take place asynchronously. Any other positive integer"
        " will cause that many processes to be spawned to execute tasks."
    ),
    required=False,
    hidden=True,
    units=u.none,
    expression="value >= -1"
)
DEM = SingleBandRasterInput(
    id="dem_path",
    name=gettext("digital elevation model"),
    about=gettext("Map of elevation above sea level."),
    data_type=float,
    units=u.meter
)
PROJECTED_DEM = DEM.model_copy(update=dict(projected=True))
THRESHOLD_FLOW_ACCUMULATION = NumberInput(
    id="threshold_flow_accumulation",
    name=gettext("threshold flow accumulation"),
    about=gettext(
        "The number of upslope pixels that must flow into a pixel before it is"
        " classified as a stream."
    ),
    units=u.pixel,
    expression="value >= 0"
)
SOIL_GROUP = SingleBandRasterInput(
    id="soil_group_path",
    name=gettext("soil hydrologic group"),
    about=gettext(
        "Map of soil hydrologic groups. Pixels may have values 1, 2, 3, or 4,"
        " corresponding to soil hydrologic groups A, B, C, or D, respectively."
    ),
    data_type=int,
    units=None
)
LULC_TABLE_COLUMN = IntegerInput(
    id="lucode",
    about=gettext(
        "LULC codes from the LULC raster. Each code must be a unique"
        " integer."
    )
)
AOI = VectorInput(
    id="aoi_path",
    name=gettext("area of interest"),
    about=gettext(
        "A map of areas over which to aggregate and summarize the final results."
    ),
    geometry_types={"POLYGON", "MULTIPOLYGON"},
    fields=[]
)
LULC = SingleBandRasterInput(
    id="lulc_bas_path",
    name=gettext("baseline LULC"),
    about=gettext(
        "A map of LULC for the baseline scenario, which must occur prior to the"
        " alternate scenario. All values in this raster must have corresponding"
        " entries in the Carbon Pools table."
    ),
    data_type=int,
    units=None
)
FLOW_DIR_ALGORITHM = OptionStringInput(
    id="flow_dir_algorithm",
    name=gettext("flow direction algorithm"),
    about=gettext("Flow direction algorithm to use."),
    options=[
        Option(key="D8", description="D8 flow direction"),
        Option(key="MFD", description="Multiple flow direction")
    ]
)

# Specs for common outputs ####################################################
TASKGRAPH_CACHE = FileOutput(
    id="taskgraph_cache",
    path="taskgraph_cache/taskgraph.db",
    about=gettext(
        "Cache that stores data between model runs. This directory contains no"
        " human-readable data and you may ignore it."
    )
)
FILLED_DEM = SingleBandRasterOutput(
    id='filled_dem',
    path='filled_dem.tif',
    about=gettext("Map of elevation after any pits are filled"),
    data_type=float,
    units=u.meter
)
FLOW_ACCUMULATION = SingleBandRasterOutput(
    id='flow_accumulation',
    path='flow_accumulation.tif',
    about=gettext("Map of flow accumulation"),
    data_type=float,
    units=u.none
)
FLOW_DIRECTION = SingleBandRasterOutput(
    id='flow_direction',
    path='flow_direction.tif',
    about=gettext(
        "MFD flow direction. Note: the pixel values should not be interpreted"
        " directly. Each 32-bit number consists of 8 4-bit numbers. Each 4-bit"
        " number represents the proportion of flow into one of the eight"
        " neighboring pixels."
    ),
    data_type=int,
    units=None
)
SLOPE = SingleBandRasterOutput(
    id="slope",
    path='slope.tif',
    about=gettext(
        "Percent slope, calculated from the pit-filled DEM. 100 is equivalent to"
        " a 45 degree slope."
    ),
    data_type=float,
    units=None
)
STREAM = SingleBandRasterOutput(
    id='stream',
    path='stream.tif',
    about=gettext(
        "Stream network, created using flow direction and flow accumulation"
        " derived from the DEM and Threshold Flow Accumulation. Values of 1"
        " represent streams, values of 0 are non-stream pixels."
    ),
    data_type=int,
    units=None
)

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


def format_type_string(arg_type):
    """Represent an arg type as a user-friendly string.

    Args:
        arg_type (str|set(str)): the type to format. May be a single type or a
            set of types.

    Returns:
        formatted string that links to a description of the input type(s)
    """
    if arg_type is RasterOrVectorInput:
        return (
            f'`{SingleBandRasterInput.display_name} <{INPUT_TYPES_HTML_FILE}#{SingleBandRasterInput.rst_section}>`__ or '
            f'`{VectorInput.display_name} <{INPUT_TYPES_HTML_FILE}#{VectorInput.rst_section}>`__')
    return f'`{arg_type.display_name} <{INPUT_TYPES_HTML_FILE}#{arg_type.rst_section}>`__'


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
        in_parentheses.append((spec.format_geometry_types_rst()))

    # Represent the required state as a string, defaulting to required
    # It doesn't make sense to include this for boolean checkboxes
    if type(spec) is not BooleanInput:
        required_string = spec.format_required_string()
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
            indented_block.append(gettext(
                'Values must be one of the following text strings:'))
            indented_block += spec.format_rst()

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

    # anchor names cannot contain underscores. sphinx will replace them
    # automatically, but lets explicitly replace them here
    anchor_name = '-'.join(arg_keys).replace('_', '-')

    # start with the spec for all args
    # narrow down to the nested spec indicated by the sequence of arg keys
    spec = module.MODEL_SPEC.get_input(arg_keys[0])
    arg_keys = arg_keys[1:]
    for i, key in enumerate(arg_keys):
        # convert raster band numbers to ints
        if i > 0 and arg_keys[i - 1] == 'bands':
            key = int(key)
        elif i > 0 and arg_keys[i - 1] == 'fields':
            spec = spec.get_field(key)
        elif i > 0 and arg_keys[i - 1] == 'contents':
            spec = spec.get_contents(key)
        elif i > 0 and arg_keys[i - 1] == 'columns':
            spec = spec.get_column(key)
        elif i > 0 and arg_keys[i - 1] == 'rows':
            spec = spec.get_row(key)
        elif key in {'bands', 'fields', 'contents', 'columns', 'rows'}:
            continue
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
        arg_name = spec.capitalize_name()
    else:
        arg_name = arg_keys[-1]

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
    try:
        resource = geometamaker.describe(datasource_path, compute_stats=True)
    except ValueError as e:
        # Don't want function to fail bc can't create metadata due to invalid filetype
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
        # field names in attr_spec are always lowercase, but the
        # actual fieldname in the data could be any case because
        # invest does not require case-sensitive fieldnames
        field_lookup = {
            field.name.lower(): field for field in resource._get_fields()}
        for nested_spec in attr_specs:
            try:
                field_metadata = field_lookup[nested_spec.id]
                # Field description only gets set if its empty, i.e. ''
                if len(field_metadata.description.strip()) < 1:
                    resource.set_field_description(
                        field_metadata.name, description=nested_spec.about)
                # units only get set if empty
                if len(field_metadata.units.strip()) < 1:
                    units = format_unit(nested_spec.units) if hasattr(
                        nested_spec, 'units') else ''
                    resource.set_field_description(
                        field_metadata.name, units=units)
            except KeyError as error:
                # fields that are in the spec but missing
                # from model results because they are conditional.
                LOGGER.debug(error)
    if isinstance(spec, SingleBandRasterInput) or isinstance(
            spec, SingleBandRasterOutput):
        if len(resource.get_band_description(1).units) < 1:
            units = format_unit(spec.units)
            resource.set_band_description(1, units=units)

    resource.write(workspace=out_workspace)
