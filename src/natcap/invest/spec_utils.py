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

from natcap.invest import utils
from natcap.invest.validation import get_message, _evaluate_expression
from . import gettext
from .unit_registry import u


LOGGER = logging.getLogger(__name__)

# accessing a file could take a long time if it's in a file streaming service
# to prevent the UI from hanging due to slow validation,
# set a timeout for these functions.
def timeout(func, *args, timeout=5, **kwargs):
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

    def wrapper_func():
        message_queue.put(func(*args, **kwargs))

    thread = threading.Thread(target=wrapper_func)
    LOGGER.debug(f'Starting file checking thread with timeout={timeout}')
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        # first arg to `check_csv`, `check_raster`, `check_vector` is the path
        warnings.warn(
            f'Validation of file {args[0]} timed out. If this file '
            'is stored in a file streaming service, it may be taking a long '
            'time to download. Try storing it locally instead.')
        return None

    else:
        LOGGER.debug('File checking thread completed.')
        # get any warning messages returned from the thread
        a = message_queue.get()
        return a

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
    return None

def get_headers_to_validate(specs):
    """Get header names to validate from a row/column/field spec dictionary.

    This module only validates row/column/field names that are static and
    always required. If `'required'` is anything besides `True`, or if the name
    contains brackets indicating it's user-defined, it is not returned.

    Args:
        specs (dict): a row/column/field spec dictionary that maps row/column/
            field names to specs for them

    Returns:
        list of expected header names to validate against
    """
    headers = []
    for spec in specs:
        # if 'required' isn't a key, it defaults to True
        if spec.required is True:
            # brackets are a special character for our args spec syntax
            # they surround the part of the key that's user-defined
            # user-defined rows/columns/fields are not validated here, so skip
            if '[' not in spec.id:
                headers.append(spec.id)
    return headers


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

    return None


class IterableWithDotAccess():
    def __init__(self, *args):
        self.args = args
        self.inputs_dict = {i.id: i for i in args}
        self.iter_index = 0

    # def __getattr__(self, key):
    #     return self.inputs_dict.get(key)

    def __iter__(self):
        return iter(self.args)

    def get(self, key):
        return self.inputs_dict[key]

    def to_json(self):
        return self.inputs_dict

    # def __next__(self):
    #     print('next')
    #     if self.iter_index < len(self.args):
    #         result = self.args[self.iter_index]
    #         self.iter_index += 1
    #         return result
    #     else:
    #         raise StopIteration

class Rows(IterableWithDotAccess):
    pass

class Columns(IterableWithDotAccess):
    pass

class Fields(IterableWithDotAccess):
    pass

class Contents(IterableWithDotAccess):
    pass

@dataclasses.dataclass
class Input:
    id: str = ''
    name: str = ''
    about: str = ''
    required: typing.Union[bool, str] = True
    allowed: typing.Union[bool, str] = True

@dataclasses.dataclass
class Output:
    id: str = ''
    about: str = ''
    created_if: typing.Union[bool, str] = True

@dataclasses.dataclass
class FileInput(Input):
    permissions: str = 'r'
    type: typing.ClassVar[str] = 'file'

    # @timeout
    def validate(self, filepath):
        """Validate a single file.

        Args:
            filepath (string): The filepath to validate.
            permissions='r' (string): A string that includes the lowercase
                characters ``r``, ``w`` and/or ``x``, indicating read, write, and
                execute permissions (respectively) required for this file.

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
    def format_column(col, base_path):
        return col.apply(
            lambda p: p if pandas.isna(p) else utils.expand_path(str(p).strip(), base_path)
        ).astype(pandas.StringDtype())

@dataclasses.dataclass
class SingleBandRasterInput(FileInput):
    band: typing.Union[Input, None] = None
    projected: typing.Union[bool, None] = None
    projection_units: typing.Union[pint.Unit, None] = None
    type: typing.ClassVar[str] = 'raster'

    # @timeout
    def validate(self, filepath):
        """Validate a GDAL Raster on disk.

        Args:
            filepath (string): The path to the raster on disk.  The file must exist
                and be readable.
            projected=False (bool): Whether the spatial reference must be projected
                in linear units.
            projection_units=None (pint.Units): The required linear units of the
                projection. If ``None``, the projection units will not be checked.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.

        """
        file_warning = FileInput.validate(self, filepath)
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

        return None

@dataclasses.dataclass
class VectorInput(FileInput):
    geometries: set = dataclasses.field(default_factory=dict)
    fields: typing.Union[Fields, None] = None
    projected: typing.Union[bool, None] = None
    projection_units: typing.Union[pint.Unit, None] = None
    type: typing.ClassVar[str] = 'vector'

    # @timeout
    def validate(self, filepath):
        """Validate a GDAL vector on disk.

        Note:
            If the provided vector has multiple layers, only the first layer will
            be checked.

        Args:
            filepath (string): The path to the vector on disk.  The file must exist
                and be readable.
            geometries (set): Set of geometry type(s) that are allowed. Options are
                'POINT', 'LINESTRING', 'POLYGON', 'MULTIPOINT', 'MULTILINESTRING',
                and 'MULTIPOLYGON'.
            fields=None (dict): A dictionary spec of field names that the vector is
                expected to have. See the docstring of ``check_headers`` for
                details on validation rules.
            projected=False (bool): Whether the spatial reference must be projected
                in linear units.  If None, the projection will not be checked.
            projection_units=None (pint.Units): The required linear units of the
                projection. If ``None``, the projection units will not be checked.

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
        for geom in self.geometries:
            allowed_geom_types += geom_map[geom]

        # NOTE: this only checks the layer geometry type, not the types of the
        # actual geometries (layer.GetGeometryTypes()). This is probably equivalent
        # in most cases, and it's more efficient than checking every geometry, but
        # we might need to change this in the future if it becomes a problem.
        # Currently not supporting ogr.wkbUnknown which allows mixed types.
        layer = gdal_dataset.GetLayer()
        if layer.GetGeomType() not in allowed_geom_types:
            return get_message('WRONG_GEOM_TYPE').format(allowed=self.geometries)

        if self.fields:
            field_patterns = get_headers_to_validate(self.fields)
            fieldnames = [defn.GetName() for defn in layer.schema]
            required_field_warning = check_headers(
                field_patterns, fieldnames, 'field')
            if required_field_warning:
                return required_field_warning

        srs = layer.GetSpatialRef()
        projection_warning = _check_projection(srs, self.projected, self.projection_units)
        return projection_warning



@dataclasses.dataclass
class RasterOrVectorInput(SingleBandRasterInput, VectorInput):
    band: typing.Union[Input, None] = None
    geometries: set = dataclasses.field(default_factory=dict)
    fields: typing.Union[Fields, None] = None
    projected: typing.Union[bool, None] = None
    projection_units: typing.Union[pint.Unit, None] = None
    type: typing.ClassVar[str] = 'raster_or_vector'

    # @timeout
    def validate(self, filepath):
        """Validate an input that may be a raster or vector.

        Args:
            filepath (string):  The path to the raster or vector.
            **kwargs: kwargs of the raster and vector spec. Will be
                passed to ``check_raster`` or ``check_vector``.

        Returns:
            A string error message if an error was found. ``None`` otherwise.
        """
        try:
            gis_type = pygeoprocessing.get_gis_type(filepath)
        except ValueError as err:
            return str(err)
        if gis_type == pygeoprocessing.RASTER_TYPE:
            return SingleBandRasterInput.validate(self, filepath)
        else:
            return VectorInput.validate(self, filepath)

@dataclasses.dataclass
class CSVInput(FileInput):
    columns: typing.Union[Columns, None] = None
    rows: typing.Union[Rows, None] = None
    index_col: typing.Union[str, None] = None
    type: typing.ClassVar[str] = 'csv'

    # @timeout
    def validate(self, filepath):
        """Validate a table.

        Args:
            filepath (string): The string filepath to the table.

        Returns:
            A string error message if an error was found. ``None`` otherwise.

        """
        file_warning = super().validate(filepath)
        if file_warning:
            return file_warning
        if self.columns or self.rows:
            try:
                self.get_validated_dataframe(filepath)
            except Exception as e:
                return str(e)

    def get_validated_dataframe(self, csv_path, read_csv_kwargs={}):
        """Read a CSV into a dataframe that is guaranteed to match the spec."""
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
    contents: typing.Union[Contents, None] = None
    permissions: str = ''
    must_exist: bool = True
    type: typing.ClassVar[str] = 'directory'

    # @timeout
    def validate(self, dirpath):
        """Validate a directory.

        Args:
            dirpath (string): The directory path to validate.
            must_exist=True (bool): If ``True``, the directory at ``dirpath``
                must already exist on the filesystem.
            permissions='rx' (string): A string that includes the lowercase
                characters ``r``, ``w`` and/or ``x``, indicating read, write, and
                execute permissions (respectively) required for this directory.

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
    units: typing.Union[pint.Unit, None] = None
    expression: typing.Union[str, None] = None
    type: typing.ClassVar[str] = 'number'

    def validate(self, value):
        """Validate numbers.

        Args:
            value: A python value. This should be able to be cast to a float.
            expression=None (string): A string expression to be evaluated with the
                intent of determining that the value is within a specific range.
                The expression must contain the string ``value``, which will
                represent the user-provided value (after it has been cast to a
                float).  Example expression: ``"(value >= 0) & (value <= 1)"``.

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
        return col.astype(float)

@dataclasses.dataclass
class IntegerInput(Input):
    type: typing.ClassVar[str] = 'integer'

    def validate(self, value):
        """Validate an integer.

        Args:
            value: A python value. This should be able to be cast to an int.

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
        return None

    @staticmethod
    def format_column(col, *args):
        return col.astype(pandas.Int64Dtype())


@dataclasses.dataclass
class RatioInput(Input):
    type: typing.ClassVar[str] = 'ratio'

    def validate(self, value):
        """Validate a ratio (a proportion expressed as a value from 0 to 1).

        Args:
            value: A python value. This should be able to be cast to a float.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.

        """
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return get_message('NOT_A_NUMBER').format(value=value)

        if as_float < 0 or as_float > 1:
            return get_message('NOT_WITHIN_RANGE').format(
                value=as_float,
                range='[0, 1]')

    @staticmethod
    def format_column(col, *args):
        return col.astype(float)

@dataclasses.dataclass
class PercentInput(Input):
    type: typing.ClassVar[str] = 'percent'

    def validate(self, value):
        """Validate a percent (a proportion expressed as a value from 0 to 100).

        Args:
            value: A python value. This should be able to be cast to a float.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return get_message('NOT_A_NUMBER').format(value=value)

        if as_float < 0 or as_float > 100:
            return get_message('NOT_WITHIN_RANGE').format(
                value=as_float,
                range='[0, 100]')

    @staticmethod
    def format_column(col, *args):
        return col.astype(float)

@dataclasses.dataclass
class BooleanInput(Input):
    type: typing.ClassVar[str] = 'boolean'

    def validate(self, value):
        """Validate a boolean value.

        If the value provided is not a python boolean, an error message is
        returned.

        Args:
            value: The value to evaluate.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        if not isinstance(value, bool):
            return get_message('NOT_BOOLEAN').format(value=value)

    @staticmethod
    def format_column(col, *args):
        return col.astype('boolean')

@dataclasses.dataclass
class StringInput(Input):
    regexp: typing.Union[str, None] = None
    type: typing.ClassVar[str] = 'string'

    def validate(self, value):
        """Validate an arbitrary string.

        Args:
            value: The value to check.  Must be able to be cast to a string.
            regexp=None (string): a string interpreted as a regular expression.

        Returns:
            A string error message if an error was found.  ``None`` otherwise.
        """
        if self.regexp:
            matches = re.fullmatch(self.regexp, str(value))
            if not matches:
                return get_message('REGEXP_MISMATCH').format(regexp=self.regexp)
        return None

    @staticmethod
    def format_column(col, *args):
        return col.apply(
            lambda s: s if pandas.isna(s) else str(s).strip().lower()
        ).astype(pandas.StringDtype())

@dataclasses.dataclass
class OptionStringInput(Input):
    options: typing.Union[list, None] = None
    type: typing.ClassVar[str] = 'option_string'

    def validate(self, value):
        """Validate that a string is in a set of options.

        Args:
            value: The value to test. Will be cast to a string before comparing
                against the allowed options.
            options (dict): option spec to validate against.

        Returns:
            A string error message if ``value`` is not in ``options``.  ``None``
            otherwise.

        """
        # if options is empty, that means it's dynamically populated
        # so validation should be left to the model's validate function.
        if self.options and str(value) not in self.options:
            return get_message('INVALID_OPTION').format(option_list=sorted(self.options))

    @staticmethod
    def format_column(col, *args):
        return col.apply(
            lambda s: s if pandas.isna(s) else str(s).strip().lower()
        ).astype(pandas.StringDtype())

@dataclasses.dataclass
class OtherInput(Input):
    def validate(self, value):
        pass

@dataclasses.dataclass
class SingleBandRasterOutput(Output):
    band: typing.Union[Input, None] = None
    projected: typing.Union[bool, None] = None
    projection_units: typing.Union[pint.Unit, None] = None

@dataclasses.dataclass
class VectorOutput(Output):
    geometries: set = dataclasses.field(default_factory=dict)
    fields: typing.Union[Fields, None] = None
    projected: typing.Union[bool, None] = None
    projection_units: typing.Union[pint.Unit, None] = None

@dataclasses.dataclass
class CSVOutput(Output):
    columns: typing.Union[Columns, None] = None
    rows: typing.Union[Rows, None] = None
    index_col: typing.Union[str, None] = None

@dataclasses.dataclass
class DirectoryOutput(Output):
    contents: typing.Union[Contents, None] = None
    permissions: str = ''
    must_exist: bool = True

@dataclasses.dataclass
class FileOutput(Output):
    pass

@dataclasses.dataclass
class NumberOutput(Output):
    units: typing.Union[pint.Unit, None] = None
    expression: typing.Union[str, None] = None

@dataclasses.dataclass
class IntegerOutput(Output):
    pass

@dataclasses.dataclass
class RatioOutput(Output):
    pass

@dataclasses.dataclass
class PercentOutput(Output):
    pass

@dataclasses.dataclass
class StringOutput(Output):
    regexp: typing.Union[str, None] = None

@dataclasses.dataclass
class OptionStringOutput(Output):
    options: typing.Union[list, None] = None

@dataclasses.dataclass
class UISpec:
    order: typing.Union[list, None] = None
    hidden: list = None
    dropdown_functions: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelSpec:
    model_id: str
    model_title: str
    userguide: str
    ui_spec: UISpec
    inputs: typing.Iterable[Input]
    outputs: typing.Iterable[Output]
    args_with_spatial_overlap: dict
    aliases: set = dataclasses.field(default_factory=set)

    def __post_init__(self):
        self.inputs_dict = {_input.id: _input for _input in self.inputs}
        self.outputs_dict = {_output.id: _output for _output in self.outputs}

    def get_input(self, key):
        return self.inputs_dict[key]


def build_model_spec(model_spec):
    inputs = [
        build_input_spec(argkey, argspec)
        for argkey, argspec in model_spec['args'].items()]
    outputs = [
        build_output_spec(argkey, argspec) for argkey, argspec in model_spec['outputs'].items()]
    ui_spec = UISpec(
        order=model_spec['ui_spec']['order'],
        hidden=model_spec['ui_spec'].get('hidden', None),
        dropdown_functions=model_spec['ui_spec'].get('dropdown_functions', None))
    return ModelSpec(
        model_id=model_spec['model_id'],
        model_title=model_spec['model_title'],
        userguide=model_spec['userguide'],
        aliases=model_spec['aliases'],
        ui_spec=ui_spec,
        inputs=inputs,
        outputs=outputs,
        args_with_spatial_overlap=model_spec.get('args_with_spatial_overlap', None))


def build_input_spec(argkey, arg):
    base_attrs = {
        'id': argkey,
        'name': arg.get('name', None),
        'about': arg.get('about', None),
        'required': arg.get('required', True),
        'allowed': arg.get('allowed', True)
    }

    t = arg['type']

    if t == 'option_string':
        return OptionStringInput(
            **base_attrs,
            options=arg['options'])

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
            band=build_input_spec('1', arg['bands'][1]),
            projected=arg.get('projected', None),
            projection_units=arg.get('projection_units', None))

    elif t == 'vector':
        return VectorInput(
            **base_attrs,
            geometries=arg['geometries'],
            fields=Fields(
                *[build_input_spec(key, field_spec) for key, field_spec in arg['fields'].items()]
            ),
            projected=arg.get('projected', None),
            projection_units=arg.get('projection_units', None))

    elif t == 'csv':
        columns = None
        rows = None
        if 'columns' in arg:
            columns = Columns(*[
                build_input_spec(col_name, col_spec)
                for col_name, col_spec in arg['columns'].items()])
        elif 'rows' in arg:
            rows = Rows(*[
                build_input_spec(row_name, row_spec)
                for row_name, row_spec in arg['rows'].items()])

        return CSVInput(
            **base_attrs,
            columns=columns,
            rows=rows,
            index_col=arg.get('index_col', None))

    elif t == 'directory':
        return DirectoryInput(
            contents=Contents(*[
                build_input_spec(k, v) for k, v in arg['contents'].items()]),
            permissions=arg.get('permissions', 'rx'),
            must_exist=arg.get('must_exist', None),
            **base_attrs)

    elif t == 'file':
        return FileInput(**base_attrs)

    elif t == {'raster', 'vector'}:
        return RasterOrVectorInput(
            **base_attrs,
            geometries=arg['geometries'],
            fields=Fields(*[
                build_input_spec(key, field_spec) for key, field_spec in arg['fields'].items()]),
            band=build_input_spec('1', arg['bands'][1]),
            projected=arg.get('projected', None),
            projection_units=arg.get('projection_units', None))

    else:
        raise ValueError


def build_output_spec(key, spec):
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
            units=spec['units'],
            expression=None)

    elif t == 'integer':
        return IntegerOutput(**base_attrs)

    elif t == 'ratio':
        return RatioOutput(**base_attrs)

    elif t == 'percent':
        return PercentOutput(**base_attrs)

    elif t == 'raster':
        return SingleBandRasterOutput(
            **base_attrs,
            band=build_output_spec(1, spec['bands'][1]),
            projected=None,
            projection_units=None)

    elif t == 'vector':
        return VectorOutput(
            **base_attrs,
            geometries=spec['geometries'],
            fields=Fields(*[
                build_output_spec(key, field_spec) for key, field_spec in spec['fields'].items()]),
            projected=None,
            projection_units=None)

    elif t == 'csv':
        return CSVOutput(
            **base_attrs,
            columns=Columns(*[
                build_output_spec(key, col_spec) for key, col_spec in spec['columns'].items()]),
            rows=None,
            index_col=spec.get('index_col', None))

    elif t == 'directory':
        return DirectoryOutput(
            contents=Contents(*[
                build_output_spec(k, v) for k, v in spec['contents'].items()]),
            permissions=None,
            must_exist=None,
            **base_attrs)

    elif t == 'freestyle_string':
        return StringOutput(
            **base_attrs,
            regexp=spec.get('regexp', None))

    elif t == 'option_string':
        return OptionStringOutput(
            **base_attrs,
            options=spec['options'])

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
    "expression": "value >= -1"
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


def serialize_args_spec(spec):
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
        # Sets are present in 'geometries' attributes of some args
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
        raise TypeError(f'fallback serializer is missing for {type(obj)}')

    spec_dict = json.loads(json.dumps(spec, default=fallback_serializer, ensure_ascii=False))
    spec_dict['args'] = spec_dict.pop('inputs')
    return json.dumps(spec_dict, ensure_ascii=False)


# accepted geometries for a vector will be displayed in this order
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


def format_geometries_string(geometries):
    """Represent a set of allowed vector geometries as user-friendly text.

    Args:
        geometries (set(str)): set of geometry names

    Returns:
        string
    """
    # sort the geometries so they always display in a consistent order
    sorted_geoms = sorted(
        geometries,
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
        vector fields and geometries, option_string options, etc.
    """
    type_string = format_type_string(spec.__class__)
    in_parentheses = [type_string]

    # For numbers and rasters that have units, display the units
    units = None
    if spec.__class__ is NumberInput:
        units = spec.units
    elif spec.__class__ is SingleBandRasterInput and spec.band.__class__ is NumberInput:
        units = spec.band.units
    if units:
        units_string = format_unit(units)
        if units_string:
            # pybabel can't find the message if it's in the f-string
            translated_units = gettext("units")
            in_parentheses.append(f'{translated_units}: **{units_string}**')

    if spec.__class__ is VectorInput:
        in_parentheses.append(format_geometries_string(spec.geometries))

    # Represent the required state as a string, defaulting to required
    # It doesn't make sense to include this for boolean checkboxes
    if spec.__class__ is not BooleanInput:
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
    if spec.__class__ is OptionStringInput:
        # may be either a dict or set. if it's empty, the options are
        # dynamically generated. don't try to document them.
        if spec.options:
            if isinstance(spec.options, dict):
                indented_block.append(gettext('Options:'))
                indented_block += format_options_string_from_dict(spec.options)
            else:
                formatted_options = format_options_string_from_list(spec.options)
                indented_block.append(gettext('Options:') + f' {formatted_options}')

    elif spec.__class__ is CSVInput:
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


def write_metadata_file(datasource_path, spec, lineage_statement, keywords_list):
    """Write a metadata sidecar file for an invest output dataset.

    Args:
        datasource_path (str) - filepath to the invest output
        spec (dict) -  the invest specification for ``datasource_path``
        lineage_statement (str) - string to describe origin of the dataset.
        keywords_list (list) - sequence of strings

    Returns:
        None

    """
    resource = geometamaker.describe(datasource_path)
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
            units = format_unit(nested_spec.units) if hasattr(nested_spec, 'units') else ''
            try:
                resource.set_field_description(
                    nested_spec.id, description=nested_spec.about, units=units)
            except KeyError as error:
                # fields that are in the spec but missing
                # from model results because they are conditional.
                LOGGER.debug(error)
    if hasattr(spec, 'band'):
        units = format_unit(spec.band.units)
        resource.set_band_description(1, units=units)

    resource.write()


def generate_metadata(model_module, args_dict):
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
            if spec_data.__class__ is DirectoryOutput:
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
                            full_path, spec_data, lineage_statement, keywords)
                    except ValueError as error:
                        # Some unsupported file formats, e.g. html
                        LOGGER.debug(error)

    _walk_spec(model_module.MODEL_SPEC.outputs, args_dict['workspace_dir'])
