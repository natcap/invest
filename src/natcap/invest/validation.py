"""Common validation utilities for InVEST models."""
import ast
import copy
import functools
import importlib
import inspect
import logging
import os
import pprint
import queue
import re
import threading
import warnings

import numpy
import pandas
import pint
import pygeoprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from . import gettext
from . import spec_utils
from . import utils

#: A flag to pass to the validation context manager indicating that all keys
#: should be checked.
CHECK_ALL_KEYS = None
LOGGER = logging.getLogger(__name__)

MESSAGES = {
    'MISSING_KEY': gettext('Key is missing from the args dict'),
    'MISSING_VALUE': gettext('Input is required but has no value'),
    'MATCHED_NO_HEADERS': gettext(
        'Expected the {header} "{header_name}" but did not find it'),
    'PATTERN_MATCHED_NONE': gettext(
        'Expected to find at least one {header} matching '
        'the pattern "{header_name}" but found none'),
    'DUPLICATE_HEADER': gettext(
        'Expected the {header} "{header_name}" only once '
        'but found it {number} times'),
    'NOT_A_NUMBER': gettext(
        'Value "{value}" could not be interpreted as a number'),
    'WRONG_PROJECTION_UNIT': gettext(
        'Layer must be projected in this unit: '
        '"{unit_a}" but found this unit: "{unit_b}"'),
    'UNEXPECTED_ERROR': gettext('An unexpected error occurred in validation'),
    'DIR_NOT_FOUND': gettext('Directory not found'),
    'NOT_A_DIR': gettext('Path must be a directory'),
    'FILE_NOT_FOUND': gettext('File not found'),
    'INVALID_PROJECTION': gettext('Dataset must have a valid projection.'),
    'NOT_PROJECTED': gettext('Dataset must be projected in linear units.'),
    'NOT_GDAL_RASTER': gettext('File could not be opened as a GDAL raster'),
    'OVR_FILE': gettext('File found to be an overview ".ovr" file.'),
    'NOT_GDAL_VECTOR': gettext('File could not be opened as a GDAL vector'),
    'REGEXP_MISMATCH': gettext(
        "Value did not match expected pattern {regexp}"),
    'INVALID_OPTION': gettext("Value must be one of: {option_list}"),
    'INVALID_VALUE': gettext('Value does not meet condition {condition}'),
    'NOT_WITHIN_RANGE': gettext('Value {value} is not in the range {range}'),
    'NOT_AN_INTEGER': gettext('Value "{value}" does not represent an integer'),
    'NOT_BOOLEAN': gettext("Value must be either True or False, not {value}"),
    'NO_PROJECTION': gettext('Spatial file {filepath} has no projection'),
    'BBOX_NOT_INTERSECT': gettext(
        'Not all of the spatial layers overlap each '
        'other. All bounding boxes must intersect: {bboxes}'),
    'NEED_PERMISSION_DIRECTORY': gettext(
        'You must have {permission} access to this directory'),
    'NEED_PERMISSION_FILE': gettext(
        'You must have {permission} access to this file'),
    'WRONG_GEOM_TYPE': gettext('Geometry type must be one of {allowed}')
}


def _evaluate_expression(expression, variable_map):
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


def get_invalid_keys(validation_warnings):
    """Get the invalid keys from a validation warnings list.

    Args:
        validation_warnings (list): A list of two-tuples where the first
            item is an iterable of string args keys affected and the second
            item is a string error message.

    Returns:
        A set of the string args keys found across all of the first elements in
        the validation tuples.

    """
    invalid_keys = set([])
    for affected_keys, error_msg in validation_warnings:
        for key in affected_keys:
            invalid_keys.add(key)
    return invalid_keys


def get_sufficient_keys(args):
    """Determine which keys in args are sufficient.

    A sufficient key is one that is:

        1. Present within ``args``
        2. Does not have a value of ``''`` or ``None``.

    Args:
        args (dict): An args dict of string keys to serializeable values.

    Returns:
        A set of keys from ``args`` that are sufficient.
    """
    sufficient_keys = set()
    for key, value in args.items():
        if value not in ('', None):
            sufficient_keys.add(key)

    return sufficient_keys


def check_directory(dirpath, must_exist=True, permissions='rx', **kwargs):
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
    if must_exist:
        if not os.path.exists(dirpath):
            return MESSAGES['DIR_NOT_FOUND']

    if os.path.exists(dirpath):
        if not os.path.isdir(dirpath):
            return MESSAGES['NOT_A_DIR']
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

    if 'r' in permissions:
        try:
            os.scandir(dirpath).close()
        except OSError:
            return MESSAGES[MESSAGE_KEY].format(permission='read')

    # Check for x access before checking for w,
    # since w operations to a dir are dependent on x access
    if 'x' in permissions:
        try:
            cwd = os.getcwd()
            os.chdir(dirpath)
        except OSError:
            return MESSAGES[MESSAGE_KEY].format(permission='execute')
        finally:
            os.chdir(cwd)

    if 'w' in permissions:
        try:
            temp_path = os.path.join(dirpath, 'temp__workspace_validation.txt')
            with open(temp_path, 'w') as temp:
                temp.close()
                os.remove(temp_path)
        except OSError:
            return MESSAGES[MESSAGE_KEY].format(permission='write')


def check_file(filepath, permissions='r', **kwargs):
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
        return MESSAGES['FILE_NOT_FOUND']

    for letter, mode, descriptor in (
            ('r', os.R_OK, 'read'),
            ('w', os.W_OK, 'write'),
            ('x', os.X_OK, 'execute')):
        if letter in permissions and not os.access(filepath, mode):
            return MESSAGES['NEED_PERMISSION_FILE'].format(permission=descriptor)


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
        return MESSAGES['INVALID_PROJECTION']

    if projected:
        if not srs.IsProjected():
            return MESSAGES['NOT_PROJECTED']

    if projection_units:
        # pint uses underscores in multi-word units e.g. 'survey_foot'
        # it is case-sensitive
        layer_units_name = srs.GetLinearUnitsName().lower().replace(' ', '_')
        try:
            # this will parse common synonyms: m, meter, meters, metre, metres
            layer_units = spec_utils.u.Unit(layer_units_name)
            # Compare pint Unit objects
            if projection_units != layer_units:
                return MESSAGES['WRONG_PROJECTION_UNIT'].format(
                    unit_a=projection_units, unit_b=layer_units_name)
        except pint.errors.UndefinedUnitError:
            return MESSAGES['WRONG_PROJECTION_UNIT'].format(
                unit_a=projection_units, unit_b=layer_units_name)

    return None


def check_raster(filepath, projected=False, projection_units=None, **kwargs):
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
    file_warning = check_file(filepath, permissions='r')
    if file_warning:
        return file_warning

    try:
        gdal_dataset = gdal.OpenEx(filepath, gdal.OF_RASTER)
    except RuntimeError:
        return MESSAGES['NOT_GDAL_RASTER']

    # Check that an overview .ovr file wasn't opened.
    if os.path.splitext(filepath)[1] == '.ovr':
        return MESSAGES['OVR_FILE']

    srs = gdal_dataset.GetSpatialRef()
    projection_warning = _check_projection(srs, projected, projection_units)
    if projection_warning:
        gdal_dataset = None
        return projection_warning

    gdal_dataset = None
    return None


def load_fields_from_vector(filepath, layer_id=0):
    """Load fieldnames from a given vector.

    Args:
        filepath (string): The path to a GDAL-compatible vector on disk.
        layer_id=0 (string or int): The identifier for the layer to use.

    Returns:
        A list of string fieldnames within the target layer.

    """
    if not os.path.exists(filepath):
        raise ValueError('File not found: %s' % filepath)

    vector = gdal.OpenEx(filepath, gdal.OF_VECTOR)
    layer = vector.GetLayer(layer_id)
    fieldnames = [defn.GetName() for defn in layer.schema]
    layer = None
    vector = None
    return fieldnames


def check_vector(filepath, geometries, fields=None, projected=False,
                 projection_units=None, **kwargs):
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
    file_warning = check_file(filepath, permissions='r')
    if file_warning:
        return file_warning

    try:
        gdal_dataset = gdal.OpenEx(filepath, gdal.OF_VECTOR)
    except RuntimeError:
        return MESSAGES['NOT_GDAL_VECTOR']

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
    for geom in geometries:
        allowed_geom_types += geom_map[geom]

    # NOTE: this only checks the layer geometry type, not the types of the
    # actual geometries (layer.GetGeometryTypes()). This is probably equivalent
    # in most cases, and it's more efficient than checking every geometry, but
    # we might need to change this in the future if it becomes a problem.
    # Currently not supporting ogr.wkbUnknown which allows mixed types.
    layer = gdal_dataset.GetLayer()
    if layer.GetGeomType() not in allowed_geom_types:
        return MESSAGES['WRONG_GEOM_TYPE'].format(allowed=geometries)

    if fields:
        field_patterns = get_headers_to_validate(fields)
        fieldnames = [defn.GetName() for defn in layer.schema]
        required_field_warning = check_headers(
            field_patterns, fieldnames, 'field')
        if required_field_warning:
            return required_field_warning

    srs = layer.GetSpatialRef()
    projection_warning = _check_projection(srs, projected, projection_units)
    return projection_warning


def check_raster_or_vector(filepath, **kwargs):
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
        return check_raster(filepath, **kwargs)
    else:
        return check_vector(filepath, **kwargs)


def check_freestyle_string(value, regexp=None, **kwargs):
    """Validate an arbitrary string.

    Args:
        value: The value to check.  Must be able to be cast to a string.
        regexp=None (string): a string interpreted as a regular expression.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    if regexp:
        matches = re.fullmatch(regexp, str(value))
        if not matches:
            return MESSAGES['REGEXP_MISMATCH'].format(regexp=regexp)
    return None


def check_option_string(value, options, **kwargs):
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
    if options and str(value) not in options:
        return MESSAGES['INVALID_OPTION'].format(option_list=sorted(options))


def check_number(value, expression=None, **kwargs):
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
        return MESSAGES['NOT_A_NUMBER'].format(value=value)

    if expression:
        # Check to make sure that 'value' is in the expression.
        if 'value' not in expression:
            raise AssertionError(
                'The variable name value is not found in the '
                f'expression: {expression}')

        # Expression is assumed to return a boolean, something like
        # "value > 0" or "(value >= 0) & (value < 1)".  An exception will
        # be raised if asteval can't evaluate the expression.
        result = _evaluate_expression(expression, {'value': float(value)})
        if not result:  # A python bool object is returned.
            return MESSAGES['INVALID_VALUE'].format(condition=expression)

    return None


def check_ratio(value, **kwargs):
    """Validate a ratio (a proportion expressed as a value from 0 to 1).

    Args:
        value: A python value. This should be able to be cast to a float.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return MESSAGES['NOT_A_NUMBER'].format(value=value)

    if as_float < 0 or as_float > 1:
        return MESSAGES['NOT_WITHIN_RANGE'].format(
            value=as_float,
            range='[0, 1]')

    return None


def check_percent(value, **kwargs):
    """Validate a percent (a proportion expressed as a value from 0 to 100).

    Args:
        value: A python value. This should be able to be cast to a float.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return MESSAGES['NOT_A_NUMBER'].format(value=value)

    if as_float < 0 or as_float > 100:
        return MESSAGES['NOT_WITHIN_RANGE'].format(
            value=as_float,
            range='[0, 100]')

    return None


def check_integer(value, **kwargs):
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
            return MESSAGES['NOT_AN_INTEGER'].format(value=value)
    except (TypeError, ValueError):
        return MESSAGES['NOT_A_NUMBER'].format(value=value)
    return None


def check_boolean(value, **kwargs):
    """Validate a boolean value.

    If the value provided is not a python boolean, an error message is
    returned.


    Args:
        value: The value to evaluate.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    if not isinstance(value, bool):
        return MESSAGES['NOT_BOOLEAN'].format(value=value)


def get_validated_dataframe(
        csv_path, columns=None, rows=None, index_col=None,
        read_csv_kwargs={}, **kwargs):
    """Read a CSV into a dataframe that is guaranteed to match the spec."""

    if not (columns or rows):
        raise ValueError('One of columns or rows must be provided')

    # build up a list of regex patterns to match columns against columns from
    # the table that match a pattern in this list (after stripping whitespace
    # and lowercasing) will be included in the dataframe
    axis = 'column' if columns else 'row'

    if rows:
        read_csv_kwargs = read_csv_kwargs.copy()
        read_csv_kwargs['header'] = None

    df = utils.read_csv_to_dataframe(csv_path, **read_csv_kwargs)

    if rows:
        # swap rows and column
        df = df.set_index(df.columns[0]).rename_axis(
            None, axis=0).T.reset_index(drop=True)

    spec = columns if columns else rows

    patterns = []
    for column in spec:
        column = column.lower()
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

    for (col_name, col_spec), pattern in zip(spec.items(), patterns):
        matching_cols = [c for c in available_cols if re.fullmatch(pattern, c)]
        if col_spec.get('required', True) is True and '[' not in col_name and not matching_cols:
            raise ValueError(MESSAGES['MATCHED_NO_HEADERS'].format(
                header=axis,
                header_name=col_name))
        available_cols -= set(matching_cols)
        for col in matching_cols:
            try:
                # frozenset needed to make the set hashable.  A frozenset and set with the same members are equal.
                if col_spec['type'] in {'csv', 'directory', 'file', 'raster', 'vector', frozenset({'raster', 'vector'})}:
                    df[col] = df[col].apply(
                        lambda p: p if pandas.isna(p) else utils.expand_path(str(p).strip(), csv_path))
                    df[col] = df[col].astype(pandas.StringDtype())
                elif col_spec['type'] in {'freestyle_string', 'option_string'}:
                    df[col] = df[col].apply(
                        lambda s: s if pandas.isna(s) else str(s).strip().lower())
                    df[col] = df[col].astype(pandas.StringDtype())
                elif col_spec['type'] in {'number', 'percent', 'ratio'}:
                    df[col] = df[col].astype(float)
                elif col_spec['type'] == 'integer':
                    df[col] = df[col].astype(pandas.Int64Dtype())
                elif col_spec['type'] == 'boolean':
                    df[col] = df[col].astype('boolean')
                else:
                    raise ValueError(f"Unknown type: {col_spec['type']}")
            except Exception as err:
                raise ValueError(
                    f'Value(s) in the "{col}" column could not be interpreted '
                    f'as {col_spec["type"]}s. Original error: {err}')

            col_type = col_spec['type']
            if isinstance(col_type, set):
                col_type = frozenset(col_type)
            if col_type in {'raster', 'vector', frozenset({'raster', 'vector'})}:
                # recursively validate the files within the column
                def check_value(value):
                    if pandas.isna(value):
                        return
                    err_msg = _VALIDATION_FUNCS[col_type](value, **col_spec)
                    if err_msg:
                        raise ValueError(
                            f'Error in {axis} "{col}", value "{value}": {err_msg}')
                df[col].apply(check_value)

    if any(df.columns.duplicated()):
        duplicated_columns = df.columns[df.columns.duplicated]
        return MESSAGES['DUPLICATE_HEADER'].format(
            header=header_type,
            header_name=expected,
            number=count)

    # set the index column, if specified
    if index_col is not None:
        index_col = index_col.lower()
        try:
            df = df.set_index(index_col, verify_integrity=True)
        except KeyError:
            # If 'index_col' is not a column then KeyError is raised for using
            # it as the index column
            LOGGER.error(f"The column '{index_col}' could not be found "
                         f"in the table {csv_path}")
            raise

    return df


def check_csv(filepath, **kwargs):
    """Validate a table.

    Args:
        filepath (string): The string filepath to the table.

    Returns:
        A string error message if an error was found. ``None`` otherwise.

    """
    file_warning = check_file(filepath, permissions='r')
    if file_warning:
        return file_warning
    if 'columns' in kwargs or 'rows' in kwargs:
        try:
            get_validated_dataframe(filepath, **kwargs)
        except Exception as e:
            return str(e)


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
            return MESSAGES['MATCHED_NO_HEADERS'].format(
                header=header_type,
                header_name=expected)
        elif count > 1:
            return MESSAGES['DUPLICATE_HEADER'].format(
                header=header_type,
                header_name=expected,
                number=count)
    return None


def check_spatial_overlap(spatial_filepaths_list,
                          different_projections_ok=False):
    """Check that the given spatial files spatially overlap.

    Args:
        spatial_filepaths_list (list): A list of files that can be opened with
            GDAL.  Must be on the local filesystem.
        different_projections_ok=False (bool): Whether it's OK for the input
            spatial files to have different projections.  If ``True``, all
            projections will be converted to WGS84 before overlap is checked.

    Returns:
        A string error message if an error is found.  ``None`` otherwise.

    """
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    wgs84_wkt = wgs84_srs.ExportToWkt()

    bounding_boxes = []
    checked_file_list = []
    for filepath in spatial_filepaths_list:
        try:
            info = pygeoprocessing.get_raster_info(filepath)
        except (ValueError, RuntimeError):
            # ValueError is raised by PyGeoprocessing < 3.4.4 when the file is
            # not a raster.
            # RuntimeError is raised by GDAL in PyGeoprocessing >= 3.4.4 when
            # the file is not a raster.
            info = pygeoprocessing.get_vector_info(filepath)

        if info['projection_wkt'] is None:
            return MESSAGES['NO_PROJECTION'].format(filepath=filepath)

        if different_projections_ok:
            bounding_box = pygeoprocessing.transform_bounding_box(
                info['bounding_box'], info['projection_wkt'], wgs84_wkt)
        else:
            bounding_box = info['bounding_box']

        if all([numpy.isinf(coord) for coord in bounding_box]):
            LOGGER.warning(
                'Skipping infinite bounding box for file %s', filepath)
            continue

        bounding_boxes.append(bounding_box)
        checked_file_list.append(filepath)

    try:
        pygeoprocessing.merge_bounding_box_list(bounding_boxes, 'intersection')
    except ValueError as error:
        LOGGER.debug(error)
        formatted_lists = _format_bbox_list(checked_file_list, bounding_boxes)
        return MESSAGES['BBOX_NOT_INTERSECT'].format(bboxes=formatted_lists)
    return None


def _format_bbox_list(file_list, bbox_list):
    """Format two lists of equal length into one string."""
    return ' | '.join(
            [a + ': ' + str(b) for a, b in zip(
                file_list, bbox_list)])


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


def get_headers_to_validate(spec):
    """Get header names to validate from a row/column/field spec dictionary.

    This module only validates row/column/field names that are static and
    always required. If `'required'` is anything besides `True`, or if the name
    contains brackets indicating it's user-defined, it is not returned.

    Args:
        spec (dict): a row/column/field spec dictionary that maps row/column/
            field names to specs for them

    Returns:
        list of expected header names to validate against
    """
    headers = []
    for key, val in spec.items():
        # if 'required' isn't a key, it defaults to True
        if ('required' not in val) or (val['required'] is True):
            # brackets are a special character for our args spec syntax
            # they surround the part of the key that's user-defined
            # user-defined rows/columns/fields are not validated here, so skip
            if '[' not in key:
                headers.append(key)
    return headers


# accessing a file could take a long time if it's in a file streaming service
# to prevent the UI from hanging due to slow validation,
# set a timeout for these functions.
_VALIDATION_FUNCS = {
    'boolean': check_boolean,
    'csv': functools.partial(timeout, check_csv),
    'file': functools.partial(timeout, check_file),
    'directory': functools.partial(timeout, check_directory),
    'freestyle_string': check_freestyle_string,
    'number': check_number,
    'ratio': check_ratio,
    'percent': check_percent,
    'integer': check_integer,
    'option_string': check_option_string,
    'raster': functools.partial(timeout, check_raster),
    'vector': functools.partial(timeout, check_vector),
    frozenset({'raster', 'vector'}): functools.partial(timeout, check_raster_or_vector),
    'other': None,  # Up to the user to define their validate()
}


def validate(args, spec, spatial_overlap_opts=None):
    """Validate an args dict against a model spec.

    Validates an arguments dictionary according to the rules laid out in
    ``spec``.  If ``spatial_overlap_opts`` is also provided, valid spatial
    inputs will be checked for spatial overlap.

    Args:
        args (dict): The InVEST model args dict to validate.
        spec (dict): The InVEST model spec dict to validate against.
        spatial_overlap_opts=None (dict): A dict.  If provided, the key
        ``"spatial_keys"`` is required to be a list of keys that may be present
            in the args dict and (if provided in args) will be checked for
            overlap with all other keys in this list.  If the key
            ``"reference_key"`` is also present in this dict, the bounding
            boxes of each of the files represented by
            ``spatial_overlap_opts["spatial_keys"]`` will be transformed to the
            SRS of the dataset at this key.

    Returns:
        A list of tuples where the first element of the tuple is an iterable of
        keys affected by the error in question and the second element of the
        tuple is the string message of the error.  If no validation errors were
        found, an empty list is returned.

    """
    validation_warnings = []

    # Phase 1: Check whether an input is required and has a value
    missing_keys = set()
    required_keys_with_no_value = set()
    expression_values = {
        input_key: args.get(input_key, False) for input_key in spec.keys()}
    keys_with_falsey_values = set()
    for key, parameter_spec in spec.items():
        # Default required to True since this is the most common
        try:
            required = parameter_spec['required']
        except KeyError:
            required = True

        if isinstance(required, str):
            required = bool(_evaluate_expression(
                expression=f'{spec[key]["required"]}',
                variable_map=expression_values))

        # At this point, required is only True or False.
        if required:
            if key not in args:
                missing_keys.add(key)
            else:
                if args[key] in ('', None):
                    required_keys_with_no_value.add(key)
        elif not expression_values[key]:
            # Don't validate falsey values or missing (None, "") values.
            keys_with_falsey_values.add(key)

    if missing_keys:
        validation_warnings.append(
            (sorted(missing_keys), MESSAGES['MISSING_KEY']))

    if required_keys_with_no_value:
        validation_warnings.append(
            (sorted(required_keys_with_no_value), MESSAGES['MISSING_VALUE']))

    # Phase 2: Check whether any input with a value validates with its
    # type-specific check function.
    invalid_keys = set()
    insufficient_keys = (
        missing_keys | required_keys_with_no_value | keys_with_falsey_values)
    for key in set(args.keys()) - insufficient_keys:
        # Extra args that don't exist in the MODEL_SPEC are okay
        # we don't need to try to validate them
        try:
            # Using deepcopy to make sure we don't modify the original spec
            parameter_spec = copy.deepcopy(spec[key])
        except KeyError:
            LOGGER.debug(f'Provided key {key} does not exist in MODEL_SPEC')
            continue

        param_type = parameter_spec['type']
        if isinstance(param_type, set):
            param_type = frozenset(param_type)
        # rewrite parameter_spec for any nested, conditional validity
        axis_keys = None
        if param_type == 'csv':
            axis_keys = ['columns', 'rows']
        elif param_type == 'vector' or 'vector' in param_type:
            axis_keys = ['fields']
        elif param_type == 'directory':
            axis_keys = ['contents']

        if axis_keys:
            for axis_key in axis_keys:
                if axis_key not in parameter_spec:
                    continue
                for nested_key, nested_spec in parameter_spec[axis_key].items():
                    if ('required' in nested_spec
                            and isinstance(nested_spec['required'], str)):
                        parameter_spec[axis_key][nested_key]['required'] = (
                            bool(_evaluate_expression(
                                nested_spec['required'], expression_values)))

        type_validation_func = _VALIDATION_FUNCS[param_type]

        if type_validation_func is None:
            # Validation for 'other' type must be performed by the user.
            continue
        try:
            # pass the entire arg spec into the validation function as kwargs
            # each type validation function allows extra kwargs with **kwargs
            warning_msg = type_validation_func(args[key], **parameter_spec)
            if warning_msg:
                validation_warnings.append(([key], warning_msg))
                invalid_keys.add(key)
        except Exception:
            LOGGER.exception(
                'Error when validating key %s with value %s',
                key, args[key])
            validation_warnings.append(([key], MESSAGES['UNEXPECTED_ERROR']))

    # Phase 3: Check spatial overlap if applicable
    if spatial_overlap_opts:
        spatial_keys = set(spatial_overlap_opts['spatial_keys'])

        # Only test for spatial overlap once all the sufficient spatial keys
        # are otherwise valid. And then only when there are at least 2.
        valid_spatial_keys = spatial_keys.difference(
            invalid_keys.union(insufficient_keys))

        if len(valid_spatial_keys) >= 2:
            spatial_files = []
            checked_keys = []
            for key in valid_spatial_keys:
                if key in args and args[key] not in ('', None):
                    spatial_files.append(args[key])
                    checked_keys.append(key)

            try:
                different_projections_ok = (
                    spatial_overlap_opts['different_projections_ok'])
            except KeyError:
                different_projections_ok = False

            spatial_overlap_error = check_spatial_overlap(
                spatial_files, different_projections_ok)
            if spatial_overlap_error:
                validation_warnings.append(
                    (checked_keys, spatial_overlap_error))

    # sort warnings alphabetically by key name
    return sorted(validation_warnings, key=lambda w: w[0][0])


def invest_validator(validate_func):
    """Decorator to enforce characteristics of validation inputs and outputs.

    Attributes of inputs and outputs that are enforced are:

        * ``args`` parameter to ``validate`` must be a ``dict``
        * ``limit_to`` parameter to ``validate`` must be either ``None`` or a
          string (``str`` or ``unicode``) that exists in the ``args`` dict.
        *  All keys in ``args`` must be strings
        * Decorated ``validate`` func must return a list of 2-tuples, where
          each 2-tuple conforms to these rules:

            * The first element of the 2-tuple is an iterable of strings.
              It is an error for the first element to be a string.
            * The second element of the 2-tuple is a string error message.

    In addition, this validates the ``n_workers`` argument if it's included.

    Raises:
        AssertionError when an invalid format is found.

    Example::

        from natcap.invest import validation
        @validation.invest_validator
        def validate(args, limit_to=None):
            # do your validation here
    """
    def _wrapped_validate_func(args, limit_to=None):
        validate_func_args = inspect.getfullargspec(validate_func)
        assert validate_func_args.args == ['args', 'limit_to'], (
            'validate has invalid parameters: parameters are: %s.' % (
                validate_func_args.args))

        assert isinstance(args, dict), 'args parameter must be a dictionary.'
        assert (isinstance(limit_to, type(None)) or
                isinstance(limit_to, str)), (
                    'limit_to parameter must be either a string key or None.')
        if limit_to is not None:
            assert limit_to in args, ('limit_to key "%s" must exist in args.'
                                      % limit_to)

        for key, value in args.items():
            assert isinstance(key, str), (
                'All args keys must be strings.')

        # Pytest in importlib mode makes it impossible for test modules to
        # import one another. This causes a problem in test_validation.py,
        # which gets imported into itself here and fails.
        # Since this decorator might not be needed in the future,
        # just ignore failed imports; assume they have no MODEL_SPEC.
        try:
            model_module = importlib.import_module(validate_func.__module__)
        except Exception:
            LOGGER.warning(
                'Unable to import module %s: assuming no MODEL_SPEC.',
                validate_func.__module__)
            model_module = None

        # If the module has an MODEL_SPEC defined, validate against that.
        if hasattr(model_module, 'MODEL_SPEC'):
            LOGGER.debug('Using MODEL_SPEC for validation')
            args_spec = getattr(model_module, 'MODEL_SPEC')['args']

            if limit_to is None:
                LOGGER.info('Starting whole-model validation with MODEL_SPEC')
                warnings_ = validate_func(args)
            else:
                LOGGER.info('Starting single-input validation with MODEL_SPEC')
                args_key_spec = args_spec[limit_to]

                args_value = args[limit_to]
                error_msg = None

                # We're only validating a single input.  This is not officially
                # supported in the validation function, but we can make it work
                # within this decorator.
                try:
                    if args_key_spec['required'] is True:
                        if args_value in ('', None):
                            error_msg = "Value is required"
                except KeyError:
                    # If required is not defined in the args_spec, we default
                    # to False.  If 'required' is an expression, we can't
                    # validate that outside of whole-model validation.
                    pass

                # If the input is not required and does not have a value, no
                # need to validate it.
                if args_value not in ('', None):
                    input_type = args_key_spec['type']
                    if isinstance(input_type, set):
                        input_type = frozenset(input_type)
                    validator_func = _VALIDATION_FUNCS[input_type]
                    error_msg = validator_func(args_value, **args_key_spec)

                if error_msg is None:
                    warnings_ = []
                else:
                    warnings_ = [([limit_to], error_msg)]
        else:  # args_spec is not defined for this function.
            LOGGER.warning('MODEL_SPEC not defined for this model')
            warnings_ = validate_func(args, limit_to)

        LOGGER.debug('Validation warnings: %s',
                     pprint.pformat(warnings_))

        return warnings_

    return _wrapped_validate_func
