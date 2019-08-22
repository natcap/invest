"""Common validation utilities for InVEST models."""
import collections
import inspect
import logging
import pprint
import os
import re

import sympy
import sympy.parsing.sympy_parser
from osgeo import gdal, osr

try:
    from builtins import basestring
except ImportError:
    # Python3 doesn't have a basestring.
    basestring = str


# Python3 doesn't know about basestring, only str.
try:
    basestring
except NameError:
    basestring = str


#: A flag to pass to the validation context manager indicating that all keys
#: should be checked.
CHECK_ALL_KEYS = None
MESSAGE_REQUIRED = 'Parameter is required but is missing or has no value'
LOGGER = logging.getLogger(__name__)


WORKSPACE_SPEC = {
    "name": "Workspace",
    "about": (
        "The folder where all intermediate and output files of the model "
        "will be written.  If this folder does not exist, it will be "
        "created."),
    "type": "directory",
    "required": True,
    "validation_options": {
        "exists": True,
        "permissions": "rwx",
    }
}

SUFFIX_SPEC = {
    "name": "File suffix",
    "about": (
        'A string that will be added to the end of all files '
        'written to the workspace.'),
    "type": "freestyle_string",
    "required": False,
    "validation_options": {
        "regexp": {
            "pattern": "[a-zA-Z0-9_-]*",
            "case_sensitive": False,
        }
    }
}

N_WORKERS_SPEC = {
    "name": "Taskgraph n_workers parameter",
    "about": (
        "The n_workers parameter to provide to taskgraph. "
        "-1 will cause all jobs to run synchronously. "
        "0 will run all jobs in the same process, but scheduling will take "
        "place asynchronously. Any other positive integer will cause that "
        "many processes to be spawned to execute tasks."),
    "type": "number",
    "required": False,
    "validation_options": {
        "regexp": {
            "pattern": "-?[0-9]+",
            "case_sensitive": False
        },
        "expression": "value >= -1"
    }
}


def check_directory(dirpath, exists=False, permissions='rx'):
    """Validate a directory.

    Parameters:
        dirpath (string): The directory path to validate.
        exists=False (bool): If ``True``, the directory at ``dirpath``
            must already exist on the filesystem.
        permissions='rx' (string): A string that includes the lowercase
            characters ``r``, ``w`` and/or ``x`` indicating required
            permissions for this folder .  See ``check_permissions`` for
            details.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    if exists:
        if not os.path.exists(dirpath):
            return "Directory not found"

    if not os.path.isdir(dirpath):
        return "Path is not a directory"

    permissions_warning = check_permissions(dirpath, permissions)
    if permissions_warning:
        return permissions_warning


def check_file(filepath, permissions='r'):
    """Validate a single file.

    Parameters:
        filepath (string): The filepath to validate.
        permissions='r' (string): A string that includes the lowercase
            characters ``r``, ``w`` and/or ``x`` indicating required
            permissions for this file.  See ``check_permissions`` for
            details.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    if not os.path.exists(filepath):
        return "File not found"

    permissions_warning = check_permissions(filepath, permissions)
    if permissions_warning:
        return permissions_warning


def check_permissions(path, permissions):
    """Validate permissions on a filesystem object.

    This function uses ``os.access`` to determine permissions access.

    Parameters:
        path (string): The path to examine for permissions.
        permissions (string): a string including the characters ``r``, ``w``
            and/or ``x`` (lowercase), indicating read, write, and execute
            permissions (respectively) that the filesystem object at ``path``
            must have.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    for letter, mode, descriptor in (
            ('r', os.R_OK, 'read'),
            ('w', os.W_OK, 'write'),
            ('x', os.X_OK, 'execute')):
        if letter in permissions and not os.access(path, mode):
            return 'You must have %s access to this file' % descriptor


def _check_projection(srs, projected, projection_units):
    if projected:
        if not srs.IsProjected():
            return "Vector must be projected in linear units."

    if projection_units:
        valid_meter_units = set(('m', 'meter', 'meters', 'metre', 'metres'))
        layer_units_name = srs.GetLinearUnitsName().lower()

        if projection_units in valid_meter_units:
            if not layer_units_name in valid_meter_units:
                return "Layer must be projected in meters"
        else:
            if layer_units_name.lower() != projection_units.lower():
                return "Layer must be projected in %s" % projection_units.lower()

    return None


def check_raster(filepath, projected=False, projection_units=None):
    file_warning = check_file(filepath, permissions='r')
    if file_warning:
        return file_warning

    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal_dataset = gdal.OpenEx(filepath, gdal.OF_RASTER)
    gdal.PopErrorHandler()

    if gdal_dataset is None:
        return "File could not be opened as a GDAL raster"

    srs = osr.SpatialReference()
    srs.ImportFromWkt(gdal_dataset.GetProjection())

    projection_warning = _check_projection(srs, projected, projection_units)
    if projection_warning:
        gdal_dataset = None
        return projection_warning

    gdal_dataset = None
    return None


def check_vector(filepath, required_fields=None, projected=False,
                 projection_units=None):
    file_warning = check_file(filepath, permissions='r')
    if file_warning:
        return file_warning

    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal_dataset = gdal.OpenEx(filepath, gdal.OF_VECTOR)
    gdal.PopErrorHandler()

    if gdal_dataset is None:
        return "File could not be opened as a GDAL vector"

    layer = gdal_dataset.GetLayer()
    srs = layer.GetSpatialRef()

    if required_fields:
        fieldnames = set([defn.GetName().upper() for defn in layer.schema])
        missing_fields = set(field.upper() for field in required_fields) - fieldnames
        if missing_fields:
            return "Fields are missing from the first layer: %s" % sorted(
                missing_fields)

    projection_warning = _check_projection(srs, projected, projection_units)
    if projection_warning:
        return projection_warning

    return None


def check_freestyle_string(value, regexp=None):
    if regexp:
        flags = 0
        if 'case_sensitive' in regexp:
            if regexp['case_sensitive']:
                flags = re.IGNORECASE
        matches = re.findall(regexp['pattern'], str(value), flags)
        if not matches:
            return "Value did not match expected pattern %s" % regexp['pattern']

    return None


def check_option_string(value, options):
    if value not in options:
        return "Value must be one of: %s" % sorted(options)


def check_number(value, expression=None):
    try:
        float(value)
    except (TypeError, ValueError):
        return "Value could not be interpreted as a number"

    if expression:
        # Check to make sure that 'value' is in the expression.
        if 'value' not in [str(x) for x in
                           sympy.parsing.sympy_parser.parse_expr(
                               expression).free_symbols]:
            raise AssertionError('Value is not used in this expression')

        # Expression is assumed to return a boolean, something like
        # "value > 0" or "(value >= 0) & (value < 1)".  An exception will
        # be raised if sympy can't evaluate the expression.
        if not sympy.lambdify(['value'], expression, 'numpy')(value):
            return "Value does not meet condition %s" % expression

    return None


def check_boolean(value):
    if isinstance(value, str):
        value = value.strip()
        if value.lower() not in ("true", "false"):
            return "Value must be one of 'True' or 'False'"

    # Python's truthiness rules allow for basically anything to be cast to a
    # boolean, so there's no error here that we can check for unless it's
    # one of the string cases above.
    return None


_VALIDATION_FUNCS = {
    'boolean': check_boolean,
    #'csv': check_csv,
    'file': check_file,
    'folder': check_directory,
    'freestyle_string': check_freestyle_string,
    'number': check_number,
    'option_string': check_option_string,
    'raster': check_raster,
    'vector': check_vector,
}


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

    Example:
        from natcap.invest import validation
        @validation.invest_validator
        def validate(args, limit_to=None):
            # do your validation here
    """
    def _wrapped_validate_func(args, limit_to=None):
        validate_func_args = inspect.getargspec(validate_func)
        assert validate_func_args.args == ['args', 'limit_to'], (
            'validate has invalid parameters: parameters are: %s.' % (
                validate_func_args.args))

        assert isinstance(args, dict), 'args parameter must be a dictionary.'
        assert (isinstance(limit_to, type(None)) or
                isinstance(limit_to, basestring)), (
                    'limit_to parameter must be either a string key or None.')
        if limit_to is not None:
            assert limit_to in args, ('limit_to key "%s" must exist in args.'
                                      % limit_to)

        for key, value in args.items():
            assert isinstance(key, basestring), (
                'All args keys must be strings.')

        # Validate n_workers if requested.
        common_warnings = []
        if limit_to in ('n_workers', None):
            try:
                n_workers_float = float(args['n_workers'])
                n_workers_int = int(n_workers_float)
                if n_workers_float != n_workers_int:
                    common_warnings.append(
                        (['n_workers'],
                         ('If provided, must be an integer ')))
            except (ValueError, KeyError):
                # ValueError When n_workers is an empty string.  Input is
                # optional, so this is not erroneous behavior.
                # KeyError for those models that don't use this input.
                pass

        warnings_ = validate_func(args, limit_to)
        LOGGER.debug('Validation warnings: %s',
                     pprint.pformat(warnings_))

        assert isinstance(warnings_, list), (
            'validate function must return a list of 2-tuples.')
        for keys_iterable, error_string in warnings_:
            assert (isinstance(keys_iterable, collections.Iterable) and not
                    isinstance(keys_iterable, basestring)), (
                        'Keys entry %s must be a non-string iterable' % (
                            keys_iterable))
            for key in keys_iterable:
                assert key in args, 'Key %s (from %s) must be in args.' % (
                    key, keys_iterable)
            assert isinstance(error_string, basestring), (
                'Error string must be a string, not a %s' % type(error_string))
        return common_warnings + warnings_

    return _wrapped_validate_func
