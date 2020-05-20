"""Common validation utilities for InVEST models."""
import ast
import codecs
import inspect
import logging
import pprint
import os
import re
import importlib

import pygeoprocessing
import pandas
import xlrd
from osgeo import gdal, osr
import numpy


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
        "exists": False,
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
        "expression": "value >= -1"
    }
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


def check_directory(dirpath, exists=False, permissions='rx'):
    """Validate a directory.

    Args:
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

    if os.path.exists(dirpath):
        if not os.path.isdir(dirpath):
            return "Path must be a directory"
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

    permissions_warning = check_permissions(dirpath, permissions)
    if permissions_warning:
        return permissions_warning


def check_file(filepath, permissions='r'):
    """Validate a single file.

    Args:
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

    Args:
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
    """Validate a GDAL projection.

    Args:
        srs (osr.SpatialReference): A GDAL Spatial Reference object
            representing the spatial reference of a GDAL dataset.
        projected (bool): Whether the spatial reference must be projected in
            linear units.
        projection_units (string): The string label (case-insensitive)
            indicating the required linear units of the projection.  Note that
            "m", "meters", "meter", "metre" and "metres" are all synonymous.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    if srs is None:
        return "Dataset must have a valid projection."

    if projected:
        if not srs.IsProjected():
            return "Dataset must be projected in linear units."

    if projection_units:
        valid_meter_units = set(('m', 'meter', 'meters', 'metre', 'metres'))
        layer_units_name = srs.GetLinearUnitsName().lower()

        if projection_units in valid_meter_units:
            if layer_units_name not in valid_meter_units:
                return "Layer must be projected in meters"
        else:
            if layer_units_name.lower() != projection_units.lower():
                return ("Layer must be projected in %s"
                        % projection_units.lower())

    return None


def check_raster(filepath, projected=False, projection_units=None):
    """Validate a GDAL Raster on disk.

    Args:
        filepath (string): The path to the raster on disk.  The file must exist
            and be readable.
        projected=False (bool): Whether the spatial reference must be projected
            in linear units.
        projection_units=None (string): The string label (case-insensitive)
            indicating the required linear units of the projection.  If
            ``None``, the projection units will not be checked.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
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


def check_vector(filepath, required_fields=None, projected=False,
                 projection_units=None):
    """Validate a GDAL vector on disk.

    Note:
        If the provided vector has multiple layers, only the first layer will
        be checked.

    Args:
        filepath (string): The path to the vector on disk.  The file must exist
            and be readable.
        required_fields=None (list): The string fieldnames (case-insensitive)
            that must be present in the vector layer's table.  If None,
            fieldnames will not be checked.
        projected=False (bool): Whether the spatial reference must be projected
            in linear units.  If None, the projection will not be checked.
        projection_units=None (string): The string label (case-insensitive)
            indicating the required linear units of the projection.  If
            ``None``, the projection units will not be checked.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
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
        missing_fields = (
            set(field.upper() for field in required_fields) - fieldnames)
        if missing_fields:
            return "Fields are missing from the first layer: %s" % sorted(
                missing_fields)

    projection_warning = _check_projection(srs, projected, projection_units)
    if projection_warning:
        return projection_warning

    return None


def check_freestyle_string(value, regexp=None):
    """Validate an arbitrary string.

    Args:
        value: The value to check.  Must be able to be cast to a string.
        regexp=None (dict): A dict representing validation parameters for a
            regular expression.  ``regexp['pattern']`` is required, and its
            value must be a string regular expression.
            ``regexp['case_sensitive']`` may also be provided and is expected
            to be a boolean value.  If ``True`` or truthy, the regular
            expression will ignore case.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    if regexp:
        flags = 0
        if 'case_sensitive' in regexp:
            if regexp['case_sensitive']:
                flags = re.IGNORECASE
        matches = re.findall(regexp['pattern'], str(value), flags)
        if not matches:
            return ("Value did not match expected pattern %s"
                    % regexp['pattern'])

    return None


def check_option_string(value, options):
    """Validate that a string is in a list of options.

    Args:
        value (string): The string value to test.
        options (list): A list of strings to test against.

    Returns:
        A string error message if ``value`` is not in ``options``.  ``None``
        otherwise.

    """
    if value not in options:
        return "Value must be one of: %s" % sorted(options)


def check_number(value, expression=None):
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
        return "Value '%s' could not be interpreted as a number" % value

    if expression:
        # Check to make sure that 'value' is in the expression.
        if 'value' not in expression:
            raise AssertionError(
                'The variable name value is not found in the '
                'expression: "%s"' % expression)

        # Expression is assumed to return a boolean, something like
        # "value > 0" or "(value >= 0) & (value < 1)".  An exception will
        # be raised if asteval can't evaluate the expression.
        result = _evaluate_expression(expression, {'value': float(value)})
        if not result:  # A python bool object is returned.
            return "Value does not meet condition %s" % expression

    return None


def check_boolean(value):
    """Validate a boolean value.

    If the value provided is not a python boolean, an error message is
    returned.


    Args:
        value: The value to evaluate.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    if not isinstance(value, bool):
        return "Value must be either True or False, not %s %s" % (
            type(value), value)


def check_csv(filepath, required_fields=None, excel_ok=False):
    """Validate a table.

    Args:
        filepath (string): The string filepath to the table.
        required_fields=None (list): A case-insensitive list of fieldnames that
            must exist in the table.  If None, fieldnames will not be checked.
        excel_ok=False (boolean): Whether it's OK for the file to be an Excel
            table.  This is not a common case.

    Returns:
        A string error message if an error was found.  ``None`` otherwise.

    """
    file_warning = check_file(filepath, permissions='r')
    if file_warning:
        return file_warning

    try:
        # Check if the file encoding is UTF-8 BOM first
        encoding = None
        with open(filepath, 'rb') as file_obj:
            first_line = file_obj.readline()
            if first_line.startswith(codecs.BOM_UTF8):
                encoding = 'utf-8-sig'
        dataframe = pandas.read_csv(
            filepath, sep=None, engine='python', encoding=encoding)
    except Exception:
        if excel_ok:
            try:
                dataframe = pandas.read_excel(filepath)
            except xlrd.biffh.XLRDError:
                return "File could not be opened as a CSV or Excel file."
        else:
            return ("File could not be opened as a CSV. "
                    "File must be encoded as a UTF-8 CSV.")

    if required_fields:
        fields_in_table = set([name.upper() for name in dataframe.columns])
        missing_fields = (
            set(field.upper() for field in required_fields) - fields_in_table)

        if missing_fields:
            return ("Fields are missing from this table: %s" %
                    sorted(missing_fields))
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
        except ValueError:
            info = pygeoprocessing.get_vector_info(filepath)
        bounding_box = info['bounding_box']

        if different_projections_ok:
            bounding_box = pygeoprocessing.transform_bounding_box(
                bounding_box, info['projection'], wgs84_wkt)

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
        formatted_lists = ' | '.join(
            [a + ': ' + str(b) for a, b in zip(
                checked_file_list, bounding_boxes)])
        message = f"Bounding boxes do not intersect: {formatted_lists}"
        return message
    return None


_VALIDATION_FUNCS = {
    'boolean': check_boolean,
    'csv': check_csv,
    'file': check_file,
    'folder': check_directory,
    'directory': check_directory,
    'freestyle_string': check_freestyle_string,
    'number': check_number,
    'option_string': check_option_string,
    'raster': check_raster,
    'vector': check_vector,
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

    # step 1: check absolute requirement
    missing_keys = set()
    keys_with_no_value = set()
    conditionally_required_keys = set()
    for key, parameter_spec in spec.items():
        try:
            required = parameter_spec['required']
        except KeyError:
            required = False
        if required is True:  # Might be an args key, can't rely on truthiness
            if key not in args:
                missing_keys.add(key)
            else:
                if args[key] in ('', None):
                    keys_with_no_value.add(key)

        # If ``required`` is a string, it must represent an expression of
        # conditional requirement based on the satisfaction of various args
        # keys.  We can only evaluate this later, after all other validation
        # happens, so add this args key to a set for later.
        elif isinstance(required, str):
            conditionally_required_keys.add(key)

    if missing_keys:
        validation_warnings.append(
            (sorted(missing_keys), "Key is missing from the args dict"))

    if keys_with_no_value:
        validation_warnings.append(
            (sorted(keys_with_no_value),
             "Input is required but has no value"))

    # Sufficiency: An input is sufficient when its key is present in args and
    # it has a value.  A sufficient input need not be valid.  Sufficiency is
    # used by the conditional requirement phase (step 3 in this function) to
    # determine whether a conditionally required input is required.
    # The only special case about sufficiency is with boolean values.
    # A boolean value absent from args is insufficient.  A boolean input that
    # is present in args but False is in sufficient.  A boolean input that is
    # present in args and True is sufficient.
    insufficient_keys = missing_keys.union(keys_with_no_value)

    # step 2: check primitive validity
    invalid_keys = set()
    for key, parameter_spec in spec.items():
        if key in invalid_keys:
            continue  # no need to validate a key we know is missing.

        # If the key isn't present, no need to validate.
        # If it's required and isn't present, we wouldn't have gotten to this
        # point in the function.
        if key not in args:
            insufficient_keys.add(key)
            continue

        # If the value is empty and it isn't required, then we don't need to
        # validate it.
        if args[key] in ('', None):
            insufficient_keys.add(key)
            continue

        # If no validation options specified, assume defaults.
        try:
            validation_options = parameter_spec['validation_options']
        except KeyError:
            validation_options = {}

        type_validation_func = _VALIDATION_FUNCS[parameter_spec['type']]
        if type_validation_func is None:
            # Validation for 'other' type must be performed by the user.
            continue

        try:
            warning_msg = type_validation_func(
                args[key], **validation_options)

            if warning_msg:
                validation_warnings.append(([key], warning_msg))
                invalid_keys.add(key)
        except Exception:
            LOGGER.exception(
                'Error when validating key %s with value %s',
                key, args[key])
            validation_warnings.append(
                ([key], 'An unexpected error occurred in validation'))

    # step 3: check conditional requirement
    # Need to evaluate sufficiency of inputs first.
    sufficient_inputs = {}
    for key in spec.keys():
        if key in insufficient_keys:
            sufficient_inputs[key] = False
        else:
            # Boolean values are special in that their T/F state is equivalent
            # to their satisfaction.  If a checkbox is checked, it is
            # considered satisfied.
            if spec[key]['type'] == 'boolean':
                sufficient_inputs[key] = args[key]

            # Any other input type must be sufficient because it is in args and
            # has a value.
            else:
                sufficient_inputs[key] = True

    for key in conditionally_required_keys:
        if key in invalid_keys:
            continue

        # An input is conditionally required when the expression given
        # evaluates to True.
        is_conditionally_required = _evaluate_expression(
            expression=spec[key]['required'],
            variable_map=sufficient_inputs)

        if is_conditionally_required:
            if key not in args:
                validation_warnings.append(
                    ([key], "Key is missing from the args dict"))
            else:
                if args[key] in ('', None):
                    validation_warnings.append(
                        ([key], "Key is required but has no value"))

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

    return validation_warnings


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

        # If the module has an ARGS_SPEC defined, validate against that.
        model_module = importlib.import_module(validate_func.__module__)
        if hasattr(model_module, 'ARGS_SPEC'):
            LOGGER.debug('Using ARG_SPEC for validation')
            args_spec = getattr(model_module, 'ARGS_SPEC')['args']

            if limit_to is None:
                LOGGER.info('Starting whole-model validation with ARGS_SPEC')
                warnings_ = validate_func(args)
            else:
                LOGGER.info('Starting single-input validation with ARGS_SPEC')
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
                    validator_func = _VALIDATION_FUNCS[input_type]

                    try:
                        validation_options = (
                            args_key_spec['validation_options'])
                    except KeyError:
                        validation_options = {}

                    error_msg = (
                        validator_func(args_value, **validation_options))

                if error_msg is None:
                    warnings_ = []
                else:
                    warnings_ = [([limit_to], error_msg)]
        else:  # args_spec is not defined for this function.
            LOGGER.warning('ARGS_SPEC not defined for this model')
            warnings_ = validate_func(args, limit_to)

        LOGGER.debug('Validation warnings: %s',
                     pprint.pformat(warnings_))

        return warnings_

    return _wrapped_validate_func
