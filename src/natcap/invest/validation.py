"""Common validation utilities for InVEST models."""
import copy
import functools
import importlib
import inspect
import logging
import os
import pprint

import numpy
import pint
import pygeoprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from . import gettext
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


def get_message(key):
    return gettext(MESSAGES[key])


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


def load_fields_from_vector(filepath, layer_id=0):
    """Load fieldnames from a given vector.

    Args:
        filepath (string): The path to a GDAL-compatible vector on disk.
        layer_id=0 (string or int): The identifier for the layer to use.

    Returns:
        A list of string fieldnames within the target layer.

    """
    vector = gdal.OpenEx(filepath, gdal.OF_VECTOR)
    layer = vector.GetLayer(layer_id)
    fieldnames = [defn.GetName() for defn in layer.schema]
    layer = None
    vector = None
    return fieldnames


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
        filepath = utils._GDALPath.from_uri(filepath).to_normalized_path()
        try:
            info = pygeoprocessing.get_raster_info(filepath)
        except (ValueError, RuntimeError):
            # ValueError is raised by PyGeoprocessing < 3.4.4 when the file is
            # not a raster.
            # RuntimeError is raised by GDAL in PyGeoprocessing >= 3.4.4 when
            # the file is not a raster.
            info = pygeoprocessing.get_vector_info(filepath)

        if info['projection_wkt'] is None:
            return get_message('NO_PROJECTION').format(filepath=filepath)

        if different_projections_ok:
            try:
                bounding_box = pygeoprocessing.transform_bounding_box(
                    info['bounding_box'], info['projection_wkt'], wgs84_wkt)
            except (ValueError, RuntimeError) as err:
                LOGGER.debug(err)
                LOGGER.warning(
                    f'Skipping spatial overlap check for {filepath}. '
                    'Bounding box cannot be transformed to EPSG:4326')
                continue

        else:
            bounding_box = info['bounding_box']

        if all([numpy.isinf(coord) for coord in bounding_box]):
            LOGGER.warning(
                f'Skipping spatial overlap check for {filepath} '
                f'because of infinite bounding box {bounding_box}')
            continue

        bounding_boxes.append(bounding_box)
        checked_file_list.append(filepath)

    try:
        pygeoprocessing.merge_bounding_box_list(bounding_boxes, 'intersection')
    except ValueError as error:
        LOGGER.debug(error)
        formatted_lists = _format_bbox_list(checked_file_list, bounding_boxes)
        return get_message('BBOX_NOT_INTERSECT').format(bboxes=formatted_lists)
    return None


def _format_bbox_list(file_list, bbox_list):
    """Format two lists of equal length into one string."""
    return ' | '.join(
            [a + ': ' + str(b) for a, b in zip(
                file_list, bbox_list)])


def validate(args, model_spec):
    """Validate an args dict against a model spec.

    Validates an arguments dictionary according to the rules laid out in
    ``spec``.

    Args:
        args (dict): The InVEST model args dict to validate.
        model_spec (dict): The InVEST model spec dict to validate against.

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
        input_spec.id: args.get(input_spec.id, False) for input_spec in model_spec.inputs}
    keys_with_falsey_values = set()
    for parameter_spec in model_spec.inputs:
        key = parameter_spec.id
        required = parameter_spec.required

        if isinstance(required, str):
            required = bool(utils.evaluate_expression(
                expression=f'{parameter_spec.required}',
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
            (sorted(missing_keys), get_message('MISSING_KEY')))

    if required_keys_with_no_value:
        validation_warnings.append(
            (sorted(required_keys_with_no_value), get_message('MISSING_VALUE')))

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
            parameter_spec = copy.deepcopy(model_spec.get_input(key))
        except KeyError:
            LOGGER.debug(f'Provided key {key} does not exist in MODEL_SPEC')
            continue

        # rewrite parameter_spec for any nested, conditional validity
        axis_keys = set(dir(parameter_spec)).intersection({'columns', 'rows', 'fields', 'contents'})

        if axis_keys:
            for axis_key in axis_keys:
                if getattr(parameter_spec, axis_key) is None:
                    continue
                for nested_spec in getattr(parameter_spec, axis_key):
                    if (isinstance(nested_spec.required, str)):
                        nested_spec.required = (
                            bool(utils.evaluate_expression(
                                nested_spec.required, expression_values)))
        try:
            # pass the entire arg spec into the validation function as kwargs
            # each type validation function allows extra kwargs with **kwargs
            warning_msg = parameter_spec.validate(args[key])
            if warning_msg:
                validation_warnings.append(([key], warning_msg))
                invalid_keys.add(key)
        except Exception:
            LOGGER.exception(f'Error when validating key {key} with value {args[key]}')
            validation_warnings.append(([key], get_message('UNEXPECTED_ERROR')))

    # Phase 3: Check spatial overlap if applicable
    if model_spec.validate_spatial_overlap:

        # validate_spatial_overlap can be True, meaning validate all spatial keys,
        # or a list, representing a subset of keys to validate
        if isinstance(model_spec.validate_spatial_overlap, list):
            spatial_keys = set(model_spec.validate_spatial_overlap)
        else:
            spatial_keys = set()
            for i in model_spec.inputs:
                if i.type in['raster', 'vector']:
                    spatial_keys.add(i.id)

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

            spatial_overlap_error = check_spatial_overlap(
                spatial_files, model_spec.different_projections_ok)
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
        # which gets imported into itself here and fails, so skip it.
        model_module = importlib.import_module(validate_func.__module__)
        if 'test_validation' in model_module.__name__:
            warnings_ = validate_func(args, limit_to)
            return warnings_

        if limit_to is None:
            LOGGER.info('Starting whole-model validation with MODEL_SPEC')
            warnings_ = validate_func(args)
        else:
            LOGGER.info('Starting single-input validation with MODEL_SPEC')
            args_key_spec = model_module.MODEL_SPEC.get_input(limit_to)

            args_value = args[limit_to]
            error_msg = None

            # We're only validating a single input.  This is not officially
            # supported in the validation function, but we can make it work
            # within this decorator.
            # If 'required' is an expression, we can't
            # validate that outside of whole-model validation.
            if args_key_spec.required is True:
                if args_value in ('', None):
                    error_msg = "Value is required"

            # If the input is not required and does not have a value, no
            # need to validate it.
            if args_value not in ('', None):
                error_msg = args_key_spec.validate(args_value)

            if error_msg is None:
                warnings_ = []
            else:
                warnings_ = [([limit_to], error_msg)]

        LOGGER.debug(f'Validation warnings: {pprint.pformat(warnings_)}')
        return warnings_

    return _wrapped_validate_func


def args_enabled(args, model_spec):
    """Get enabled/disabled status of arg fields given their values and spec.

    Args:
        args (dict): Dict mapping arg keys to user-provided values
        model_spec (dict): MODEL_SPEC dictionary

    Returns:
        Dictionary mapping each arg key to a boolean value - True if the
        arg field should be enabled, False otherwise
    """
    enabled = {}
    expression_values = {
        arg_spec.id: args.get(arg_spec.id, False) for arg_spec in model_spec.inputs}
    for arg_spec in model_spec.inputs:
        if isinstance(arg_spec.allowed, str):
            enabled[arg_spec.id] = bool(utils.evaluate_expression(
                arg_spec.allowed, expression_values))
        else:
            enabled[arg_spec.id] = True
    return enabled
