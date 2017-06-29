import contextlib
import collections
import inspect
import logging
import pprint
import functools

from osgeo import gdal


#: A flag to pass to the validation context manager indicating that all keys
#: should be checked.
CHECK_ALL_KEYS = None
LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def append_gdal_warnings(warnings_list):
    """Append GDAL warnings within this context manager to a list.

    Parameters:
        warnings_list (list): A list to which formatted GDAL warnings will
            be appended.

    Example:
        # Show an example here.
    """

    def _append_gdal_warnings(err_level, err_no, err_msg):
        warnings_list.append('[errno {err}] {msg}'.format(
            err=err_no, msg=err_msg.replace('\n', ' ')))

    gdal.PushErrorHandler(_append_gdal_warnings)
    yield
    gdal.PopErrorHandler()


class ValidationContext:
    def __init__(self, args, limit_to):
        self.args = args
        self.limit_to = limit_to
        self.warnings = []

    def warn(self, message, keys):
        if isinstance(keys, basestring):
            keys = (keys,)
        self.warnings.append((keys, message))

    def is_arg_complete(self, key, require=False):
        if (key not in self.args or
                self.limit_to not in (key, None)):
            return False

        try:
            value = self.args[key]
            if isinstance(value, basestring):
                value = value.strip()
        except KeyError:
            return False

        if value in ('', None) and require:
            self.require(key)
            return False
        return True

    def require(self, *keys):
        for key in keys:
            if key not in self.args or self.args[key] in ('', None):
                self.warn(('Parameter is required but is missing '
                           'or has no value'), keys=(key,))


def validator(validate_func):
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

    Raises:
        AssertionError when an invalid format is found.

    Example:
        from natcap.invest import validation
        @validation.validator
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
            assert limit_to in args, 'limit_to key must exist in args.'

        for key, value in args.iteritems():
            assert isinstance(key, basestring), (
                'All args keys must be strings.')

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
        return warnings_

    return _wrapped_validate_func
