import contextlib
import collections
import inspect
import logging
import pprint

from osgeo import gdal


#: A flag to pass to the validation context manager indicating that all keys
#: should be checked.
CHECK_ALL_KEYS = None
LOGGER = logging.getLogger(__name__)


def require(key, args, warnings_):
    """Append an error to warnings_ when a required key is missing.

    A warning is recorded about a required key when either of these conditions
    are met:

        * The key is missing from ``args``.
        * The key's value is one of ``''`` or ``None``.

    Parameters:
        key (string): The string key to index into args.
        args (dict): The full args dict.
        warnings_ (list): the warnings list.

    Returns:
        ``None``
    """
    try:
        if args[key] in ('', None):
            raise KeyError
    except KeyError:
        warnings_.append((key, 'Key is required'))


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


def build_validation_contextmanager(warnings, limit_to):
    def _test_validity(*keys):
        return test_validity(keys, warnings=warnings, limit_to=limit_to)
    return _test_validity


@contextlib.contextmanager
def test_validity(target_keys, warnings, limit_to):
    def _add_to_warnings(message, keys=target_keys, exit=False):
        warnings.append((keys, message))
        if exit:
            raise ValueError()

    if limit_to in tuple(target_keys) + (None,):
        try:
            yield _add_to_warnings
        except KeyError as missing_key:
            _add_to_warnings('Args key %s is required.' % missing_key)
        except ValueError:
            pass


class ValidatorContext(object):
    class Validator(object):
        def __init__(self, key, args, limit_to, warnings):
            self.args = args
            self.warnings = warnings
            self.limit_to = limit_to
            self.key = key

        def warn(self, message, keys=None):
            if not keys:
                keys = (self.key,)
            self.warnings.append((keys, message))

        def require(self, *keys):
            for key in keys:
                if key not in self.args or self.args[key] in ('', None):
                    self.warn('Args key %s is required' % key, keys=(key,))

    def __init__(self, args, limit_to):
        self.args = args
        self.limit_to = limit_to
        self.warnings = []

    def check(self, key, require=False):
        if key not in self.args:
            return None
        elif self.limit_to not in (key, None):
            return None
        elif require and self.args[key] in ('', None):
            return None
        else:
            return self.Validator(key, self.args, self.limit_to, self.warnings)

    def warn(self, message, keys):
        self.warnings.append((keys, message))


class ValidationContext(object):
    def __init__(self, warnings, limit_to):
        self._warnings = warnings
        self._limit_to = limit_to
        self._keys = None

    def add_warning(self, message, keys=None):
        if not keys:
            keys = self._keys
        self._warnings.append((keys, message))

    def __call__(self, *keys):
        self._keys = keys
        if self._limit_to in tuple(self._keys) + (None,):
            return self
        return False

    def __enter__(self):
        return self.add_warning

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is exc_value is traceback is None:
            return
        elif exc_type is StopIteration:
            return True
        else:
            # simulate LOGGER.exception since we're not actually within an
            # exception handler.
            LOGGER.error('Exception encountered while validating keys %s',
                         self._keys, exc_info=(exc_type, exc_value, traceback))
            if exc_type is KeyError:
                self.add_warning(
                    [exc_value], 'Args key %s is required' % exc_value)


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

        _wrapped_validate_func.context = ValidatorContext(args, limit_to)
        validation_warnings = validate_func(args, limit_to)
        LOGGER.debug('Validation warnings: %s',
                     pprint.pformat(validation_warnings))

        assert isinstance(validation_warnings, list), (
            'validate function must return a list of 2-tuples.')
        for keys_iterable, error_string in validation_warnings:
            assert (isinstance(keys_iterable, collections.Iterable) and not
                    isinstance(keys_iterable, basestring)), (
                        'Keys entry %s must be a non-string iterable' % (
                            keys_iterable))
            for key in keys_iterable:
                assert key in args, 'Key %s (from %s) must be in args.' % (
                    key, keys_iterable)
            assert isinstance(error_string, basestring), (
                'Error string must be a string, not a %s' % type(error_string))
        return validation_warnings

    return _wrapped_validate_func
