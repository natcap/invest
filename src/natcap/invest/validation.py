import contextlib
import collections

from osgeo import gdal


#: A flag to pass to the validation context manager indicating that all keys
#: should be checked.
CHECK_ALL_KEYS = None


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


def build_validation_contextmanager(warnings, limit_to):
    def _test_validity(*keys):
        return test_validity(keys, warnings=warnings, limit_to=limit_to)
    return _test_validity


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


def validator(validate_func):
    def _wrapped_validate_func(args, limit_to=None):
        assert isinstance(args, dict), 'args parameter must be a dictionary.'
        assert (isinstance(limit_to, type(None)) or
                isinstance(limit_to, basestring)), (
                    'limit_to parameter must be either a string key or None.')
        if limit_to is not None:
            assert limit_to in args, 'limit_to key must exist in args.'

        for key, value in args.iteritems():
            assert isinstance(key, basestring), (
                'All args keys must be strings.')

        return_value = validate_func(args, limit_to)

        assert isinstance(return_value, list), (
            'validate function must return a list of 2-tuples.')
        for keys_iterable, error_string in return_value:
            assert (isinstance(keys_iterable, collections.Iterable) and not
                    isinstance(keys_iterable, basestring)), (
                        'Keys entry %s must be a non-string iterable' % (
                            keys_iterable))
            for key in keys_iterable:
                assert key in args, 'Key %s (from %s) must be in args.' % (
                    key, keys_iterable)

    return _wrapped_validate_func
