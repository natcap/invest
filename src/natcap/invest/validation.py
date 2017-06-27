import contextlib

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
def _capture_gdal_warnings(warnings_list):

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
