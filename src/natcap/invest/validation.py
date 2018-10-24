"""Common validation utilities for InVEST models."""
import collections
import inspect
import logging
import pprint


#: A flag to pass to the validation context manager indicating that all keys
#: should be checked.
CHECK_ALL_KEYS = None
MESSAGE_REQUIRED = 'Parameter is required but is missing or has no value'
LOGGER = logging.getLogger(__name__)


class ValidationContext(object):
    """An object to represent a validation context.

    A validation context reduces the amount of boilerplate code needed within
    an InVEST validation function to produce validation warnings that are
    consistent with the InVEST Validation API.
    """
    def __init__(self, args, limit_to):
        """Create a ValidationContext object.

        Parameters:
            args (dict): The args dict to validate.
            limit_to (string or None): If a string, the args key that should be
                validated.  If ``None``, all args key-value pairs will be
                validated.
        """
        self.args = args
        self.limit_to = limit_to
        self.warnings = []

    def warn(self, message, keys):
        """Record a warning in the internal warnings list.

        Parameters:
            message (string): The message of the warning to log.
            keys (iterable): An iterable of string keys that the message
                refers to.
        """
        if isinstance(keys, basestring):
            keys = (keys,)
        keys = tuple(sorted(keys))
        self.warnings.append((keys, message))

    def is_arg_complete(self, key, require=False):
        """Test if a given argument is complete and should be validated.

        An argument is complete if:

            * The value associated with ``key`` is neither ``''`` or ``None``
            * The key-value pair is in ``self.args``
            * The key should be validated (the key matches the value of
              ``self.limit_to`` or ``self.limit_to == None``)

        If the argument is incomplete and ``require == True``, a warning is
        recorded in the ValidationContext's warnings list.

        Parameters:
            key (string): The key to test.
            require=False (bool): Whether the parameter is required.

        Returns:
            A ``bool``, indicating whether the argument is complete.
        """
        try:
            value = self.args[key]
            if isinstance(value, basestring):
                value = value.strip()
        except KeyError:
            value = None

        if (value in ('', None) or
                self.limit_to not in (key, None)):
            if require:
                self.warn(
                    'Parameter is required but is missing or has no value',
                    keys=(key,))
            return False
        return True


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

        for key, value in args.iteritems():
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
