import unittest


class ValidatorTest(unittest.TestCase):
    def test_args_wrong_type(self):
        """Validation: check for error when args is the wrong type."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args=123)

    def test_limit_to_wrong_type(self):
        """Validation: check for error when limit_to is the wrong type."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={}, limit_to=1234)

    def test_limit_to_not_in_args(self):
        """Validation: check for error when limit_to is not a key in args."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={}, limit_to='bar')

    def test_args_keys_must_be_strings(self):
        """Validation: check for error when args keys are not all strings."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={1: 'foo'})

    def test_invalid_return_value(self):
        """Validation: check for error when the return value type is wrong."""
        from natcap.invest import validation

        for invalid_value in (1, True, None):
            @validation.invest_validator
            def validate(args, limit_to=None):
                return invalid_value

            with self.assertRaises(AssertionError):
                validate({})

    def test_invalid_keys_iterable(self):
        """Validation: check for error when return keys not an iterable."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return [('a', 'error 1')]

        with self.assertRaises(AssertionError):
            validate({'a': 'foo'})

    def test_return_keys_in_args(self):
        """Validation: check for error when return keys not all in args."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return [(('a',), 'error 1')]

        with self.assertRaises(AssertionError):
            validate({})

    def test_error_string_wrong_type(self):
        """Validation: check for error when error message not a string."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return [(('a',), 1234)]

        with self.assertRaises(AssertionError):
            validate({'a': 'foo'})

    def test_wrong_parameter_names(self):
        """Validation: check for error when wrong function signature used."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(foo):
            pass

        with self.assertRaises(AssertionError):
            validate({})

    def test_return_value(self):
        """Validation: validation errors should be returned from decorator."""
        from natcap.invest import validation

        errors = [(('a', 'b'), 'Error!')]

        @validation.invest_validator
        def validate(args, limit_to=None):
            return errors

        validation_errors = validate({'a': 'foo', 'b': 'bar'})
        self.assertEqual(validation_errors, errors)

    def test_n_workers(self):
        """Validation: validation error returned on invalid n_workers."""
        from natcap.invest import validation

        @validation.invest_validator
        def validate(args, limit_to=None):
            return []

        validation_errors = validate({'n_workers': 1.5})
        self.assertEqual(len(validation_errors), 1)
        self.assertTrue(validation_errors[0][0] == ['n_workers'])
        self.assertTrue('must be an integer' in validation_errors[0][1])
