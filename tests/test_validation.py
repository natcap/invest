import unittest

class ValidatorTest(unittest.TestCase):
    def test_args_wrong_type(self):
        from natcap.invest import validation

        @validation.validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args=123)

    def test_limit_to_wrong_type(self):
        from natcap.invest import validation

        @validation.validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={}, limit_to=1234)

    def test_limit_to_not_in_args(self):
        from natcap.invest import validation

        @validation.validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={}, limit_to='bar')

    def test_args_keys_must_be_strings(self):
        from natcap.invest import validation

        @validation.validator
        def validate(args, limit_to=None):
            pass

        with self.assertRaises(AssertionError):
            validate(args={1: 'foo'})

    def test_invalid_return_value(self):
        from natcap.invest import validation

        for invalid_value in (1, True, None):
            @validation.validator
            def validate(args, limit_to=None):
                return invalid_value

            with self.assertRaises(AssertionError):
                validate({})

    def test_invalid_keys_iterable(self):
        from natcap.invest import validation

        @validation.validator
        def validate(args, limit_to=None):
            return [('a', 'error 1')]

        with self.assertRaises(AssertionError):
            validate({'a': 'foo'})

    def test_return_keys_in_args(self):
        from natcap.invest import validation

        @validation.validator
        def validate(args, limit_to=None):
            return [(('a',), 'error 1')]

        with self.assertRaises(AssertionError):
            validate({})

    def test_error_string_wrong_type(self):
        from natcap.invest import validation

        @validation.validator
        def validate(args, limit_to=None):
            return [(('a',), 1234)]

        with self.assertRaises(AssertionError):
            validate({'a': 'foo'})

    def test_wrong_parameter_names(self):
        from natcap.invest import validation

        @validation.validator
        def validate(foo):
            pass

        with self.assertRaises(AssertionError):
            validate({})
