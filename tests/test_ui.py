import unittest

import six


class UtilitiesTest(unittest.TestCase):
    def test_print_args(self):
        from natcap.invest.cli import _format_args

        args_iterable = ('foo', 'bar', 'baz')

        args_dict = {
            'some_arg': [1, 2, 3, 4],
            'foo': 'bar',
        }

        args_string = _format_args(args_iterable=args_iterable,
                                   args_dict=args_dict)
        expected_string = six.text_type(
            'Arguments:\n'
            'foo      bar\n'
            'some_arg [1, 2, 3, 4]\n')
        self.assertEqual(args_string, expected_string)
