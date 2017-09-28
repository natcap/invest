# TODO: this test module is so short, can you roll it into `test_ui_inputs.py`?
import unittest

import six


class UtilitiesTest(unittest.TestCase):
    def test_print_args(self):
        from natcap.invest.scenarios import format_args_dict

        args_dict = {
            'some_arg': [1, 2, 3, 4],
            'foo': 'bar',
        }

        args_string = format_args_dict(args_dict=args_dict)
        expected_string = six.text_type(
            'Arguments:\n'
            'foo      bar\n'
            'some_arg [1, 2, 3, 4]\n')
        self.assertEqual(args_string, expected_string)
