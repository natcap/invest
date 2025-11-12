import importlib
import os
import shutil
import subprocess
import unittest
from unittest.mock import MagicMock

from natcap.invest import rst_generator
from docutils.nodes import emphasis, Node, paragraph, reference, strong


class TestRSTGenerator(unittest.TestCase):

    def test_parse_rst(self):
        """parse_rst should create a correct list of docutils nodes."""
        nodes = rst_generator.parse_rst(
            '**Bar** (`number <input_types.html#number>`__, '
            'units: **m³/month**, *required*): Description')
        # should be a list of one paragraph node
        self.assertEqual(len(nodes), 1)
        self.assertEqual(type(nodes[0]), paragraph)
        # that paragraph node should have child nodes corresponding to parts
        # of the text
        nodes = nodes[0].children
        self.assertEqual(type(nodes[0]), strong)
        self.assertEqual(nodes[0].children[0], 'Bar')
        self.assertEqual(nodes[1], ' (')
        self.assertEqual(type(nodes[2]), reference)
        self.assertEqual(nodes[2].children[0], 'number')
        self.assertEqual(nodes[3], ', units: ')
        self.assertEqual(type(nodes[4]), strong)
        self.assertEqual(nodes[4].children[0], 'm³/month')
        self.assertEqual(nodes[5], ', ')
        self.assertEqual(type(nodes[6]), emphasis)
        self.assertEqual(nodes[6].children[0], 'required')
        self.assertEqual(nodes[7], '): Description')

    def test_invest_spec(self):
        """invest_spec role function should return what sphinx expects."""
        mock_inliner = MagicMock()
        mock_inliner.document.settings.env.app.config.language = 'en'
        nodes, messages = rst_generator.invest_spec(
            None, None, 'carbon discount_rate', None, mock_inliner)
        self.assertEqual(len(nodes), 2)
        for node in nodes:
            self.assertTrue(isinstance(node, Node))
        self.assertEqual(messages, [])

    def test_real_model_spec(self):
        from natcap.invest import carbon
        out = rst_generator.describe_input(
            'natcap.invest.carbon', ['carbon_pools_path', 'columns', 'lucode'])
        expected_rst = (
            '.. _carbon-pools-path-columns-lucode:\n\n' +
            '**lucode** (`integer <input_types.html#integer>`__, *required*): ' +
            carbon.MODEL_SPEC.get_input('carbon_pools_path').get_column('lucode').about
        )
        self.assertEqual(repr(out), repr(expected_rst))


if __name__ == '__main__':
    unittest.main()
