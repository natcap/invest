import os
import shutil
import subprocess
import unittest
from unittest.mock import MagicMock

import investspec
from docutils.nodes import emphasis, Node, paragraph, reference, strong

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
BUILD_DIR = os.path.join(TEST_DIR, 'build')


class TestInvestSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Install mock module."""
        subprocess.run([
            'pip', 'install', os.path.join(TEST_DIR, 'test_module')])

    @classmethod
    def tearDownClass(cls):
        """Remove mock module build directory."""
        shutil.rmtree(BUILD_DIR)

    def test_parse_rst(self):
        """parse_rst should create a correct list of docutils nodes."""
        nodes = investspec.parse_rst(
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
        mock_inliner.document.settings.env.app.config.investspec_module_prefix = 'test_module'
        mock_inliner.document.settings.env.app.config.language = 'en'
        nodes, messages = investspec.invest_spec(
            None, None, 'test_module number_input', None, mock_inliner)
        self.assertEqual(len(nodes), 2)
        for node in nodes:
            self.assertTrue(isinstance(node, Node))
        self.assertEqual(messages, [])

    def test_investspec_integration(self):
        """Built html should contain generated arg documentation."""
        subprocess.run([
            'sphinx-build',
            '-W',  # fail on warning
            '-a',  # write all files, not just new or changed files
            '-b', 'html',  # build html
            TEST_DIR, BUILD_DIR])
        expected_html = (
            '<strong>Foo</strong> (<a class="reference external" '
            'href="input_types.html#number">number</a>, units: '
            '<strong>m³/month</strong>, <em>required</em>): Numbers have '
            'units that are displayed in a human-readable way.')
        with open(os.path.join(BUILD_DIR, 'index.html')) as file:
            actual_html = file.read()
        self.assertIn(expected_html, actual_html)


if __name__ == '__main__':
    unittest.main()
