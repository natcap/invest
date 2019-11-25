"""General InVEST tests."""

import unittest
import os


class FileRegistryTests(unittest.TestCase):
    """Tests for the InVEST file registry builder."""

    def test_build_file_registry_duplicate_paths(self):
        """InVEST test that file registry recognizes duplicate paths."""
        from natcap.invest import utils
        with self.assertRaises(ValueError):
            utils.build_file_registry(
                [({'a': 'a.tif'}, ''), ({'b': 'a.tif'}, '')], '')

    def test_build_file_registry_duplicate_keys(self):
        """InVEST test that file registry recognizes duplicate keys."""
        from natcap.invest import utils
        with self.assertRaises(ValueError):
            utils.build_file_registry(
                [({'a': 'a.tif'}, ''), ({'a': 'b.tif'}, '')], '')

    def test_build_file_registry(self):
        """InVEST test a complicated file registry creation."""
        from natcap.invest import utils

        dict_a = {
            'a': 'aggregated_results.shp',
            'b': 'P.tif',
            '': 'CN.tif',
            'l_avail_path': ''}

        dict_b = {
            'apple': '.shp',
            'bear': 'tif',
            'cat': 'CN.tif'}

        dict_c = {}

        result = utils.build_file_registry(
            [(dict_a, ''), (dict_b, 'foo'), (dict_c, 'garbage')], '')

        expected_dict = {
            'a': 'aggregated_results.shp',
            'b': 'P.tif',
            '': 'CN.tif',
            'l_avail_path': '',
            'apple': os.path.join('foo', '.shp'),
            'bear': os.path.join('foo', 'tif'),
            'cat': os.path.join('foo', 'CN.tif'),
            }

        unexpected_paths = []
        for key, result_path in expected_dict.items():
            expected_path = os.path.normpath(result[key])
            if os.path.normpath(result_path) != expected_path:
                unexpected_paths.append(
                    (key, expected_path, os.path.normpath(result_path)))

        extra_keys = set(result.keys()).difference(set(expected_dict.keys()))

        if len(unexpected_paths) > 0 or len(extra_keys) > 0:
            raise AssertionError(
                "Unexpected paths or keys: %s %s" % (
                    str(unexpected_paths), str(extra_keys)))
