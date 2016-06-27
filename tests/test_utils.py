"""Module for testing the natcap.invest.utils module.."""
import unittest
import os
import tempfile
import shutil

from pygeoprocessing.testing import scm
import pygeoprocessing.testing


class SuffixUtilsTests(unittest.TestCase):
    """Tests for natcap.invest.utils.make_suffix_string."""

    def test_suffix_string(self):
        """Utils: test suffix_string."""
        from natcap.invest import utils

        args = {'foo': 'bar', 'file_suffix': 'suff'}
        suffix = utils.make_suffix_string(args, 'file_suffix')
        self.assertEqual(suffix, '_suff')

    def test_suffix_string_underscore(self):
        """Utils: test suffix_string underscore."""
        from natcap.invest import utils

        args = {'foo': 'bar', 'file_suffix': '_suff'}
        suffix = utils.make_suffix_string(args, 'file_suffix')
        self.assertEqual(suffix, '_suff')

    def test_suffix_string_empty(self):
        """Utils: test empty suffix_string."""
        from natcap.invest import utils

        args = {'foo': 'bar', 'file_suffix': ''}
        suffix = utils.make_suffix_string(args, 'file_suffix')
        self.assertEqual(suffix, '')

    def test_suffix_string_no_entry(self):
        """Utils: test no suffix entry in args."""
        from natcap.invest import utils

        args = {'foo': 'bar'}
        suffix = utils.make_suffix_string(args, 'file_suffix')
        self.assertEqual(suffix, '')


class FileRegistryUtilsTests(unittest.TestCase):
    """Tests for natcap.invest.utils.file_registry."""

    def test_exp_decay_kernel_raster(self):
        """Utils: test exponential_decay_kernel_raster."""
        pass

    def test_build_file_registry(self):
        """Utils: test build_file_registry on simple case."""
        from natcap.invest import utils

        base_dict = {'foo': 'bar', 'baz': '/bart/bam.txt'}
        file_registry = utils.build_file_registry([(base_dict, '')], '')

        self.assertEqual(
            FileRegistryUtilsTests._norm_dict(base_dict),
            FileRegistryUtilsTests._norm_dict(file_registry))

    def test_build_file_registry_suffix(self):
        """Utils: test build_file_registry on suffix."""
        from natcap.invest import utils

        base_dict = {'foo': 'bar', 'baz': '/bart/bam.txt'}
        file_registry = utils.build_file_registry([
            (base_dict, '')], '_suff')
        expected_dict = {
            'foo': 'bar_suff',
            'baz': '/bart/bam_suff.txt'
        }

        self.assertEqual(
            FileRegistryUtilsTests._norm_dict(expected_dict),
            FileRegistryUtilsTests._norm_dict(file_registry))

    def test_build_file_registry_list_suffix(self):
        """Utils: test build_file_registry on list of files w/ suffix."""
        from natcap.invest import utils

        base_dict = {
            'foo': ['bar', '/bart/bam.txt']
        }
        file_registry = utils.build_file_registry([
            (base_dict, '')], '_suff')
        expected_dict = {
            'foo': ['bar_suff', '/bart/bam_suff.txt']
        }

        self.assertEqual(
            FileRegistryUtilsTests._norm_dict(expected_dict),
            FileRegistryUtilsTests._norm_dict(file_registry))

    def test_build_file_registry_path(self):
        """Utils: test build_file_registry on path."""
        from natcap.invest import utils

        base_dict = {
            'foo': 'bar',
            'baz': '/bart/bam.txt',
            'jab': 'jim'
        }
        file_registry = utils.build_file_registry([
            (base_dict, 'newpath')], '')
        expected_dict = {
            'foo': 'newpath/bar',
            'jab': 'newpath/jim',
            'baz': '/bart/bam.txt',
        }

        self.assertEqual(
            FileRegistryUtilsTests._norm_dict(expected_dict),
            FileRegistryUtilsTests._norm_dict(file_registry))

    def test_build_file_registry_duppath(self):
        """Utils: test build_file_registry ValueError on duplicate paths."""
        from natcap.invest import utils

        base_dict = {
            'foo': 'bar',
            'jab': 'bar'
        }
        with self.assertRaises(ValueError):
            _ = utils.build_file_registry([
                (base_dict, 'newpath')], '')

    def test_build_file_registry_dupkeys(self):
        """Utils: test build_file_registry ValueError on duplicate keys."""
        from natcap.invest import utils

        base_dict1 = {
            'foo': 'bar',
        }
        base_dict2 = {
            'foo': 'bar2',
        }
        with self.assertRaises(ValueError):
            _ = utils.build_file_registry([
                (base_dict1, ''), (base_dict2, '')], '')

    def test_build_file_registry_invalid_value(self):
        """Utils: test build_file_registry with invalid path type."""
        from natcap.invest import utils

        base_dict = {
            'foo': 'bar',
            'baz': None
        }
        with self.assertRaises(ValueError):
            _ = utils.build_file_registry([(base_dict, 'somepath')], '')

    @staticmethod
    def _norm_dict(path_dict):
        """Take a dictionary of paths and normalize the paths."""
        result_dict = {}
        for key, path in path_dict.iteritems():
            if isinstance(path, str):
                result_dict[key] = os.path.normpath(path)
            elif isinstance(path, list):
                result_dict[key] = [
                    os.path.normpath(list_path) for list_path in path]
            else:
                raise ValueError("Unexpected path value: %s", path)
        return result_dict


class ExponentialDecayUtilsTests(unittest.TestCase):
    """Tests for natcap.invest.utils.exponential_decay_kernel_raster."""

    _REGRESSION_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'invest-test-data',
        'exp_decay_kernel')

    def setUp(self):
        """Setup workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(_REGRESSION_PATH)
    def test_exp_decay_kernel_raster(self):
        """Utils: test exponential_decay_kernel_raster."""
        from natcap.invest import utils
        expected_distance = 100  # 10 pixels
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel_100.tif')
        utils.exponential_decay_kernel_raster(
            expected_distance, kernel_filepath)

        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(
                ExponentialDecayUtilsTests._REGRESSION_PATH,
                'kernel_100.tif'), kernel_filepath, 1e-6)
