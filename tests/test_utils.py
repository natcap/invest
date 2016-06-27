"""Module for testing the natcap.invest.utils module.."""
import unittest


class UtilsTests(unittest.TestCase):
    """Tests for natcap.invest.utils."""

    def test_suffix_string(self):
        """Utils: test suffix_string."""
        pass

    def test_exp_decay_kernel_raster(self):
        """Utils: test exponential_decay_kernel_raster."""
        pass

    def test_build_file_registry(self):
        """Utils: test build_file_registry."""
        from natcap.invest import utils

        base_dict = {'foo': 'bar', 'baz': '/bart/bam.txt'}
        file_registry = utils.build_file_registry([(base_dict, '')], '')

        self.assertEqual(base_dict, file_registry)
