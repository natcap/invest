import json
import os
import shutil
import tempfile
import unittest

from natcap.invest.file_registry import FileRegistry
from natcap.invest import spec
import numpy


class FileRegistryTests(unittest.TestCase):

    def setUp(self):
        """Overriding setUp func. to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_basic_file_registry(self):
        """Test basic functionality of file registry."""
        f_reg = FileRegistry([
            spec.FileOutput(
                id='foo',
                path='foo_result.txt'
            ),
            spec.FileOutput(
                id='bar',
                path='intermediate/bar_result.txt'
            )
        ], self.workspace_dir)
        # indexing adds items to the registry
        self.assertEqual(f_reg.registry, {})
        self.assertEqual(
            f_reg['foo'], os.path.join(self.workspace_dir, 'foo_result.txt'))
        self.assertEqual(f_reg.registry, {
            'foo': os.path.join(self.workspace_dir, 'foo_result.txt')
        })
        self.assertEqual(
            f_reg['bar'], os.path.join(self.workspace_dir, 'intermediate', 'bar_result.txt'))
        self.assertEqual(f_reg.registry, {
            'foo': os.path.join(self.workspace_dir, 'foo_result.txt'),
            'bar': os.path.join(self.workspace_dir, 'intermediate', 'bar_result.txt')
        })

    def test_file_registry_invalid_key(self):
        """Test error when indexing by invalid key."""
        f_reg = FileRegistry([
            spec.FileOutput(
                id='foo',
                path='foo_result.txt'
            ),
            spec.FileOutput(
                id='bar',
                path='intermediate/bar_result.txt'
            )
        ], self.workspace_dir)
        with self.assertRaises(KeyError):
            _ = f_reg['x']
        self.assertEqual(f_reg.registry, {})

    def test_file_registry_error_duplicate_outputs(self):
        """Test error when outputs have duplicate ids or paths."""
        with self.assertRaises(ValueError):  # error on duplicate ids
            FileRegistry([
                spec.FileOutput(
                    id='foo',
                    path='foo_result.txt'
                ),
                spec.FileOutput(
                    id='foo',
                    path='intermediate/bar_result.txt'
                )
            ], self.workspace_dir)

        with self.assertRaises(ValueError):  # error on duplicate paths
            FileRegistry([
                spec.FileOutput(
                    id='foo',
                    path='foo_result.txt'
                ),
                spec.FileOutput(
                    id='bar',
                    path='foo_result.txt'
                )
            ], self.workspace_dir)

    def test_file_registry_with_patterns(self):
        """Test indexing and registry with file patterns."""
        f_reg = FileRegistry([
            spec.FileOutput(
                id='foo_[VAR]',
                path='foo_[VAR]_result.txt'
            ),
            spec.FileOutput(
                id='[X]_[Y]_[Z]',
                path='[Z]-[Y]-[X].txt'
            ),
        ], self.workspace_dir)

        self.assertEqual(f_reg['foo_[VAR]', '1'],
            os.path.join(self.workspace_dir, 'foo_1_result.txt'))
        self.assertEqual(f_reg['foo_[VAR]', '2'],
            os.path.join(self.workspace_dir, 'foo_2_result.txt'))
        self.assertEqual(f_reg['[X]_[Y]_[Z]', 'foo', 'bar', 'baz'],
            os.path.join(self.workspace_dir, 'baz-bar-foo.txt'))
        print(dict(f_reg.registry))
        self.assertEqual(f_reg.registry, {
            'foo_[VAR]': {
                '1': os.path.join(self.workspace_dir, 'foo_1_result.txt'),
                '2': os.path.join(self.workspace_dir, 'foo_2_result.txt')
            },
            '[X]_[Y]_[Z]': {
                'foo': {
                    'bar': {
                        'baz': os.path.join(
                            self.workspace_dir, 'baz-bar-foo.txt')
                    }
                }
            }
        })

    def test_file_registry_invalid_indexing(self):
        """Test errors on indexing with the wrong number of keys."""
        f_reg = FileRegistry([
            spec.FileOutput(
                id='foo',
                path='foo_result.txt'
            ),
            spec.FileOutput(
                id='foo_[VAR]',
                path='foo_[VAR]_result.txt'
            )
        ], self.workspace_dir)
        with self.assertRaises(KeyError):
            _ = f_reg['foo', 'x']
        with self.assertRaises(KeyError):
            _ = f_reg['foo_[VAR]']
        with self.assertRaises(KeyError):
            _ = f_reg['foo_[VAR]', 'x', 'y']

    def test_file_registry_jsonifiable(self):
        """Test that file registry dict can be converted to JSON."""
        f_reg = FileRegistry([
            spec.FileOutput(
                id='foo',
                path='foo_result.txt'
            ),
            spec.FileOutput(
                id='foo_[A]_[B]',
                path='foo_[A]_[B]_result.txt'
            )
        ], self.workspace_dir)
        # Keys should be cast to strings
        _ = f_reg['foo']
        _ = f_reg['foo_[A]_[B]', 'bar', numpy.float32(1.5)]
        self.assertEqual(
            json.dumps(f_reg.registry),
            json.dumps({
                'foo': os.path.join(self.workspace_dir, 'foo_result.txt'),
                'foo_[A]_[B]': {
                    'bar': {
                        '1.5': os.path.join(self.workspace_dir, 'foo_bar_1.5_result.txt')
                    }
                }
            }))
