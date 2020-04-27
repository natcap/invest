"""Module for testing the natcap.invest.utils module.."""
import codecs
import unittest
import os
import tempfile
import shutil
import logging
import threading
import warnings
import re
import glob
import textwrap

from pygeoprocessing.testing import scm
import pygeoprocessing.testing
from osgeo import gdal


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
        for key, path in path_dict.items():
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
                'kernel_100.tif'), kernel_filepath, abs_tol=1e-6)


class SandboxTempdirTests(unittest.TestCase):
    def setUp(self):
        """Setup workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_sandbox_manager(self):
        from natcap.invest import utils

        with utils.sandbox_tempdir(suffix='foo',
                                   prefix='bar',
                                   dir=self.workspace_dir) as new_dir:
            self.assertTrue(new_dir.startswith(self.workspace_dir))
            basename = os.path.basename(new_dir)
            self.assertTrue(basename.startswith('bar'))
            self.assertTrue(basename.endswith('foo'))

            # trigger the exception handling for coverage.
            shutil.rmtree(new_dir)


class TimeFormattingTests(unittest.TestCase):
    def test_format_time_hours(self):
        from natcap.invest.utils import _format_time

        seconds = 3667
        self.assertEqual(_format_time(seconds), '1h 1m 7s')

    def test_format_time_minutes(self):
        from natcap.invest.utils import _format_time

        seconds = 67
        self.assertEqual(_format_time(seconds), '1m 7s')

    def test_format_time_seconds(self):
        from natcap.invest.utils import _format_time

        seconds = 7
        self.assertEqual(_format_time(seconds), '7s')


class LogToFileTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_log_to_file_all_threads(self):
        """Utils: Verify that we can capture messages from all threads."""
        from natcap.invest.utils import log_to_file

        logfile = os.path.join(self.workspace, 'logfile.txt')

        def _log_from_other_thread():
            thread_logger = logging.getLogger()
            thread_logger.info('this is from a thread')

        local_logger = logging.getLogger()

        # create the file before we log to it, so we know a warning should
        # be logged.
        with open(logfile, 'w') as new_file:
            new_file.write(' ')

        with log_to_file(logfile) as handler:
            thread = threading.Thread(target=_log_from_other_thread)
            thread.start()
            local_logger.info('this should be logged')
            local_logger.info('this should also be logged')

            thread.join()
            handler.flush()

        with open(logfile) as opened_logfile:
            messages = [msg for msg in opened_logfile.read().split('\n')
                        if msg if msg]
        self.assertEqual(len(messages), 3)

    def test_log_to_file_from_thread(self):
        """Utils: Verify that we can filter from a threading.Thread."""
        from natcap.invest.utils import log_to_file

        logfile = os.path.join(self.workspace, 'logfile.txt')

        def _log_from_other_thread():
            thread_logger = logging.getLogger()
            thread_logger.info('this should not be logged')
            thread_logger.info('neither should this message')

        local_logger = logging.getLogger()

        thread = threading.Thread(target=_log_from_other_thread)

        with log_to_file(logfile, exclude_threads=[thread.name]) as handler:
            thread.start()
            local_logger.info('this should be logged')

            thread.join()
            handler.flush()

        with open(logfile) as opened_logfile:
            messages = [msg for msg in opened_logfile.read().split('\n')
                        if msg if msg]
        self.assertEqual(len(messages), 1)


class ThreadFilterTests(unittest.TestCase):
    def test_thread_filter_same_thread(self):
        from natcap.invest.utils import ThreadFilter

        # name, level, pathname, lineno, msg, args, exc_info, func=None
        record = logging.LogRecord(
            name='foo',
            level=logging.INFO,
            pathname=__file__,
            lineno=500,
            msg='some logging message',
            args=(),
            exc_info=None,
            func='test_thread_filter_same_thread')
        filterer = ThreadFilter(threading.currentThread().name)

        # The record comes from the same thread.
        self.assertEqual(filterer.filter(record), False)

    def test_thread_filter_different_thread(self):
        from natcap.invest.utils import ThreadFilter

        # name, level, pathname, lineno, msg, args, exc_info, func=None
        record = logging.LogRecord(
            name='foo',
            level=logging.INFO,
            pathname=__file__,
            lineno=500,
            msg='some logging message',
            args=(),
            exc_info=None,
            func='test_thread_filter_same_thread')
        filterer = ThreadFilter('Thread-nonexistent')

        # The record comes from the same thread.
        self.assertEqual(filterer.filter(record), True)


class BuildLookupFromCsvTests(unittest.TestCase):
    """Tests for natcap.invest.utils.build_lookup_from_csv."""

    def setUp(self):
        """Make temporary directory for workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_build_lookup_from_csv(self):
        """utils: test build_lookup_from_csv."""
        from natcap.invest import utils
        table_str = 'a,b,foo,bar,_\n0.0,x,-1,bar,apple\n'
        table_path = os.path.join(self.workspace_dir, 'table.csv')
        with open(table_path, 'w') as table_file:
            table_file.write(table_str)
        result = utils.build_lookup_from_csv(
            table_path, 'a', to_lower=True)
        expected_dict = {
            0.0: {
                'a': 0.0,
                'b': 'x',
                'foo': -1.0,
                'bar': 'bar',
                '_': 'apple'
                },
            }
        self.assertEqual(result, expected_dict)


class MakeDirectoryTests(unittest.TestCase):
    """Tests for natcap.invest.utils.make_directories."""

    def setUp(self):
        """Make temporary directory for workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_make_directories(self):
        """utils: test that make directories works as expected."""
        from natcap.invest import utils
        directory_list = [
            os.path.join(self.workspace_dir, x) for x in [
                'apple', 'apple/pie', 'foo/bar/baz']]
        utils.make_directories(directory_list)
        for path in directory_list:
            self.assertTrue(os.path.isdir(path))

    def test_make_directories_on_existing(self):
        """utils: test that no error if directory already exists."""
        from natcap.invest import utils
        path = os.path.join(self.workspace_dir, 'foo', 'bar', 'baz')
        os.makedirs(path)
        utils.make_directories([path])
        self.assertTrue(os.path.isdir(path))

    def test_make_directories_on_file(self):
        """utils: test that value error raised if file exists on directory."""
        from natcap.invest import utils
        dir_path = os.path.join(self.workspace_dir, 'foo', 'bar')
        os.makedirs(dir_path)
        file_path = os.path.join(dir_path, 'baz')
        file = open(file_path, 'w')
        file.close()
        with self.assertRaises(OSError):
            utils.make_directories([file_path])

    def test_make_directories_wrong_type(self):
        """utils: test that ValueError raised if value not a list."""
        from natcap.invest import utils
        with self.assertRaises(ValueError):
            utils.make_directories(self.workspace_dir)


class GDALWarningsLoggingTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_log_warnings(self):
        """utils: test that we can capture GDAL warnings to logging."""
        from natcap.invest import utils

        logfile = os.path.join(self.workspace, 'logfile.txt')

        # this warning should go to stdout.
        gdal.Open('this_file_should_not_exist.tif')

        with utils.log_to_file(logfile) as handler:
            with utils.capture_gdal_logging():
                # warning should be captured.
                gdal.Open('file_file_should_also_not_exist.tif')
            handler.flush()

        # warning should go to stdout
        gdal.Open('this_file_should_not_exist.tif')

        with open(logfile) as opened_logfile:
            messages = [msg for msg in opened_logfile.read().split('\n')
                        if msg if msg]

        self.assertEqual(len(messages), 1)


class PrepareWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_prepare_workspace(self):
        """utils: test that prepare_workspace does what is expected."""
        from natcap.invest import utils

        workspace = os.path.join(self.workspace, 'foo')
        try:
            with utils.prepare_workspace(workspace,
                                         'some_model'):
                warnings.warn('deprecated', UserWarning)
                gdal.Open('file should not exist')
        except Warning as warning_raised:
            self.fail('Warning was not captured: %s' % warning_raised)

        self.assertTrue(os.path.exists(workspace))
        logfile_glob = glob.glob(os.path.join(workspace, '*.txt'))
        self.assertEqual(len(logfile_glob), 1)
        self.assertTrue(
            os.path.basename(logfile_glob[0]).startswith('InVEST-some_model'))
        with open(logfile_glob[0]) as logfile:
            logfile_text = logfile.read()
            # all the following strings should be in the logfile.
            expected_string = 'file should not exist: No such file or directory'
            self.assertTrue(expected_string in logfile_text)  # gdal error captured
            self.assertEqual(len(re.findall('WARNING', logfile_text)), 1)
            self.assertTrue('Elapsed time:' in logfile_text)


class BuildLookupFromCSVTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_key_not_in_header(self):
        """utils: test that ValueError is raised when key not in header."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace, 'csv.csv')
        with open(csv_file, 'w') as file_obj:
            file_obj.write(textwrap.dedent(
                """header1,header2,header3
                1,2,3
                4,FOO,bar
                """
            ))

        with self.assertRaises(ValueError):
            utils.build_lookup_from_csv(csv_file, 'some_key')

    def test_results_lowercase_non_numeric(self):
        """utils: text handling of converting to lowercase."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace, 'csv.csv')
        with open(csv_file, 'w') as file_obj:
            file_obj.write(textwrap.dedent(
                """
                header1,HEADER2,header3
                1,2,bar
                4,5,FOO
                """
            ).strip())

        lookup_dict = utils.build_lookup_from_csv(
            csv_file, 'header1', to_lower=True)

        self.assertEqual(lookup_dict[4]['header3'], 'foo')
        self.assertEqual(lookup_dict[1]['header2'], 2)

    def test_results_uppercase_numeric_cast(self):
        """utils: test handling of uppercase, num. casting, blank values."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace, 'csv.csv')
        with open(csv_file, 'w') as file_obj:
            file_obj.write(textwrap.dedent(
                """
                header1,HEADER2,header3,missing_column,
                1,2,3,
                4,FOO,bar,
                """
            ).strip())

        lookup_dict = utils.build_lookup_from_csv(
            csv_file, 'header1', to_lower=False)

        self.assertEqual(lookup_dict[4]['HEADER2'], 'FOO')
        self.assertEqual(lookup_dict[4]['header3'], 'bar')
        self.assertEqual(lookup_dict[1]['header1'], 1)

    def test_csv_dialect_detection_semicolon_delimited(self):
        """utils: test that we can parse semicolon-delimited CSVs."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace, 'csv.csv')
        with open(csv_file, 'w') as file_obj:
            file_obj.write(textwrap.dedent(
                """
                header1;HEADER2;header3;
                1;2;3;
                4;FOO;bar;
                """
            ).strip())

        lookup_dict = utils.build_lookup_from_csv(
            csv_file, 'header1', to_lower=False)

        self.assertEqual(lookup_dict[4]['HEADER2'], 'FOO')
        self.assertEqual(lookup_dict[4]['header3'], 'bar')
        self.assertEqual(lookup_dict[1]['header1'], 1)

    def test_csv_utf8_bom_encoding(self):
        """utils: test that CSV read correctly with UTF-8 BOM encoding."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace, 'csv.csv')
        # writing with utf-8-sig will prepend the BOM
        with open(csv_file, 'w', encoding='utf-8-sig') as file_obj:
            file_obj.write(textwrap.dedent(
                """
                header1,HEADER2,header3
                1,2,bar
                4,5,FOO
                """
            ).strip())
        # confirm that the file has the BOM prefix
        with open(csv_file, 'rb') as file_obj:
            self.assertTrue(file_obj.read().startswith(codecs.BOM_UTF8))

        lookup_dict = utils.build_lookup_from_csv(
            csv_file, 'header1')
        # assert the BOM prefix was correctly parsed and skipped
        self.assertEqual(lookup_dict[4]['header2'], 5)
        self.assertEqual(lookup_dict[4]['header3'], 'foo')
        self.assertEqual(lookup_dict[1]['header1'], 1)

    def test_csv_latin_1_encoding(self):
        """utils: test that CSV read correctly with Latin-1 encoding."""
        from natcap.invest import utils

        csv_file = os.path.join(self.workspace, 'csv.csv')
        with codecs.open(csv_file, 'w', encoding='iso-8859-1') as file_obj:
            file_obj.write(textwrap.dedent(
                """
                header 1,HEADER 2,header 3
                1,2,bar1
                4,5,FOO
                """
            ).strip())

        lookup_dict = utils.build_lookup_from_csv(
            csv_file, 'header 1')

        self.assertEqual(lookup_dict[4]['header 2'], 5)
        self.assertEqual(lookup_dict[4]['header 3'], 'foo')
        self.assertEqual(lookup_dict[1]['header 1'], 1)
